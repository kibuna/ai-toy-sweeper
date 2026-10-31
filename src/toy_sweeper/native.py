"""学習環境の Rust 版（native/）を Python から使うための部品。

cfg_json: 設定（AllCfg の sim と task）を Rust 版に渡す JSON にする。Rust 版は知らない項目があると読む時点で例外にする。
episode_dict: EpisodeSpec を Rust 版に渡す dict にする（toy_sweeper_native.TaskSampler.sample が返すのと同じ形）。
rng_state: numpy の Generator(PCG64) の状態を Rust 版の NpRng に渡す形 (state, inc, 32 ビットの残りか None) にする。
NativeVecEnv: SB3 の VecEnv。多数の env を Rust で並列に進める。make_vec_env(..., seed=seed) + Monitor と同じ動き。
NativeEnv: 評価用の 1 つの env（ToySweeperEnv の Sim2D と同じ動き。evaluate.run_episode が使うものだけ）。
"""
from __future__ import annotations

import dataclasses
import json
import time

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from .config import AllCfg
from .sim2d import effective_friction_radius
from .spaces import N_ACTIONS, ObservationBuilder
from .task import EpisodeSpec

# info["failure"] の番号（Rust の env::Failure）
FAILURE_NAMES = ("out_of_area", "non_target_moved")


def cfg_json(cfg: AllCfg) -> str:
    sim = dataclasses.asdict(cfg.sim)
    task = dataclasses.asdict(cfg.task)
    derived = {"effective_friction_radius": {name: effective_friction_radius(o.size_m[0], o.size_m[1])
                                             for name, o in cfg.sim.objects.items()}}
    return json.dumps({"sim": sim, "task": task, "derived": derived})


def episode_dict(ep: EpisodeSpec) -> dict:
    def rect(r):
        return (tuple(r.center), tuple(r.size))
    return {
        "stage": ep.stage,
        "robot": (ep.robot.x, ep.robot.y, ep.robot.yaw),
        "objects": [(o.name, int(o.type), o.pose.x, o.pose.y, o.pose.yaw, o.color) for o in ep.objects],
        "target_names": list(ep.target_names),
        "reach_point": ep.reach_point,
        "goal_name": ep.goal_name,
        "goal": rect(ep.goal),
        "zones": [rect(z) for z in ep.zones],
        "home_overrides": [(n, i) for n, i in ep.home_overrides],
        "area_size": None if ep.area_size is None else tuple(ep.area_size),
    }


def rng_state(g: np.random.Generator) -> tuple[int, int, int | None]:
    s = g.bit_generator.state
    if s["bit_generator"] != "PCG64":
        raise ValueError(f"Rust 版の乱数は PCG64 だけ: {s['bit_generator']}")
    return s["state"]["state"], s["state"]["inc"], (s["uinteger"] if s["has_uint32"] else None)


def seed_state(seed: int) -> tuple[int, int, int | None]:
    """env.reset(seed=seed) の後の np_random の状態（gymnasium の seeding.np_random と同じ）。"""
    return rng_state(seeding.np_random(seed)[0])


class NativeEnv:
    """評価用: ToySweeperEnv(cfg, stage)（Sim2D）の Rust 版を 1 つ（scripts/evaluate_policies.py --env native）。

    evaluate.run_episode と、観測だけを見る Policy が使うもの（reset、step、_progress_distance、空間）だけを持つ。
    同じ Seed なら観測・報酬・終わり方・info がビット単位で同じ（test/test_native_env.py、test_native_evaluate.py）。
    盤面や物体の姿勢など env の中身（episode、observed_poses など）は持たない（読もうとすると AttributeError）。
    """

    def __init__(self, cfg: AllCfg, stage: int):
        import toy_sweeper_native as nat
        self.cfg = cfg
        self.stage = stage
        self._native = nat.Env(cfg_json(cfg), stage)
        self.observation_space = ObservationBuilder(cfg.sim, cfg.task).observation_space
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self._seeded = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """reset(seed=s) は gymnasium の Env.reset(seed=s) と同じ乱数（seeding.np_random(s)）。Seed なしは今の乱数のまま。"""
        if options:
            raise ValueError(f"NativeEnv は reset の options を受け付けない: {options}")
        if seed is None and not self._seeded:
            # Python 版は OS の乱数で Seed を決める。同じにできないので Seed を求める
            raise ValueError("NativeEnv の最初の reset には Seed が要る")
        self._seeded = True
        return self._native.reset(None if seed is None else seed_state(seed))

    def step(self, action):
        return self._native.step(int(action))

    def _progress_distance(self) -> float:
        """ToySweeperEnv._progress_distance() と同じ（Stage 0 は指定点まで、ほかは全対象がゴールに入るまでの距離の和）。"""
        return self._native.progress_distance()

    def true_robot_yaw(self) -> float:
        """ToySweeperEnv.true_robot_yaw() と同じ（観測のノイズを掛けない真の向き。評価の切り返しの回数に使う）。"""
        return self._native.robot_pose()[2]

    def render(self):
        raise NotImplementedError("NativeEnv は描画しない（描画は Python 版の ToySweeperEnv を使う）")


class NativeVecEnv(VecEnv):
    """n_envs 個の ToySweeperEnv（Rust 版）を threads 本のスレッドで並列に進める SB3 の VecEnv。

    make_vec_env(lambda: ToySweeperEnv(cfg, stage), n_envs, seed=seed, monitor_kwargs={"info_keywords": ("is_success",)})
    と同じ乱数・同じ動き（test/test_native_vecenv.py）。env i の最初の reset は Seed seed + i（SB3 の VecEnv.seed）、
    その後の reset（終わった env をその場で reset）は Seed なしで env の乱数のまま。
    info: 終わった env は Python の env の info 全部と Monitor の episode、terminal_observation、TimeLimit.truncated。
    終わっていない env は is_success、stage、TimeLimit.truncated だけ（SB3 と train.py が読むのはこれだけ。全部の
    dict を毎歩作ると Python の時間が env の時間を超えるため）。
    報酬は float64（SubprocVecEnv と同じ。DummyVecEnv は float32 に丸める）。
    失敗: step / reset の途中の失敗は一部の env だけ進んだ半端な状態にするので、Rust 側が止まった状態にし、以後の
    step / reset / set_curriculum は最初の失敗を付けて例外にする。Python 側の失敗（info を作る途中など）も同じく止める。
    """

    def __init__(self, cfg: AllCfg, n_envs: int, seed: int | None, threads: int, stage: int,
                 disjoint_seeds: bool = False):
        """disjoint_seeds: env i の最初の Seed を seed × n_envs + i にする（rl.training.disjoint_env_seeds）。
        False なら SB3 と同じ seed + i（Python 版との突き合わせの試験用）。"""
        import toy_sweeper_native as nat
        self._disjoint = disjoint_seeds
        if n_envs < 1 or threads < 1:
            raise ValueError(f"n_envs と threads は 1 以上: n_envs={n_envs}, threads={threads}")
        self._n = n_envs
        ob = ObservationBuilder(cfg.sim, cfg.task)
        # 最初の reset で seed() の Seed に入れ替えるので、ここの乱数は使われない（作るときに要るだけ）
        first = seed if seed is not None else 0
        self._native = nat.VecEnv(cfg_json(cfg), [seed_state(first + i) for i in range(n_envs)], stage, threads)
        self._threads = threads
        self._actions: np.ndarray | None = None
        self._error: BaseException | None = None   # Python 側の失敗（あれば以後は進めない）
        self.t_start = time.time()   # Monitor の t_start（エピソードの t はここからの秒）
        super().__init__(n_envs, ob.observation_space, gym.spaces.Discrete(N_ACTIONS))
        if seed is not None:
            # make_vec_env と同じ: 行動の空間と、最初の reset の Seed
            self.action_space.seed(seed)
        self.seed(seed)

    def seed(self, seed: int | None = None):
        if not self._disjoint:
            return super().seed(seed)
        if seed is None:
            raise ValueError("disjoint_seeds のときは Seed を決めて渡す（黙って乱数の Seed にしない）")
        self._seeds = [seed * self._n + i for i in range(self._n)]
        return self._seeds

    def _alive(self) -> None:
        if self._error is not None:
            raise RuntimeError("NativeVecEnv は前の失敗で止まっている（半端な状態なので続けない。作り直す）: "
                               f"{self._error!r}") from self._error

    # ---------- VecEnv ----------
    def reset(self):
        self._alive()
        if any(self._options):
            raise ValueError(f"NativeVecEnv は reset の options を受け付けない: {self._options}")
        states = [None if s is None else seed_state(int(s)) for s in self._seeds]
        obs, stages = self._native.reset(states)
        self.reset_infos = [{"stage": int(s)} for s in stages]
        self._reset_seeds()
        self._reset_options()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._alive()
        a = np.asarray(actions)
        if a.shape != (self._n,) or not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"actions は ({self._n},) の整数: shape {a.shape}, dtype {a.dtype}")
        self._actions = np.ascontiguousarray(a, dtype=np.int64)

    def step_wait(self):
        self._alive()
        if self._actions is None:
            raise RuntimeError("step_async が先")
        out = self._native.step(self._actions)   # 途中の失敗は Rust 側が止まった状態にする
        self._actions = None
        try:
            return self._step_result(out)
        except BaseException as e:
            # env はもう進んだので、結果を返せなかったら続けられない
            self._error = e
            raise

    def _step_result(self, out: dict):
        term, trunc = out["terminated"], out["truncated"]
        dones = term | trunc
        names = self._native.reward_term_names
        infos = []
        now = time.time()
        for i in range(self._n):
            if not dones[i]:
                infos.append({"is_success": False, "stage": int(out["stage"][i]), "TimeLimit.truncated": False})
                continue
            f = int(out["failure"][i])
            info = {"is_success": bool(out["is_success"][i]), "failure": None if f < 0 else FAILURE_NAMES[f],
                    "stage": int(out["stage"][i]),
                    "reward_terms": dict(zip(names, (float(x) for x in out["reward_terms"][i]))),
                    "steps": int(out["steps"][i]), "n_delivered": int(out["n_delivered"][i]),
                    "n_targets": int(out["n_targets"][i]), "n_foreign_in_goal": int(out["n_foreign_in_goal"][i]),
                    "n_home_out": int(out["n_home_out"][i]), "n_chassis_contact": int(out["n_chassis_contact"][i])}
            # Monitor（info_keywords=("is_success",)）
            info["episode"] = {"r": round(float(out["episode_return"][i]), 6), "l": int(out["episode_length"][i]),
                               "t": round(now - self.t_start, 6), "is_success": info["is_success"]}
            info["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
            info["terminal_observation"] = out["terminal_obs"][i].copy()
            self.reset_infos[i] = {"stage": int(out["new_stage"][i])}
            infos.append(info)
        return out["obs"], out["reward"], dones, infos

    def close(self) -> None:
        self._native = None

    def get_attr(self, attr_name: str, indices=None) -> list:
        idx = self._get_indices(indices)
        if attr_name == "render_mode":
            return [None for _ in idx]
        raise AttributeError(f"NativeVecEnv は属性 {attr_name!r} を読めない")

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        raise AttributeError(f"NativeVecEnv は属性 {attr_name!r} を書けない")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list:
        self._alive()
        idx = list(self._get_indices(indices))
        if method_name == "set_curriculum":
            stage, previous, p = _curriculum_args(*method_args, **method_kwargs)
            self._native.set_curriculum(int(stage), [int(s) for s in previous], float(p), idx)
            return [None for _ in idx]
        raise AttributeError(f"NativeVecEnv は env の関数 {method_name!r} を呼べない")

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices=None) -> list[bool]:
        # 各 env は Monitor で包んだのと同じ（episode の info を出す）
        return [issubclass(Monitor, wrapper_class) for _ in self._get_indices(indices)]

    def get_images(self):
        raise NotImplementedError("NativeVecEnv は描画しない（評価は Python 版の ToySweeperEnv を使う）")


def _curriculum_args(stage, previous, rehearsal_probability):
    return stage, previous, rehearsal_probability
