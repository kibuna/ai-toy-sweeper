"""Rust 版の env（native/src/env.rs）が Python 版（toy_sweeper.env.ToySweeperEnv、Sim2D）と同じ観測・報酬・終わり方・
info・乱数の状態になる。

全 Stage、設定の組（基本、壁なし、v21 の不完全さ、車体の接触の減点なし、遅延と車輪の利得と対象外の失敗、復習あり）で、
同じ Seed から reset し、台本（MixedHeuristicPolicy）と乱数の行動を混ぜて進める。終わったら Seed なしで reset して続ける
（乱数は env のまま）。毎歩、観測（f32 のビット）、報酬（f64 のビット）、terminated / truncated、info（鍵の順・型・値）、
乱数の状態（state・inc・32 ビットの残り）を比べる。
"""
import dataclasses
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils import seeding

import toy_sweeper_native as nat
from toy_sweeper.config import load_all
from toy_sweeper.env import ToySweeperEnv
from toy_sweeper.geometry import Pose2D
from toy_sweeper.native import cfg_json, episode_dict, rng_state, seed_state
from toy_sweeper.policies import MixedHeuristicPolicy
from toy_sweeper.sim2d import ObjectSpec
from toy_sweeper.spaces import N_ACTIONS, Action, ObjectType
from toy_sweeper.task import STAGES, TaskSampler

TEST_CFG = Path(__file__).resolve().parent / "configs"   # テストだけで使う設定の差し替え

EXP = Path(__file__).resolve().parents[1] / "config" / "experiments"
STEPS = 250
# 通した場面の数
_COVER = {"steps": 0, "episodes": 0, "success": 0, "out_of_area": 0, "non_target_moved": 0, "truncated": 0,
          "grasps": 0, "grasp_fails": 0, "occluded": 0, "noise_draws": 0, "chassis_contact": 0, "delivered": 0,
          "rehearsal": 0, "latency": 0}


def _replace(cfg, **kw):
    sim, task = cfg.sim, cfg.task
    if "walls" in kw:
        sim = dataclasses.replace(sim, walls=dataclasses.replace(sim.walls, enabled=kw["walls"]))
    if "effects" in kw:
        sim = dataclasses.replace(sim, real_world_effects=dataclasses.replace(sim.real_world_effects, **kw["effects"]))
    if "reward" in kw:
        task = dataclasses.replace(task, reward=dataclasses.replace(task.reward, **kw["reward"]))
    if "episode" in kw:
        task = dataclasses.replace(task, episode=dataclasses.replace(task.episode, **kw["episode"]))
    return dataclasses.replace(cfg, sim=sim, task=task)


def _configs():
    base = load_all()
    return {
        "base": base,
        "no_walls": _replace(base, walls=False),
        "imperfect": load_all(overrides=(EXP / "imperfect.yaml",)),
        "no_chassis_penalty": load_all(overrides=(EXP / "no_chassis_penalty.yaml",)),
        # 遅延と車輪の利得（0.025 s 刻みの丸めの境目も通る範囲）、対象外を動かしすぎたら失敗、壁なし、
        # 時間切れを通すための短いエピソード
        "effects": _replace(base, walls=False, effects={"command_latency_s": (0.0, 0.1), "wheel_gain": (0.85, 1.15)},
                            reward={"non_target_failure": True, "non_target_tolerance_m": 0.04},
                            episode={"max_steps": 40, "max_steps_per_extra_target": 15}),
        # v21 の不完全さ + 遅延 + 壁なし
        "imperfect_effects": _replace(load_all(overrides=(EXP / "imperfect.yaml",)), walls=False,
                                      effects={"command_latency_s": (0.0, 0.05), "wheel_gain": (0.9, 1.1)}),
        "rehearsal": base,
        # v25: 縁までの距離を 4 隅で測る（不完全さあり）
        "corner_edge": load_all(overrides=(EXP / "imperfect.yaml", EXP / "corner_edge.yaml")),
        # v26: 後片付けへの手がかり（主対象の切り替え、近づく項と直すべき距離）
        "cleanup_guidance": load_all(overrides=(EXP / "imperfect.yaml", EXP / "corner_edge.yaml", EXP / "cleanup_guidance.yaml")),
        # 爪がふさがれたら止まる（sim2d.jaw_stall）
        "jaw_stall": load_all(overrides=(EXP / "imperfect.yaml", EXP / "cleanup_guidance.yaml", TEST_CFG / "jaw_stall.yaml")),
        # v28: Stage 7・8 の対象にミニカーが混ざる
        "multi_toy_car": load_all(overrides=(EXP / "imperfect.yaml", EXP / "cleanup_guidance.yaml", EXP / "multi_toy_car.yaml")),
        # v28: 時間の罰を重く、開閉の罰
        "efficient": load_all(overrides=(EXP / "imperfect.yaml", EXP / "cleanup_guidance.yaml", EXP / "efficient.yaml")),
        # v29: Stage 7・8 の対象を近くに固める
        "target_group": load_all(overrides=(EXP / "imperfect.yaml", EXP / "cleanup_guidance.yaml", EXP / "multi_toy_car.yaml",
                                            EXP / "efficient.yaml", TEST_CFG / "target_group.yaml")),
        # v30: 盤面ごとの作業領域、壁なしで外へ 0.10 m まで、観測に作業領域の大きさ
        "free_area": load_all(overrides=(EXP / "imperfect.yaml", EXP / "cleanup_guidance.yaml", EXP / "multi_toy_car.yaml",
                                         EXP / "efficient.yaml", EXP / "free_area.yaml")),
        # v30: 実機の速さに合わせて 8 Hz・速さ 2 倍、遅れと左右の利得の DR
        "robot_8hz": load_all(overrides=(EXP / "imperfect.yaml", EXP / "cleanup_guidance.yaml", EXP / "multi_toy_car.yaml",
                                         EXP / "efficient.yaml", EXP / "free_area.yaml", EXP / "robot_8hz.yaml")),
        # v31: 実測した車体・速さ・物体、速さの倍率の DR
        "hw_1022": load_all(overrides=(EXP / "imperfect.yaml", EXP / "cleanup_guidance.yaml", EXP / "multi_toy_car.yaml",
                                       EXP / "efficient.yaml", EXP / "free_area.yaml", EXP / "robot_8hz.yaml",
                                       EXP / "hw_1022.yaml")),
        # 全対象が入れば成功（ゴールの対象外・置き場から出た対象外を問わない）、歩数の上限 2 倍
        "targets_only": load_all(overrides=(EXP / "imperfect.yaml", EXP / "cleanup_guidance.yaml", EXP / "multi_toy_car.yaml",
                                            EXP / "efficient.yaml", EXP / "free_area.yaml", EXP / "robot_8hz.yaml",
                                            EXP / "hw_1022.yaml", EXP / "steps_x2.yaml", TEST_CFG / "targets_only.yaml")),
        # v34: 全対象が入れば成功 + 終わりに残った直すべき物の減点
        "no_push_in": load_all(overrides=(EXP / "imperfect.yaml", EXP / "cleanup_guidance.yaml", EXP / "multi_toy_car.yaml",
                                          EXP / "efficient.yaml", EXP / "free_area.yaml", EXP / "robot_8hz.yaml",
                                          EXP / "hw_1022.yaml", TEST_CFG / "no_push_in.yaml")),
        # v35: 届けた対象に接した「入れてしまった」対象外
        # v36: 旋回の惰性（v32_s0_x2 の設定 + motor_lag.yaml）
        "motor_lag": load_all(overrides=(EXP / "imperfect.yaml", EXP / "cleanup_guidance.yaml", EXP / "multi_toy_car.yaml",
                                         EXP / "efficient.yaml", EXP / "free_area.yaml", EXP / "robot_8hz.yaml",
                                         EXP / "hw_1022.yaml", EXP / "steps_x2.yaml", EXP / "motor_lag.yaml")),
        "wedged": load_all(overrides=(EXP / "imperfect.yaml", EXP / "cleanup_guidance.yaml", EXP / "multi_toy_car.yaml",
                                      EXP / "efficient.yaml", EXP / "free_area.yaml", EXP / "robot_8hz.yaml",
                                      EXP / "hw_1022.yaml", TEST_CFG / "wedged.yaml")),
    }


_CFGS = {}


def cfgs():
    if not _CFGS:
        _CFGS.update(_configs())
    return _CFGS


def canon(x):
    """比べる形: float は hex（-0.0 と 0.0 も区別）、bool と int も区別、dict は鍵の順も比べる。"""
    if isinstance(x, bool):
        return ("b", x)
    if isinstance(x, float):
        return ("f", x.hex())
    if isinstance(x, dict):
        return ("d", tuple((k, canon(v)) for k, v in x.items()))
    if isinstance(x, (tuple, list)):
        return tuple(canon(v) for v in x)
    assert x is None or type(x) in (int, str), type(x)
    return x


def _instrument(py: ToySweeperEnv):
    """数えるための包み（Python 版の動きは変えない）。"""
    occ, fails = py._occluded, py.backend._grasp_fails

    def occluded(robot, p):
        r = occ(robot, p)
        _COVER["occluded"] += r
        return r

    def grasp_fails(name):
        r = fails(name)
        _COVER["grasp_fails" if r else "grasps"] += 1
        return r
    py._occluded = occluded
    py.backend._grasp_fails = grasp_fails


def _same_obs(po, ro, where):
    assert ro.dtype == np.float32 and po.dtype == np.float32, where
    assert np.array_equal(po.view(np.uint32), ro.view(np.uint32)), (where, np.flatnonzero(po != ro), po, ro)


def _same_reset(py, rs, po, pinfo, ro, rinfo, where):
    _same_obs(po, ro, where)
    assert canon(pinfo) == canon(rinfo), where
    assert rs.rng_state() == rng_state(py.np_random), where
    assert rs.latency_s == py.latency_s and rs.wheel_gain == py.wheel_gain, where
    assert rs.max_steps() == py.max_steps(), where
    _COVER["episodes"] += 1
    _COVER["latency"] += py.latency_s > 0


def _run(cfg, py, rs, po, steps, seed, where):
    """steps 歩進める（終わったら両方とも Seed なしで reset）。"""
    policy = MixedHeuristicPolicy(cfg)
    rng = np.random.default_rng(seed)
    noise = cfg.sim.real_world_effects.obs_position_noise_m != 0 or cfg.sim.real_world_effects.obs_yaw_noise_rad != 0
    for k in range(steps):
        a = int(rng.integers(N_ACTIONS)) if rng.random() < 0.3 else int(policy.predict(po))
        w = (*where, k, a)
        try:
            po, pr, pterm, ptrunc, pinfo = py.step(a)
        except (RuntimeError, ValueError) as e:
            with pytest.raises(type(e)):
                rs.step(a)
            pytest.fail(f"Python 版が例外を出した（Rust 版も同じ種類の例外）: {w}: {e}")
        ro, rr, rterm, rtrunc, rinfo = rs.step(a)
        _same_obs(po, ro, w)
        assert type(rr) is float and rr.hex() == float(pr).hex(), (w, pr, rr)
        assert (rterm, rtrunc) == (pterm, ptrunc), w
        assert canon(rinfo) == canon(pinfo), (w, pinfo, rinfo)
        assert rs.rng_state() == rng_state(py.np_random), w
        _COVER["steps"] += 1
        _COVER["noise_draws"] += 3 * (len(py.episode.objects) + 1) if noise else 0
        _COVER["chassis_contact"] += pinfo["n_chassis_contact"]
        _COVER["delivered"] += pinfo["reward_terms"]["delivery"] > 0
        if pterm or ptrunc:
            _COVER["success"] += pinfo["is_success"]
            _COVER["truncated"] += ptrunc
            if pinfo["failure"]:
                _COVER[pinfo["failure"]] += 1
            po, pinfo = py.reset()
            ro, rinfo = rs.reset()
            _same_reset(py, rs, po, pinfo, ro, rinfo, (*w, "reset"))
    return po


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("name", ["base", "no_walls", "imperfect", "no_chassis_penalty", "effects", "imperfect_effects",
                                  "rehearsal", "corner_edge", "cleanup_guidance", "jaw_stall", "multi_toy_car", "efficient",
                                  "target_group", "free_area", "robot_8hz", "hw_1022", "motor_lag"])
def test_env_matches_python(name, stage):
    cfg = cfgs()[name]
    py = ToySweeperEnv(cfg, stage)
    rs = nat.Env(cfg_json(cfg), stage)
    if name == "rehearsal" and stage > 0:
        # rl.curriculum と同じ形: 前の Stage を確率 0.5 で
        prev = tuple(STAGES[:stage])
        py.set_curriculum(stage, prev, 0.5)
        rs.set_curriculum(stage, list(prev), 0.5)
    _instrument(py)
    seed = 1000 * stage + sum(map(ord, name))
    po, pinfo = py.reset(seed=seed)
    ro, rinfo = rs.reset(seed_state(seed))
    _same_reset(py, rs, po, pinfo, ro, rinfo, (name, stage, "reset"))
    _run(cfg, py, rs, po, STEPS, seed, (name, stage))
    if name == "rehearsal":
        _COVER["rehearsal"] += 1


def _stage9_forced(cfg, delivered):
    sp = dataclasses.replace(cfg.task.sampling, stage9_mistake_probability=1.0,
                             stage9_all_delivered_probability=delivered)
    return dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, sampling=sp))


@pytest.mark.parametrize("delivered", [0.0, 1.0])
@pytest.mark.parametrize("name", ["base", "imperfect", "imperfect_effects", "corner_edge", "cleanup_guidance", "jaw_stall",
                                  "targets_only", "no_push_in", "wedged"])
def test_env_stage9_mistakes_match_python(name, delivered):
    """Stage 9（v24）でやり残しを必ず加える盤面: 置き場の指定（入れてしまった・出してしまった・置き直した対象）に従う
    観測・報酬・終わり方・info が一致する。盤面を渡す reset（episode_dict に home_overrides）も同じ。"""
    cfg = _stage9_forced(cfgs()[name], delivered)
    py = ToySweeperEnv(cfg, 9)
    rs = nat.Env(cfg_json(cfg), 9)
    _instrument(py)
    seed = 9100 + int(delivered) + sum(map(ord, name))
    po, pinfo = py.reset(seed=seed)
    ro, rinfo = rs.reset(seed_state(seed))
    _same_reset(py, rs, po, pinfo, ro, rinfo, (name, delivered, "reset"))
    assert py.episode.home_overrides
    _run(cfg, py, rs, po, STEPS, seed, (name, delivered))
    for s in range(seed, seed + 3):
        g = seeding.np_random(s)[0]
        ep = TaskSampler(cfg.sim, cfg.task).sample(9, g)
        assert ep.home_overrides
        po, pinfo = py.reset(seed=s, options={"episode": ep})
        ro, rinfo = rs.reset_with_episode(episode_dict(ep), seed_state(s))
        _same_reset(py, rs, po, pinfo, ro, rinfo, (name, delivered, "episode", s))
        _run(cfg, py, rs, po, 30, s, (name, delivered, "episode", s))


@pytest.mark.parametrize("stage", STAGES)
def test_reset_with_episode(stage):
    """盤面を渡す reset: Python の盤面と、盤面を引いた直後の乱数の状態を渡すと、Seed からの reset と同じになる。"""
    cfg = cfgs()["imperfect"]
    py = ToySweeperEnv(cfg, stage)
    seed = 7000 + stage
    po, pinfo = py.reset(seed=seed)
    g = seeding.np_random(seed)[0]
    ep = TaskSampler(cfg.sim, cfg.task).sample(stage, g)
    assert ep == py.episode
    rs = nat.Env(cfg_json(cfg), 0)   # Stage は盤面が決める
    ro, rinfo = rs.reset_with_episode(episode_dict(ep), rng_state(g))
    _same_reset(py, rs, po, pinfo, ro, rinfo, ("episode", stage))
    _run(cfg, py, rs, po, 60, seed, ("episode", stage))


def test_coverage():
    """上の一致の試験が、成功・失敗（両方の種類）・時間切れ・つかむ・挟み損ね・隠れ・観測ノイズ・復習を通している。"""
    print(f"\n一致: {_COVER}")
    if _COVER["steps"] == 0:
        pytest.skip("一致の試験が走っていない")
    for k in ("success", "out_of_area", "non_target_moved", "truncated", "grasps", "grasp_fails", "occluded",
              "chassis_contact", "delivered", "latency"):
        assert _COVER[k] >= 3, (k, _COVER)
    assert _COVER["noise_draws"] > 0 and _COVER["rehearsal"] > 0, _COVER


# ---------- 例外 ----------
def _ep_with(cfg, objects, robot=Pose2D(0.0, 0.0, 0.0)):
    ep = TaskSampler(cfg.sim, cfg.task).sample(3, np.random.default_rng(0))
    return dataclasses.replace(ep, robot=robot, objects=tuple(objects), target_names=(objects[0].name,), zones=())


def test_out_of_area_with_walls_raises_like_python():
    """壁があるのに物体が外にいたら、Python 版と同じく RuntimeError（盤面と行動の列を付ける）。"""
    cfg = cfgs()["base"]
    ep = _ep_with(cfg, [ObjectSpec("a", ObjectType.LEGO, Pose2D(0.1, 0.1, 0.0), "red"),
                        ObjectSpec("b", ObjectType.LEGO, Pose2D(0.45, 0.0, 0.0), "red")])
    py = ToySweeperEnv(cfg, 3)
    py.reset(seed=1, options={"episode": ep})
    rs = nat.Env(cfg_json(cfg), 3)
    rs.reset_with_episode(episode_dict(ep), seed_state(1))
    with pytest.raises(RuntimeError, match="作業領域の外"):
        py.step(0)
    with pytest.raises(RuntimeError, match=r"壁があるのに \['b'\] が作業領域の外にいる\nepisode:"):
        rs.step(0)


def test_obs_out_of_range_raises_like_python():
    """観測が値域 ±3 の外なら、Python 版と同じく ValueError。"""
    cfg = cfgs()["no_walls"]
    ep = _ep_with(cfg, [ObjectSpec("a", ObjectType.LEGO, Pose2D(0.1, 0.1, 0.0), "red"),
                        ObjectSpec("b", ObjectType.LEGO, Pose2D(1.5, 0.0, 0.0), "red")])
    py = ToySweeperEnv(cfg, 3)
    with pytest.raises(ValueError, match="値域"):
        py.reset(seed=1, options={"episode": ep})
    rs = nat.Env(cfg_json(cfg), 3)
    with pytest.raises(ValueError, match="値域"):
        rs.reset_with_episode(episode_dict(ep), seed_state(1))


def test_errors():
    cfg = cfgs()["base"]
    rs = nat.Env(cfg_json(cfg), 1)
    with pytest.raises(RuntimeError, match="reset"):
        rs.step(0)
    with pytest.raises(ValueError, match="Stage"):
        nat.Env(cfg_json(cfg), 10)
    with pytest.raises(ValueError, match="Stage"):
        rs.set_curriculum(2, [10], 0.5)
    with pytest.raises(ValueError, match="復習"):
        rs.set_curriculum(2, [], 0.5)
    with pytest.raises(ValueError, match="復習"):
        rs.set_curriculum(2, [1], 1.5)
    rs.reset(seed_state(0))
    for a in (-1, N_ACTIONS):
        with pytest.raises(ValueError, match="not a valid Action"):
            rs.step(a)
        with pytest.raises(ValueError):
            Action(a)
    bad = episode_dict(_ep_with(cfg, [ObjectSpec("a", ObjectType.LEGO, Pose2D(0.1, 0.1, 0.0), "red")]))
    with pytest.raises(ValueError, match="対象"):
        rs.reset_with_episode({**bad, "target_names": ["x"]}, seed_state(0))
    with pytest.raises(ValueError, match="未知の鍵"):
        rs.reset_with_episode({**bad, "extra": 1}, seed_state(0))
    # 置き場の指定（v24）: 盤面に無い物体、重複、番号が範囲外は、Python 版と同じく ValueError
    py = ToySweeperEnv(cfg, 1)
    ep = _ep_with(cfg, [ObjectSpec("a", ObjectType.LEGO, Pose2D(0.1, 0.1, 0.0), "red")])
    for over in ((("x", -1),), (("a", -1), ("a", -1)), (("a", 1),), (("a", -2),)):
        with pytest.raises(ValueError, match="置き場"):
            py.reset(seed=0, options={"episode": dataclasses.replace(ep, home_overrides=over)})
        with pytest.raises(ValueError, match="置き場"):
            rs.reset_with_episode({**bad, "home_overrides": list(over)}, seed_state(0))


def test_python_sum():
    """報酬の和は CPython 3.12 の sum()（Neumaier の総和）と同じ。"""
    r = np.random.default_rng(0)
    for _ in range(2000):
        n = int(r.integers(0, 14))
        xs = [float(x) for x in r.choice([1.0, 1e-8, 1e8, 1e16], size=n) * r.uniform(-1, 1, size=n)]
        assert nat._py_sum(xs).hex() == float(sum(xs)).hex(), xs


@pytest.mark.parametrize("name", ["crash_v25_s2", "crash_v27_s0"])
def test_jaw_stall_crash_replay_matches_python(name):
    """爪がふさがれたら止まる（sim2d.jaw_stall）: 学習中の例外の盤面と行動の列（爪が止まる場面がある）で、Rust 版と Python 版の
    観測・報酬・終わり方がビット単位で一致する。"""
    from toy_sweeper.crash_replay import parse, reset_like_crash, rng_and_curriculum
    text = (Path(__file__).resolve().parent / "data" / f"{name}.txt").read_text()
    cfg = load_all(overrides=tuple(EXP / f for f in ("set_encoder.yaml", "chassis03.yaml", "imperfect.yaml", "gpu_ppo.yaml",
                                                    "cleanup.yaml", "corner_edge.yaml", "cleanup_guidance.yaml")) + (TEST_CFG / "jaw_stall.yaml",))
    spec, acts = parse(text)
    py = ToySweeperEnv(cfg, spec.stage)
    stalls = []
    orig = py.backend._turn_jaws
    def turn(angle):
        before = py.backend._jaw_angle
        orig(angle)
        stalls.append(py.backend._jaw_angle == before and angle != before)
    py.backend._turn_jaws = turn
    assert reset_like_crash(py, text)
    rng, cur = rng_and_curriculum(text)
    rs = nat.Env(cfg_json(cfg), spec.stage)
    rs.set_curriculum(cur[0], list(cur[1]), cur[2])
    ro, _ = rs.reset(rng)
    for k, a in enumerate(acts):
        po, pr, pt, ptr, pinfo = py.step(a)
        ro, rr, rt, rtr, rinfo = rs.step(a)
        _same_obs(po, ro, ("step", k))
        assert pr == rr and pt == rt and ptr == rtr, k
        if pt or ptr:
            break
    assert any(stalls), "この行動の列で爪が止まる場面がない（試験になっていない）"
