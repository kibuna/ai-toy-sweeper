"""おもちゃ片付けの Gymnasium 環境。物理はバックエンド（2D / Gazebo）に任せる。"""
from __future__ import annotations

import math
from typing import Protocol

import gymnasium as gym
import numpy as np

from .config import AllCfg
from .geometry import Pose2D, Rect, box_corners, dist, signed_distance_to_rect, wrap_angle
from .sim2d import ObjectSpec, Sim2D
from .spaces import (N_ACTIONS, OBJECT_TYPE_NAMES, Action, GripperState, ObservationBuilder, WorldSnapshot,
                     drive_velocity)
from .robot_model import body_center_x
from .snapshot import approach_distance, build_snapshot, fix_approach_distance, inside_goal, pending_fixes
from .task import STAGES, EpisodeSpec, TaskSampler, episode_area, gripper_point, start_home_index


class Backend(Protocol):
    """物理バックエンドの共通インターフェース（Sim2D, GazeboBackend）。"""
    def reset(self, robot: Pose2D, objects: list[ObjectSpec], gripper: GripperState) -> None: ...
    def drive(self, linear: float, angular: float, duration: float) -> None: ...
    def actuate_gripper(self, close: bool) -> None: ...
    def robot_pose(self) -> Pose2D: ...
    def object_pose(self, name: str) -> Pose2D: ...
    def gripper_state(self) -> GripperState: ...


# 指令遅延を丸める刻み。2D（1/120 s）と Gazebo（1 ms）の物理刻みの両方の整数倍
LATENCY_GRID_S = 0.025


class ToySweeperEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, cfg: AllCfg, stage: int, backend: Backend | None = None, render_mode: str | None = None):
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"未対応の render_mode: {render_mode}")
        self.cfg = cfg
        self.render_mode = render_mode
        self.backend: Backend = backend if backend is not None else Sim2D(cfg.sim)
        self.sampler = TaskSampler(cfg.sim, cfg.task)
        self.obs_builder = ObservationBuilder(cfg.sim, cfg.task)
        self.observation_space = self.obs_builder.observation_space
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.area = Rect(center=(0.0, 0.0), size=tuple(cfg.sim.work_area.size_m))   # 盤面ごとに reset で置き換える（v30）
        self.set_stage(stage)
        self.episode: EpisodeSpec | None = None
        self._observed: tuple[dict[str, Pose2D], Pose2D] | None = None

    def set_stage(self, stage: int) -> None:
        """評価用: いつもこの Stage の盤面を出す（復習なし）。"""
        self.set_curriculum(stage, (), 0.0)

    def set_curriculum(self, stage: int, previous: tuple[int, ...], rehearsal_probability: float) -> None:
        """学習用: 確率 rehearsal_probability で previous の Stage（一様）、それ以外は stage の盤面を出す。"""
        for s in (stage, *previous):
            if s not in STAGES:
                raise ValueError(f"未定義の Stage: {s}")
        if not 0.0 <= rehearsal_probability <= 1.0 or (rehearsal_probability > 0 and not previous):
            raise ValueError(f"復習の設定が不正: previous={previous}, p={rehearsal_probability}")
        self.stage, self._previous, self._rehearsal = stage, tuple(previous), rehearsal_probability

    # ---------- 状態の読み出し ----------
    def _spec(self, name: str) -> ObjectSpec:
        return next(o for o in self.episode.objects if o.name == name)

    def _is_target(self, name: str) -> bool:
        return name in self.episode.target_names

    def _noisy(self, p: Pose2D) -> Pose2D:
        """観測ノイズ（real_world_effects）。報酬と判定には使わない。"""
        e = self.cfg.sim.real_world_effects
        if e.obs_position_noise_m == 0 and e.obs_yaw_noise_rad == 0:
            return p
        n = self.np_random.normal(size=3)
        return Pose2D(p.x + e.obs_position_noise_m * n[0], p.y + e.obs_position_noise_m * n[1],
                      p.yaw + e.obs_yaw_noise_rad * n[2])

    def _inside_goal(self, spec: ObjectSpec, pose: Pose2D) -> bool:
        return inside_goal(self.cfg.sim, self.episode.goal, spec.type, pose)

    def _approach_to(self, robot: Pose2D, target: Pose2D) -> float:
        return approach_distance(self.cfg.sim, self.cfg.task, self.episode.goal, robot, target)

    def snapshot_from(self, poses: dict[str, Pose2D], robot: Pose2D,
                      extra_obstacles: tuple = ()) -> WorldSnapshot:
        """観測した姿勢（真値＋ノイズ、またはカメラの追跡）から WorldSnapshot を作る（中身は snapshot.build_snapshot）。

        poses に無い対象外の物体は渡さない（見えていない）。extra_obstacles は、物体名と結び付いていない追跡（SlotObject）。
        主対象（v7）は、全対象がゴールに入ったら前の主対象を持ち越すので env が覚えておく。
        直前 1 ステップの移動は、前回の観測したロボットの姿勢との差（v9）。
        """
        prev = self._last_obs_robot if self._last_obs_robot is not None else robot
        c, sn = math.cos(prev.yaw), math.sin(prev.yaw)
        dx, dy = robot.x - prev.x, robot.y - prev.y
        motion = (c * dx + sn * dy, -sn * dx + c * dy, wrap_angle(robot.yaw - prev.yaw))
        self._last_obs_robot = robot
        time_left = (self.max_steps() - self._steps) / self.max_steps()
        snap, self._last_primary = build_snapshot(self.episode, self.cfg.sim, self.cfg.task, poses, robot,
                                                  self.backend.gripper_state(), self._prev_action, self._last_primary,
                                                  extra_obstacles=tuple(extra_obstacles), obs_age_s=0.0,
                                                  robot_motion=motion, time_left=time_left)
        return snap

    def _occluded(self, robot: Pose2D, p: Pose2D) -> bool:
        """仮想の斜め上のカメラから見て、車体の陰か閉じた爪の間にある（task.observation.occlusion、v21）。"""
        oc = self.cfg.task.observation.occlusion
        sim = self.cfg.sim
        if self.backend.gripper_state() == GripperState.CLOSED:
            if dist(gripper_point(sim, robot), p.xy) < oc.held_radius_m:
                return True
        cx, cy, cz = oc.camera_xyz_m
        bx, by = sim.robot.body_size_m[0] / 2 + oc.margin_m, sim.robot.body_size_m[1] / 2 + oc.margin_m
        bc = body_center_x(sim)
        c, s = math.cos(robot.yaw), math.sin(robot.yaw)
        zo = oc.object_mid_height_m
        for z in np.linspace(zo, oc.robot_height_m, 4):       # カメラから物体への線が、車体の箱を通るか
            t = (cz - z) / (cz - zo)
            x, y = cx + t * (p.x - cx), cy + t * (p.y - cy)
            lx, ly = c * (x - robot.x) + s * (y - robot.y), -s * (x - robot.x) + c * (y - robot.y)
            if abs(lx - bc) <= bx and abs(ly) <= by:
                return True
        return False

    def _snapshot(self) -> WorldSnapshot:
        b = self.backend
        poses = {o.name: self._noisy(b.object_pose(o.name)) for o in self.episode.objects}
        robot = self._noisy(b.robot_pose())
        if self.cfg.task.observation.occlusion.enabled:
            # 隠れた物体は最後に見えた位置のまま（現場の追跡と同じ）。一度も見えていなければ今の位置（開始時の同定で見えている）
            for n, p in poses.items():
                if self._occluded(robot, p) and n in self._last_seen:
                    poses[n] = self._last_seen[n]
                else:
                    self._last_seen[n] = p
        self._observed = (dict(poses), robot)
        return self.snapshot_from(poses, robot)

    def observed_poses(self) -> tuple[dict[str, Pose2D], Pose2D]:
        """直近の Observation を作った (物体名 → 姿勢, ロボットの姿勢)。観測ノイズの設定があればノイズ込み。

        全状態を使う計画器（toy_sweeper.planner）が、Observation と同じ観測値を読むため。
        """
        if self._observed is None:
            raise RuntimeError("reset() の前に観測を読もうとした")
        poses, robot = self._observed
        return dict(poses), robot

    def _true_poses(self) -> dict[str, Pose2D]:
        return {o.name: self.backend.object_pose(o.name) for o in self.episode.objects}

    # ---------- 判定とポテンシャル（v9） ----------
    def _entry_distance(self, spec: ObjectSpec, pose: Pose2D) -> float:
        """物体の 4 隅がすべてゴールに入るまでの距離（4 隅のうち最も外の隅の、ゴールまでの距離。入っていれば 0）。"""
        size = self.cfg.sim.objects[OBJECT_TYPE_NAMES[spec.type]].size_m[:2]
        return max(max(0.0, signed_distance_to_rect(c, self.episode.goal)) for c in box_corners(pose, size))

    def _fix_distance(self, spec: ObjectSpec, pose: Pose2D) -> float:
        """対象外の「直すべき距離」（v26）: 置き場を持つ物体は 4 隅のうちいちばん外の角の置き場の縁からのはみ出し、
        持たない物体は中心のゴールの中への深さ。直すべきでなければ 0。"""
        home = self._homes[spec.name]
        if home is not None:
            size = self.cfg.sim.objects[OBJECT_TYPE_NAMES[spec.type]].size_m[:2]
            return max(max(0.0, signed_distance_to_rect(c, home)) for c in box_corners(pose, size))
        return max(0.0, -signed_distance_to_rect(pose.xy, self.episode.goal))

    def true_robot_yaw(self) -> float:
        """観測のノイズを掛けない真の向き（評価の旋回の切り返しの回数に使う、evaluate.run_episode）。"""
        return self.backend.robot_pose().yaw

    def _progress_distance(self) -> float:
        """Stage 0 はグリッパー中心→指定点、それ以外は全対象の _entry_distance の合計（ゴール中心までではない）。
        cleanup_shaping（v26）では、対象外の _fix_distance の合計を足す。"""
        ep = self.episode
        if not ep.target_names:
            return dist(gripper_point(self.cfg.sim, self.backend.robot_pose()), ep.reach_point)
        d = sum(self._entry_distance(self._spec(n), self.backend.object_pose(n)) for n in ep.target_names)
        if self.cfg.task.reward.cleanup_shaping:
            d += sum(self._fix_distance(o, self.backend.object_pose(o.name)) for o in ep.objects
                     if not self._is_target(o.name))
        return d

    def _approach_distance(self) -> float:
        """未配達の対象の接近点までの距離の最小。Stage 0 は 0。全対象配達済みは 0、cleanup_shaping（v26）では
        直すべき対象外（snapshot.pending_fixes）の接近点までの距離の最小（無ければ 0）。"""
        ep = self.episode
        robot = self.backend.robot_pose()
        left = [n for n in ep.target_names if not self._inside_goal(self._spec(n), self.backend.object_pose(n))]
        if left:
            return min(self._approach_to(robot, self.backend.object_pose(n)) for n in left)
        if not ep.target_names or not self.cfg.task.reward.cleanup_shaping:
            return 0.0
        poses = self._true_poses()
        ret, rem = pending_fixes(ep, self.cfg.sim, poses)
        if not ret and not rem:
            return 0.0
        return min(fix_approach_distance(self.cfg.sim, self.cfg.task, ep.goal, self._homes[n], robot, poses[n])
                   for n in ret + rem)

    def _n_delivered(self) -> int:
        return sum(self._inside_goal(self._spec(n), self.backend.object_pose(n)) for n in self.episode.target_names)

    def _started_in_goal(self, spec: ObjectSpec) -> bool:
        """もともとゴールの中にあった対象外か = 置き場が今のゴール（番号 0）か（v24）。置き場の指定（home_overrides）が
        無ければ、これまでの「開始時に 4 隅がゴールの中」と同じ（start_home_index はゴールを最初に調べる）。"""
        return self._home_index[spec.name] == 0

    def _n_foreign_in_goal(self) -> int:
        """もともとゴールの外にあった対象外で、今は中心がゴール内のものの数（v9: 入れても失敗にせず、残れば成功にしない）。"""
        ep = self.episode
        return sum(not self._is_target(o.name) and not self._started_in_goal(o)
                   and ep.goal.contains(self.backend.object_pose(o.name).xy) for o in ep.objects)

    def _non_target_displacements(self) -> list[float]:
        """対象外ごとの移動の量。置き場（v18。開始時に 4 隅が入っていた今のゴールか zones）を持つ物体は、中心が置き場の
        外へはみ出した距離（中では 0）。"""
        out = []
        for o in self.episode.objects:
            if self._is_target(o.name):
                continue
            now = self.backend.object_pose(o.name)
            home = self._homes[o.name]
            if home is not None:
                out.append(max(0.0, signed_distance_to_rect(now.xy, home)))
            else:
                out.append(dist(now.xy, o.pose.xy))
        return out

    def _n_home_out(self) -> int:
        """置き場を持つ対象外のうち、外形の 4 隅が置き場に入っていないものの数（v18）。"""
        return sum(home is not None and not inside_goal(self.cfg.sim, home, self._spec(n).type, self.backend.object_pose(n))
                   for n, home in self._homes.items())

    def _non_target_near_goal(self) -> float:
        """もともとゴールの外にあった対象外の物体の、ゴールへの近さの合計（v8 の項、基本設定は無効）。"""
        ep = self.episode
        band = self.cfg.task.reward.non_target_near_goal_band_m
        return sum(max(0.0, band - max(0.0, signed_distance_to_rect(self.backend.object_pose(o.name).xy, ep.goal)))
                   for o in ep.objects if not self._is_target(o.name) and not self._started_in_goal(o))

    def _potentials(self) -> dict[str, float]:
        """ポテンシャル Φ の各項（合計が Φ）。報酬は γΦ(s′) − Φ(s)、終端では Φ(s′) = 0（v9）。"""
        rw = self.cfg.task.reward
        has_targets = bool(self.episode.target_names)
        return {
            "progress": -rw.progress_per_m * self._progress_distance(),
            "approach": -rw.approach_per_m * self._approach_distance(),
            "delivery": rw.delivery_bonus * (self._n_delivered() if has_targets else 0),
            "foreign_in_goal": -rw.foreign_in_goal_penalty * self._n_foreign_in_goal(),
            "non_target": -rw.non_target_per_m * sum(self._non_target_displacements()),
            "non_target_near_goal": -rw.non_target_near_goal_per_m * self._non_target_near_goal(),
            "home_out": -rw.home_out_penalty * self._n_home_out(),
        }

    def _is_success(self) -> bool:
        ep = self.episode
        if not ep.target_names:
            return self._progress_distance() <= self.cfg.task.success.reach_tolerance_m
        if not self.cfg.task.success.require_cleanup:
            return self._n_delivered() == len(ep.target_names)
        return self._n_delivered() == len(ep.target_names) and self._n_foreign_in_goal() == 0 and self._n_home_out() == 0

    def _failure(self) -> str | None:
        # 壁ありでは、ロボットや物体の中心が外に出るのは物理モデルの不具合。黙って失敗扱いにせず止める。
        # 壁なし（v19nw）では、外に出たら失敗。v30 は縁から out_of_area_margin_m までは失敗にしない
        outside = [n for n in ["robot", *(o.name for o in self.episode.objects)]
                   if not self.area.contains((self.backend.robot_pose() if n == "robot" else self.backend.object_pose(n)).xy)]
        if outside:
            if self.cfg.sim.walls.enabled:
                raise RuntimeError(f"壁があるのに {outside} が作業領域の外にいる\n{self._diagnostics()}")
            m = self.cfg.sim.walls.out_of_area_margin_m
            outer = Rect(center=self.area.center, size=(self.area.size[0] + 2 * m, self.area.size[1] + 2 * m))
            if any(not outer.contains((self.backend.robot_pose() if n == "robot" else self.backend.object_pose(n)).xy)
                   for n in outside):
                return "out_of_area"
        rw = self.cfg.task.reward
        if rw.non_target_failure and any(d > rw.non_target_tolerance_m for d in self._non_target_displacements()):
            return "non_target_moved"
        return None

    def max_steps(self) -> int:
        n = max(len(self.episode.target_names), 1)
        e = self.cfg.task.episode
        return e.max_steps + e.max_steps_per_extra_target * (n - 1)

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        unknown = set(options) - {"episode"}
        if unknown:
            raise ValueError(f"未知の reset option: {unknown}")
        ep = options.get("episode")
        if ep is None:
            stage = self.stage
            if self._previous and self.np_random.random() < self._rehearsal:
                stage = self._previous[int(self.np_random.integers(len(self._previous)))]
            ep = self.sampler.sample(stage, self.np_random)
        names = [o.name for o in ep.objects]
        unknown = [n for n, _ in ep.home_overrides if n not in names]
        if unknown:
            raise ValueError(f"置き場の指定の物体 {unknown} が盤面の物体にない")
        if len({n for n, _ in ep.home_overrides}) != len(ep.home_overrides):
            raise ValueError(f"置き場の指定に重複がある: {ep.home_overrides}")
        bad = [(n, i) for n, i in ep.home_overrides if not -1 <= i <= len(ep.zones)]
        if bad:
            raise ValueError(f"置き場の番号が範囲外（−1〜{len(ep.zones)}）: {bad}")
        if ep.area_size is not None and self.cfg.sim.walls.enabled and tuple(ep.area_size) != tuple(self.cfg.sim.work_area.size_m):
            raise ValueError(f"壁ありで、盤面の作業領域 {ep.area_size} が sim.work_area と違う（壁は盤面ごとに作り直さない）")
        self.episode = ep
        self.area = Rect(center=(0.0, 0.0), size=episode_area(self.cfg.sim, ep))
        # 対象外の置き場（v18）。開始時の位置か、盤面の指定（v24）で決まるので、エピソードの間は変わらない
        rects = (ep.goal, *ep.zones)
        self._home_index = {o.name: start_home_index(self.cfg.sim, ep, o) for o in ep.objects
                            if o.name not in ep.target_names}
        self._homes = {n: (None if i < 0 else rects[i]) for n, i in self._home_index.items()}
        if hasattr(self.backend, "rng"):    # 挟み損ね（Sim2D、v21）の乱数。Seed から決まる
            self.backend.rng = self.np_random
        self.backend.reset(ep.robot, list(ep.objects), GripperState.OPEN)
        self._last_seen: dict[str, Pose2D] = {}
        # 実機の誤差（Domain Randomization）をエピソードごとに引く。盤面を引いた後なので、同じ Seed なら盤面は変わらない
        e = self.cfg.sim.real_world_effects
        self.latency_s = round(float(self.np_random.uniform(*e.command_latency_s)) / LATENCY_GRID_S) * LATENCY_GRID_S
        self.wheel_gain = (float(self.np_random.uniform(*e.wheel_gain)), float(self.np_random.uniform(*e.wheel_gain)))
        # 左右共通の速さの倍率（v31）。幅 0 なら乱数を引かない（今までの設定で乱数の列を変えない）
        lo, hi = e.speed_gain
        sg = float(self.np_random.uniform(lo, hi)) if lo != hi else lo
        self.wheel_gain = (self.wheel_gain[0] * sg, self.wheel_gain[1] * sg)
        # 車輪の速さが指令に追いつく時定数（v36）。幅 0 なら乱数を引かない
        lo, hi = e.motor_lag_s
        tau = float(self.np_random.uniform(lo, hi)) if lo != hi else lo
        if hasattr(self.backend, "motor_lag_s"):
            self.backend.motor_lag_s = tau
        elif tau > 0:
            raise ValueError(f"このバックエンド（{type(self.backend).__name__}）には旋回の惰性（motor_lag_s {tau}）を入れられない")
        self._applied_cmd = (0.0, 0.0)  # 車輪に効いている（遅延後の）指令
        self._prev_action = Action.STOP
        self._history: list[Action] = []
        self._steps = 0
        self._last_primary = ep.target_names[0] if ep.target_names else None
        self._last_obs_robot = None
        self._phi = self._potentials()
        self._last_terms: dict = {}
        return self.obs_builder.build(self._snapshot()), {"stage": ep.stage}

    def step(self, action):
        assert self.episode is not None, "reset() が先"
        a = Action(int(action))
        c, rw = self.cfg, self.cfg.task.reward
        before_gripper = self.backend.gripper_state()
        redundant = False
        T = c.task.control.action_duration_s
        # 状態の変わる開閉（v28 の gripper_toggle_penalty）
        toggled = a in (Action.OPEN_GRIPPER, Action.CLOSE_GRIPPER) and before_gripper != (
            GripperState.CLOSED if a == Action.CLOSE_GRIPPER else GripperState.OPEN)
        if a in (Action.OPEN_GRIPPER, Action.CLOSE_GRIPPER):
            want = GripperState.CLOSED if a == Action.CLOSE_GRIPPER else GripperState.OPEN
            redundant = before_gripper == want
            if self.latency_s > 0 and self._applied_cmd != (0.0, 0.0):
                self.backend.drive(*self._applied_cmd, self.latency_s)  # 停止が届くまで前の指令が続く
            self.backend.actuate_gripper(close=(a == Action.CLOSE_GRIPPER))
            self._applied_cmd = (0.0, 0.0)
        else:
            v, w = drive_velocity(c.sim.robot, a)
            cmd = self._wheel_effects(v, w)
            if self.latency_s > 0:
                self.backend.drive(*self._applied_cmd, self.latency_s)
            self.backend.drive(*cmd, T - self.latency_s)
            self._applied_cmd = cmd
        # 車体の形（アームは除く）に触れた対象の数（v19）。減点が 0 なら数えない（Gazebo のバックエンドは数えられない）
        n_chassis = (len(self.backend.chassis_contacts() & set(self.episode.target_names))
                     if c.task.reward.chassis_contact_penalty > 0 else 0)
        self._prev_action = a
        self._history.append(a)
        self._steps += 1

        failure = self._failure()
        success = failure is None and self._is_success()
        terminated = success or failure is not None
        # 形のつく項: kΦ(s′) − Φ(s)、k = reward.shaping_discount（v17 から 1。理由は config/task.yaml）。
        # 終端では Φ(s′) = 0（Ng et al. 1999）。時間切れは終端ではない
        k_phi = rw.shaping_discount
        phi_new = self._potentials()
        terms = {k: (0.0 if terminated else k_phi * phi_new[k]) - self._phi[k] for k in phi_new}
        terms.update({
            "success": rw.success_bonus if success else 0.0,
            "failure": -rw.failure_penalty if failure else 0.0,
            "redundant_gripper": -rw.redundant_gripper_penalty if redundant else 0.0,
            "chassis_contact": -rw.chassis_contact_penalty * n_chassis,
            "time": -rw.time_penalty,
            "gripper_toggle": -rw.gripper_toggle_penalty if toggled else 0.0,
            # v34: 終わったときに残っている直すべき物（ポテンシャルの foreign_in_goal・home_out は終端で戻るので）
            "leftover": (-rw.leftover_penalty * (self._n_foreign_in_goal() + self._n_home_out())
                         if terminated and rw.leftover_penalty > 0 else 0.0),
        })
        self._phi = phi_new
        self._last_terms = terms
        reward = float(sum(terms.values()))
        truncated = (not terminated) and self._steps >= self.max_steps()
        N = self._n_delivered() if self.episode.target_names else 0
        info = {"is_success": success, "failure": failure, "stage": self.episode.stage, "reward_terms": terms,
                "steps": self._steps, "n_delivered": N, "n_targets": len(self.episode.target_names),
                "n_foreign_in_goal": self._n_foreign_in_goal(), "n_home_out": self._n_home_out(), "n_chassis_contact": n_chassis}
        try:
            obs = self.obs_builder.build(self._snapshot())
        except ValueError as e:
            # 値域の外は物理の不具合のことが多い。盤面と行動の列を付けて、reset(options={"episode": ...}) で再生できるようにする
            raise ValueError(f"{e}\n{self._diagnostics()}") from e
        return obs, reward, terminated, truncated, info

    def _diagnostics(self) -> str:
        """物理の不具合を調べるための状況（例外のメッセージに付ける）。"""
        b = self.backend
        lines = [f"episode: {self.episode}", f"steps: {self._steps}", f"actions: {[int(a) for a in self._history]}",
                 f"robot: {b.robot_pose()}", f"gripper: {b.gripper_state()}"]
        for name in ("jaw_angle", "grasped_object"):
            if hasattr(b, name):
                lines.append(f"{name}: {getattr(b, name)()}")
        lines += [f"{o.name}: {b.object_pose(o.name)}" for o in self.episode.objects]
        return "\n".join(lines)

    def _wheel_effects(self, v: float, w: float) -> tuple[float, float]:
        """左右のモーター利得を掛けた実際の (v, ω)。差動二輪の運動学で左右の車輪速度に分けて掛ける。"""
        b = self.cfg.sim.robot.wheel_separation_m
        gl, gr = self.wheel_gain
        vl, vr = (v - w * b / 2) * gl, (v + w * b / 2) * gr
        return ((vl + vr) / 2, (vr - vl) / b)

    def render(self):
        if self.render_mode != "rgb_array":
            raise RuntimeError("render_mode='rgb_array' で作った環境だけ描画できる")
        from .render import draw_frame
        return draw_frame(self)
