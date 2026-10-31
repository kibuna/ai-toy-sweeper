"""幾何計画の基準 Policy。毎ステップ全状態を読み、7 つの離散行動の 1 つを返す。

流れ（毎ステップ判定し直す閉ループ）:
1. 爪が閉じている: 持っている物体（姿勢から推定）があれば、行き先（ゴール内の空き / ゴール外の元の近く）へ運び、着いたら開く。
   持っていなければ開く。
2. 爪が開いている: 仕事を 1 つ選んで続ける。ゴールに入り込んだ対象外を先に出し、次に未配達の対象。
   LEGO は挟む姿勢（物体が爪の中心線上）へ計画して近づき、閉じる。ミニカーは押せる直線（ロボットの向きで
   取れる向き）の手前へ計画して近づき、ゴールの中まで真っすぐ押す。
3. 経路は行動の格子上の A*（search.py）。前の計画は、ロボットが予測どおりに動き、物体が動いていない間だけ続ける。
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import AllCfg, ConfigError
from ..geometry import Pose2D, dist, signed_distance_to_rect, wrap_angle
from ..robot_model import SIDES, body_center_x, front_x, gripper_center_x, jaw_inner_half_gap, jaw_polygons
from ..spaces import Action, GripperState, ObjectType
from .config import PlannerCfg
from .cspace import (CSpace, Footprint, Grid, KernelCache, box_polygon, convex_overlap, disc_polygon, robot_polygons,
                     transform)
from .search import GeodesicHeuristic, SearchResult, SegmentHeuristic, geodesic_field, kinematic_step, lattice_astar
from .world import Geometry, ObjState, WorldState, to_local, world_state_from_env

log = logging.getLogger("toy_sweeper.planner")

DRIVE = (Action.FORWARD, Action.BACKWARD, Action.TURN_LEFT, Action.TURN_RIGHT)


@dataclass
class Task:
    kind: str    # "grasp"（LEGO を挟みに行く）/ "push"（ミニカーを押す）
    name: str
    dest: str    # "goal"（対象をゴールへ）/ "out"（入り込んだ対象外をゴールの外へ）
    via: tuple[float, float] | None = None  # ミニカーの 2 段の押し: 1 段目で運ぶ先（ミニカー中心）
    n_via: int = 0                           # 2 段の押しの 1 段目を使った回数
    pivot: tuple[float, int] | None = None  # 端を押して回す: (押す向き, 残りの前進の回数)
    n_pivot: int = 0                         # 端を押して回した回数


@dataclass(frozen=True)
class PushLine:
    """ミニカーを押す 1 本の直線（向き theta、距離 length）。via があれば、そこまで押した後に押し直す。

    offset はミニカー中心の、ロボットの中心線からの横の位置（0 = 中心で押す。端を押して回すときは ±push.pivot_offset_m）。
    """
    theta: float
    length: float
    via: tuple[float, float] | None
    offset: float = 0.0
    pivot: bool = False


@dataclass
class Plan:
    key: tuple
    actions: list[Action]
    poses: list[Pose2D]                 # poses[i]: actions[i] を出す前の名目の姿勢
    objects: dict[str, Pose2D]          # 計画時の物体の姿勢（持っている物体は除く）
    i: int = 0


@dataclass
class EpisodeStats:
    steps: int = 0
    plans: int = 0
    plan_failures: dict = field(default_factory=dict)  # 理由 → 回数
    expansions: int = 0
    max_expansions: int = 0
    success_expansions: list = field(default_factory=list)  # 見つかった計画ごとの展開数
    plan_time_s: list = field(default_factory=list)         # 探索 1 回の時間
    step_time_s: list = field(default_factory=list)
    drop_relaxations: int = 0
    no_task_steps: int = 0
    stuck_events: int = 0
    carry_failures: int = 0

    def fail(self, reason: str) -> None:
        self.plan_failures[reason] = self.plan_failures.get(reason, 0) + 1

    def as_row(self) -> dict:
        t = np.array(self.step_time_s) if self.step_time_s else np.zeros(1)
        return {"steps": self.steps, "plans": self.plans, "expansions": self.expansions,
                "max_expansions": self.max_expansions,
                "success_expansions_max": max(self.success_expansions, default=0),
                "success_expansions": " ".join(map(str, self.success_expansions)),
                "plan_time_max_ms": float(max(self.plan_time_s, default=0.0) * 1e3),
                "plan_failures": sum(self.plan_failures.values()),
                "plan_failures_by_reason": dict(self.plan_failures),
                "drop_relaxations": self.drop_relaxations, "no_task_steps": self.no_task_steps,
                "stuck_events": self.stuck_events, "carry_failures": self.carry_failures,
                "step_time_mean_ms": float(t.mean() * 1e3), "step_time_p95_ms": float(np.percentile(t, 95) * 1e3),
                "step_time_max_ms": float(t.max() * 1e3)}


class PlannerPolicy:
    """全状態を読む幾何計画の Policy。使い方: PlannerPolicy(cfg, load_planner_cfg()).bind(env)。

    env から読むのは observed_poses()（Observation と同じ観測値）、episode（初期姿勢・対象・ゴール）、
    area（作業領域）、backend.gripper_state()（指令した開閉）だけ（world.py）。
    """
    name = "planner"

    def __init__(self, cfg: AllCfg, pcfg: PlannerCfg):
        if cfg.task.reward.non_target_failure and pcfg.foreign.max_relocation_m >= cfg.task.reward.non_target_tolerance_m:
            raise ConfigError("planner.foreign.max_relocation_m は task.reward.non_target_tolerance_m 未満にする")
        self.cfg, self.p = cfg, pcfg
        sim = cfg.sim
        self.geo = Geometry(sim, pcfg.held.lateral_tol_m)
        T = cfg.task.control.action_duration_s
        self.step_m = sim.robot.linear_speed_mps * T
        self.turn_rad = sim.robot.angular_speed_radps * T
        self.grip_x = gripper_center_x(sim)
        self.base_polys = robot_polygons(sim)
        # 押す接触の判定は、実際に開いている爪と車体だけで行う
        bx, by = sim.robot.body_size_m[0], sim.robot.body_size_m[1]
        cx = body_center_x(sim)
        self.open_polys = [np.array([(cx - bx / 2, -by / 2), (cx + bx / 2, -by / 2), (cx + bx / 2, by / 2),
                                     (cx - bx / 2, by / 2)])]
        for side in SIDES:
            self.open_polys += [np.array(q) for q in jaw_polygons(sim, side, sim.gripper.jaw_open_angle_rad)]
        self.kernels = KernelCache(pcfg.grid.resolution_m, pcfg.grid.heading_bins)
        # 外形: 爪を開く指令の後は開いた爪（開閉は 1 行動で終わる）。閉じた後は角度が物体の幅で決まり観測しないので、
        # 開いた爪と閉じた爪の両方を含む外形にする
        self.open_fp = Footprint(tuple(self.open_polys), ("open",))
        self.body_fp = Footprint((self.open_polys[0],), ("body",))  # 車体だけ（接している隣は爪でだけ押してよい）
        self.closed_fp = Footprint(tuple(self.base_polys), ("closed",))
        car_r = self.geo.radius[ObjectType.TOY_CAR]
        pp = pcfg.push
        self.push_fp = Footprint(tuple(self.open_polys) + (disc_polygon((pp.disc_x_m, 0.0), car_r + pp.disc_extra_m),),
                                 ("push",))
        self.pad = max(self.push_fp.radius, self.closed_fp.radius + 2 * max(self.geo.radius.values())) + 0.02
        self._contact_cache: dict[tuple, float] = {}
        # 開いた爪の梁の間の前後の範囲（2D の把持の判定は、閉じた瞬間に物体中心がここにあること）。両端から close_x_inset_m 内側
        a_open = sim.gripper.jaw_open_angle_rad
        xs = [x for x in np.arange(0.0, 0.2, 0.0001) if jaw_inner_half_gap(sim, a_open, float(x)) > -math.inf]
        if not xs:
            raise ConfigError("開いた爪の梁の範囲が求まらない（gripper の寸法を確認）")
        ins = pcfg.grasp.close_x_inset_m
        self.grasp_x_range = (float(xs[0]) + ins, float(xs[-1]) - ins)
        # 開いた爪の籠（中心がここにある物体は、閉じると対象と一緒に挟み込まれて引きずられる）: 前後は車体前面から閉じた爪の先、
        # 左右は開いた爪の先の内側
        self.cage_x = (front_x(sim), max(float(q[:, 0].max()) for q in self.base_polys))
        self.cage_half_y = max(jaw_inner_half_gap(sim, a_open, float(x)) for x in xs)
        self.grasp_fail: dict[str, int] = {}
        self.env = None
        self._episode = None
        self.episode_log: list[dict] = []
        self.stats: EpisodeStats | None = None

    # ---------- 接続 ----------
    def bind(self, env) -> "PlannerPolicy":
        self.env = env
        return self

    def reset(self) -> None:
        if self.env is None:
            raise RuntimeError("PlannerPolicy は bind(env) してから使う")
        if self.env.episode is None:
            raise RuntimeError("env.reset() の後に policy.reset() を呼ぶ")
        if self.env.episode.stage == 9 or self.env.episode.home_overrides:
            # 計画器は「もともとゴールの中」を開始の位置から決め、置き場（zones）も読まない。Stage 9（v24）の置き場の指定に
            # 従えないので、黙って違う判定で動かさない
            raise ValueError("計画器は Stage 9（置き場の指定 home_overrides のある盤面、v24）に対応していない")
        self.finish()
        self._episode = self.env.episode
        self.grid = Grid(self.env.area, self.pad, self.p.grid.resolution_m)
        self.walls = self.grid.outside_area()
        self.task: Task | None = None
        self.plan: Plan | None = None
        self.blocked: dict[str, int] = {}
        self.carry_role: dict[str, str] = {}
        self.carry_level: dict[str, int] = {}
        self.car_push_only: set[str] = set()
        self.t = 0
        self.history: deque = deque(maxlen=self.p.recovery.stuck_steps + 1)
        self._cs_cache: dict = {}
        self.stats = EpisodeStats()

    def finish(self) -> None:
        """直前のエピソードの統計を episode_log に移す（reset でも呼ばれる）。"""
        if self.stats is not None:
            self.episode_log.append(self.stats.as_row())
            self.stats = None

    def predict(self, obs) -> int:
        if self.env is None or self._episode is None:
            raise RuntimeError("bind(env) と reset() の後に predict() を呼ぶ")
        if self.env.episode is not self._episode:
            raise RuntimeError("env が reset されたのに policy.reset() が呼ばれていない")
        t0 = time.perf_counter()
        w = world_state_from_env(self.env)
        self._check_stuck(w)
        a = self._decide(w)
        self.history.append((w.robot, a))
        self.t += 1
        self.stats.steps += 1
        self.stats.step_time_s.append(time.perf_counter() - t0)
        return int(a)

    # ---------- 判定 ----------
    def _decide(self, w: WorldState) -> Action:
        if w.gripper not in (GripperState.OPEN, GripperState.CLOSED):
            raise RuntimeError(f"グリッパーの状態が開閉のどちらでもない: {w.gripper!r}")
        held = self.geo.held_object(w)
        if w.gripper == GripperState.CLOSED:
            if held is None:
                log.debug("t=%d 閉じているが何も持っていない → 開く", self.t)
                self.plan = None
                if self.task is not None and self.task.kind == "grasp":
                    n = self.grasp_fail.get(self.task.name, 0) + 1
                    self.grasp_fail[self.task.name] = n
                    if n >= self.p.grasp.max_failures and w.obj(self.task.name).type == ObjectType.TOY_CAR:
                        # ミニカーを挟み損ねが続く: 押す方へ切り替える（外さない）
                        self._car_to_push(self.task.name, "car_grasp_failed")
                        self.grasp_fail[self.task.name] = 0
                        self.task = None
                    elif n >= self.p.grasp.max_failures:
                        # 同じ物体で挟み損ねが続く: しばらく外す（記録する）
                        self.stats.fail("grasp_failed")
                        self.blocked[self.task.name] = self.t + self.p.recovery.blocked_retry_steps
                        self.grasp_fail[self.task.name] = 0
                        self.task = None
                return Action.OPEN_GRIPPER
            return self._carry(w, held)
        # 押している最中なら、仕事の選び直しより先に押し続けるか判定する
        if self.task is not None and self.task.kind == "push":
            car = w.obj(self.task.name)
            if self.task.pivot is not None and (self.plan is None or self.plan.i >= len(self.plan.actions)):
                # 接近の計画を終えた: 端を押して回す
                if self._pivot_forward_ok(w, car, self.task.pivot):
                    th, n = self.task.pivot
                    self.task.pivot = (th, n - 1) if n > 1 else None
                    self.plan = None
                    return Action.FORWARD
                log.debug("t=%d %s の端を押して回すのを終える", self.t, car.name)
                self.task.pivot = None
            if self._push_forward_ok(w, car, self.task.dest, self.task.via):
                self.plan = None
                return Action.FORWARD
            if self.task.via is not None and self._in_push_contact(w, car):
                # 1 段目を押し終えた（または先へ押せない）: 押し直す
                log.debug("t=%d %s の 1 段目の押しを終える", self.t, car.name)
                self.task.via = None
                self.plan = None
        task = self._current_task(w)
        if task is None:
            return self._no_task(w)
        o = w.obj(task.name)
        if task.kind == "grasp":
            if self._grasp_ready(w, o):
                self.carry_role[o.name] = task.dest
                self.plan = None
                return Action.CLOSE_GRIPPER
            a = self._follow_or_plan(("grasp", o.name), w, lambda: self._plan_grasp(w, o))
        else:
            a = self._follow_or_plan(("push", o.name), w, lambda: self._plan_push_approach(w, o, task))
        if a is None and task.kind == "grasp" and o.type == ObjectType.TOY_CAR:
            self._car_to_push(o.name, "car_grasp_unreachable")
            self.task = None
            return self._decide(w)
        if a is None:
            # 接近の計画に失敗: その仕事を外し、選び直す（外した物体は候補から消えるので、繰り返しは物体の数まで）
            self.blocked[o.name] = self.t + self.p.recovery.blocked_retry_steps
            self.task = None
            log.debug("t=%d %s への接近の計画に失敗 → %d ステップ外して選び直す", self.t, o.name,
                      self.p.recovery.blocked_retry_steps)
            return self._decide(w)
        return a

    def _needs_work(self, w: WorldState, o: ObjState) -> bool:
        if o.is_target:
            return not self.geo.inside_goal(w.goal, o)
        return self.geo.is_foreign_in_goal(w.goal, o)

    def _current_task(self, w: WorldState) -> Task | None:
        if self.task is not None:
            o = w.obj(self.task.name)
            if self._needs_work(w, o) and self.blocked.get(o.name, -1) < self.t:
                return self.task
            self.task, self.plan = None, None
        gp = Pose2D(*w.robot.transform_point((self.grip_x, 0.0)), 0.0).xy
        foreign = [o for o in w.objects if self.geo.is_foreign_in_goal(w.goal, o)]
        targets = [o for o in w.objects if o.is_target and not self.geo.inside_goal(w.goal, o)]
        group = foreign if foreign else targets
        cands = sorted((o for o in group if self.blocked.get(o.name, -1) < self.t),
                       key=lambda o: dist(gp, o.pose.xy) + 0.5 * dist(o.pose.xy, w.goal.center))
        for o in cands:
            dest = "goal" if o.is_target else "out"
            task = Task("grasp" if self._grasp_kind(o, dest) else "push", o.name, dest)
            plan_fn = ((lambda o=o: self._plan_grasp(w, o)) if task.kind == "grasp"
                       else (lambda o=o, task=task: self._plan_push_approach(w, o, task)))
            if task.kind == "push" and self._push_forward_ok(w, o, dest):
                self.task = task
                return task
            if task.kind == "grasp" and self._grasp_ready(w, o):
                self.task = task
                return task
            res = plan_fn()
            if (res is None or res.actions is None) and task.kind == "grasp" and o.type == ObjectType.TOY_CAR:
                # ミニカーを挟みに行けない（接近がふさがっている）: 押す
                self._car_to_push(o.name, "car_grasp_unreachable")
                task = Task("push", o.name, dest)
                res = self._plan_push_approach(w, o, task)
            if res is not None and res.actions is not None:
                self.task = task
                self._store_plan((task.kind, o.name), w, res.actions, exclude=None)
                return task
            self.blocked[o.name] = self.t + self.p.recovery.blocked_retry_steps
            log.debug("t=%d %s の計画に失敗（%s）。%d ステップ外す", self.t, o.name,
                      None if res is None else res.reason, self.p.recovery.blocked_retry_steps)
        return None

    def _no_task(self, w: WorldState) -> Action:
        """どの仕事も計画できない（全部が一時的に外れている）。記録し、その場で止まって待つ。"""
        self.stats.no_task_steps += 1
        log.debug("t=%d 計画できる仕事がない（外している: %s）", self.t, self.blocked)
        return Action.STOP

    def _check_stuck(self, w: WorldState) -> None:
        h = self.history
        n = self.p.recovery.stuck_steps
        if len(h) < n or self.task is None and w.gripper == GripperState.OPEN:
            return
        recent = list(h)[-n:]
        if not all(a in (Action.FORWARD, Action.BACKWARD) for _, a in recent):
            return
        moved = max(dist(recent[0][0].xy, p.xy) for p, _ in recent[1:] + [(w.robot, None)])
        if moved < self.p.recovery.stuck_motion_m:
            self.stats.stuck_events += 1
            name = self.task.name if self.task is not None else None
            log.debug("t=%d 詰まった（%d ステップで %.1f mm）。仕事 %s を外して計画を捨てる", self.t, n, moved * 1e3, name)
            if name is not None:
                self.blocked[name] = self.t + self.p.recovery.blocked_retry_steps
            self.task, self.plan = None, None
            h.clear()

    # ---------- 運ぶ ----------
    def _carry(self, w: WorldState, held: ObjState) -> Action:
        role = self.carry_role.get(held.name)
        if role is None:
            if held.is_target and not self.geo.inside_goal(w.goal, held):
                role = "goal"
            elif self.geo.is_foreign_in_goal(w.goal, held):
                role = "out"
        if role is None:
            log.debug("t=%d 運ぶ予定のない %s を持っている → 開く", self.t, held.name)
            self.plan = None
            return Action.OPEN_GRIPPER
        self.carry_role[held.name] = role
        rel = self._held_rel(w, held)
        masks = self._dest_masks(w, held, role)
        cx, cy = w.robot.transform_point(rel[:2])
        level = self.carry_level.get(held.name, 0)
        r, c = self.grid.cell(cx, cy)
        if masks[level][0][r, c]:
            log.debug("t=%d %s を置く（%s、空き %.3f m）", self.t, held.name, role, masks[level][1])
            self.plan = None
            self.carry_role.pop(held.name, None)
            self.carry_level.pop(held.name, None)
            return Action.OPEN_GRIPPER
        a = self._follow_or_plan(("carry", held.name, role), w, lambda: self._plan_carry(w, held, rel, masks),
                                 exclude=held.name)
        if a is None:
            # 運べない: その場で離し、その物体をしばらく外す（記録する）
            self.stats.carry_failures += 1
            self.blocked[held.name] = self.t + self.p.recovery.blocked_retry_steps
            self.carry_role.pop(held.name, None)
            log.debug("t=%d %s を運ぶ経路がない → その場で離す", self.t, held.name)
            return Action.OPEN_GRIPPER
        return a

    def _held_rel(self, w: WorldState, held: ObjState) -> tuple[float, float, float]:
        lx, ly = to_local(w.robot, held.pose.xy)
        # キャッシュのため 2 mm / 0.05 rad に丸める（外形の判定は hard_margin_m で余裕を持つ）
        return (round(lx / 0.002) * 0.002, round(ly / 0.002) * 0.002,
                round(wrap_angle(held.pose.yaw - w.robot.yaw) / 0.05) * 0.05)

    def _dest_masks(self, w: WorldState, o: ObjState, role: str) -> list[tuple[np.ndarray, float]]:
        """置き場所（物体中心を置いてよい画素）の候補を、他の物体との空きの広い順に返す（空の段階も含む）。"""
        g = self.grid
        X, Y = np.meshgrid(g.xs, g.ys)
        r = self.geo.radius[o.type]
        goal, area = w.goal, w.area
        others = [box_polygon(q.pose, self.geo.size[q.type]) for q in w.objects if q.name != o.name]
        occ = g.raster(others)
        dist_px = cv2.distanceTransform((1 - occ).astype(np.uint8), cv2.DIST_L2, 5) * g.res
        dx = np.abs(X - goal.center[0]) - goal.size[0] / 2
        dy = np.abs(Y - goal.center[1]) - goal.size[1] / 2
        if role == "goal":
            e = r + self.p.drop.goal_edge_margin_m
            base = (dx <= -e) & (dy <= -e)
        else:
            f = self.p.foreign
            out_d = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
            e = r + f.wall_margin_m
            in_area = ((np.abs(X - area.center[0]) <= area.size[0] / 2 - e)
                       & (np.abs(Y - area.center[1]) <= area.size[1] / 2 - e))
            near_start = np.hypot(X - o.initial.x, Y - o.initial.y) <= f.max_relocation_m
            base = (out_d >= r + f.out_clearance_m) & in_area & near_start
        out = []
        for clr in self.p.drop.object_clearance_m:
            m = base & (dist_px >= r + clr)
            if role == "goal":
                # 詰め方: まずゴールの縁に沿った帯（edge_band_m）に置き、真ん中を後の対象のために空けておく
                band = m & ((dx >= -e - self.p.drop.edge_band_m) | (dy >= -e - self.p.drop.edge_band_m))
                out.append((band.astype(np.uint8), clr))
            out.append((m.astype(np.uint8), clr))
        return out

    def _plan_carry(self, w: WorldState, held: ObjState, rel, masks) -> SearchResult | None:
        fp = Footprint(tuple(self.closed_fp.polys) + (box_polygon(Pose2D(*rel), self.geo.size[held.type]),),
                       ("held", held.type, rel))
        cs = self._cspace(w, fp, exclude={held.name})
        res = None
        for level, (m, clr) in enumerate(masks):
            if not m.any():
                continue
            dist_px = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5)
            dgrid, step, g = dist_px * self.grid.res / self.step_m, self.step_m, self.grid
            hx, hy = rel[0], rel[1]

            def center(x, y, th):
                c, s = math.cos(th), math.sin(th)
                return x + c * hx - s * hy, y + s * hx + c * hy

            def goal_test(x, y, th, m=m):
                px, py = center(x, y, th)
                r, c = g.cell(px, py)
                return [] if m[r, c] else None

            ys_, xs_ = np.nonzero(m)
            geo = self._geodesic(cs, list(zip(g.xs[xs_], g.ys[ys_])), offset=math.hypot(hx, hy))

            def heuristic(x, y, th, dgrid=dgrid, geo=geo):
                px, py = center(x, y, th)
                r, c = g.cell(px, py)
                return max(float(dgrid[r, c]), geo(x, y, th))

            res = self._search(w, cs, goal_test, heuristic)
            if res.actions is not None:
                self.carry_level[held.name] = level
                if level > 0:
                    self.stats.drop_relaxations += 1
                    log.debug("t=%d %s の置き場所の空きを %.3f m に緩めた", self.t, held.name, clr)
                return res
        return res

    # ---------- 挟む ----------
    def _extent_x(self, o: ObjState, th: float) -> float:
        """物体の外形の、向き th 方向への半分の長さ。"""
        sx, sy = self.geo.size[o.type]
        phi = o.pose.yaw - th
        return sx / 2 * abs(math.cos(phi)) + sy / 2 * abs(math.sin(phi))

    def _grasp_kind(self, o: ObjState, dest: str) -> bool:
        """挟んで運ぶ仕事にするか。LEGO は常に。ミニカーは grasp.toy_car_first のとき、ゴールへ運ぶ対象で、まだ押す方へ
        切り替えていなければ（比較用の設定）。"""
        if o.type == ObjectType.LEGO:
            return True
        return self.p.grasp.toy_car_first and dest == "goal" and o.name not in self.car_push_only

    def _car_to_push(self, name: str, reason: str) -> None:
        self.car_push_only.add(name)
        self.stats.fail(reason)
        log.debug("t=%d %s を挟めない（%s）→ 押す", self.t, name, reason)

    def _grasp_ready(self, w: WorldState, o: ObjState) -> bool:
        """閉じれば挟める: 物体中心が開いた爪の梁の間（前後 grasp_x_range、左右 close_lateral_tol_m）。

        ミニカー（grasp.toy_car_first のとき）は梁の間まで入らないので、爪の付け根に当たったら閉じてみる。
        """
        lx, ly = to_local(w.robot, o.pose.xy)
        if abs(ly) > self.p.grasp.close_lateral_tol_m:
            return False
        if o.type == ObjectType.TOY_CAR:
            return 0.0 < lx <= self._contact_x(o.type, o.pose.yaw - w.robot.yaw) + 0.01
        lo, hi = self.grasp_x_range
        return lo <= lx <= hi

    def _displacement(self, w: WorldState, q: ObjState) -> float:
        """失敗の判定に使う移動の量（env と同じ: 開始時にゴール内なら中心のはみ出し、それ以外は初期位置からの距離）。"""
        if self.geo.started_in_goal(w.goal, q):
            return max(0.0, signed_distance_to_rect(q.pose.xy, w.goal))
        return dist(q.pose.xy, q.initial.xy)

    def _touching(self, w: WorldState, o: ObjState) -> set[str]:
        """o に接している（外形の隙間が grasp.neighbor_gap_m 以内の）物体。配達済みの対象と、対象外ですでに
        grasp.neighbor_max_displacement_m 以上動いたものは除く（繰り返し押して失敗の判定に近づけない）。"""
        gp = self.p.grasp
        h = gp.neighbor_gap_m / 2
        po = box_polygon(o.pose, self.geo.size[o.type], h)
        out = set()
        for q in w.objects:
            if q.name == o.name or (q.is_target and self.geo.inside_goal(w.goal, q)):
                continue
            if not q.is_target and self._displacement(w, q) >= gp.neighbor_max_displacement_m:
                continue
            if dist(q.pose.xy, o.pose.xy) < self.geo.radius[q.type] + self.geo.radius[o.type] + 2 * h and \
                    convex_overlap(po, box_polygon(q.pose, self.geo.size[q.type], h)):
                out.add(q.name)
        return out

    def _plan_grasp(self, w: WorldState, o: ObjState) -> SearchResult:
        gp = self.p.grasp
        # 近づく物体そのものは膨らませない（触れる直前まで寄る。重なる姿勢は C-space が除く）
        cs = self._cspace(w, self.open_fp, exclude=set(), exact={o.name})
        # 最後の直進では、対象に接している物体は爪で数 mm 押しのけてよい（車体は当てない）
        touching = self._touching(w, o)
        tail_cs = self._cspace(w, self.open_fp, exclude={o.name} | touching, hard_margin=self.p.margins.tail_margin_m)
        tail_body = (self._cspace(w, self.body_fp, exclude={q.name for q in w.objects} - touching,
                                  hard_margin=self.p.margins.tail_margin_m) if touching else None)
        ox, oy = o.pose.xy
        hi_close = self.grasp_x_range[1]
        near_centers = [q.pose.xy for q in w.objects if q.name != o.name and dist(q.pose.xy, o.pose.xy) < 0.1]
        free_cache: dict[int, float] = {}
        blockers = [box_polygon(q.pose, self.geo.size[q.type]) for q in w.objects
                    if q.name != o.name and q.name not in touching and dist(q.pose.xy, o.pose.xy) < 0.12]

        def free_ahead(th: float) -> float:
            """物体を向き th へ押して、壁か（接していない）他の物体に当たるまでの距離（2 mm 刻み、3 cm まで）。"""
            key = int(round(wrap_angle(th) / 0.01))
            if key not in free_cache:
                c, s = math.cos(th), math.sin(th)
                d = 0.0
                while d < gp.max_push_m:
                    pose = Pose2D(o.pose.x + (d + 0.002) * c, o.pose.y + (d + 0.002) * s, o.pose.yaw)
                    poly = box_polygon(pose, self.geo.size[o.type])
                    if not w.area.contains_all([tuple(q) for q in poly]) or any(convex_overlap(poly, b) for b in blockers):
                        break
                    d += 0.002
                free_cache[key] = d
            return free_cache[key]
        step = self.step_m

        def final_test(x, y, th):
            c, s = math.cos(th), math.sin(th)
            dx, dy = ox - x, oy - y
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            if abs(ly) > gp.approach_lateral_tol_m:
                return None
            # 接近の範囲は「物体に触れる位置」から approach_extra_m まで。それより手前で物体と重なる姿勢は C-space が除く
            near = self._contact_x(o.type, o.pose.yaw - th)
            if not (near <= lx <= near + gp.approach_extra_m):
                return None
            # 前進して爪の付け根の V に当たると物体は押されて前に留まるので、梁の範囲の奥（hi_close）を越えて進めばよい
            n = max(0, math.ceil((lx - hi_close) / step))
            if o.type == ObjectType.TOY_CAR:  # ミニカーは梁の間に入らない: 付け根に当たるまで
                n = max(0, math.floor((lx - near) / step)) + 1
            # ただし押される量（触れてからさらに進む分）は、物体の前の空き（壁・他の物体まで）以内。壁際で押し込むと回って
            # 壁に沿い、挟めなくなる（2026-10-17、Stage 5 Seed 10010）
            if near - (lx - n * step) > free_ahead(th):
                return None
            if not self._straight_free(tail_cs, x, y, th, n):
                return None
            if tail_body is not None and not self._straight_free(tail_body, x, y, th, n):
                return None
            # 閉じる姿勢で、対象以外の物体の中心が爪の籠の中にあれば一緒に挟み込むので不可（外形が触れるだけなら押しのける）
            fx, fy = x + n * step * c, y + n * step * s
            for qx, qy in near_centers:
                qlx, qly = c * (qx - fx) + s * (qy - fy), -s * (qx - fx) + c * (qy - fy)
                if self.cage_x[0] <= qlx <= self.cage_x[1] and abs(qly) <= self.cage_half_y:
                    return None
            return [Action.FORWARD] * n

        goal_test = self._with_lateral_shift(cs, final_test, (ox, oy), gp.approach_extra_m + 0.1)

        segs = []
        for k in range(-12, 13):
            th = w.robot.yaw + k * self.turn_rad
            if o.type == ObjectType.TOY_CAR and abs(math.sin(o.pose.yaw - th)) > math.sin(gp.toy_car_max_rel_angle_rad):
                continue  # ミニカーは車軸の向きから差し込む（幅 30 mm を挟む）
            c, s = math.cos(th), math.sin(th)
            near = self._contact_x(o.type, o.pose.yaw - th)
            far = near + gp.approach_extra_m
            segs.append((ox - far * c, oy - far * s, ox - near * c, oy - near * s, th))
        segs = self._feasible_segments(cs, segs, goal_test)
        if not segs:
            self.stats.fail("no_goal_pose")
            return None
        return self._search(w, cs, goal_test, self._approach_heuristic(cs, segs))

    def _geodesic(self, cs: CSpace, sources_xy, offset: float = 0.0) -> GeodesicHeuristic:
        """C-space の障害物（膨らませる前の物体と壁）から search.heuristic_clearance_m 以上離れた所を通る 2D の距離場。"""
        g, f = self.grid, 2  # 1 cm 格子（5 mm の 2 倍）で計算する
        d = cv2.distanceTransform((1 - cs.hard).astype(np.uint8), cv2.DIST_L2, 5) * g.res
        free = (d > self.p.search.heuristic_clearance_m)[::f, ::f]
        cell = g.res * f
        src = {(int((y - g.y0) / cell), int((x - g.x0) / cell)) for x, y in sources_xy}
        field = geodesic_field(free, list(src), cell)
        # 通れない画素（物体のそばにいる開始位置など）には、3 cm 以内の届く画素の値 + 3 cm を入れる
        fin = np.where(np.isfinite(field), field, np.float32(1e6)).astype(np.float32)
        near = cv2.erode(fin, np.ones((7, 7), np.uint8)) + np.float32(3 * cell)
        field = np.where(np.isfinite(field), field, np.where(near < 1e5, near, np.inf)).astype(np.float32)
        return GeodesicHeuristic(field, g.x0, g.y0, cell, self.step_m, offset)

    def _approach_heuristic(self, cs: CSpace, segs):
        """目標の線分への、向きを考えた距離（SegmentHeuristic）と、障害物を避けた 2D の距離の大きい方。"""
        seg = SegmentHeuristic(segs, self.step_m, self.turn_rad)
        pts = []
        for ax, ay, bx, by, _ in segs:
            n = max(1, int(math.hypot(bx - ax, by - ay) / 0.01))
            pts += [(ax + (bx - ax) * i / n, ay + (by - ay) * i / n) for i in range(n + 1)]
        geo = self._geodesic(cs, pts)
        return lambda x, y, th: max(seg(x, y, th), geo(x, y, th))

    def _turn_free(self, cs: CSpace, x: float, y: float, th: float, dk: int) -> bool:
        """その場で旋回 1 回（dk = ±1）の掃引が通れるか（探索と同じく 4 分割で確かめる）。"""
        return all(cs.cost_at(x, y, th + dk * self.turn_rad * i / 4) < math.inf for i in range(1, 5))

    def _with_lateral_shift(self, cs: CSpace, final_test, obj_xy, max_lx: float):
        """目標の判定 final_test を「横ずらし（旋回 k 回・前進 n 回・逆に旋回 k 回）の後に通る」まで広げる。

        格子の A* は重複判定（cell_m）で横の細かい位置を失い、横 ±approach_lateral_tol_m の直線に乗れないことがある
        （2026-10-17、Stage 6 で展開数の上限に当たる失敗の多く）。旋回 1 回で 1 前進あたり横 4.9 mm ずれる。
        """
        sp = self.p.search
        step, turn = self.step_m, self.turn_rad
        ox, oy = obj_xy

        def goal_test(x, y, th):
            r = final_test(x, y, th)
            if r is not None:
                return r
            c, s = math.cos(th), math.sin(th)
            dx, dy = ox - x, oy - y
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            if not (abs(ly) <= sp.lateral_shift_max_m and 0.0 < lx <= max_lx):
                return None
            sgn = 1 if ly > 0 else -1
            for k in sp.lateral_shift_turns:
                lat = step * math.sin(k * turn)
                n = int(round(abs(ly) / lat))
                if n < 1:
                    continue
                th1 = th + sgn * k * turn
                if not all(self._turn_free(cs, x, y, th + sgn * i * turn, sgn) for i in range(k)):
                    continue
                if not self._straight_free(cs, x, y, th1, n):
                    continue
                x2, y2 = x + n * step * math.cos(th1), y + n * step * math.sin(th1)
                if not all(self._turn_free(cs, x2, y2, th1 - sgn * i * turn, -sgn) for i in range(k)):
                    continue
                r2 = final_test(x2, y2, th)
                if r2 is not None:
                    a1, a2 = (Action.TURN_LEFT, Action.TURN_RIGHT) if sgn > 0 else (Action.TURN_RIGHT, Action.TURN_LEFT)
                    return [a1] * k + [Action.FORWARD] * n + [a2] * k + r2
            return None
        return goal_test

    def _feasible_segments(self, cs: CSpace, segs, goal_test) -> list:
        """目標の線分を 1 cm 刻みで調べ、通れて目標の判定を通る姿勢がある部分（最初から最後の点まで）だけを返す。

        空なら探索しても無駄。ヒューリスティックも、ふさがった向きの線分に引かれないよう、この部分だけで作る。
        """
        out = []
        for ax, ay, bx, by, th in segs:
            n = max(1, int(math.hypot(bx - ax, by - ay) / 0.01))
            ok = [i for i in range(n + 1)
                  if cs.cost_at(ax + (bx - ax) * i / n, ay + (by - ay) * i / n, th) < math.inf
                  and goal_test(ax + (bx - ax) * i / n, ay + (by - ay) * i / n, th) is not None]
            if ok:
                i0, i1 = ok[0], ok[-1]
                out.append((ax + (bx - ax) * i0 / n, ay + (by - ay) * i0 / n, ax + (bx - ax) * i1 / n,
                            ay + (by - ay) * i1 / n, th))
        return out

    def _straight_free(self, cs: CSpace, x: float, y: float, th: float, n: int, sign: float = 1.0) -> bool:
        c, s = math.cos(th), math.sin(th)
        j = cs.bin(th)
        for i in range(1, 2 * n + 1):
            d = sign * 0.5 * i * self.step_m
            if cs.cost(x + d * c, y + d * s, j) == math.inf:
                return False
        return True

    # ---------- 押す（ミニカー） ----------
    def _contact_x(self, otype: ObjectType, phi: float, ly: float = 0.0) -> float:
        """物体（ロボットとのなす角 phi）を前方の横 ly に置いたとき、車体・開いた爪に触れる最小の前方距離。"""
        key = (otype, int(round(wrap_angle(phi) / 0.02)), int(round(ly / 0.002)))
        if key not in self._contact_cache:
            size = self.geo.size[otype]
            lo, hi = 0.0, 0.3
            for _ in range(30):
                mid = (lo + hi) / 2
                car = box_polygon(Pose2D(mid, key[2] * 0.002, key[1] * 0.02), size)
                if any(convex_overlap(car, q) for q in self.open_polys):
                    lo = mid
                else:
                    hi = mid
            self._contact_cache[key] = hi
        return self._contact_cache[key]

    def _final_region(self, w: WorldState, o: ObjState, dest: str):
        """押し込む先（ミニカー中心を置いてよい点）の判定関数。"""
        r = self.geo.radius[o.type]
        goal = w.goal
        if dest == "goal":
            e = r + self.p.push.goal_margin_m
            return lambda x, y: abs(x - goal.center[0]) <= goal.size[0] / 2 - e and abs(y - goal.center[1]) <= goal.size[1] / 2 - e
        f = self.p.foreign
        area = w.area
        e = r + f.wall_margin_m

        def ok(x, y):
            return (signed_distance_to_rect((x, y), goal) >= r + f.out_clearance_m
                    and abs(x - area.center[0]) <= area.size[0] / 2 - e and abs(y - area.center[1]) <= area.size[1] / 2 - e
                    and math.hypot(x - o.initial.x, y - o.initial.y) <= f.max_relocation_m)
        return ok

    def _push_length(self, cs: CSpace, o: ObjState, th: float, final_ok, start_xy=None) -> float | None:
        """ミニカーを向き th へ真っすぐ押して、押し込む先に入るまでの距離（途中で当たるなら None）。"""
        pp = self.p.push
        c, s = math.cos(th), math.sin(th)
        j = cs.bin(th)
        x0, y0 = start_xy if start_xy is not None else o.pose.xy
        n = int(pp.max_push_m / pp.sweep_step_m)
        for i in range(n + 1):
            d = i * pp.sweep_step_m
            px, py = x0 + d * c, y0 + d * s
            if cs.cost(px - pp.disc_x_m * c, py - pp.disc_x_m * s, j) == math.inf:
                return None
            if final_ok(px, py):
                return d
        return None

    def _sideways(self, car_yaw: float, th: float, extra: float = 0.0) -> bool:
        """押す向き th とミニカーの向きのなす角が min_rel_angle_rad（+ extra）以上か（横向きに押す）。"""
        return abs(math.sin(car_yaw - th)) >= math.sin(min(self.p.push.min_rel_angle_rad + extra, math.pi / 2))

    def _settled_yaws(self, car_yaw: float, th: float) -> list[float]:
        """向き th へ押した後のミニカーの向きの見込み（押す向きとのなす角が settled_rel_angle_rad の範囲に落ち着く）。"""
        phi = wrap_angle(car_yaw - th)
        phi -= math.pi * round(phi / math.pi)  # 箱は 180° 対称
        sgn = 1.0 if phi >= 0 else -1.0
        lo, hi = self.p.push.settled_rel_angle_rad
        return [th + sgn * a for a in (lo, (lo + hi) / 2, hi)]

    def _in_push_contact(self, w: WorldState, o: ObjState) -> bool:
        pp = self.p.push
        lx, ly = to_local(w.robot, o.pose.xy)
        cx = self._contact_x(o.type, o.pose.yaw - w.robot.yaw)
        return pp.contact_x_window_m[0] <= lx - cx <= pp.contact_x_window_m[1] and abs(ly) <= pp.contact_lateral_tol_m

    def _free_push(self, cs: CSpace, th: float, start_xy, max_d: float) -> float:
        """ミニカー中心を start_xy から向き th へ押して、当たらずに進める距離（sweep_step_m 刻み、max_d まで）。"""
        pp = self.p.push
        c, s = math.cos(th), math.sin(th)
        j = cs.bin(th)
        d = 0.0
        while d <= max_d + 1e-9:
            px, py = start_xy[0] + d * c, start_xy[1] + d * s
            if cs.cost(px - pp.disc_x_m * c, py - pp.disc_x_m * s, j) == math.inf:
                return d - pp.sweep_step_m
            d += pp.sweep_step_m
        return max_d

    def _push_forward_ok(self, w: WorldState, o: ObjState, dest: str, via=None) -> bool:
        """押している最中（ミニカーが前で接していて、今の向きのまま押せば行き先に入る）なら True。"""
        if w.gripper != GripperState.OPEN or o.type != ObjectType.TOY_CAR or not self._in_push_contact(w, o):
            return False
        pp = self.p.push
        cs = self._cspace(w, self.push_fp, exclude={o.name})
        th = w.robot.yaw
        c, s = math.cos(th), math.sin(th)
        # 今のロボット位置から押した場合のミニカー中心の通り道（中心線上に disc_x_m 前）
        start = (w.robot.x + pp.disc_x_m * c, w.robot.y + pp.disc_x_m * s)
        if via is not None:
            rem = (via[0] - o.pose.x) * c + (via[1] - o.pose.y) * s
            return rem > pp.sweep_step_m / 2 and self._free_push(cs, th, start, rem) >= rem - 1e-9
        final_ok = self._final_region(w, o, dest)
        if final_ok(*o.pose.xy):
            return False
        return self._push_length(cs, o, th, final_ok, start_xy=start) is not None

    def _lattice_headings(self, w: WorldState) -> list[float]:
        """今の向きから旋回で取れる向き（約 1 周）。"""
        return [w.robot.yaw + k * self.turn_rad for k in range(-13, 14)]

    def _push_lines(self, w: WorldState, o: ObjState, dest: str, push_cs: CSpace, two_stage: bool) -> list[PushLine]:
        """押せる直線。two_stage なら 2 段（1 段目を押して向きと位置を変え、押し直してゴールへ）の 1 段目。"""
        final_ok = self._final_region(w, o, dest)
        heads = self._lattice_headings(w)
        if not two_stage:
            lines = []
            for th in heads:
                if self._sideways(o.pose.yaw, th):
                    L = self._push_length(push_cs, o, th, final_ok)
                    if L is not None:
                        lines.append(PushLine(th, L, None))
            return lines
        if dest != "goal":
            return []
        pp = self.p.push
        best: dict[int, tuple[float, PushLine]] = {}
        for k1, th1 in enumerate(heads):
            if not self._sideways(o.pose.yaw, th1):
                continue
            free = self._free_push(push_cs, th1, o.pose.xy, max(pp.via_lengths_m))
            yaws = self._settled_yaws(o.pose.yaw, th1)
            c1, s1 = math.cos(th1), math.sin(th1)
            for d1 in pp.via_lengths_m:
                if d1 > free:
                    break
                p1 = (o.pose.x + d1 * c1, o.pose.y + d1 * s1)
                moved = ObjState(o.name, o.type, Pose2D(p1[0], p1[1], yaws[1]), o.initial, o.is_target)
                for th2 in heads:
                    if not all(self._sideways(yw, th2, pp.via_rel_angle_extra_rad) for yw in yaws):
                        continue
                    L2 = self._push_length(push_cs, moved, th2, final_ok)
                    if L2 is not None and (k1 not in best or d1 + L2 < best[k1][0]):
                        best[k1] = (d1 + L2, PushLine(th1, d1, p1))
        return [ln for _, ln in best.values()]

    def _pivot_fp(self, offset: float) -> Footprint:
        pp = self.p.push
        r = self.geo.radius[ObjectType.TOY_CAR] + pp.disc_extra_m
        return Footprint(tuple(self.open_polys) + (disc_polygon((pp.disc_x_m, offset), r),), ("pivot", round(offset, 3)))

    def _pivot_lines(self, w: WorldState, o: ObjState) -> list[PushLine]:
        """端を押して回す直線（車軸とほぼ直角に、ミニカーの端寄りを pivot_steps 回押す。1 回で約 0.7〜1.3 rad 回る）。

        1 本でも 2 段でもゴールへ押せないとき（車軸の向きにゴールがあるなど）に使う。回った後は改めて押す直線を探す。
        """
        pp = self.p.push
        lines = []
        L = pp.pivot_steps * self.step_m
        for th in self._lattice_headings(w):
            if abs(math.sin(o.pose.yaw - th)) < math.sin(pp.pivot_min_rel_angle_rad):
                continue
            c, s = math.cos(th), math.sin(th)
            for off in (pp.pivot_offset_m, -pp.pivot_offset_m):
                cs = self._cspace(w, self._pivot_fp(off), exclude={o.name})
                # ミニカー中心がロボットの中心線から横 off にあるとき、車軸中点は中心から disc_x 後ろ、off 右
                ok = True
                for i in range(int(L / pp.sweep_step_m) + 1):
                    d = i * pp.sweep_step_m
                    ax = o.pose.x + d * c - pp.disc_x_m * c + off * s
                    ay = o.pose.y + d * s - pp.disc_x_m * s - off * c
                    if cs.cost_at(ax, ay, th) == math.inf:
                        ok = False
                        break
                if ok:
                    lines.append(PushLine(th, L, None, off, True))
        return lines

    def _pivot_forward_ok(self, w: WorldState, o: ObjState, pivot: tuple[float, int]) -> bool:
        """端を押して回している最中（向きが押す向きのまま、ミニカーが前で接している）なら True。"""
        th, _ = pivot
        if w.gripper != GripperState.OPEN or abs(wrap_angle(w.robot.yaw - th)) > self.p.replan.yaw_tol_rad:
            return False
        lx, ly = to_local(w.robot, o.pose.xy)
        pp = self.p.push
        return 0.0 < lx <= pp.disc_x_m + pp.contact_x_window_m[1] + 0.02 and abs(ly) <= pp.pivot_offset_m + 0.03

    def _plan_push_approach(self, w: WorldState, o: ObjState, task: Task) -> SearchResult | None:
        """押し始める姿勢への計画。1 本で押せる直線を先に試し、その接近がすべてふさがっていれば 2 段の押しを試す。"""
        pp = self.p.push
        push_cs = self._cspace(w, self.push_fp, exclude={o.name})
        # 近づく物体そのものは膨らませない（触れる直前まで寄る。重なる姿勢は C-space が除く）
        cs = self._cspace(w, self.open_fp, exclude=set(), exact={o.name})
        tail_cs = self._cspace(w, self.open_fp, exclude={o.name}, hard_margin=self.p.margins.tail_margin_m)
        ox, oy = o.pose.xy
        step = self.step_m
        res, reason = None, "no_push_line"
        for mode in ("direct", "via", "pivot"):
            if mode == "via" and task.n_via >= pp.max_via:
                reason = "via_limit"
                continue
            if mode == "pivot" and (task.n_pivot >= pp.max_pivot or task.dest != "goal"):
                continue
            lines = (self._pivot_lines(w, o) if mode == "pivot"
                     else self._push_lines(w, o, task.dest, push_cs, mode == "via"))
            if not lines:
                continue
            by_key: dict[int, list[PushLine]] = {}
            for ln in lines:
                by_key.setdefault(int(round(wrap_angle(ln.theta) / 0.01)), []).append(ln)

            def match(x, y, th, by_key=by_key):
                """姿勢 (x, y, th) が押し始めの範囲に入る直線と、そこから接触までの前進の回数。"""
                c, s = math.cos(th), math.sin(th)
                dx, dy = ox - x, oy - y
                lx, ly = c * dx + s * dy, -s * dx + c * dy
                for ln in by_key.get(int(round(wrap_angle(th) / 0.01)), ()):
                    if abs(ly - ln.offset) > pp.approach_lateral_tol_m:
                        continue
                    cx = self._contact_x(o.type, o.pose.yaw - th, ln.offset)
                    if cx <= lx <= cx + pp.approach_extra_m:
                        return ln, max(0, math.floor((lx - cx) / step))
                return None

            def final_test(x, y, th, match=match):
                m = match(x, y, th)
                if m is None or not self._straight_free(tail_cs, x, y, th, m[1]):
                    return None
                return [Action.FORWARD] * (m[1] + 1)  # 最後の 1 歩で接触する

            goal_test = (final_test if mode == "pivot"
                         else self._with_lateral_shift(cs, final_test, (ox, oy), pp.approach_extra_m + 0.15))
            segs = []
            for ln in lines:
                th = ln.theta
                c, s = math.cos(th), math.sin(th)
                cx = self._contact_x(o.type, o.pose.yaw - th, ln.offset)
                far = cx + pp.approach_extra_m
                bx, by = ox + ln.offset * s, oy - ln.offset * c  # 横 offset だけずらした直線
                segs.append((bx - far * c, by - far * s, bx - cx * c, by - cx * s, th))
            segs = self._feasible_segments(cs, segs, goal_test)
            if not segs:
                reason = "no_goal_pose"
                continue
            res = self._search(w, cs, goal_test, self._approach_heuristic(cs, segs))
            if res.actions is None:
                continue
            # 直進の部分（接触までの前進）を除いた、押し始めの姿勢でどの直線かを決める
            k = len(res.actions)
            while k > 0 and res.actions[k - 1] == Action.FORWARD:
                k -= 1
            end = w.robot
            m = None
            for i, a in enumerate(res.actions):
                if i >= k:
                    m = m or match(end.x, end.y, end.yaw)
                end = kinematic_step(end, a, self.step_m, self.turn_rad)
            if m is None:
                start = w.robot
                for a in res.actions[:k]:
                    start = kinematic_step(start, a, self.step_m, self.turn_rad)
                m = match(start.x, start.y, start.yaw)
            if m is None:
                raise RuntimeError(f"計画の終わりの姿勢が押し始めの範囲にない（{o.name}）")
            ln = m[0]
            if ln.via is not None and task.via is None:
                task.n_via += 1
                log.debug("t=%d %s を 2 段で押す（1 段目 %.2f m、向き %.2f）", self.t, o.name, ln.length, ln.theta)
            task.via = ln.via
            task.pivot = None
            if ln.pivot:
                task.n_pivot += 1
                task.pivot = (end.yaw, self.p.push.pivot_steps)
                log.debug("t=%d %s の端を押して回す（向き %.2f、横 %.3f）", self.t, o.name, ln.theta, ln.offset)
            return res
        if res is None:
            self.stats.fail(reason)
        return res

    # ---------- 計画の共通部分 ----------
    def _search(self, w: WorldState, cs: CSpace, goal_test, heuristic) -> SearchResult:
        """重み付き A* を search.heuristic_weights の順に（それぞれ max_expansions まで）試す。最初に見つかった計画を返す。"""
        t0 = time.perf_counter()
        sp = self.p.search
        for wt, limit in zip(sp.heuristic_weights, sp.max_expansions):
            res = lattice_astar(w.robot, cs, goal_test, heuristic, sp, self.step_m, self.turn_rad, wt, limit)
            if res.actions is not None or res.reason == "exhausted":
                break
        self.stats.plan_time_s.append(time.perf_counter() - t0)
        if res.actions is not None:
            self.stats.success_expansions.append(res.expansions)
        self.stats.plans += 1
        self.stats.expansions += res.expansions
        self.stats.max_expansions = max(self.stats.max_expansions, res.expansions)
        if res.actions is None:
            self.stats.fail(res.reason)
        return res

    def _cspace(self, w: WorldState, fp: Footprint, exclude: set[str], hard_margin: float | None = None,
                exact: set[str] = frozenset()) -> CSpace:
        """障害物（exclude 以外の全物体と壁）の C-space。今の外形と重なっている物体は overlap 層に移す。

        hard_margin を渡さなければ margins.hard_margin_m（最後の直進の判定だけ margins.tail_margin_m を渡す）。
        exact の物体は margins.tail_margin_m で膨らませる（近づく対象）。
        """
        hm_all = self.p.margins.hard_margin_m if hard_margin is None else hard_margin
        sm = self.p.margins.soft_margin_m
        robot_world = transform(list(fp.polys), w.robot)
        hard_polys, soft_polys, over_polys, over_names = [], [], [], []
        sig = []
        for o in w.objects:
            if o.name in exclude:
                continue
            size = self.geo.size[o.type]
            hm = self.p.margins.tail_margin_m if o.name in exact else hm_all
            poly = box_polygon(o.pose, size, hm)
            sig.append((o.name, round(o.pose.x, 3), round(o.pose.y, 3), round(o.pose.yaw, 2)))
            # 「今重なっている」の判定は、格子の量子化（± 1 画素）と向きの刻みを見込んで 2 画素広く取る。狭いと、多角形では
            # 離れているのに格子では重なる位置から、どこへも動けなくなる（2026-10-17、Stage 7 Seed 10072）
            touch = box_polygon(o.pose, size, hm + 2 * self.grid.res)
            if dist(o.pose.xy, w.robot.xy) < fp.radius + self.geo.radius[o.type] + hm + 2 * self.grid.res and \
                    any(convex_overlap(touch, q) for q in robot_world):
                over_polys.append(poly)
                over_names.append(o.name)
            else:
                hard_polys.append(poly)
            soft_polys.append(box_polygon(o.pose, size, sm))
        key = (fp.key, hm_all, tuple(sorted(exclude)), tuple(sorted(exact)), tuple(over_names), tuple(sig))
        cs = self._cs_cache.get(key)
        if cs is None:
            if len(self._cs_cache) > 32:
                self._cs_cache.clear()
            g = self.grid
            hard = g.raster(hard_polys) | self.walls
            soft = g.raster(soft_polys)
            over = g.raster(over_polys) if over_polys else None
            cs = CSpace(g, self.kernels, fp, hard, soft, over, self.p.search.soft_cost, self.p.search.overlap_cost)
            self._cs_cache[key] = cs
        return cs

    def _store_plan(self, key: tuple, w: WorldState, actions: list[Action], exclude: str | None) -> None:
        poses = [w.robot]
        for a in actions:
            poses.append(kinematic_step(poses[-1], a, self.step_m, self.turn_rad))
        objs = {o.name: o.pose for o in w.objects if o.name != exclude}
        self.plan = Plan(key, list(actions), poses, objs)

    def _plan_valid(self, key: tuple, w: WorldState) -> bool:
        pl = self.plan
        if pl is None or pl.key != key or pl.i >= len(pl.actions):
            return False
        exp = pl.poses[pl.i]
        rp = self.p.replan
        if dist(exp.xy, w.robot.xy) > rp.pose_tol_m or abs(wrap_angle(exp.yaw - w.robot.yaw)) > rp.yaw_tol_rad:
            return False
        for name, pose in pl.objects.items():
            if dist(pose.xy, w.obj(name).pose.xy) > rp.object_move_tol_m:
                return False
        return True

    def _follow_or_plan(self, key: tuple, w: WorldState, plan_fn, exclude: str | None = None) -> Action | None:
        if not self._plan_valid(key, w):
            res = plan_fn()
            if res is None or res.actions is None:
                self.plan = None
                return None
            if not res.actions:
                # 目標の状態にいるのに、判定（閉じる・押す・置く）を通らなかった: 計画と判定の食い違いは不具合
                raise RuntimeError(f"計画が空（{key}）なのに次の行動が決まらない: robot={w.robot}")
            self._store_plan(key, w, res.actions, exclude)
        pl = self.plan
        a = pl.actions[pl.i]
        pl.i += 1
        return a
