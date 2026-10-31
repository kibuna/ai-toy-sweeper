"""Stage ごとの盤面生成と、タスクに関する幾何。"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from .config import SimCfg, TaskCfg
from .geometry import Pose2D, Rect, box_corners, dist, signed_distance_to_rect
from .robot_model import front_x, gripper_center_x, robot_radius
from .sim2d import ObjectSpec
from .spaces import OBJECT_TYPE_NAMES, ObjectType

STAGES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# 描画と VLM 用画像のための色の候補（物理と Observation には使わない）。2026-10-18、実物のおもちゃにある
# 黄緑・水色・ピンク・オレンジを足した。盤面の乱数の引き方（回数）は変わらないので、Seed ごとの物体の位置は前と同じ
COLORS = ("red", "blue", "yellow", "green", "white", "black", "lime", "light_blue", "pink", "orange")


@dataclass(frozen=True)
class EpisodeSpec:
    stage: int
    robot: Pose2D
    objects: tuple[ObjectSpec, ...]
    target_names: tuple[str, ...]            # 対象の集合（v7）。Stage 0 は空
    reach_point: tuple[float, float] | None  # Stage 0 のみ
    goal_name: str
    goal: Rect
    # 今のゴールとは別の置き場（v18、前の指示のゴールなど）。開始時に 4 隅が入っている対象外は、そこから出さない
    zones: tuple[Rect, ...] = ()
    # 置き場の指定（v24、Stage 9 の「やり残し」）: (物体名, 置き場の番号)。番号は 0 = 今のゴール、1〜 = zones[番号 − 1]、
    # −1 = 置き場なし（もともとゴールの外）。指定の無い物体は開始の位置から決める（start_home_index）
    home_overrides: tuple[tuple[str, int], ...] = ()
    # 作業領域の (幅, 奥行き)（v30、sampling.work_area_size_range_m）。None は sim.work_area.size_m（固定の作業領域）
    area_size: tuple[float, float] | None = None


def episode_area(sim: SimCfg, ep: EpisodeSpec) -> tuple[float, float]:
    """盤面の作業領域の (幅, 奥行き)（v30）。"""
    return tuple(sim.work_area.size_m) if ep.area_size is None else ep.area_size


def start_home_index(sim: SimCfg, ep: EpisodeSpec, o: ObjectSpec) -> int:
    """物体の置き場の番号（v18、v24）。ep.home_overrides に指定があればそれ。無ければ開始時に外形の 4 隅が入っている
    長方形を、今のゴール（0）、zones（1〜）の順に調べた最初のもの。どこにも入っていなければ −1（置き場なし）。

    今のゴールを最初に調べるので、指定の無い物体では「番号が 0」と「開始時に 4 隅が今のゴールの中」は同じ意味。
    """
    for name, i in ep.home_overrides:
        if name == o.name:
            if not -1 <= i <= len(ep.zones):
                raise ValueError(f"置き場の番号 {i} が範囲外（−1〜{len(ep.zones)}）: {o.name}")
            return i
    corners = box_corners(o.pose, sim.objects[OBJECT_TYPE_NAMES[o.type]].size_m[:2])
    return next((i for i, z in enumerate((ep.goal, *ep.zones)) if z.contains_all(corners)), -1)


def gripper_point(sim: SimCfg, robot: Pose2D) -> tuple[float, float]:
    """爪の間の中心。到達判定と接近報酬の基準点。"""
    return robot.transform_point((gripper_center_x(sim), 0.0))


def approach_point(obj_xy, goal_xy, offset: float) -> tuple[float, float]:
    """物体中心からゴールと反対方向へ offset 離れた点（p_approach）。"""
    dx, dy = goal_xy[0] - obj_xy[0], goal_xy[1] - obj_xy[1]
    n = math.hypot(dx, dy)
    if n == 0:
        raise ValueError("物体がゴール中心と一致していて押す方向が決まらない")
    return (obj_xy[0] - offset * dx / n, obj_xy[1] - offset * dy / n)


def goal_rects(sim: SimCfg) -> dict[str, Rect]:
    return {k: Rect(center=tuple(z.center_m), size=tuple(z.size_m)) for k, z in sim.goal_zones.items()}


def goal_rects_for_area(sim: SimCfg, size: tuple[float, float]) -> dict[str, Rect]:
    """名前つきのゴール（sim.work_area での goal_zones）を、大きさ size の作業領域へ写す（v30）。軸ごとに、近い縁からの
    距離を保つ（角と辺に接する 0.2 m 四方は接したまま）。中心 0 の軸は 0 のまま。大きさはそのまま。"""
    out = {}
    for k, z in sim.goal_zones.items():
        c = []
        for a in range(2):
            ref, new = sim.work_area.size_m[a] / 2, size[a] / 2
            v = z.center_m[a]
            c.append(0.0 if v == 0.0 else math.copysign(new - (ref - abs(v)), v))
        out[k] = Rect(center=(c[0], c[1]), size=tuple(z.size_m))
    return out


def max_objects(task: TaskCfg) -> int:
    """盤面の物体数の上限（全 Stage）。Stage 6 以前は対象 1 + 対象外、Stage 7 は対象 + 対象外、Stage 8・9 は置き場の物体 +
    対象 + 置き場の外の対象外（Stage 9 は Stage 8 の物体を置き直すだけ）。Gazebo の World の物体の枠の数に使う。"""
    sp = task.sampling
    return max(1 + sp.clutter_range[1], sp.stage7_targets[1] + sp.stage7_clutter[1],
               sp.stage8_zones[1] * sp.stage8_zone_objects[1] + sp.stage7_targets[1] + sp.stage8_clutter[1])


class TaskSampler:
    def __init__(self, sim: SimCfg, task: TaskCfg):
        self.sim = sim
        self.sp = task.sampling
        self.goals = goal_rects(sim)
        self.goal_names = sorted(self.goals)
        self.half = (sim.work_area.size_m[0] / 2, sim.work_area.size_m[1] / 2)
        self.robot_r = robot_radius(sim)
        self.obj_r = {t: math.hypot(*sim.objects[OBJECT_TYPE_NAMES[t]].size_m[:2]) / 2 for t in ObjectType}

    def sample(self, stage: int, rng: np.random.Generator) -> EpisodeSpec:
        if stage not in STAGES:
            raise ValueError(f"未定義の Stage: {stage}（定義済み: {STAGES}）")
        area = None
        if self.sp.work_area_size_range_m:
            # v30: 盤面ごとの作業領域（幅 → 奥行きの順）。この盤面の中だけ half と名前つきのゴールを置き換える
            area = tuple(float(rng.uniform(*r)) for r in self.sp.work_area_size_range_m)
            self.half = (area[0] / 2, area[1] / 2)
            self.goals = goal_rects_for_area(self.sim, area)
        for _ in range(self.sp.max_tries):
            ep = self._try(stage, rng)
            if ep is not None:
                return ep if area is None else dataclasses.replace(ep, area_size=area)
        raise RuntimeError(f"Stage {stage}: {self.sp.max_tries} 回試しても有効な盤面が引けない。設定を見直すこと")

    # ---------- 部品 ----------
    def _robot_ok(self, x: float, y: float) -> bool:
        m = self.sp.robot_margin_m
        return abs(x) <= self.half[0] - m and abs(y) <= self.half[1] - m

    def _xy(self, rng) -> tuple[float, float]:
        m = self.sp.object_margin_m
        return (float(rng.uniform(-self.half[0] + m, self.half[0] - m)),
                float(rng.uniform(-self.half[1] + m, self.half[1] - m)))

    def _type(self, rng) -> ObjectType:
        return ObjectType.TOY_CAR if rng.random() < self.sp.toy_car_probability else ObjectType.LEGO

    def _target_type(self, rng) -> ObjectType:
        """Stage 7・8 の対象の種類（v28）。multi_target_toy_car_probability が 0 なら LEGO（乱数を引かない）。"""
        p = self.sp.multi_target_toy_car_probability
        return ObjectType.TOY_CAR if p > 0.0 and rng.random() < p else ObjectType.LEGO

    def _clutter_colors(self, t: ObjectType, color: str) -> list[str]:
        """対象外の色の候補。LEGO は対象の色以外。ミニカーはどの色でもよいが、対象にミニカーが混ざりうるとき（v28）は
        対象の色以外（指示「その色のもの全部」が 1 つに決まるように）。"""
        if t == ObjectType.LEGO or self.sp.multi_target_toy_car_probability > 0.0:
            return [c for c in COLORS if c != color]
        return list(COLORS)

    def _robot_near(self, rng, o: tuple[float, float], to_goal: float, stage: int) -> Pose2D | None:
        sp = self.sp
        if stage == 1:
            r = front_x(self.sim) + float(rng.uniform(*sp.stage1_front_gap_m))
            rx, ry = o[0] - r * math.cos(to_goal), o[1] - r * math.sin(to_goal)
            ryaw = to_goal + float(rng.uniform(-sp.stage1_yaw_noise_rad, sp.stage1_yaw_noise_rad))
        else:
            if stage in (2, 4):
                u = to_goal + math.pi + float(rng.uniform(-sp.stage2_max_offset_rad, sp.stage2_max_offset_rad))
            else:
                u = float(rng.uniform(-math.pi, math.pi))
            r = float(rng.uniform(*sp.robot_object_distance_m))
            rx, ry = o[0] + r * math.cos(u), o[1] + r * math.sin(u)
            ryaw = float(rng.uniform(-math.pi, math.pi))
        if not self._robot_ok(rx, ry):
            return None
        return Pose2D(rx, ry, math.remainder(ryaw, 2 * math.pi))

    def _clear_of_goal(self, o: ObjectSpec, goal: Rect) -> bool:
        """対象外の置き方の条件（v9）: 4 隅がゴール内か、外形の 4 隅がゴールから goal_clearance_m 以上離れている。

        縁にまたがる配置や、縁のすぐ外の「ひと押しでゴールに入る」配置を除く。
        """
        corners = box_corners(o.pose, self.sim.objects[OBJECT_TYPE_NAMES[o.type]].size_m[:2])
        if goal.contains_all(corners):
            return True
        return min(signed_distance_to_rect(c, goal) for c in corners) >= self.sp.goal_clearance_m

    def _neighbor(self, rng, target: ObjectSpec, objects: list[ObjectSpec], name: str,
                  otype: ObjectType | None = None, color: str | None = None) -> ObjectSpec | None:
        """対象に接する物体（同じ向きで、対象の前後か左右に隙間 cluster_gap_m で並べる）。

        otype / color を渡さなければ乱数で決める（Stage 6 の対象外）。Stage 7 の対象どうしは同じ種類・色を渡す。
        """
        t = self._type(rng) if otype is None else otype
        ts = self.sim.objects[OBJECT_TYPE_NAMES[target.type]].size_m[:2]
        ns = self.sim.objects[OBJECT_TYPE_NAMES[t]].size_m[:2]
        axis = int(rng.integers(2))           # 0: 対象の x（長辺）方向に並べる, 1: y 方向
        sign = 1.0 if rng.random() < 0.5 else -1.0
        gap = float(rng.uniform(*self.sp.cluster_gap_m))
        along = sign * (ts[axis] / 2 + ns[axis] / 2 + gap)
        other = 1 - axis
        shift = float(rng.uniform(-ts[other] / 2, ts[other] / 2))
        local = (along, shift) if axis == 0 else (shift, along)
        x, y = target.pose.transform_point(local)
        m = self.sp.object_margin_m
        if abs(x) > self.half[0] - m or abs(y) > self.half[1] - m:
            return None
        if any(dist((x, y), q.pose.xy) < self.obj_r[t] + self.obj_r[q.type] for q in objects if q.name != target.name):
            return None
        c = COLORS[int(rng.integers(len(COLORS)))] if color is None else color
        return ObjectSpec(name, t, Pose2D(x, y, target.pose.yaw), c)

    def _group(self, rng) -> bool:
        """Stage 7・8 の対象を近くに置くか（v29）。target_group_probability が 0 なら False（乱数を引かない）。"""
        p = self.sp.target_group_probability
        return p > 0.0 and rng.random() < p

    def _near(self, rng, base: ObjectSpec, objects: list[ObjectSpec], name: str, t: ObjectType,
              color: str) -> ObjectSpec | None:
        """対象の近く（接しない）に置く（v29）。中心の距離 = 外接円の半径の和 + target_group_gap_m、方向と向きは一様。
        作業領域の余白の外か、他の物体と重なれば None。"""
        a = float(rng.uniform(-math.pi, math.pi))
        d = self.obj_r[base.type] + self.obj_r[t] + float(rng.uniform(*self.sp.target_group_gap_m))
        x, y = base.pose.x + d * math.cos(a), base.pose.y + d * math.sin(a)
        m = self.sp.object_margin_m
        if abs(x) > self.half[0] - m or abs(y) > self.half[1] - m:
            return None
        if any(dist((x, y), q.pose.xy) < self.obj_r[t] + self.obj_r[q.type] + self.sp.object_gap_m for q in objects):
            return None
        return ObjectSpec(name, t, Pose2D(x, y, float(rng.uniform(-math.pi, math.pi))), color)

    # ---------- 盤面 ----------
    def _goal(self, rng) -> tuple[str, Rect]:
        if self.sp.goal_mode == "zones":
            name = self.goal_names[int(rng.integers(len(self.goal_names)))]
            return name, self.goals[name]
        w, h = (float(rng.uniform(*self.sp.random_goal_size_m)) for _ in range(2))
        cx = float(rng.uniform(-self.half[0] + w / 2, self.half[0] - w / 2))
        cy = float(rng.uniform(-self.half[1] + h / 2, self.half[1] - h / 2))
        return "random", Rect(center=(cx, cy), size=(w, h))

    def _try_stage5(self, rng) -> EpisodeSpec | None:
        """壁際（作業領域の端の近く）の LEGO を、内側のゴールへ。押すだけでは運べない配置。"""
        sp = self.sp
        side = int(rng.integers(4))  # 0: 右, 1: 上, 2: 左, 3: 下 の端
        inward = [(-1.0, 0.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0)][side]
        d = float(rng.uniform(*sp.stage5_edge_distance_m))
        along = float(rng.uniform(-self.half[0] + sp.object_margin_m, self.half[0] - sp.object_margin_m))
        if side in (0, 2):
            o = (-inward[0] * (self.half[0] - d), along)
        else:
            o = (along, -inward[1] * (self.half[1] - d))
        goal_name, goal = self._goal(rng)
        gx, gy = goal.center[0] - o[0], goal.center[1] - o[1]
        if goal.contains(o) or (gx * inward[0] + gy * inward[1]) < math.cos(sp.stage5_goal_max_angle_rad) * math.hypot(gx, gy):
            return None
        r = float(rng.uniform(*sp.stage5_robot_distance_m))
        rx, ry = o[0] + inward[0] * r, o[1] + inward[1] * r
        if not self._robot_ok(rx, ry):
            return None
        yaw = math.atan2(-inward[1], -inward[0]) + float(rng.uniform(-sp.stage5_yaw_noise_rad, sp.stage5_yaw_noise_rad))
        target = ObjectSpec("obj0", ObjectType.LEGO, Pose2D(o[0], o[1], float(rng.uniform(-math.pi, math.pi))),
                            COLORS[int(rng.integers(len(COLORS)))])
        return EpisodeSpec(5, Pose2D(rx, ry, math.remainder(yaw, 2 * math.pi)), (target,), ("obj0",), None, goal_name, goal)

    def _place(self, rng, objects: list[ObjectSpec], t: ObjectType, name: str, color: str,
               ok=None) -> ObjectSpec | None:
        """作業領域内の、他の物体と重ならない位置に置く。ok（ObjectSpec → bool）を渡すと、それも満たす位置を探す（v18）。"""
        for _ in range(self.sp.max_tries):
            p = self._xy(rng)
            if all(dist(p, q.pose.xy) >= self.obj_r[t] + self.obj_r[q.type] + self.sp.object_gap_m for q in objects):
                o = ObjectSpec(name, t, Pose2D(p[0], p[1], float(rng.uniform(-math.pi, math.pi))), color)
                if ok is None or ok(o):
                    return o
        return None

    def _try_stage7(self, rng) -> EpisodeSpec | None:
        """同じ色の LEGO を 2〜5 個まとめて運ぶ（v7）。対象外は色か種類が違う。"""
        sp = self.sp
        goal_name, goal = self._goal(rng)
        color = COLORS[int(rng.integers(len(COLORS)))]
        n_t = int(rng.integers(sp.stage7_targets[0], sp.stage7_targets[1] + 1))
        n_o = int(rng.integers(sp.stage7_clutter[0], sp.stage7_clutter[1] + 1))
        objects: list[ObjectSpec] = []
        for i in range(n_t):
            tt = self._target_type(rng)
            if i > 0 and rng.random() < sp.cluster_probability:
                # v9: 対象どうしをくっつけて置く（設計書 v7 の記述どおり）
                base = objects[int(rng.integers(len(objects)))]
                o = self._neighbor(rng, base, objects, f"obj{i}", tt, color)
            elif i > 0 and self._group(rng):
                # v29: それまでの対象の近く（接しない）に置く。まとめて押す練習
                o = self._near(rng, objects[int(rng.integers(len(objects)))], objects, f"obj{i}", tt, color)
            else:
                o = self._place(rng, objects, tt, f"obj{i}", color)
            if o is None or goal.contains(o.pose.xy):
                return None
            objects.append(o)
        for i in range(n_t, n_t + n_o):
            t = self._type(rng)
            others = self._clutter_colors(t, color)
            o = self._place(rng, objects, t, f"obj{i}", others[int(rng.integers(len(others)))])
            if o is None or not self._clear_of_goal(o, goal):
                return None
            objects.append(o)
        m = sp.robot_margin_m
        rx = float(rng.uniform(-self.half[0] + m, self.half[0] - m))
        ry = float(rng.uniform(-self.half[1] + m, self.half[1] - m))
        if any(dist((rx, ry), q.pose.xy) < self.robot_r + self.obj_r[q.type] + sp.object_gap_m for q in objects):
            return None
        robot = Pose2D(rx, ry, float(rng.uniform(-math.pi, math.pi)))
        return EpisodeSpec(7, robot, tuple(objects), tuple(f"obj{i}" for i in range(n_t)), None, goal_name, goal)

    def _clear_of(self, o: ObjectSpec, rect: Rect) -> bool:
        """外形の 4 隅が長方形から goal_clearance_m 以上離れている。"""
        corners = box_corners(o.pose, self.sim.objects[OBJECT_TYPE_NAMES[o.type]].size_m[:2])
        return min(signed_distance_to_rect(c, rect) for c in corners) >= self.sp.goal_clearance_m

    def _rects_apart(self, a: Rect, b: Rect) -> bool:
        """2 つの長方形の間に goal_clearance_m 以上の隙間がある（x か y のどちらかの方向で）。"""
        g = self.sp.goal_clearance_m
        return (abs(a.center[0] - b.center[0]) >= (a.size[0] + b.size[0]) / 2 + g
                or abs(a.center[1] - b.center[1]) >= (a.size[1] + b.size[1]) / 2 + g)

    def _zones(self, rng, goal_name: str, goal: Rect) -> tuple[Rect, ...] | None:
        """Stage 8 の置き場（v18）。zones: 今のゴール以外の名前付きゴールから選ぶ。random: 乱数の長方形。"""
        n = int(rng.integers(self.sp.stage8_zones[0], self.sp.stage8_zones[1] + 1))
        if self.sp.goal_mode == "zones":
            rest = [k for k in self.goal_names if k != goal_name]
            return tuple(self.goals[rest[i]] for i in sorted(rng.choice(len(rest), size=n, replace=False)))
        zones: list[Rect] = []
        for _ in range(n):   # 置き場ごとに、ゴールと他の置き場から離れた長方形が出るまで引き直す
            for _ in range(self.sp.max_tries):
                _, z = self._goal(rng)
                if all(self._rects_apart(z, q) for q in (goal, *zones)):
                    zones.append(z)
                    break
            else:
                return None
        return tuple(zones)

    def _place_in(self, rng, objects: list[ObjectSpec], t: ObjectType, name: str, color: str,
                  zone: Rect, ok=None) -> ObjectSpec | None:
        """外形の 4 隅が zone に入り、他の物体と重ならない位置に置く。ok（ObjectSpec → bool）を渡すと、それも満たす
        位置を探す（v24。向きを引いた後に判定するので、ok が無いときと乱数の引き方は同じ）。"""
        r = self.obj_r[t]
        hx, hy = zone.size[0] / 2 - r, zone.size[1] / 2 - r
        if hx <= 0 or hy <= 0:
            raise ValueError(f"置き場 {zone} が物体（外接円の半径 {r} m）より小さい")
        for _ in range(self.sp.max_tries):
            p = (zone.center[0] + float(rng.uniform(-hx, hx)), zone.center[1] + float(rng.uniform(-hy, hy)))
            if all(dist(p, q.pose.xy) >= r + self.obj_r[q.type] + self.sp.object_gap_m for q in objects):
                o = ObjectSpec(name, t, Pose2D(p[0], p[1], float(rng.uniform(-math.pi, math.pi))), color)
                if ok is None or ok(o):
                    return o
        return None

    def _try_stage8(self, rng) -> EpisodeSpec | None:
        """連続の色分けの 1 回分（v18）: Stage 7 に、前の指示で運んだ物体の入った置き場を足す。"""
        sp = self.sp
        goal_name, goal = self._goal(rng)
        zones = self._zones(rng, goal_name, goal)
        if zones is None:
            return None
        color = COLORS[int(rng.integers(len(COLORS)))]
        others = [c for c in COLORS if c != color]
        objects: list[ObjectSpec] = []
        k = 0
        for z in zones:   # 置き場の物体: 置き場ごとに 1 色の LEGO（ミニカーは toy_car_probability）
            zc = others[int(rng.integers(len(others)))]
            for _ in range(int(rng.integers(sp.stage8_zone_objects[0], sp.stage8_zone_objects[1] + 1))):
                o = self._place_in(rng, objects, self._type(rng), f"obj{k}", zc, z)
                if o is None:
                    return None
                objects.append(o)
                k += 1
        def target_ok(o):
            return not goal.contains(o.pose.xy) and all(self._clear_of(o, z) for z in zones)

        def clutter_ok(o):
            return self._clear_of_goal(o, goal) and all(self._clear_of(o, z) for z in zones)

        # 物体は 1 個ずつ、条件を満たす位置が出るまで引き直す（物体が多く、盤面ごと引き直すとほとんど通らない）。
        # 接して置く（確率 cluster_probability）位置が条件を外れたら、自由な位置に置く
        n_t = int(rng.integers(sp.stage7_targets[0], sp.stage7_targets[1] + 1))
        targets: list[ObjectSpec] = []
        for i in range(n_t):
            o = None
            tt = self._target_type(rng)
            if i > 0 and rng.random() < sp.cluster_probability:
                o = self._neighbor(rng, targets[int(rng.integers(len(targets)))], objects, f"obj{k}", tt, color)
            elif i > 0 and self._group(rng):   # v29: それまでの対象の近く（接しない）
                o = self._near(rng, targets[int(rng.integers(len(targets)))], objects, f"obj{k}", tt, color)
            if o is None or not target_ok(o):
                o = self._place(rng, objects, tt, f"obj{k}", color, target_ok)
            if o is None:
                return None
            objects.append(o)
            targets.append(o)
            k += 1
        for _ in range(int(rng.integers(sp.stage8_clutter[0], sp.stage8_clutter[1] + 1))):
            t = self._type(rng)
            pool = self._clutter_colors(t, color)
            c = pool[int(rng.integers(len(pool)))]
            o = None
            if rng.random() < sp.cluster_probability:   # 対象に接して置く（Stage 6 と同じ。引き離す練習）
                o = self._neighbor(rng, targets[int(rng.integers(len(targets)))], objects, f"obj{k}", t, c)
            if o is None or not clutter_ok(o):
                o = self._place(rng, objects, t, f"obj{k}", c, clutter_ok)
            if o is None:
                return None
            objects.append(o)
            k += 1
        m = sp.robot_margin_m
        for _ in range(sp.max_tries):
            rx = float(rng.uniform(-self.half[0] + m, self.half[0] - m))
            ry = float(rng.uniform(-self.half[1] + m, self.half[1] - m))
            if all(dist((rx, ry), q.pose.xy) >= self.robot_r + self.obj_r[q.type] + sp.object_gap_m for q in objects):
                break
        else:
            return None
        robot = Pose2D(rx, ry, float(rng.uniform(-math.pi, math.pi)))
        return EpisodeSpec(8, robot, tuple(objects), tuple(o.name for o in targets), None, goal_name, goal, zones)

    def _robot_clear(self, robot: Pose2D, o: ObjectSpec) -> bool:
        """ロボットと物体の外接円の間に object_gap_m 以上（Stage 8 のロボットの置き方と同じ式）。"""
        return dist(robot.xy, o.pose.xy) >= self.robot_r + self.obj_r[o.type] + self.sp.object_gap_m

    def _in_margin(self, x: float, y: float) -> bool:
        """中心が作業領域の端から object_margin_m 以上内側（_xy と同じ範囲）。"""
        m = self.sp.object_margin_m
        return abs(x) <= self.half[0] - m and abs(y) <= self.half[1] - m

    def _relocated_ok(self, robot: Pose2D):
        """ゴールの中へ置き直す物体（v24）の、_place_in の重なりの判定に足す条件: 作業領域の余白の内側、ロボットと重ならない。"""
        return lambda o: self._in_margin(o.pose.x, o.pose.y) and self._robot_clear(robot, o)

    def _place_out(self, rng, objects: list[ObjectSpec], o: ObjectSpec, zone: Rect, avoid: tuple[Rect, ...],
                   robot: Pose2D) -> ObjectSpec | None:
        """置き場 zone の縁の外、中心の符号付き距離が stage9_home_out_m の範囲に置き直す（v24「出してしまった」）。

        zone を stage9_home_out_m の上限だけ広げた長方形から、中心 x、y、向きの順に毎回 3 つ引き、作業領域の
        object_margin_m の内側、他の物体・ロボットと重ならない、avoid（今のゴールと他の置き場）にかからない
        （_clear_of）位置が出るまで max_tries 回まで引き直す。
        """
        lo, hi = self.sp.stage9_home_out_m
        r = self.obj_r[o.type]
        wx, wy = zone.size[0] / 2 + hi, zone.size[1] / 2 + hi
        for _ in range(self.sp.max_tries):
            x = zone.center[0] + float(rng.uniform(-wx, wx))
            y = zone.center[1] + float(rng.uniform(-wy, wy))
            cand = ObjectSpec(o.name, o.type, Pose2D(x, y, float(rng.uniform(-math.pi, math.pi))), o.color)
            if (lo <= signed_distance_to_rect((x, y), zone) <= hi
                    and self._in_margin(x, y)
                    and all(dist((x, y), q.pose.xy) >= r + self.obj_r[q.type] + self.sp.object_gap_m for q in objects)
                    and self._robot_clear(robot, cand) and all(self._clear_of(cand, z) for z in avoid)):
                return cand
        return None

    def _wedge(self, rng, objects: list[ObjectSpec], o: ObjectSpec, targets: list[ObjectSpec], goal: Rect,
               robot: Pose2D) -> ObjectSpec | None:
        """届けた対象に挟まった対象外（v35）: 毎回 targets から integers で 1 個選び、_neighbor でそれに接して置く。4 隅が
        goal の中、作業領域の余白の内側、ロボットと重ならない（_relocated_ok）位置が出るまで max_tries 回まで引き直す。"""
        ok = self._relocated_ok(robot)
        size = self.sim.objects[OBJECT_TYPE_NAMES[o.type]].size_m[:2]
        for _ in range(self.sp.max_tries):
            t = targets[int(rng.integers(len(targets)))]
            cand = self._neighbor(rng, t, objects, o.name, o.type, o.color)
            if cand is not None and goal.contains_all(box_corners(cand.pose, size)) and ok(cand):
                return cand
        return None

    def _try_stage9(self, rng) -> EpisodeSpec | None:
        """後片付けから始まる盤面（v24）: Stage 8 の盤面に、確率 stage9_mistake_probability で「やり残し」を加える。

        乱数の引き方（Rust 版 native/src/sampler.rs も同じ順）:
        1. Stage 8 の盤面（_try_stage8）。
        2. random() < stage9_mistake_probability でなければ、そのまま返す（stage だけ 9）。
        3. random() < stage9_all_delivered_probability なら、対象を target_names の順に今のゴールの中へ置き直す
           （_place_in。作業領域の余白の内側、ロボットと重ならない）。
        4. やり残しの数 integers(1, 3)。
        5. 1 個ずつ: 候補は置き場の無い対象外（「入れてしまった」）と、置き場が zones の物体（「出してしまった」）。
           番号は Stage 8 の盤面の開始の位置から決め、選んだ物体は除く。両方あれば random() < 0.5 で「入れてしまった」、
           片方だけならそれ（乱数なし）、両方無ければ None。候補から integers(候補の数) で 1 個選び、「入れてしまった」は
           手順 3 と同じ条件で今のゴールの中へ、「出してしまった」は _place_out で元の置き場の縁の外へ置き直す。
           ただし手順 3 で対象を置き直し、stage9_wedged_probability > 0 で random() < それなら、「入れてしまった」は
           挟まった置き方（_wedge、v35）にする。
        置けなければ None（sample() が盤面ごと引き直す）。置き場の指定（home_overrides、物体の順）: 入れてしまった物体と
        置き直した対象は −1、出してしまった物体は元の置き場の番号。
        """
        sp = self.sp
        base = self._try_stage8(rng)
        if base is None:
            return None
        if not rng.random() < sp.stage9_mistake_probability:
            return dataclasses.replace(base, stage=9)
        goal, zones, robot = base.goal, base.zones, base.robot
        objects = list(base.objects)
        index = {o.name: i for i, o in enumerate(objects)}
        targets = set(base.target_names)
        over: dict[str, int] = {}

        def others(name):
            return [q for q in objects if q.name != name]

        delivered = rng.random() < sp.stage9_all_delivered_probability
        if delivered:
            for n in base.target_names:
                t = objects[index[n]]
                o = self._place_in(rng, others(n), t.type, n, t.color, goal, self._relocated_ok(robot))
                if o is None:
                    return None
                objects[index[n]] = o
                over[n] = -1
        home = {o.name: start_home_index(self.sim, base, o) for o in base.objects if o.name not in targets}
        for _ in range(int(rng.integers(1, 3))):
            ins = [n for n, h in home.items() if h == -1 and n not in over]
            outs = [n for n, h in home.items() if h >= 1 and n not in over]
            if not ins and not outs:
                return None
            put_in = (rng.random() < 0.5) if (ins and outs) else bool(ins)
            pool = ins if put_in else outs
            n = pool[int(rng.integers(len(pool)))]
            b = objects[index[n]]
            if put_in and delivered and sp.stage9_wedged_probability > 0 and rng.random() < sp.stage9_wedged_probability:
                o = self._wedge(rng, others(n), b, [objects[index[t]] for t in base.target_names], goal, robot)
            elif put_in:
                o = self._place_in(rng, others(n), b.type, n, b.color, goal, self._relocated_ok(robot))
            else:
                zone = zones[home[n] - 1]
                avoid = (goal, *(z for i, z in enumerate(zones) if i != home[n] - 1))
                o = self._place_out(rng, others(n), b, zone, avoid, robot)
            if o is None:
                return None
            objects[index[n]] = o
            over[n] = -1 if put_in else home[n]
        return dataclasses.replace(base, stage=9, objects=tuple(objects),
                                   home_overrides=tuple((o.name, over[o.name]) for o in objects if o.name in over))

    def _try(self, stage: int, rng) -> EpisodeSpec | None:
        if stage == 9:
            return self._try_stage9(rng)
        if stage == 5:
            return self._try_stage5(rng)
        if stage == 7:
            return self._try_stage7(rng)
        if stage == 8:
            return self._try_stage8(rng)
        goal_name, goal = self._goal(rng)
        o = self._xy(rng)
        if goal.contains(o):
            return None
        to_goal = math.atan2(goal.center[1] - o[1], goal.center[0] - o[0])

        if stage == 0:
            robot = self._robot_near(rng, o, to_goal, 0)
            return None if robot is None else EpisodeSpec(0, robot, (), (), o, goal_name, goal)

        ttype = {1: ObjectType.LEGO, 2: ObjectType.LEGO, 3: ObjectType.LEGO, 4: ObjectType.TOY_CAR}.get(stage)
        if ttype is None:  # Stage 6
            ttype = self._type(rng)
        target = ObjectSpec("obj0", ttype, Pose2D(o[0], o[1], float(rng.uniform(-math.pi, math.pi))),
                            COLORS[int(rng.integers(len(COLORS)))])
        objects = [target]

        if stage != 6:
            robot = self._robot_near(rng, o, to_goal, stage)
            if robot is None:
                return None
        else:
            n = int(rng.integers(self.sp.clutter_range[0], self.sp.clutter_range[1] + 1))
            neighbors = 0
            if rng.random() < self.sp.cluster_probability:
                neighbors = min(n, int(rng.integers(1, self.sp.cluster_max_neighbors + 1)))
            for i in range(1, neighbors + 1):
                nb = self._neighbor(rng, target, objects, f"obj{i}")
                if nb is None or not self._clear_of_goal(nb, goal):
                    return None
                objects.append(nb)
            for i in range(neighbors + 1, n + 1):
                for _ in range(self.sp.max_tries):
                    t = self._type(rng)
                    p = self._xy(rng)
                    cand = ObjectSpec(f"obj{i}", t, Pose2D(p[0], p[1], float(rng.uniform(-math.pi, math.pi))),
                                      COLORS[int(rng.integers(len(COLORS)))])
                    if all(dist(p, q.pose.xy) >= self.obj_r[t] + self.obj_r[q.type] + self.sp.object_gap_m
                           for q in objects) and self._clear_of_goal(cand, goal):
                        break
                else:
                    return None
                objects.append(cand)
            m = self.sp.robot_margin_m
            rx = float(rng.uniform(-self.half[0] + m, self.half[0] - m))
            ry = float(rng.uniform(-self.half[1] + m, self.half[1] - m))
            if any(dist((rx, ry), q.pose.xy) < self.robot_r + self.obj_r[q.type] + self.sp.object_gap_m for q in objects):
                return None
            robot = Pose2D(rx, ry, float(rng.uniform(-math.pi, math.pi)))
        return EpisodeSpec(stage, robot, tuple(objects), ("obj0",), None, goal_name, goal)
