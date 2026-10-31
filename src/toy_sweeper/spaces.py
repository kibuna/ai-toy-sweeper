"""実機・Gazebo・2D シミュレータで共通の Observation / Action 定義。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from gymnasium import spaces

from .config import RobotCfg, SimCfg, TaskCfg
from .geometry import Pose2D, Rect, box_corners, signed_distance_to_rect, wrap_angle


class Action(IntEnum):
    STOP = 0
    FORWARD = 1
    BACKWARD = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    OPEN_GRIPPER = 5
    CLOSE_GRIPPER = 6


N_ACTIONS = len(Action)
DRIVE_ACTIONS = (Action.STOP, Action.FORWARD, Action.BACKWARD, Action.TURN_LEFT, Action.TURN_RIGHT)


def drive_velocity(robot: RobotCfg, a: Action) -> tuple[float, float]:
    """走行の Action の指令速度 (linear m/s, angular rad/s)。学習環境と現場のランタイムで共通。"""
    if a not in DRIVE_ACTIONS:
        raise ValueError(f"走行の Action でない: {a!r}")
    return {
        Action.STOP: (0.0, 0.0),
        Action.FORWARD: (robot.linear_speed_mps, 0.0),
        Action.BACKWARD: (-robot.linear_speed_mps, 0.0),
        Action.TURN_LEFT: (0.0, robot.angular_speed_radps),
        Action.TURN_RIGHT: (0.0, -robot.angular_speed_radps),
    }[a]


class GripperState(IntEnum):
    OPEN = 0
    CLOSED = 1
    MOVING = 2
    UNKNOWN = 3


class ObjectType(IntEnum):
    LEGO = 0
    TOY_CAR = 1


OBJECT_TYPE_NAMES = {ObjectType.LEGO: "lego", ObjectType.TOY_CAR: "toy_car"}


class TaskMode(IntEnum):
    """主対象について、いまの仕事（v26、task.observation.cleanup_primary）。"""
    DELIVER = 0   # 対象をゴールへ運ぶ
    RETURN = 1    # 置き場から出た対象外を置き場へ戻す
    REMOVE = 2    # ゴールの中の置き場を持たない対象外を外へ出す


@dataclass(frozen=True)
class SlotObject:
    """物体スロットに入れる 1 個（v9）。"""
    pose: Pose2D
    type: ObjectType
    displacement: tuple[float, float]   # 開始時の位置からの移動（世界座標、m）。初期位置が分からない追跡は呼び出し側が決める
    # 置き場（v18）: 開始時に外形の 4 隅が入っていた今のゴールか EpisodeSpec.zones の長方形。無ければ None。
    # v17 までの「開始時にゴール内」の一般化で、置き場が今のゴールのときは同じ意味。盤面に置き場の指定
    # （EpisodeSpec.home_overrides、v24）があればそれに従う
    home: Rect | None


@dataclass(frozen=True)
class WorldSnapshot:
    """Observation を作るための入力。Gazebo の真値でもカメラ推定でもよい。"""
    robot: Pose2D
    target: Pose2D            # Stage 0 は指定点（yaw は 0）
    target_type: ObjectType | None  # Stage 0 は None
    goal: Rect
    gripper: GripperState
    prev_action: Action
    obs_age_s: float
    other_targets: tuple[SlotObject, ...]  # 主対象以外の対象（順不同。Builder がロボットに近い順に並べる）
    obstacles: tuple[SlotObject, ...]      # 対象外の物体（同上）
    n_remaining: int                       # ゴールに入っていない対象の数
    robot_motion: tuple[float, float, float]  # 直前 1 ステップのロボットの移動（前のロボット座標で Δx, Δy, Δyaw）（v9）
    time_left: float                       # 残りステップ / 打ち切りのステップ数（v9）
    target_held: bool                      # 主対象を挟んでいるか（観測した位置から計算）（v9）
    # v26: 主対象の行き先（頭のゴールの欄）と、いまの仕事。運ぶ・出すでは今のゴール、戻すでは主対象の置き場。
    # Stage 0 は task_mode が None。cleanup_primary が無効なら、いつも destination == goal・DELIVER（か None）
    destination: Rect
    task_mode: TaskMode | None
    # v30: 盤面の作業領域の (幅, 奥行き)（task.episode_area）。観測に入れるのは task.observation.area_size のときだけ
    area_size: tuple[float, float]


# (名前, 次元) の並び
LAYOUT: tuple[tuple[str, int], ...] = (
    ("robot_xy", 2),
    ("robot_heading", 2),
    ("target_xy", 2),
    ("target_heading2", 2),
    ("goal_center", 2),
    ("goal_size", 2),
    ("robot_to_target", 3),
    ("target_to_goal", 3),
    ("target_type", 2),
    ("gripper", 4),
    ("prev_action", N_ACTIONS),
    ("obs_age", 1),
    ("robot_motion", 3),        # v9
    ("time_left", 1),           # v9
    ("target_goal_sdist", 1),   # v9: 主対象の中心からゴールの縁までの符号付き距離（中は負）
    ("target_held", 1),         # v9
)
BASE_DIM = sum(n for _, n in LAYOUT)
EXTRA_DIM = 1     # 残りの対象の数（v7）
MODE_DIM = 3      # いまの仕事 TaskMode の one-hot（v26、cleanup_primary のときだけ。残りの対象の数の後）
AREA_DIM = 2      # 作業領域の半分の幅・奥行き / position_scale_m（v30、area_size のときだけ。頭の終わり）
# 相対位置 2、向き 2、種類 2、開始時からの変位 2（ロボット座標）、ゴール（置き場を持つ物体は置き場）の縁までの
# 符号付き距離 1、置き場を持つ 1（v18、v17 までは「開始時にゴール内」）、対象か 1、存在マスク 1（v9）
SLOT_DIM = 12
SLICES: dict[str, slice] = {}
_i = 0
for _name, _n in LAYOUT:
    SLICES[_name] = slice(_i, _i + _n)
    _i += _n

# 観測の値域。正規化後の位置は作業領域内で [-1, 1]。領域外に出たらエピソードは終わるので
# ±3 を超える値は不正な入力とみなして例外にする（黙って切り詰めない）。
OBS_BOUND = 3.0


def obs_bound(sim: SimCfg, task: TaskCfg) -> float:
    """観測の値域（v30）。±OBS_BOUND は 0.8 m 四方・尺度 0.4 m で決めた（相対位置は最大 0.8√2 / 0.4 ≒ 2.83）。盤面の作業領域が
    広くなると相対位置も大きくなる（1.0 m 四方の外へ 0.10 m ずつ出てよいと最大 1.7 m / 0.4 ≒ 4.2）ので、いちばん広い
    作業領域（壁なしなら外へ出てよい分を足す）と尺度の比で広げる。今までの設定（0.8 m 四方）では ±3 のまま。"""
    ar = task.sampling.work_area_size_range_m
    ext = max(ar[0][1], ar[1][1]) if ar else max(sim.work_area.size_m)
    if not sim.walls.enabled:
        ext += 2 * sim.walls.out_of_area_margin_m
    return OBS_BOUND * max(1.0, ext / (2 * min(task.observation.position_scale_m)))


def _one_hot(i: int | None, n: int) -> list[float]:
    v = [0.0] * n
    if i is not None:
        v[i] = 1.0
    return v


def _decode_mode(v: np.ndarray) -> TaskMode | None:
    return None if v.sum() == 0 else TaskMode(int(np.argmax(v)))


class ObservationBuilder:
    def __init__(self, sim: SimCfg, task: TaskCfg):
        # 位置の尺度（v30 で sim.work_area から分けた。盤面ごとに作業領域が変わっても同じ 1 m が同じ値になる）
        self.half = tuple(task.observation.position_scale_m)
        self.size = (2 * self.half[0], 2 * self.half[1])
        self.diag = math.hypot(*self.size)
        self.action_duration = task.control.action_duration_s
        self.n_target_slots = task.observation.n_target_slots
        self.n_obstacle_slots = task.observation.n_obstacle_slots
        self.max_targets = task.sampling.stage7_targets[1]
        # 直前 1 ステップの移動の正規化: 1 Action で進む距離・回る角度
        self.step_m = sim.robot.linear_speed_mps * task.control.action_duration_s
        self.step_rad = sim.robot.angular_speed_radps * task.control.action_duration_s
        self.cleanup = task.observation.cleanup_primary
        self.area = task.observation.area_size
        self.head_dim = BASE_DIM + EXTRA_DIM + (MODE_DIM if self.cleanup else 0) + (AREA_DIM if self.area else 0)
        self.dim = self.head_dim + SLOT_DIM * (self.n_target_slots + self.n_obstacle_slots)
        self.bound = obs_bound(sim, task)
        self.observation_space = spaces.Box(-self.bound, self.bound, shape=(self.dim,), dtype=np.float32)
        self.corners = task.observation.edge_distance == "corners"
        self.sizes = {t: sim.objects[OBJECT_TYPE_NAMES[t]].size_m[:2] for t in ObjectType}

    def _edge_distance(self, p: Pose2D, otype: ObjectType, zone: Rect, by_corners: bool) -> float:
        """縁までの符号付き距離（外は正）。by_corners なら外形の 4 隅のうちいちばん外の角（v25。0 以下 ⟺ 4 隅が中）。"""
        if not by_corners:
            return signed_distance_to_rect(p.xy, zone)
        return max(signed_distance_to_rect(c, zone) for c in box_corners(p, self.sizes[otype]))

    def _slot_features(self, r: Pose2D, goal: Rect, objs, n_slots: int, is_target: bool) -> list[float]:
        c, s = math.cos(r.yaw), math.sin(r.yaw)
        rel = []
        for o in objs:
            p = o.pose
            dx, dy = p.x - r.x, p.y - r.y
            mx, my = o.displacement
            # 置き場を持つ物体は置き場の縁まで、持たない物体は今のゴールの縁までの符号付き距離（v18）。
            # corners（v25）では、成功の条件が 4 隅で決まる物体（置き場を持つ物体、対象）をいちばん外の角で測る
            zone = goal if o.home is None else o.home
            sd = self._edge_distance(p, o.type, zone, self.corners and (is_target or o.home is not None))
            rel.append((math.hypot(dx, dy), c * dx + s * dy, -s * dx + c * dy, wrap_angle(p.yaw - r.yaw), o.type,
                        c * mx + s * my, -s * mx + c * my, sd, o.home is not None))
        rel.sort(key=lambda e: e[0])
        v: list[float] = []
        for _, lx, ly, a, t, mx, my, sd, sig in rel[:n_slots]:
            v += [lx / self.half[0], ly / self.half[1], math.sin(2 * a), math.cos(2 * a), *_one_hot(int(t), 2),
                  mx / self.diag, my / self.diag, sd / self.half[0], 1.0 if sig else 0.0,
                  1.0 if is_target else 0.0, 1.0]
        v += [0.0] * (SLOT_DIM * (n_slots - min(len(rel), n_slots)))
        return v

    def build(self, s: WorldSnapshot) -> np.ndarray:
        if not self.cleanup and (s.task_mode not in (None, TaskMode.DELIVER) or s.destination != s.goal):
            raise ValueError(f"cleanup_primary が無効なのに、運ぶ以外の仕事か今のゴール以外の行き先: {s.task_mode}, {s.destination}")
        # 頭のゴールの欄は行き先（v26）。スロットの縁までの距離（置き場を持たない物体）は今のゴール
        r, t, g = s.robot, s.target, s.destination
        dx, dy = t.x - r.x, t.y - r.y
        rel_bearing = wrap_angle(math.atan2(dy, dx) - r.yaw)
        gx, gy = g.center[0] - t.x, g.center[1] - t.y
        goal_bearing = math.atan2(gy, gx)
        v = [
            r.x / self.half[0], r.y / self.half[1],
            math.sin(r.yaw), math.cos(r.yaw),
            t.x / self.half[0], t.y / self.half[1],
            math.sin(2 * t.yaw), math.cos(2 * t.yaw),
            g.center[0] / self.half[0], g.center[1] / self.half[1],
            g.size[0] / self.size[0], g.size[1] / self.size[1],
            math.hypot(dx, dy) / self.diag, math.sin(rel_bearing), math.cos(rel_bearing),
            math.hypot(gx, gy) / self.diag, math.sin(goal_bearing), math.cos(goal_bearing),
            *_one_hot(None if s.target_type is None else int(s.target_type), 2),
            *_one_hot(int(s.gripper), 4),
            *_one_hot(int(s.prev_action), N_ACTIONS),
            s.obs_age_s / self.action_duration,
            s.robot_motion[0] / self.step_m, s.robot_motion[1] / self.step_m, s.robot_motion[2] / self.step_rad,
            s.time_left,
            # 出す（v26）は「中心がゴールの外」が条件なので中心で測る
            self._edge_distance(t, s.target_type, g, self.corners and s.target_type is not None
                                and s.task_mode != TaskMode.REMOVE) / self.half[0],
            1.0 if s.target_held else 0.0,
            s.n_remaining / self.max_targets,
            *(_one_hot(None if s.task_mode is None else int(s.task_mode), MODE_DIM) if self.cleanup else []),
            *((s.area_size[0] / 2 / self.half[0], s.area_size[1] / 2 / self.half[1]) if self.area else ()),
            *self._slot_features(r, s.goal, s.other_targets, self.n_target_slots, True),
            *self._slot_features(r, s.goal, s.obstacles, self.n_obstacle_slots, False),
        ]
        obs = np.asarray(v, dtype=np.float32)
        assert obs.shape == (self.dim,)
        if not np.all(np.isfinite(obs)) or np.any(np.abs(obs) > self.bound):
            raise ValueError(f"Observation が値域 ±{self.bound} の外: {s}")
        return obs

    def decode(self, obs: np.ndarray) -> dict:
        """Observation を意味のある値へ戻す（Heuristic Policy とテスト用）。"""
        o = np.asarray(obs, dtype=np.float64)
        h = np.array(self.half)
        tt = o[SLICES["target_type"]]
        return {
            "robot_xy": o[SLICES["robot_xy"]] * h,
            "robot_yaw": math.atan2(*o[SLICES["robot_heading"]]),
            "target_xy": o[SLICES["target_xy"]] * h,
            "target_axis": math.atan2(*o[SLICES["target_heading2"]]) / 2,   # 長辺の向き（180° の区別なし）
            "goal_center": o[SLICES["goal_center"]] * h,
            "goal_size": o[SLICES["goal_size"]] * np.array(self.size),
            "robot_to_target_dist": o[SLICES["robot_to_target"]][0] * self.diag,
            "robot_to_target_bearing": o[SLICES["robot_to_target"]][1:],
            "target_type": None if tt.sum() == 0 else ObjectType(int(np.argmax(tt))),
            "gripper": GripperState(int(np.argmax(o[SLICES["gripper"]]))),
            "prev_action": Action(int(np.argmax(o[SLICES["prev_action"]]))),
            "robot_motion": tuple(o[SLICES["robot_motion"]] * np.array([self.step_m, self.step_m, self.step_rad])),
            "time_left": float(o[SLICES["time_left"]][0]),
            "target_goal_sdist": float(o[SLICES["target_goal_sdist"]][0]) * self.half[0],
            "target_held": bool(o[SLICES["target_held"]][0]),
            "n_remaining": int(round(o[BASE_DIM] * self.max_targets)),
            **({"task_mode": _decode_mode(o[BASE_DIM + EXTRA_DIM:BASE_DIM + EXTRA_DIM + MODE_DIM])} if self.cleanup else {}),
            **({"area_size": tuple(o[self.head_dim - AREA_DIM:self.head_dim] * 2 * h)} if self.area else {}),
            "other_targets": self._decode_slots(o, 0, self.n_target_slots),
            "obstacles": self._decode_slots(o, self.n_target_slots, self.n_obstacle_slots),
        }

    def _decode_slots(self, o: np.ndarray, first: int, n: int) -> list[dict]:
        out = []
        base = self.head_dim
        for k in range(first, first + n):
            e = o[base + SLOT_DIM * k: base + SLOT_DIM * (k + 1)]
            if e[11] == 0:
                continue
            out.append({"local_xy": (e[0] * self.half[0], e[1] * self.half[1]),
                        "type": ObjectType(int(np.argmax(e[4:6]))),
                        "displacement_local": (e[6] * self.diag, e[7] * self.diag),
                        "goal_sdist": e[8] * self.half[0], "has_home": bool(e[9]), "is_target": bool(e[10])})
        return out
