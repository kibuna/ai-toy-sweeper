"""計画器が読む世界の状態（俯瞰カメラの追跡で得られる量だけ）と、その上の幾何の判定。

読むもの:
- ロボットの姿勢、物体ごとの姿勢と種類（直近の Observation を作った観測値。観測ノイズを入れた設定ならノイズ込み）
- 物体ごとの開始時の姿勢、どれが対象か、ゴールの長方形、作業領域（タスク開始時に決まる量）
- グリッパーの開閉の状態（指令した状態。カメラと指令の記録で分かる）
読まないもの: Sim2D の把持状態（grasped_object）、爪の角度、物体の速度。「持っているか」は姿勢から幾何で推定する。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import SimCfg
from ..geometry import Pose2D, Rect, box_corners, signed_distance_to_rect
from ..robot_model import front_x
from ..spaces import OBJECT_TYPE_NAMES, GripperState, ObjectType


@dataclass(frozen=True)
class ObjState:
    name: str
    type: ObjectType
    pose: Pose2D
    initial: Pose2D
    is_target: bool


@dataclass(frozen=True)
class WorldState:
    robot: Pose2D
    gripper: GripperState
    objects: tuple[ObjState, ...]
    goal: Rect
    area: Rect

    def obj(self, name: str) -> ObjState:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(f"物体 {name!r} が観測にない")


def world_state_from_env(env) -> WorldState:
    """ToySweeperEnv から WorldState を作る。姿勢は env.observed_poses()（Observation と同じ観測値）。"""
    if env.episode is None:
        raise RuntimeError("env.reset() の前に状態を読もうとした")
    poses, robot = env.observed_poses()
    ep = env.episode
    targets = set(ep.target_names)
    missing = [o.name for o in ep.objects if o.name not in poses]
    if missing:
        raise KeyError(f"観測に無い物体がある: {missing}")
    objs = tuple(ObjState(o.name, o.type, poses[o.name], o.pose, o.name in targets) for o in ep.objects)
    return WorldState(robot, env.backend.gripper_state(), objs, ep.goal, env.area)


def to_local(robot: Pose2D, p: tuple[float, float]) -> tuple[float, float]:
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    dx, dy = p[0] - robot.x, p[1] - robot.y
    return c * dx + s * dy, -s * dx + c * dy


class Geometry:
    """物体の寸法と、ゴール・把持の判定（設定から決まる量をまとめる）。"""

    def __init__(self, sim: SimCfg, held_lateral_tol_m: float):
        self.sim = sim
        self.size = {t: tuple(sim.objects[OBJECT_TYPE_NAMES[t]].size_m[:2]) for t in ObjectType}
        self.radius = {t: math.hypot(*self.size[t]) / 2 for t in ObjectType}
        g = sim.gripper
        self.jaw_x0 = front_x(sim) + g.pivot_x_from_front_m
        self.jaw_x1 = self.jaw_x0 + g.jaw_length_m
        self.held_tol = held_lateral_tol_m

    def corners(self, o: ObjState, pose: Pose2D | None = None):
        return box_corners(pose if pose is not None else o.pose, self.size[o.type])

    def inside_goal(self, goal: Rect, o: ObjState, pose: Pose2D | None = None) -> bool:
        """配達の判定（env と同じ: 4 隅がゴール内）。"""
        return goal.contains_all(self.corners(o, pose))

    def goal_depth(self, goal: Rect, o: ObjState) -> float:
        """4 隅のうち最もゴールの縁に近い隅の、縁までの距離（中は正、外は負）。"""
        return min(-signed_distance_to_rect(c, goal) for c in self.corners(o))

    def started_in_goal(self, goal: Rect, o: ObjState) -> bool:
        return self.inside_goal(goal, o, o.initial)

    def is_foreign_in_goal(self, goal: Rect, o: ObjState) -> bool:
        """env の成功条件で数える「入り込んだ対象外」（もともと外にあった対象外で、中心がゴール内）。"""
        return (not o.is_target) and (not self.started_in_goal(goal, o)) and goal.contains(o.pose.xy)

    def held_object(self, w: WorldState) -> ObjState | None:
        """閉じた爪の間（前後は梁の根元の手前から先端、左右 held_lateral_tol_m）にある物体。複数なら中心線に近いもの。"""
        if w.gripper != GripperState.CLOSED:
            return None
        best, best_d = None, math.inf
        for o in w.objects:
            lx, ly = to_local(w.robot, o.pose.xy)
            if self.jaw_x0 <= lx <= self.jaw_x1 and abs(ly) <= self.held_tol and abs(ly) < best_d:
                best, best_d = o, abs(ly)
        return best
