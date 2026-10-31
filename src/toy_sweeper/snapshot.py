"""観測した姿勢から WorldSnapshot（Observation の入力）を作る純粋関数（v7）。

学習環境（ToySweeperEnv）と現場の非同期ランタイム（toy_sweeper.field）の両方が使う。env もバックエンドも要らない。
主対象の選び方: ゴールに入っていない対象のうち、グリッパー中心から接近点が最も近いもの。全対象がゴールに入ったら
前の主対象を持ち越す（呼び出し側が last_primary として持つ）。
"""
from __future__ import annotations

from .config import SimCfg, TaskCfg
from .geometry import Pose2D, Rect, box_corners, dist
from .spaces import OBJECT_TYPE_NAMES, Action, GripperState, ObjectType, SlotObject, TaskMode, WorldSnapshot
from .sim2d import ObjectSpec
from .task import EpisodeSpec, approach_point, episode_area, gripper_point, start_home_index


def inside_goal(sim: SimCfg, goal: Rect, otype: ObjectType, pose: Pose2D) -> bool:
    """物体の外形の 4 隅がすべてゴールの中にあるか（配達の判定）。"""
    size = sim.objects[OBJECT_TYPE_NAMES[otype]].size_m[:2]
    return goal.contains_all(box_corners(pose, size))


def approach_distance(sim: SimCfg, task: TaskCfg, goal: Rect, robot: Pose2D, target: Pose2D) -> float:
    """グリッパー中心から、対象の接近点（ゴールと反対側 approach_offset_m）までの距離。"""
    ap = approach_point(target.xy, goal.center, task.reward.approach_offset_m)
    return dist(gripper_point(sim, robot), ap)


def select_primary(ep: EpisodeSpec, sim: SimCfg, task: TaskCfg, poses: dict[str, Pose2D], robot: Pose2D,
                   last_primary: str) -> str:
    types = {o.name: o.type for o in ep.objects}
    left = [n for n in ep.target_names if not inside_goal(sim, ep.goal, types[n], poses[n])]
    if not left:
        return last_primary
    return min(left, key=lambda n: approach_distance(sim, task, ep.goal, robot, poses[n]))


def target_held(sim: SimCfg, task: TaskCfg, robot: Pose2D, gripper: GripperState, target: Pose2D) -> bool:
    """主対象を挟んでいるか。閉じた爪の間（左右 held_lateral_tolerance_m、前後は梁の根元の手前から先端）にあるか。"""
    if gripper != GripperState.CLOSED:
        return False
    import math
    from .robot_model import front_x
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    dx, dy = target.x - robot.x, target.y - robot.y
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    g = sim.gripper
    x0 = front_x(sim) + g.pivot_x_from_front_m
    return abs(ly) <= task.observation.held_lateral_tolerance_m and x0 <= lx <= x0 + g.jaw_length_m


def home_zone(sim: SimCfg, ep: EpisodeSpec, o: ObjectSpec) -> Rect | None:
    """物体の置き場（v18）: 開始時に外形の 4 隅が入っている、今のゴールか ep.zones の長方形。無ければ None。
    盤面に置き場の指定（ep.home_overrides、v24）があればそれに従う（task.start_home_index）。"""
    i = start_home_index(sim, ep, o)
    return None if i < 0 else (ep.goal, *ep.zones)[i]


def pending_fixes(ep: EpisodeSpec, sim: SimCfg, poses: dict[str, Pose2D]) -> tuple[list[str], list[str]]:
    """直すべき対象外（v26）: (置き場から出た物体, ゴールの中の置き場を持たない対象外)。どちらも物体の順。

    戻す: 置き場を持ち、外形の 4 隅が置き場に入っていない（env._n_home_out と同じ）。
    出す: 置き場を持たず、中心が今のゴールの中（env._n_foreign_in_goal のうち置き場を持たないもの。置き場を持つ物体が
    ゴールに入ったときは「戻す」に入れる。戻せばゴールからも出るので）。poses に無い物体（見えていない）は数えない。
    """
    targets = set(ep.target_names)
    ret, rem = [], []
    for o in ep.objects:
        if o.name in targets or o.name not in poses:
            continue
        home = home_zone(sim, ep, o)
        if home is not None:
            if not inside_goal(sim, home, o.type, poses[o.name]):
                ret.append(o.name)
        elif ep.goal.contains(poses[o.name].xy):
            rem.append(o.name)
    return ret, rem


def fix_destination_xy(goal: Rect, home: Rect | None, pose: Pose2D) -> tuple[float, float]:
    """直すべき対象外を押す先の点（v26。接近点はこの点と反対側）: 戻すは置き場の中心、出すはゴールの中心から物体の
    向こう側（物体をゴールの中心から遠ざける向き）。"""
    if home is not None:
        return home.center
    return (2 * pose.x - goal.center[0], 2 * pose.y - goal.center[1])


def fix_approach_distance(sim: SimCfg, task: TaskCfg, goal: Rect, home: Rect | None, robot: Pose2D, pose: Pose2D) -> float:
    """グリッパー中心から、直すべき対象外の接近点までの距離（v26。approach_distance の行き先を fix_destination_xy に）。"""
    ap = approach_point(pose.xy, fix_destination_xy(goal, home, pose), task.reward.approach_offset_m)
    return dist(gripper_point(sim, robot), ap)


def slot_object(sim: SimCfg, ep: EpisodeSpec, name: str, pose: Pose2D) -> SlotObject:
    """名前の分かっている物体のスロット（開始時の位置は EpisodeSpec の初期姿勢）。"""
    o = next(x for x in ep.objects if x.name == name)
    return SlotObject(pose, o.type, (pose.x - o.pose.x, pose.y - o.pose.y), home_zone(sim, ep, o))


def build_snapshot(ep: EpisodeSpec, sim: SimCfg, task: TaskCfg, poses: dict[str, Pose2D], robot: Pose2D,
                   gripper: GripperState, prev_action: Action, last_primary: str | None, *,
                   extra_obstacles: tuple[SlotObject, ...], obs_age_s: float,
                   robot_motion: tuple[float, float, float], time_left: float) -> tuple[WorldSnapshot, str | None]:
    """観測した姿勢から WorldSnapshot を作り、(snapshot, 今回の主対象) を返す。

    poses: 物体名 → 姿勢。対象は全部必要（無ければ KeyError）。対象外で poses に無いものは見えていないとして渡さない。
    extra_obstacles: 物体名と結び付いていない追跡（初期位置の扱いは呼び出し側が決めて SlotObject にする）。
    last_primary: 前回の主対象（最初は対象の先頭）。Stage 0（対象なし）は None。
    obs_age_s: 学習では常に 0（ToySweeperEnv）。robot_motion: 直前 1 ステップの移動。time_left: 残り時間の割合（0〜1）。
    """
    if ep.target_names and last_primary not in ep.target_names:
        raise ValueError(f"last_primary {last_primary!r} が対象 {ep.target_names} にない")
    if not ep.target_names and last_primary is not None:
        raise ValueError(f"対象のない盤面で last_primary が {last_primary!r}")
    if not 0.0 <= time_left <= 1.0:
        raise ValueError(f"time_left は 0〜1: {time_left}")
    if task.observation.area_size and ep.area_size is None:
        # 実機では校正の大きさを入れる。sim.work_area の固定に黙って落とさない（v30）
        raise ValueError("observation.area_size が有効なのに、盤面に作業領域の大きさ（EpisodeSpec.area_size）が無い")
    common = dict(robot=robot, goal=ep.goal, gripper=gripper, prev_action=prev_action, obs_age_s=obs_age_s,
                  robot_motion=robot_motion, time_left=time_left, area_size=episode_area(sim, ep))
    if not ep.target_names:  # Stage 0: 指定点
        return WorldSnapshot(target=Pose2D(ep.reach_point[0], ep.reach_point[1], 0.0), target_type=None,
                             other_targets=(), obstacles=tuple(extra_obstacles), n_remaining=0, target_held=False,
                             destination=ep.goal, task_mode=None, **common), None
    types = {o.name: o.type for o in ep.objects}
    primary = select_primary(ep, sim, task, poses, robot, last_primary)
    remaining = sum(not inside_goal(sim, ep.goal, types[n], poses[n]) for n in ep.target_names)
    # v26: 全対象が入っていて直すべき対象外があれば、この観測の主対象をそのうち接近点の最も近いもの（最初の最小）にする。
    # 返す主対象（次の last_primary）は対象のまま
    head, mode, dest = primary, TaskMode.DELIVER, ep.goal
    if task.observation.cleanup_primary and remaining == 0:
        ret, rem = pending_fixes(ep, sim, poses)
        specs = {o.name: o for o in ep.objects}
        cands = [(n, home_zone(sim, ep, specs[n])) for n in ret] + [(n, None) for n in rem]
        if cands:
            head, home = min(cands, key=lambda c: fix_approach_distance(sim, task, ep.goal, c[1], robot, poses[c[0]]))
            mode, dest = (TaskMode.RETURN, home) if home is not None else (TaskMode.REMOVE, ep.goal)
    others = tuple(slot_object(sim, ep, n, poses[n]) for n in ep.target_names if n != head)
    targets = set(ep.target_names)
    obstacles = tuple(slot_object(sim, ep, o.name, poses[o.name]) for o in ep.objects
                      if o.name not in targets and o.name in poses and o.name != head) + tuple(extra_obstacles)
    held = target_held(sim, task, robot, gripper, poses[head])
    return WorldSnapshot(target=poses[head], target_type=types[head], other_targets=others,
                         obstacles=obstacles, n_remaining=remaining, target_held=held, destination=dest,
                         task_mode=mode, **common), primary
