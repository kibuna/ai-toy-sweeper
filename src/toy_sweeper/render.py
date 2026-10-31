"""俯瞰描画（動画・デバッグ用）。OpenCV で RGB 画像を返す。

爪は Sim2D なら実際の角度、それ以外（Gazebo）はグリッパー状態から開閉どちらかの角度で描く。
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import Pose2D, box_corners
from .robot_model import SIDES, body_center_x, front_x, jaw_polygons
from .spaces import OBJECT_TYPE_NAMES, GripperState, ObjectType

# 描画の解像度。動画の見やすさだけで決めた値で、学習には影響しない
PX_PER_M = 700
MARGIN_PX = 40
PANEL_PX = 60

_BG = (245, 245, 240)
_AREA = (225, 215, 195)
_GOAL = (205, 225, 200)
_GOAL_ACTIVE = (150, 215, 150)
_ROBOT = (60, 90, 200)
_JAW = (30, 40, 120)
_TEXT = (20, 20, 20)
_APPROACH = (240, 160, 0)
_COLORS = {"red": (220, 50, 50), "blue": (40, 90, 220), "yellow": (240, 200, 30), "green": (40, 160, 70),
           "white": (250, 250, 250), "black": (40, 40, 40), "lime": (150, 205, 40), "light_blue": (100, 195, 240),
           "pink": (245, 150, 200), "orange": (250, 135, 30)}


def object_rgb(color: str) -> tuple[int, int, int]:
    """物体の色名の描画の色（RGB）。"""
    if color not in _COLORS:
        raise KeyError(f"描画の色がない: {color}")
    return _COLORS[color]


def _to_px(env, p):
    hx, hy = env.area.size[0] / 2, env.area.size[1] / 2
    return (int(MARGIN_PX + (p[0] + hx) * PX_PER_M), int(MARGIN_PX + (hy - p[1]) * PX_PER_M))


def _poly(img, env, pts, color, thickness=-1):
    arr = np.array([_to_px(env, p) for p in pts], dtype=np.int32)
    if thickness < 0:
        cv2.fillPoly(img, [arr], color, lineType=cv2.LINE_AA)
    else:
        cv2.polylines(img, [arr], True, color, thickness, lineType=cv2.LINE_AA)


def draw_frame(env, caption: str = "SIM (2D)") -> np.ndarray:
    from .task import approach_point, goal_rects_for_area, gripper_point

    cfg = env.cfg
    hx, hy = env.area.size[0] / 2, env.area.size[1] / 2   # 盤面の作業領域（v30）
    w = int(2 * MARGIN_PX + 2 * hx * PX_PER_M)
    h = int(2 * MARGIN_PX + 2 * hy * PX_PER_M) + PANEL_PX
    img = np.full((h, w, 3), _BG, np.uint8)
    _poly(img, env, box_corners(Pose2D(0, 0, 0), (2 * hx, 2 * hy)), _AREA)
    ep = env.episode
    for name, rect in goal_rects_for_area(cfg.sim, env.area.size).items():
        _poly(img, env, box_corners(Pose2D(*rect.center, 0.0), rect.size), _GOAL)
        cv2.putText(img, name, _to_px(env, (rect.center[0] - 0.03, rect.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (90, 120, 90), 1, cv2.LINE_AA)
    _poly(img, env, box_corners(Pose2D(*ep.goal.center, 0.0), ep.goal.size), _GOAL_ACTIVE)  # 今回のゴール

    for o in ep.objects:
        pose = env.backend.object_pose(o.name)
        size = cfg.sim.objects[OBJECT_TYPE_NAMES[o.type]].size_m[:2]
        _poly(img, env, box_corners(pose, size), _COLORS[o.color])
        _poly(img, env, box_corners(pose, size), (60, 60, 60), 1)
        if o.type == ObjectType.TOY_CAR:  # 窓（前後方向が分かるように）
            _poly(img, env, box_corners(Pose2D(*pose.transform_point((0.0, 0.0)), pose.yaw),
                                        (size[0] * 0.4, size[1] * 0.7)), (180, 200, 220))
        if o.name in ep.target_names:
            _poly(img, env, box_corners(pose, (size[0] + 0.012, size[1] + 0.012)), (20, 20, 20), 2)
    if not ep.target_names:
        cv2.drawMarker(img, _to_px(env, ep.reach_point), _APPROACH, cv2.MARKER_CROSS, 18, 3)
    else:
        for n in ep.target_names:
            t = env.backend.object_pose(n).xy
            if not ep.goal.contains(t):
                cv2.circle(img, _to_px(env, approach_point(t, ep.goal.center, cfg.task.reward.approach_offset_m)), 4,
                           _APPROACH, -1, lineType=cv2.LINE_AA)

    r = env.backend.robot_pose()
    bx, by = cfg.sim.robot.body_size_m[0], cfg.sim.robot.body_size_m[1]
    body = Pose2D(*r.transform_point((body_center_x(cfg.sim), 0.0)), r.yaw)
    _poly(img, env, box_corners(body, (bx, by)), _ROBOT)
    cv2.line(img, _to_px(env, r.xy), _to_px(env, r.transform_point((front_x(cfg.sim), 0))), (255, 255, 255), 2, cv2.LINE_AA)
    if hasattr(env.backend, "jaw_angle"):
        angle = env.backend.jaw_angle()
    else:
        g = cfg.sim.gripper
        angle = g.jaw_closed_angle_rad if env.backend.gripper_state() == GripperState.CLOSED else g.jaw_open_angle_rad
    for side in SIDES:
        for poly in jaw_polygons(cfg.sim, side, angle):
            _poly(img, env, [r.transform_point(p) for p in poly], _JAW)
    cv2.circle(img, _to_px(env, gripper_point(cfg.sim, r)), 3, (255, 255, 255), -1)

    terms = getattr(env, "_last_terms", {}) or {}
    total = sum(terms.values()) if terms else 0.0
    targets = [o for o in ep.objects if o.name in ep.target_names]
    task = "reach point" if not targets else (
        f"{targets[0].color} {OBJECT_TYPE_NAMES[targets[0].type]} x{len(targets)} -> {ep.goal_name}")
    lines = [
        f"{caption}  stage {ep.stage}  step {env._steps}/{env.max_steps()}  task: {task}",
        f"action {env._prev_action.name:<13s} gripper {env.backend.gripper_state().name:<7s} reward {total:+.3f}",
    ]
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, h - PANEL_PX + 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT, 1, cv2.LINE_AA)
    return img


# ---------- VLM 用の盤面画像（文字・目印なし） ----------
# 想定したカメラ（640 px で作業領域 0.8 m）と同じ分解能
SCENE_PX_PER_M = 800
_FLOOR = (196, 160, 118)
_STUD_DARK = 0.75


def draw_scene(cfg, robot, objects, poses, jaw_angle: float | None = None) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """真上から見た盤面（実カメラ画像の代わり）。物体ごとの画素の外接矩形 (x0, y0, x1, y1) も返す。"""
    from .robot_model import SIDES, body_center_x, jaw_polygons

    hx, hy = cfg.sim.work_area.size_m[0] / 2, cfg.sim.work_area.size_m[1] / 2
    w, h = int(2 * hx * SCENE_PX_PER_M), int(2 * hy * SCENE_PX_PER_M)

    def px(p):
        return (int(round((p[0] + hx) * SCENE_PX_PER_M)), int(round((hy - p[1]) * SCENE_PX_PER_M)))

    def poly(img, pts, color):
        cv2.fillPoly(img, [np.array([px(p) for p in pts], np.int32)], color, lineType=cv2.LINE_AA)

    img = np.full((h, w, 3), _FLOOR, np.uint8)
    # 木目風の筋（床らしさのため。物体の判定には関係しない）
    for y in range(0, h, 40):
        cv2.line(img, (0, y), (w, y), (184, 148, 108), 1)
    boxes = {}
    for o in objects:
        pose = poses[o.name]
        sx, sy = cfg.sim.objects[OBJECT_TYPE_NAMES[o.type]].size_m[:2]
        col = _COLORS[o.color]
        if o.type == ObjectType.LEGO:
            poly(img, box_corners(pose, (sx, sy)), col)
            dark = tuple(int(c * _STUD_DARK) for c in col)
            for i in range(4):
                for j in range(2):
                    c = pose.transform_point(((i - 1.5) * sx / 4, (j - 0.5) * sy / 2))
                    cv2.circle(img, px(c), max(2, int(0.0024 * SCENE_PX_PER_M)), dark, 1, cv2.LINE_AA)
        else:
            for i in (-1, 1):
                for j in (-1, 1):  # タイヤ
                    c = Pose2D(*pose.transform_point((i * sx * 0.3, j * sy * 0.5)), pose.yaw)
                    poly(img, box_corners(c, (sx * 0.22, sy * 0.25)), (25, 25, 25))
            poly(img, box_corners(pose, (sx, sy * 0.9)), col)
            win = Pose2D(*pose.transform_point((-sx * 0.05, 0.0)), pose.yaw)
            poly(img, box_corners(win, (sx * 0.45, sy * 0.65)), (170, 200, 225))
        pts = np.array([px(p) for p in box_corners(pose, (sx * 1.05, sy * 1.2))])
        boxes[o.name] = (int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max()))
    bx, by = cfg.sim.robot.body_size_m[:2]
    poly(img, box_corners(Pose2D(*robot.transform_point((body_center_x(cfg.sim), 0.0)), robot.yaw), (bx, by)), (40, 120, 60))
    for side in SIDES:
        for pl in jaw_polygons(cfg.sim, side, cfg.sim.gripper.jaw_open_angle_rad if jaw_angle is None else jaw_angle):
            poly(img, [robot.transform_point(p) for p in pl], (60, 60, 60))
    return img, boxes
