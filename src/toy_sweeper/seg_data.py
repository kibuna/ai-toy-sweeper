"""学習する検出器のデータ作り: Gazebo の条件（カメラ・照明・床）と物体の置き方を引き、
物体ごとのマスクのカメラの labels_map から YOLO のラベルを作る。Gazebo を動かす部分は scripts/gen_seg_data.py。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import AllCfg
from .geometry import Pose2D, box_corners
from .perception_env import robot_polygons
from .robot_model import gripper_center_x
from .sim2d import ObjectSpec
from .spaces import GripperState, ObjectType

_E = Path(__file__).resolve().parents[2] / "config" / "experiments"
# データを撮る設定: リハーサル（scripts/eval_rehearsal.py）と同じ（車体の寸法・作業領域・8 Hz が同じ
# 描画になるように）。この後に世界ごとの条件（sample_conditions）を重ねる
BASE_OVERRIDES = tuple(_E / f"{n}.yaml" for n in (
    "gazebo_oblique", "no_chassis_penalty", "set_encoder", "imperfect", "corner_edge", "cleanup_guidance", "multi_toy_car",
    "efficient", "free_area", "robot_8hz", "hw_1022", "steps_x2", "gazebo_area08"))

# 条件の範囲。実機のカメラ（data/calibration/camera.yaml: 位置 (−0.10, −0.87, 0.66) m、
# 水平画角 1.21 rad）と、今までの cam_*・light_*・floor_* の条件を含む。広さは見た目で決めた（未検証）
CONDITION_RANGES = {
    "cam_x": (-0.25, 0.25), "cam_y": (-1.0, -0.7), "cam_z": (0.55, 1.0),
    "aim": (-0.1, 0.1),             # 向く点（床、作業領域の中心からの x・y）
    "roll": (-0.03, 0.03),
    "hfov": (1.05, 1.4),
    "sun": (0.45, 1.0),             # 太陽の明るさ（light_dim 0.45〜light_bright 1.0）
    "sun_dir_z": (-0.95, -0.35),    # 太陽の向きの z（light_low_side −0.35〜ほぼ真上）
    "ambient": (0.2, 0.6),          # 環境光（light_dim 0.2〜light_bright 0.6）
    "tint": (-1.0, 1.0),            # 色味: −1 寒色（light_cool）、0 白、+1 暖色（light_warm）
}
P_SHADOWS = 0.8                     # 影ありの割合（実機の部屋の照明は影が出る見込み。light_no_shadow も混ぜる）
WARM, COOL = np.array([1.0, 0.75, 0.5]), np.array([0.75, 0.85, 1.0])    # light_warm・light_cool の太陽の色
FLOORS = [(0.55, 0.45, 0.35), (0.77, 0.63, 0.45), (0.5, 0.5, 0.5), (0.25, 0.22, 0.2)]          # 木目、オーク、灰、暗い
WORK_AREAS = [(0.88, 0.84, 0.76), (0.95, 0.95, 0.95), (0.6, 0.6, 0.6), (0.35, 0.35, 0.35), (0.77, 0.63, 0.45)]
COLOR_JITTER = 0.05


def look_at_pose(cam: tuple[float, float, float], target: tuple[float, float, float], roll: float) -> list[float]:
    """Gazebo のカメラ（自分の +x を向く）の姿勢 [x, y, z, roll, pitch, yaw] を、target を向くように。"""
    d = np.subtract(target, cam)
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(-d[2], math.hypot(d[0], d[1]))
    return [float(cam[0]), float(cam[1]), float(cam[2]), float(roll), float(pitch), float(yaw)]


def _tint(t: float) -> np.ndarray:
    return 1.0 + (t * (WARM - 1.0) if t >= 0 else -t * (COOL - 1.0))


def sample_conditions(rng: np.random.Generator) -> dict:
    """Gazebo の World の条件を引き、設定の差し替え（config/experiments の形の dict）で返す。"""
    r = CONDITION_RANGES
    u = lambda k: float(rng.uniform(*r[k]))
    cam = (u("cam_x"), u("cam_y"), u("cam_z"))
    pose = look_at_pose(cam, (u("aim"), u("aim"), 0.0), u("roll"))
    tint = _tint(u("tint"))
    z = u("sun_dir_z")
    a = rng.uniform(0, 2 * math.pi)
    h = math.sqrt(1 - z * z)
    jit = lambda c: [round(float(np.clip(v + rng.uniform(-COLOR_JITTER, COLOR_JITTER), 0, 1)), 4) for v in c]
    rnd = lambda v: [round(float(x), 4) for x in v]
    g = {
        "camera_pose": rnd(pose), "camera_hfov_rad": round(u("hfov"), 4), "camera_image_px": [1920, 1080],
        "segmentation_camera": True,
        "light_diffuse": rnd(np.clip(u("sun") * tint, 0, 1)),
        "light_direction": rnd((h * math.cos(a), h * math.sin(a), z)),
        "light_cast_shadows": bool(rng.random() < P_SHADOWS),
        "scene_ambient": rnd(np.clip(u("ambient") * tint, 0, 1)),
        "floor_rgb": jit(FLOORS[rng.integers(len(FLOORS))]),
        "work_area_rgb": jit(WORK_AREAS[rng.integers(len(WORK_AREAS))]),
    }
    return {"sim": {"gazebo": g}}


def convex_overlap(a, b) -> bool:
    """2 つの凸多角形が重なるか（分離軸。辺で接するだけは重ならない）。"""
    for poly in (a, b):
        p = np.asarray(poly, float)
        for i in range(len(p)):
            e = p[(i + 1) % len(p)] - p[i]
            n = np.array([-e[1], e[0]])
            pa, pb = np.asarray(a, float) @ n, np.asarray(b, float) @ n
            if pa.max() <= pb.min() + 1e-12 or pb.max() <= pa.min() + 1e-12:
                return False
    return True


@dataclass
class Layout:
    robot: Pose2D
    gripper: GripperState
    objects: list[ObjectSpec]
    held: str | None                                      # 爪の中に置いた物体
    touching: list[tuple[str, str]] = field(default_factory=list)   # 接して置いた (物体, 隣)


# 置き方（割合はどれも未検証）
OUTSIDE_M = 0.05            # 作業領域の外にはみ出して置く幅（実機の決まりは外の余白 10 cm まで物を探す）
ROBOT_INSIDE_M = 0.08       # ロボットの中心を作業領域の縁からこれだけ内側に
P_CLOSED = 0.4              # 爪を閉じている割合
GAP_M = (0.0005, 0.003)     # 接して置くすき間
MAX_TRIES = 2000


def sample_layout(cfg: AllCfg, objects: list[ObjectSpec], rng: np.random.Generator, p_touch: float = 0.5,
                  p_held: float = 0.3) -> Layout:
    """objects から 1 個以上を選び、重ならずに置く。p_touch の割合で既に置いた物体に接して置き、p_held の割合で爪の中に 1 個。"""
    hx, hy = (v / 2 for v in cfg.sim.work_area.size_m)
    size = lambda o: tuple(cfg.sim.objects[o.type.name.lower()].size_m[:2])
    chosen = [objects[i] for i in rng.permutation(len(objects))[:rng.integers(1, len(objects) + 1)]]
    robot = Pose2D(float(rng.uniform(-hx + ROBOT_INSIDE_M, hx - ROBOT_INSIDE_M)),
                   float(rng.uniform(-hy + ROBOT_INSIDE_M, hy - ROBOT_INSIDE_M)), float(rng.uniform(-math.pi, math.pi)))
    held = None
    gripper = GripperState.CLOSED if rng.random() < P_CLOSED else GripperState.OPEN
    placed: list[ObjectSpec] = []
    if rng.random() < p_held:
        o = chosen[0]
        if o.type == ObjectType.TOY_CAR:
            gripper = GripperState.OPEN          # ミニカーは挟まず、開いた爪で押す
        gc = robot.transform_point((gripper_center_x(cfg.sim), 0.0))
        placed.append(ObjectSpec(o.name, o.type, Pose2D(gc[0], gc[1], robot.yaw), o.color))
        held = o.name
    g = cfg.sim.gripper
    jaw = g.jaw_closed_angle_rad if gripper == GripperState.CLOSED else g.jaw_open_angle_rad
    robot_polys = robot_polygons(cfg, robot, jaw)
    boxes = [box_corners(p.pose, size(p)) for p in placed]

    def free(box) -> bool:
        return not any(convex_overlap(box, b) for b in boxes) and not any(convex_overlap(box, p) for p in robot_polys)

    touching = []
    for o in chosen[len(placed):]:
        s = size(o)
        for _ in range(MAX_TRIES):
            yaw = float(rng.uniform(-math.pi, math.pi))
            if placed and rng.random() < p_touch:
                k = int(rng.integers(len(placed)))
                anchor, abox = placed[k], boxes[k]
                gap = float(rng.uniform(*GAP_M))
                a = rng.uniform(0, 2 * math.pi)
                d = np.array([math.cos(a), math.sin(a)])
                lo, hi = 0.0, math.hypot(*s) + math.hypot(*size(anchor)) + gap
                for _ in range(40):    # 隣に（すき間 gap だけ広げた箱で）ちょうど接する距離
                    t = (lo + hi) / 2
                    pose = Pose2D(anchor.pose.x + t * d[0], anchor.pose.y + t * d[1], yaw)
                    if convex_overlap(box_corners(pose, (s[0] + 2 * gap, s[1] + 2 * gap)), abox):
                        lo = t
                    else:
                        hi = t
                pose = Pose2D(anchor.pose.x + hi * d[0], anchor.pose.y + hi * d[1], yaw)
                partner = anchor.name
            else:
                pose = Pose2D(float(rng.uniform(-hx - OUTSIDE_M, hx + OUTSIDE_M)),
                              float(rng.uniform(-hy - OUTSIDE_M, hy + OUTSIDE_M)), yaw)
                partner = None
            box = box_corners(pose, s)
            if abs(pose.x) > hx + OUTSIDE_M or abs(pose.y) > hy + OUTSIDE_M or not free(box):
                continue
            placed.append(ObjectSpec(o.name, o.type, pose, o.color))
            boxes.append(box)
            if partner is not None:
                touching.append((o.name, partner))
            break
        else:
            raise RuntimeError(f"{o.name} を {MAX_TRIES} 回で置けない（物体 {len(chosen)} 個）")
    return Layout(robot, gripper, placed, held, touching)


def instances_from_labels(labels: np.ndarray, min_pixels: int) -> list[tuple[int, np.ndarray, int]]:
    """segmentation カメラの labels_map（画素ごとに (インスタンスの番号, 0, ラベル)）から、物体ごとの
    (クラス（ラベル − 1）, 輪郭の画素の座標 (N, 2), 画素の数)。2 つ以上に割れた物体は一番大きい部分だけ。
    画素が min_pixels より少ない物体は除く。"""
    key = labels[..., 2].astype(np.int32) * 65536 + labels[..., 1].astype(np.int32) * 256 + labels[..., 0]
    out = []
    for k in np.unique(key):
        if k == 0:
            continue
        label = int(k // 65536)
        if label not in (1, 2):
            raise ValueError(f"想定外のラベル {label}（sdf_gen.SEG_LABELS は 1、2）")
        mask = (key == k).astype(np.uint8)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        npx = int(stats[big, cv2.CC_STAT_AREA])
        if npx < min_pixels:
            continue
        cnts, _ = cv2.findContours((cc == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out.append((label - 1, max(cnts, key=len).reshape(-1, 2), npx))
    return out
