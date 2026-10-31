"""床とロボットの ArUco マーカーの配置（config/markers.yaml、doc/camera_setup.md）。

Gazebo の描画（gazebo.sdf_gen）とカメラの校正（scripts/calibrate_app.py）が同じ配置を使う。
床の座標は作業領域の中心が原点、x は右、y は奥（カメラから遠い方）。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import CONFIG_DIR, SimCfg, load_yaml, strict_dataclass_from_dict

# 角の名前 → 作業領域の角の向き（x, y の符号）
CORNER_SIGNS = {"near_left": (-1, -1), "near_right": (1, -1), "far_right": (1, 1), "far_left": (-1, 1)}


@dataclass(frozen=True)
class FieldMarkersCfg:
    dictionary: str
    corner_marker_size_m: float
    paper_margin_m: float
    wall_thickness_m: float
    corner_ids: dict[str, int]
    robot_id: int
    robot_marker_size_m: float
    robot_marker_offset_m: tuple[float, ...]
    robot_marker_height_m: float


@dataclass(frozen=True)
class IntrinsicsBoardCfg:
    dictionary: str
    squares: tuple[int, ...]
    square_size_m: float
    marker_size_m: float


@dataclass(frozen=True)
class MarkersCfg:
    field: FieldMarkersCfg
    intrinsics: IntrinsicsBoardCfg


def load_markers_cfg(config_dir=None) -> MarkersCfg:
    c = strict_dataclass_from_dict(MarkersCfg, load_yaml("markers.yaml", config_dir or CONFIG_DIR), "markers")
    if set(c.field.corner_ids) != set(CORNER_SIGNS):
        raise ValueError(f"markers.yaml の corner_ids は {sorted(CORNER_SIGNS)}: {sorted(c.field.corner_ids)}")
    if len(c.field.robot_marker_offset_m) != 2:
        raise ValueError(f"markers.yaml の robot_marker_offset_m は 2 要素（前、左）: {c.field.robot_marker_offset_m}")
    ids = list(c.field.corner_ids.values()) + [c.field.robot_id]
    if len(set(ids)) != len(ids):
        raise ValueError(f"マーカーの ID が重なっている: {ids}")
    return c


def aruco_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"未知の ArUco の辞書: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def corner_marker_centers(m: MarkersCfg, sim: SimCfg, wall_thickness_m: float) -> dict[int, tuple[float, float]]:
    """床の 4 隅のマーカーの中心（ID → 床の座標、config/markers.yaml の置き方）。
    奥: 作業領域の内側の角に、紙の 2 辺を角の 2 辺に押し当てる。
    手前: 手前の壁の外側に、紙の角を壁の外側の角に合わせ、紙の 1 辺を壁の外側の面に押し当てる（壁なしは厚み 0）。
    wall_thickness_m: 実機は markers.yaml の wall_thickness_m、Gazebo は sim.yaml の walls.thickness_m。"""
    hx, hy = sim.work_area.size_m[0] / 2, sim.work_area.size_m[1] / 2
    half = m.field.paper_margin_m + m.field.corner_marker_size_m / 2   # 紙の端から黒い四角の中心まで
    t = wall_thickness_m
    if t < 0:
        raise ValueError(f"壁の厚みが負: {t}")
    out = {}
    for name, i in m.field.corner_ids.items():
        sx, sy = CORNER_SIGNS[name]
        if sy > 0:   # 奥: 内側
            out[i] = (sx * (hx - half), hy - half)
        else:        # 手前: 壁の外側の角から、x は内へ、y は手前へ
            out[i] = (sx * (hx + t - half), -(hy + t + half))
    return out


def corner_marker_square(m: MarkersCfg, sim: SimCfg, wall_thickness_m: float, marker_id: int) -> np.ndarray:
    """床のマーカーの黒い四角の 4 つの角（床の座標、[4, 2]）。マーカーの向きは問わないので、並びは決まっていない
    （校正では、中心から求めた姿勢で投影して、近い検出の角と対応させる）。"""
    cx, cy = corner_marker_centers(m, sim, wall_thickness_m)[marker_id]
    h = m.field.corner_marker_size_m / 2
    return np.array([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])


def marker_image(m: MarkersCfg, marker_id: int, side_px: int, margin_px: int) -> np.ndarray:
    """マーカーの画像（白い余白つき、uint8 のグレー）。Gazebo の床の模様に使う。"""
    img = cv2.aruco.generateImageMarker(aruco_dictionary(m.field.dictionary), marker_id, side_px)
    return cv2.copyMakeBorder(img, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=255)
