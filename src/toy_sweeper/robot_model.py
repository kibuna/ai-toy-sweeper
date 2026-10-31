"""ロボットの形状をロボット座標系（原点 = 車軸中点 = 旋回中心、x 前方、y 左）で表す。

Pose2D でロボットの姿勢を持つときは、常に車軸中点の位置を指す（config/sim.yaml robot.axle_offset_x_m）。
グリッパーは左右対称に回る 2 本の爪（M5Stack Unit Catch）。1 本の爪は軸から外へ曲がる「腕」と、
その先の「梁」からなり、閉じた状態（角度 0）で左右の梁は前方に平行。角度は外向きに測る。
"""
from __future__ import annotations

import math

from .config import SimCfg

SIDES = (1.0, -1.0)  # 左爪, 右爪


def body_center_x(sim: SimCfg) -> float:
    """車体の箱の中心の x。"""
    return -sim.robot.axle_offset_x_m


def front_x(sim: SimCfg) -> float:
    """車体前面の x。"""
    return body_center_x(sim) + sim.robot.body_size_m[0] / 2


def pivot_xy(sim: SimCfg, side: float) -> tuple[float, float]:
    """爪の回転軸の位置。side は +1（左）か -1（右）。"""
    g = sim.gripper
    return (front_x(sim) + g.pivot_x_from_front_m, side * g.pivot_half_spacing_m)


def _jaw_point(sim: SimCfg, side: float, angle: float, d: float, e: float) -> tuple[float, float]:
    """爪座標（軸から前へ d、外へ e。閉じた状態の向き）の点を、角度 angle のロボット座標へ。"""
    px, py = pivot_xy(sim, side)
    a = side * angle
    ly = side * e
    return (px + d * math.cos(a) - ly * math.sin(a), py + d * math.sin(a) + ly * math.cos(a))


def jaw_polygons(sim: SimCfg, side: float, angle: float) -> list[list[tuple[float, float]]]:
    """爪 1 本の外形（腕と梁の 2 つの凸多角形）。"""
    g = sim.gripper
    hw, e, d0, L = g.jaw_width_m / 2, g.jaw_lateral_offset_m, g.jaw_beam_start_m, g.jaw_length_m
    beam = [(d0, e - hw), (L, e - hw), (L, e + hw), (d0, e + hw)]
    arm = [(0.0, -hw), (d0, e - hw), (d0, e + hw), (0.0, hw)]
    return [[_jaw_point(sim, side, angle, d, y) for d, y in poly] for poly in (arm, beam)]


def jaw_polygon_points(sim: SimCfg, side: float, angle: float) -> list[tuple[float, float]]:
    return [p for poly in jaw_polygons(sim, side, angle) for p in poly]


def gripper_center_x(sim: SimCfg) -> float:
    """爪の間の中心（到達判定と接近報酬の基準点）の x。梁の中央。"""
    g = sim.gripper
    return front_x(sim) + g.pivot_x_from_front_m + (g.jaw_beam_start_m + g.jaw_length_m) / 2


def jaw_inner_half_gap(sim: SimCfg, angle: float, x: float) -> float:
    """ロボット座標の前方位置 x で、左右の梁の内側の縁の間隔の半分（梁の範囲外なら -inf）。"""
    g = sim.gripper
    ei = g.jaw_lateral_offset_m - g.jaw_width_m / 2
    x0, y0 = _jaw_point(sim, 1.0, angle, g.jaw_beam_start_m, ei)
    t = (x - x0) / math.cos(angle)
    if t < 0 or t > g.jaw_length_m - g.jaw_beam_start_m:
        return -math.inf
    return y0 + t * math.sin(angle)


def angle_touching(sim: SimCfg, x: float, half_extent: float) -> float:
    """前方位置 x にある幅 2*half_extent の物体に、梁の内側の縁がちょうど触れる角度（閉〜開の範囲に収める）。"""
    g = sim.gripper
    lo, hi = g.jaw_closed_angle_rad, g.jaw_open_angle_rad

    def gap(a):
        return jaw_inner_half_gap(sim, a, x)
    if gap(lo) >= half_extent:
        return lo
    for _ in range(60):  # 二分法。角度が開くほど隙間は広がる
        mid = (lo + hi) / 2
        if gap(mid) < half_extent:
            lo = mid
        else:
            hi = mid
    return hi


def robot_radius(sim: SimCfg) -> float:
    """車軸中点から、車体と開いた爪の最も遠い点までの距離（配置の重なり判定用）。"""
    bx, by = sim.robot.body_size_m[0], sim.robot.body_size_m[1]
    cx = body_center_x(sim)
    pts = [(cx + sx * bx / 2, sy * by / 2) for sx in (1, -1) for sy in (1, -1)]
    for side in SIDES:
        pts += jaw_polygon_points(sim, side, sim.gripper.jaw_open_angle_rad)
    return max(math.hypot(x, y) for x, y in pts)
