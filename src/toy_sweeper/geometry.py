"""平面幾何の小道具。座標系は作業領域中心原点、m、rad。"""
from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_angle(a: float) -> float:
    """[-pi, pi) に折り返す。"""
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    def transform_point(self, local: tuple[float, float]) -> tuple[float, float]:
        """ローカル座標の点を世界座標へ。"""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.x + c * local[0] - s * local[1], self.y + s * local[0] + c * local[1])


@dataclass(frozen=True)
class Rect:
    """軸平行の長方形。"""
    center: tuple[float, float]
    size: tuple[float, float]

    def contains(self, p: tuple[float, float]) -> bool:
        return (abs(p[0] - self.center[0]) <= self.size[0] / 2
                and abs(p[1] - self.center[1]) <= self.size[1] / 2)

    def contains_all(self, pts) -> bool:
        return all(self.contains(p) for p in pts)


def box_corners(pose: Pose2D, size_xy: tuple[float, float]) -> list[tuple[float, float]]:
    hx, hy = size_xy[0] / 2, size_xy[1] / 2
    return [pose.transform_point(c) for c in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))]


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def signed_distance_to_rect(p: tuple[float, float], r: Rect) -> float:
    """点から長方形の縁までの符号付き距離（外は正、中は負）。"""
    dx = abs(p[0] - r.center[0]) - r.size[0] / 2
    dy = abs(p[1] - r.center[1]) - r.size[1] / 2
    if dx > 0 or dy > 0:
        return math.hypot(max(dx, 0.0), max(dy, 0.0))
    return max(dx, dy)
