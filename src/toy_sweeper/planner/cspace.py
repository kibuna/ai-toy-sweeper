"""占有格子と C-space（ロボットの形と向きを考えた衝突判定）。

- 格子: 作業領域の外へ pad だけ広げた正方格子（壁の外は障害物）。画素の中心が多角形の内側なら占有。
- C-space: 向きの刻み j ごとに、障害物の画像をロボットの外形（車軸中点を原点にした画像）で膨張（cv2.dilate）したもの。
  画素 (row, col) が 1 なら、車軸中点をその画素に置いた向き j のロボットは障害物と重なる。必要な向きだけ遅延して作る。
- コスト: 通れない（hard）、近い（soft、soft_cost）、開始時に重なっていた物体と重なる（overlap_cost）の 3 層を 1 枚の
  float 配列にまとめる（inf が通れない）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import SimCfg
from ..geometry import Pose2D, Rect
from ..robot_model import SIDES, body_center_x, jaw_polygons

_SHIFT = 8  # cv2.fillPoly の小数点以下のビット数


def convex_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """凸多角形 a, b（(n, 2) 配列）が重なるか（分離軸定理。接しているだけなら重ならない）。"""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            axis = np.array([-e[1], e[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    return True


def box_polygon(pose: Pose2D, size: tuple[float, float], margin: float = 0.0) -> np.ndarray:
    hx, hy = size[0] / 2 + margin, size[1] / 2 + margin
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    local = np.array([(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)])
    return local @ np.array([[c, s], [-s, c]]) + np.array(pose.xy)


def disc_polygon(center: tuple[float, float], r: float, n: int = 16) -> np.ndarray:
    """円に外接する正 n 角形（円を含む）。"""
    rr = r / math.cos(math.pi / n)
    a = np.arange(n) * 2 * math.pi / n
    return np.stack([center[0] + rr * np.cos(a), center[1] + rr * np.sin(a)], axis=1)


def robot_polygons(sim: SimCfg) -> list[np.ndarray]:
    """車体と、開いた爪・閉じた爪の両方（ロボット座標）。爪の角度は観測しないので両方を含む外形で判定する。"""
    bx, by = sim.robot.body_size_m[0], sim.robot.body_size_m[1]
    cx = body_center_x(sim)
    polys = [np.array([(cx - bx / 2, -by / 2), (cx + bx / 2, -by / 2), (cx + bx / 2, by / 2), (cx - bx / 2, by / 2)])]
    for angle in (sim.gripper.jaw_open_angle_rad, sim.gripper.jaw_closed_angle_rad):
        for side in SIDES:
            polys += [np.array(p) for p in jaw_polygons(sim, side, angle)]
    return polys


def transform(polys: list[np.ndarray], pose: Pose2D) -> list[np.ndarray]:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    R = np.array([[c, s], [-s, c]])
    return [p @ R + np.array(pose.xy) for p in polys]


@dataclass(frozen=True)
class Footprint:
    """ロボットの外形（ロボット座標の凸多角形の集まり）。key は C-space のキャッシュ用。"""
    polys: tuple[np.ndarray, ...]
    key: tuple

    @property
    def radius(self) -> float:
        return max(float(np.hypot(p[:, 0], p[:, 1]).max()) for p in self.polys)


class Grid:
    def __init__(self, area: Rect, pad: float, res: float):
        self.res = res
        self.x0 = area.center[0] - area.size[0] / 2 - pad
        self.y0 = area.center[1] - area.size[1] / 2 - pad
        self.n = int(math.ceil((max(area.size) + 2 * pad) / res))
        idx = (np.arange(self.n) + 0.5) * res
        self.xs = self.x0 + idx  # 列 col の中心の x
        self.ys = self.y0 + idx  # 行 row の中心の y
        self.area = area

    def cell(self, x: float, y: float) -> tuple[int, int]:
        """(row, col)。格子の外は IndexError（pad が足りない設定の不具合を黙って丸めない）。"""
        col, row = int((x - self.x0) / self.res), int((y - self.y0) / self.res)
        if not (0 <= row < self.n and 0 <= col < self.n) or x < self.x0 or y < self.y0:
            raise IndexError(f"格子の外: ({x:.3f}, {y:.3f})")
        return row, col

    def _px(self, pts: np.ndarray) -> np.ndarray:
        u = (pts[:, 0] - self.x0) / self.res - 0.5
        v = (pts[:, 1] - self.y0) / self.res - 0.5
        return np.round(np.stack([u, v], axis=1) * (1 << _SHIFT)).astype(np.int32)

    def raster(self, polys) -> np.ndarray:
        img = np.zeros((self.n, self.n), np.uint8)
        if polys:
            cv2.fillPoly(img, [self._px(p) for p in polys], 1, lineType=cv2.LINE_8, shift=_SHIFT)
        return img

    def outside_area(self) -> np.ndarray:
        """作業領域の外（壁）の画素。"""
        a = self.area
        X, Y = np.meshgrid(self.xs, self.ys)
        inside = (np.abs(X - a.center[0]) <= a.size[0] / 2) & (np.abs(Y - a.center[1]) <= a.size[1] / 2)
        return (~inside).astype(np.uint8)


class KernelCache:
    """外形 × 向きの刻み → 膨張に使う画像（車軸中点が画像の中心画素）。"""

    def __init__(self, res: float, n_bins: int):
        self.res, self.n_bins = res, n_bins
        self._cache: dict[tuple, tuple[np.ndarray, tuple[int, int]]] = {}

    def get(self, fp: Footprint, j: int) -> tuple[np.ndarray, tuple[int, int]]:
        k = (fp.key, j)
        if k not in self._cache:
            c = int(math.ceil(fp.radius / self.res)) + 1
            th = 2 * math.pi * j / self.n_bins
            img = np.zeros((2 * c + 1, 2 * c + 1), np.uint8)
            pts = []
            for p in transform(list(fp.polys), Pose2D(0.0, 0.0, th)):
                q = np.stack([c + p[:, 0] / self.res, c + p[:, 1] / self.res], axis=1)
                pts.append(np.round(q * (1 << _SHIFT)).astype(np.int32))
            cv2.fillPoly(img, pts, 1, lineType=cv2.LINE_8, shift=_SHIFT)
            self._cache[k] = (img, (c, c))
        return self._cache[k]


class CSpace:
    """向きの刻みごとのコスト画像（遅延して作る）。cost(x, y, j) が inf なら通れない。"""

    def __init__(self, grid: Grid, kernels: KernelCache, fp: Footprint, hard: np.ndarray,
                 soft: np.ndarray | None, overlap: np.ndarray | None, soft_cost: float, overlap_cost: float):
        self.grid, self.kernels, self.fp = grid, kernels, fp
        self.hard, self.soft, self.overlap = hard, soft, overlap
        self.soft_cost, self.overlap_cost = soft_cost, overlap_cost
        self.n_bins = kernels.n_bins
        self._layers: dict[int, np.ndarray] = {}

    def bin(self, theta: float) -> int:
        return int(round(theta / (2 * math.pi) * self.n_bins)) % self.n_bins

    def layer(self, j: int) -> np.ndarray:
        L = self._layers.get(j)
        if L is None:
            k, anchor = self.kernels.get(self.fp, j)
            L = np.zeros(self.hard.shape, np.float32)
            if self.soft is not None and self.soft_cost > 0:
                L += self.soft_cost * cv2.dilate(self.soft, k, anchor=anchor)
            if self.overlap is not None:
                L += self.overlap_cost * cv2.dilate(self.overlap, k, anchor=anchor)
            L[cv2.dilate(self.hard, k, anchor=anchor) > 0] = np.inf
            self._layers[j] = L
        return L

    def cost(self, x: float, y: float, j: int) -> float:
        """格子の外（壁の外側 pad より遠い）は、車軸中点が作業領域の外なので通れない（inf）。"""
        g = self.grid
        col, row = int((x - g.x0) / g.res), int((y - g.y0) / g.res)
        if not (0 <= row < g.n and 0 <= col < g.n) or x < g.x0 or y < g.y0:
            return math.inf
        return float(self.layer(j)[row, col])

    def cost_at(self, x: float, y: float, theta: float) -> float:
        return self.cost(x, y, self.bin(theta))
