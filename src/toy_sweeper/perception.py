"""俯瞰カメラ画像からの物体検出と追跡。

VLM は使わない（毎フレーム動かすには遅い）。床との色の差で物体を切り出し、輪郭の最小外接矩形から
床面上の位置と向き（180° の区別なし）を出す。前フレームの追跡と位置で対応付けて ID を保つ。
画素から床面への変換は射影変換（ホモグラフィ）。実機ではキャリブレーション（doc/camera_setup.md）で求める。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import Pose2D, wrap_angle


@dataclass(frozen=True)
class PerceptionCfg:
    floor_rgb: tuple[int, ...]        # 床の色（実機では空の作業領域を撮って決める）
    color_threshold: float            # 床の色との距離（RGB）がこれ以上の画素を物体とみなす
    min_area_px: int                  # これより小さい塊は雑音として捨てる
    robot_mask_margin_m: float        # ロボット（ArUco の姿勢から分かる）の周りを除く余白
    gate_m: float                     # 追跡の対応付けで許す移動量（1 フレームあたり）
    max_missed: int                   # これだけ連続で見つからなければ追跡を消す
    toy_car_min_length_m: float       # VLM で分類していない塊の種類を長辺で決める閾値


def load_perception_cfg() -> PerceptionCfg:
    from .config import load_yaml, strict_dataclass_from_dict
    return strict_dataclass_from_dict(PerceptionCfg, load_yaml("perception.yaml"), "perception")


def chromaticity(rgb) -> tuple[float, float, float]:
    """明るさを除いた色（r, g, b を合計で割る）。影や照明の明るさで平均の色が暗く・明るくなっても変わりにくい
    （2026-10-19、Gazebo のリハーサルで、爪から離れた物体の色度の変化は 0.5 以下、RGB は 27 まで変わった）。"""
    t = sum(rgb)
    if t <= 0:
        raise ValueError(f"色の合計が 0 以下: {rgb}")
    return (rgb[0] / t, rgb[1] / t, rgb[2] / t)


def chroma_distance(a, b) -> float:
    """色度の距離 ×255（RGB と同じくらいの数の大きさにする）。"""
    return 255 * math.dist(chromaticity(a), chromaticity(b))


@dataclass(frozen=True)
class Detection:
    pose: Pose2D            # yaw は長辺の向き（180° の区別なし）
    size_m: tuple[float, float]
    mean_rgb: tuple[float, float, float]
    bbox_px: tuple[int, int, int, int]


@dataclass
class Track:
    id: int
    pose: Pose2D
    size_m: tuple[float, float]
    mean_rgb: tuple[float, float, float]   # 最後に対応した検出の画素の平均色（同一性の確認に使う。toy_sweeper.field）
    missed: int = 0
    history: list = field(default_factory=list)
    hits: int = 1      # 対応が付いたフレームの数（生まれたフレームを含む。短命な塊を同定に出さないため、2026-10-28）


class Homography:
    """画素 (u, v) ↔ 床面 (x, y) [m] の射影変換。"""

    def __init__(self, px_pts, world_pts):
        self.H = cv2.getPerspectiveTransform(np.float32(px_pts), np.float32(world_pts))
        self.Hinv = np.linalg.inv(self.H)

    def to_world(self, uv: np.ndarray) -> np.ndarray:
        p = cv2.perspectiveTransform(np.float32(uv).reshape(-1, 1, 2), self.H)
        return p.reshape(-1, 2)

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        p = cv2.perspectiveTransform(np.float32(xy).reshape(-1, 1, 2), np.float32(self.Hinv))
        return p.reshape(-1, 2)


def detect_objects(img: np.ndarray, hom: Homography, robot_polygons_m: list[list[tuple[float, float]]],
                   cfg: PerceptionCfg) -> list[Detection]:
    """RGB 画像から物体を検出する。robot_polygons_m はロボットの部品（車体、爪）の外形（床面座標）で、その領域は除く。

    部品ごとに除くので、爪の間にある物体は見える（真上のカメラと同じ）。
    """
    diff = np.linalg.norm(img.astype(np.float32) - np.float32(cfg.floor_rgb), axis=2)
    mask = (diff >= cfg.color_threshold).astype(np.uint8)
    robot = np.zeros_like(mask)
    for poly in robot_polygons_m:
        cv2.fillPoly(robot, [np.int32(np.round(hom.to_px(np.array(poly))))], 1)
    m_px = max(1, int(round(cfg.robot_mask_margin_m * np.linalg.norm(hom.to_px(np.array([[0.0, 0.0], [1.0, 0.0]]))[1]
                                                                      - hom.to_px(np.array([[0.0, 0.0]]))[0]))))
    robot = cv2.dilate(robot, np.ones((2 * m_px + 1, 2 * m_px + 1), np.uint8))
    mask[robot > 0] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < cfg.min_area_px:
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        (cu, cv_), (w, h), _ = cv2.minAreaRect(pts)
        box_px = cv2.boxPoints(((cu, cv_), (w, h), _))
        box_w = hom.to_world(box_px)
        center = hom.to_world(np.array([[cu, cv_]]))[0]
        e1, e2 = box_w[1] - box_w[0], box_w[2] - box_w[1]
        long_e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        yaw = wrap_angle(math.atan2(long_e[1], long_e[0]))
        size = (float(max(np.linalg.norm(e1), np.linalg.norm(e2))), float(min(np.linalg.norm(e1), np.linalg.norm(e2))))
        mean = tuple(float(v) for v in img[ys, xs].mean(axis=0))
        x0, y0, bw, bh = stats[i, :4]
        out.append(Detection(Pose2D(float(center[0]), float(center[1]), yaw), size, mean,
                             (int(x0), int(y0), int(x0 + bw), int(y0 + bh))))
    return out


class Tracker:
    """近い順の貪欲な対応付けで ID を保つ追跡。見失っても max_missed フレームは最後の位置を保持する。
    chroma_gate > 0 なら、前のフレームと色度（chroma_distance）がそれ以上違う検出にはつながない（くっついた別の物の塊へ
    追跡が移らないように、2026-10-25。perception_oblique の track_chroma_gate）。0 なら色を見ない。"""

    def __init__(self, cfg: PerceptionCfg, chroma_gate: float):
        if chroma_gate < 0:
            raise ValueError(f"chroma_gate は 0 以上: {chroma_gate}")
        self.cfg = cfg
        self.chroma_gate = chroma_gate
        self.tracks: dict[int, Track] = {}
        self._next = 0

    def update(self, dets: list[Detection]) -> dict[int, Track]:
        pairs = sorted(((math.dist(t.pose.xy, d.pose.xy), tid, j) for tid, t in self.tracks.items()
                        for j, d in enumerate(dets)), key=lambda e: e[0])
        used_t, used_d = set(), set()
        for dd, tid, j in pairs:
            if dd > self.cfg.gate_m or tid in used_t or j in used_d:
                continue
            t, d = self.tracks[tid], dets[j]
            if self.chroma_gate > 0 and chroma_distance(t.mean_rgb, d.mean_rgb) >= self.chroma_gate:
                continue
            # 向きは 180° の区別がないので、前の向きに近い方を選ぶ
            yaw = d.pose.yaw if abs(wrap_angle(d.pose.yaw - t.pose.yaw)) <= math.pi / 2 else wrap_angle(d.pose.yaw + math.pi)
            t.pose, t.size_m, t.mean_rgb, t.missed = Pose2D(d.pose.x, d.pose.y, yaw), d.size_m, d.mean_rgb, 0
            t.hits += 1
            used_t.add(tid)
            used_d.add(j)
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.cfg.max_missed:
                    del self.tracks[tid]
        for j, d in enumerate(dets):
            if j not in used_d:
                self.tracks[self._next] = Track(self._next, d.pose, d.size_m, d.mean_rgb)
                self._next += 1
        return self.tracks
