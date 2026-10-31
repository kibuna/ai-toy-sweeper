"""斜め上のカメラの画像からの物体の検出（背景との差。doc/camera_setup.md）。

真上の合成画像用の detect_objects（床の 1 色との差）は、斜めの実画像では使えない: 床に模様（木目、照り返し、Gazebo の
ゴールの面）があり、物体の側面も写る。ここでは「校正する」ときに撮る何も置いていない床の画像を背景として、
- 背景との差（Lab。明るさの差は小さく見て、照り返しや影に強くする）で物体を切り出し、
- ロボットは高さのある箱として画像に投影した範囲を除き、
- 塊の輪郭を物体の高さの半分の水平面に戻して（CameraModel.to_floor）、床の上の中心・向き・大きさを出す。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import CameraModel
from .geometry import Pose2D, wrap_angle
from .perception import Detection


class BlobTooLarge(RuntimeError):
    """背景との差の塊が max_area_m2 より大きい（照明の変化、カメラが動いた、人や影が入った）。bbox_px は塊の画像の外接矩形。"""

    def __init__(self, msg: str, area_m2: float, bbox_px: tuple[int, int, int, int]):
        super().__init__(msg)
        self.area_m2, self.bbox_px = area_m2, bbox_px


# 色で分けるとき、1 つの色に要る色を持つ画素の数の下限（k-means が意味を持つ程度。未検証）
SPLIT_MIN_PIXELS = 30


@dataclass(frozen=True)
class BackgroundDetectCfg:
    luma_weight: float          # Lab の L の差の重み（a, b は 1）。照り返し・影は主に明るさの差
    threshold: float            # 重みつきの Lab の距離がこれ以上の画素を物体とみなす
    min_area_m2: float          # 床に戻した塊の面積がこれより小さければ雑音
    max_area_m2: float          # これより大きい塊は物体ではない（照明の変化、カメラが動いた）。例外にする
    object_mid_height_m: float  # 塊の輪郭を戻す水平面の高さ（物体の高さの半分）
    robot_height_m: float       # ロボットを除く範囲の高さ（車体の上まで）
    robot_margin_m: float       # 車体を除く範囲の余白（ArUco の姿勢の誤差）
    jaw_margin_m: float         # 爪を除く範囲の余白
    held_radius_m: float        # 挟んでいる物体とみなす、爪の中心からの距離（field.rehearsal）
    reassociate_max_m: float    # 見失った対象を付け直す距離の上限（field.rehearsal）
    reassociate_size_ratio: float  # 付け直す追跡の長辺の、対象の長辺に対する比の上限
    merged_max_length_m: float  # 同じ色の物とくっついた塊に付け直すときの塊の長辺の上限（0 はしない。field.target_tracking）
    track_chroma_gate: float    # 追跡を前のフレームと色度がこれ以上違う検出につながない（0 は色を見ない。perception.Tracker）
    recover_back_steps: int     # 見失ったときに下がる歩数（field.rehearsal）
    recover_search_m: float     # 仕切り直しで探す距離の上限
    recover_max: int            # 仕切り直しの回数の上限
    recover_unseen_steps: int   # 対象が見えない状態がこの歩数続いたら仕切り直す
    open_px: int                # 雑音を消す（開く）カーネルの大きさ
    close_px: int               # 開いた後に、近い画素をつなぐ（閉じる）円のカーネルの大きさ。0 はしない
    color_split: bool           # くっついた違う色の物体の塊を、色と場所で 2 つに分ける（split_by_color）
    split_chroma_min: float     # 色を持つとみなす画素の、Lab の a・b の距離の下限（窓・タイヤ・影の暗い所は色を持たない）
    split_min_ab: float         # 2 つの色の中心の a・b の距離の下限
    split_min_fraction: float   # 少ない側の色の画素の割合の下限
    split_min_separation: float # 2 つの色の画素の重心の距離 ÷ 塊の大きさ（画素数の平方根）の下限


def robot_mask(shape: tuple[int, int], cam: CameraModel, polys_m: list[list[tuple[float, float]]],
               heights_m: list[float], margins_m: list[float]) -> np.ndarray:
    """ロボットの部品（床の多角形）を、部品ごとの高さ 0〜heights_m[i] の柱として画像に投影した範囲（uint8、1 が除く画素）。
    爪は低い（床から 約 1 cm）ので、車体と同じ高さで除くと、挟んだ物体まで隠れる（2026-10-18、Gazebo で見失った）。"""
    if not len(heights_m) == len(margins_m) == len(polys_m):
        raise ValueError(f"部品の数 {len(polys_m)}、高さの数 {len(heights_m)}、余白の数 {len(margins_m)} が違う")
    mask = np.zeros(shape, np.uint8)
    for poly, height_m, margin_m in zip(polys_m, heights_m, margins_m):
        p = np.asarray(poly, float)
        c = p.mean(axis=0)
        d = p - c
        n = np.linalg.norm(d, axis=1, keepdims=True)
        p = c + d * (1 + margin_m / np.maximum(n, 1e-9))     # 中心から外へ margin だけ広げる
        pts = np.vstack([cam.to_px(p, 0.0), cam.to_px(p, height_m)])
        cv2.fillConvexPoly(mask, cv2.convexHull(np.int32(np.round(pts))), 1)
    return mask


class BackgroundDetector:
    """現場のランタイムの Perception に差し込む検出（detector(frame, robot, jaw_angle)）。"""

    def __init__(self, cam: CameraModel, background_rgb: np.ndarray, cfg: BackgroundDetectCfg, robot_polygons,
                 jaw_top_m: float, roi_half_m: tuple[float, float] | None = None,
                 keep_half_m: tuple[float, float] | None = None, drop_large: bool = False,
                 body_heights_m: tuple[float, ...] | None = None):
        """robot_polygons(robot_pose, jaw_angle) → ロボットの部品の床の多角形（perception_env.robot_polygons: 車体、爪…）。
        jaw_top_m: 爪の上端の高さ（車体は cfg.robot_height_m、爪はこの高さまで除く）。
        roi_half_m: 物を探す床の長方形（中心 0）の半分の幅・奥行き。外の画素は物と見ない（実機: 柵・ソファ・部屋の奥。
        床より上の画素を床に戻すと計算があふれる）。None は画像全体（Gazebo のリハーサル）。
        keep_half_m: 物とみなす塊の中心の範囲（中心 0 の長方形の半分）。外に中心がある塊は、大きくても物と見ない（止めない。
        実機: 作業領域の外の帯にかかるソファの脚・影）。None は全部（Gazebo のリハーサル）。
        drop_large: 大きすぎる塊（max_area_m2 超え）を、例外にせず物でないとして捨て、last_large に残す（実機: カメラが少し動いた・
        人や影が入ったとき、ほかの物は見続ける）。False は例外（Gazebo のリハーサル）。
        body_heights_m: 先頭から車体の部品の高さ（残りは爪）。None は先頭の 1 つが車体で cfg.robot_height_m（実機は後ろと、
        低いグリッパーの取り付け部の 2 つ）。"""
        self.jaw_top_m = jaw_top_m
        self.body_heights_m = (cfg.robot_height_m,) if body_heights_m is None else tuple(body_heights_m)
        if not self.body_heights_m or min(self.body_heights_m) <= 0:
            raise ValueError(f"body_heights_m は 1 つ以上の正の高さ: {body_heights_m}")
        if background_rgb.shape[1::-1] != tuple(cam.intr.size):
            raise ValueError(f"背景の大きさ {background_rgb.shape[1::-1]} がカメラの {cam.intr.size} と違う")
        self.cam, self.cfg, self.robot_polygons = cam, cfg, robot_polygons
        self.bg = self._lab(background_rgb)
        self.background_rgb = background_rgb
        self.roi = None if roi_half_m is None else self._roi(roi_half_m, background_rgb.shape[:2])
        self.keep_half = keep_half_m
        self.drop_large = drop_large
        self.last_large: list[tuple[float, tuple[int, int, int, int]]] = []   # 直前の detect で捨てた大きすぎる塊（面積、画像の枠）

    def _roi(self, half: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
        """床の長方形を画像に投影した範囲（uint8、1 が探す画素）。辺は細かく分けて投影する（レンズの歪みで曲がる）。"""
        hx, hy = half
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        pts = np.vstack([np.linspace(a, b, 50, endpoint=False) for a, b in zip(corners, corners[1:] + corners[:1])])
        roi = np.zeros(shape, np.uint8)
        cv2.fillPoly(roi, [np.round(self.cam.to_px(pts)).astype(np.int32)], 1)
        return roi

    def _lab(self, rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)

    def foreground(self, rgb: np.ndarray, robot_mask_u8: np.ndarray | None) -> np.ndarray:
        d = self._lab(rgb) - self.bg
        dist = np.sqrt((self.cfg.luma_weight * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
        fg = (dist >= self.cfg.threshold).astype(np.uint8)
        if self.roi is not None:
            fg[self.roi == 0] = 0
        if robot_mask_u8 is not None:
            fg[robot_mask_u8 > 0] = 0
        k = np.ones((self.cfg.open_px, self.cfg.open_px), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
        if self.cfg.close_px > 0:   # 窓・タイヤの暗い帯で割れたミニカーをつなぐ（実機）。除いた範囲には広げない
            c = self.cfg.close_px
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)))
            if robot_mask_u8 is not None:
                fg[robot_mask_u8 > 0] = 0
            if self.roi is not None:
                fg[self.roi == 0] = 0
        return fg

    def split_by_color(self, lab: np.ndarray, blob: np.ndarray) -> list[np.ndarray]:
        """くっついた違う色の物体の塊を 2 つに分ける（分けなければ [blob]）。色を持つ画素（a・b の距離が split_chroma_min 以上）を
        a・b で 2 つに分け（k-means）、色の中心が split_min_ab 以上離れ、少ない側が split_min_fraction 以上、2 つの重心が
        塊の大きさの split_min_separation 以上離れているときだけ分ける。1 つの物の縁の色ずれ（床と混ざる）は中心を囲む輪に
        なるので重心が離れず、ミニカーの暗い窓・タイヤは色を持たないので分けない。色を持たない画素は近い方の色の画素の側に入れる。
        2026-10-22、実機の画像 3 枚の 21 個の塊で決めた（くっついた LEGO 2 個の 3 例だけが分かれる。未検証）。"""
        ys, xs = np.nonzero(blob)
        ab = lab[ys, xs, 1:3] - 128.0
        sat = np.hypot(ab[:, 0], ab[:, 1]) >= self.cfg.split_chroma_min
        if sat.sum() < 2 * SPLIT_MIN_PIXELS:
            return [blob]
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        cv2.setRNGSeed(0)   # k-means の初期値を毎回同じにする（同じ画像で同じ結果）
        _, lbl, ctr = cv2.kmeans(ab[sat].astype(np.float32), 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lbl = lbl.ravel()
        frac = min((lbl == 0).mean(), (lbl == 1).mean())
        px = np.stack([xs[sat], ys[sat]], axis=1).astype(float)
        sep = np.linalg.norm(px[lbl == 0].mean(axis=0) - px[lbl == 1].mean(axis=0)) / math.sqrt(len(ys))
        if (np.linalg.norm(ctr[0] - ctr[1]) < self.cfg.split_min_ab or frac < self.cfg.split_min_fraction
                or sep < self.cfg.split_min_separation):
            return [blob]
        # 塊の画素を、近い方の色の画素の側に分ける（距離変換）
        dist = []
        for k in (0, 1):
            seed = np.ones_like(blob)
            seed[ys[sat][lbl == k], xs[sat][lbl == k]] = 0
            dist.append(cv2.distanceTransform(seed, cv2.DIST_L2, 3))
        part0 = (blob > 0) & (dist[0] <= dist[1])
        return [part0.astype(np.uint8), ((blob > 0) & ~part0).astype(np.uint8)]

    def detect(self, rgb: np.ndarray, robot_mask_u8: np.ndarray | None) -> list[Detection]:
        return self.measure(rgb, self.blobs(rgb, robot_mask_u8))

    def blobs(self, rgb: np.ndarray, robot_mask_u8: np.ndarray | None) -> list[tuple[np.ndarray, int, int]]:
        """物体ごとの画素の塊（切り出し（uint8）、切り出しの左上 x, y）。ここは背景との差（学習する検出器は seg_detector）。"""
        fg = self.foreground(rgb, robot_mask_u8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        # 塊ごとに、その外接矩形だけを切り出して調べる（画像全体の大きさの配列を塊ごとに作ると、フル HD で 1 枚 約 100 ms
        # かかり、カメラの 10 fps に追いつかなかった。2026-10-22）
        blobs = []   # (塊の切り出し, 切り出しの左上 x, y)
        for i in range(1, n):
            x0, y0, bw, bh = (int(v) for v in stats[i, :4])
            sl = (slice(y0, y0 + bh), slice(x0, x0 + bw))
            blob = (labels[sl] == i).astype(np.uint8)
            if self.cfg.color_split:
                lab = cv2.cvtColor(rgb[sl], cv2.COLOR_RGB2LAB).astype(np.float32)
                blobs += [(p, x0, y0) for p in self.split_by_color(lab, blob)]
            else:
                blobs.append((blob, x0, y0))
        return blobs

    def measure(self, rgb: np.ndarray, blobs: list[tuple[np.ndarray, int, int]]) -> list[Detection]:
        """塊ごとに、輪郭を物体の高さの半分の面に戻して、床の中心・向き・大きさと画素の平均の色を出す。"""
        out = []
        self.last_large = []
        for blob, ox, oy in blobs:
            cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = max(cnts, key=len).reshape(-1, 2).astype(float) + (ox, oy)
            floor = self.cam.to_floor(contour, self.cfg.object_mid_height_m)
            area = abs(cv2.contourArea(floor.astype(np.float32)))
            if area < self.cfg.min_area_m2:
                continue
            if self.keep_half is not None:
                mx, my = floor.mean(axis=0)
                if abs(mx) > self.keep_half[0] or abs(my) > self.keep_half[1]:
                    continue
            if area > self.cfg.max_area_m2:
                ys, xs = np.nonzero(blob)
                box_px = (int(xs.min()) + ox, int(ys.min()) + oy, int(xs.max()) + 1 + ox, int(ys.max()) + 1 + oy)
                if self.drop_large:
                    self.last_large.append((area, box_px))
                    continue
                raise BlobTooLarge(f"背景との差の塊が大きすぎる（{area:.4f} m²、画像の ({int(xs.min()) + ox}, {int(ys.min()) + oy})〜"
                                   f"({int(xs.max()) + ox}, {int(ys.max()) + oy})）。照明が変わったか、カメラが動いたか、人・影が"
                                   "入った。何も置いていない床で校正し直す", area,
                                   (int(xs.min()) + ox, int(ys.min()) + oy, int(xs.max()) + 1 + ox, int(ys.max()) + 1 + oy))
            (cx, cy), (w, h), _ = cv2.minAreaRect(floor.astype(np.float32))
            box = cv2.boxPoints(((cx, cy), (w, h), _))
            e1, e2 = box[1] - box[0], box[2] - box[1]
            long_e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
            yaw = wrap_angle(math.atan2(long_e[1], long_e[0]))
            ys, xs = np.nonzero(blob)
            mean = tuple(float(v) for v in rgb[ys + oy, xs + ox].mean(axis=0))
            out.append(Detection(Pose2D(float(cx), float(cy), yaw), (float(max(w, h)), float(min(w, h))), mean,
                                 (int(xs.min()) + ox, int(ys.min()) + oy, int(xs.max()) + 1 + ox, int(ys.max()) + 1 + oy)))
        return out

    def mask(self, shape: tuple[int, int], robot: Pose2D, jaw_angle: float) -> np.ndarray:
        """ロボット（車体と爪）が画像で占める範囲。この範囲の物体は見えない（検出から除き、隠れている物体の判定にも使う）。"""
        polys = self.robot_polygons(robot, jaw_angle)
        nb = len(self.body_heights_m)                    # 先頭の nb 個が車体の部品、残りが爪
        if len(polys) < nb:
            raise ValueError(f"ロボットの部品 {len(polys)} 個が、車体の部品の数 {nb} より少ない")
        heights = list(self.body_heights_m) + [self.jaw_top_m] * (len(polys) - nb)
        margins = [self.cfg.robot_margin_m] * nb + [self.cfg.jaw_margin_m] * (len(polys) - nb)
        return robot_mask(shape, self.cam, polys, heights, margins)

    def hidden_by_robot(self, mask: np.ndarray, xy: tuple[float, float]) -> bool:
        """床の点（物体の高さの半分）が、ロボットの陰（mask）に入っているか。"""
        u, v = np.round(self.cam.to_px(np.array([xy]), self.cfg.object_mid_height_m)[0]).astype(int)
        h, w = mask.shape
        return 0 <= u < w and 0 <= v < h and bool(mask[v, u])

    def __call__(self, frame, robot: Pose2D, jaw_angle: float) -> list[Detection]:
        return self.detect(frame.image, self.mask(frame.image.shape[:2], robot, jaw_angle))
