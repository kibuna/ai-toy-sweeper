"""カメラの校正（doc/camera_setup.md）。

- 内部パラメータ（焦点距離、画像の中心、レンズの歪み）: ChArUco ボードの写真から。
- 外部パラメータ（床に対するカメラの姿勢）: 床の 4 隅の ArUco マーカーの写真から（solvePnP）。
- CameraModel: 床の座標（任意の高さの水平面）と画素の行き来。真上から見ないカメラでは、高さのある点（ロボットの上の
  マーカー、物体の上面）は、その高さの平面で計算しないと床の位置がずれる。
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import SimCfg
from .markers import CORNER_SIGNS, MarkersCfg, aruco_dictionary, corner_marker_centers, corner_marker_square


# ボードの写真で使う角の数の下限（6 × 8 マスの内側の角 35 個の約 1/3）。少ない写真は姿勢が不安定（未検証）
MIN_BOARD_CORNERS = 12
# レンズのモデル: 画像の中心を固定し、fx = fy、接線の歪み 0、k3 0（k1・k2 と焦点距離だけ求める）。2026-10-22、C920 の
# ボードの写真 5 枚で全部を求めると fx 1800 / fy 4846・k3 −15 に壊れ（再投影誤差は 0.32 px で見かけは良い）、このモデルでは
# f 1381 px（公称の対角 78° から 約 1360 px）、床のマーカー間の距離もそろった。Web カメラは画像の中心がレンズの中心に近く、
# 画素は正方形という前提（C920 で中心のずれは未確認）。写真を 15〜20 枚撮って全部を求めたものとの比較は未実施
SIMPLE_LENS_FLAGS = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
                     | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
# 床のマーカーの角の合わせ方。自由な長方形（estimate_extrinsics_free_rect）では縮尺がマーカーの大きさ（60 mm）だけで
# 決まり、角の偏りがそのまま作業領域の大きさに出る。合成画像（1920×1080、0.75 × 0.59 m）で比べた（2026-10-22）:
# なし 0.760 × 0.599（0.51 px）、SUBPIX 0.753 × 0.593（0.24 px）、CONTOUR 0.761 × 0.599（0.20 px）、APRILTAG 0.751 × 0.590
# （0.07 px）。実カメラの写真での比較は未実施。1920×1080 で 1 枚 111 ms（SUBPIX は 8 ms）と遅いので、校正だけに使う
FIELD_CORNER_REFINEMENT = cv2.aruco.CORNER_REFINE_APRILTAG
# 焦点距離の初期値（画像の幅に対する比）。0.75 は水平の視野角 67° ほど。最適化の出発点で、結果は写真で決まる
FOCAL_GUESS_PER_WIDTH = 0.75


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Intrinsics:
    K: np.ndarray            # [3, 3]
    dist: np.ndarray         # [k1, k2, p1, p2, k3]
    size: tuple[int, int]    # (幅, 高さ) px
    rms_px: float            # 校正の再投影誤差（ボードの角）。視野角から作ったものは 0


def intrinsics_from_fov(size: tuple[int, int], hfov_rad: float) -> Intrinsics:
    """歪みのない理想のカメラ（Gazebo のカメラ）の内部パラメータ。"""
    w, h = size
    f = (w / 2) / math.tan(hfov_rad / 2)
    return Intrinsics(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float), np.zeros(5), size, 0.0)


def camera_from_gazebo_pose(pose: list[float], hfov_rad: float, size: tuple[int, int]) -> CameraModel:
    """Gazebo のカメラ（姿勢 [x, y, z, roll, pitch, yaw]、カメラは自分の +x を向き +z が上）の CameraModel（歪みなし）。
    マーカーで校正しない（見えない画角でも使える）。学習する検出器のデータの評価用（scripts/eval_seg_detector.py）。"""
    x, y, z, r, p, yw = pose
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(yw), math.sin(yw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    # OpenCV のカメラの軸（右, 下, 前）を Gazebo のカメラの軸（前 x, 左 y, 上 z）で表したもの
    world_from_cv = Rz @ Ry @ Rx @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    R = world_from_cv.T
    tvec = -R @ np.array([x, y, z], float)
    return CameraModel(intrinsics_from_fov(size, hfov_rad), cv2.Rodrigues(R)[0].reshape(3), tvec, 0.0,
                       (math.nan, math.nan), math.nan)


def charuco_board(m: MarkersCfg):
    b = m.intrinsics
    return cv2.aruco.CharucoBoard(tuple(b.squares), b.square_size_m, b.marker_size_m, aruco_dictionary(b.dictionary))


def calibrate_intrinsics(images: list[np.ndarray], m: MarkersCfg, min_corners: int) -> tuple[Intrinsics, list[float]]:
    """ボードの写真（BGR）から内部パラメータ（SIMPLE_LENS_FLAGS のモデル）。戻り値の 2 つめは写真ごとの再投影誤差
    （px、使えなかった写真は nan）。"""
    if not images:
        raise CalibrationError("ボードの写真がない")
    board = charuco_board(m)
    det = cv2.aruco.CharucoDetector(board)
    size = (images[0].shape[1], images[0].shape[0])
    objp, imgp, used = [], [], []
    for k, img in enumerate(images):
        if (img.shape[1], img.shape[0]) != size:
            raise CalibrationError(f"写真 {k} の大きさ {img.shape[1]}x{img.shape[0]} が {size} と違う（解像度をそろえる）")
        ch, ids, _, _ = det.detectBoard(img)
        if ch is None or len(ch) < min_corners:
            continue
        o, i = board.matchImagePoints(ch, ids)
        objp.append(o)
        imgp.append(i)
        used.append(k)
    if len(used) < 5:
        raise CalibrationError(f"角が {min_corners} 個以上読めたボードの写真が {len(used)} 枚（5 枚以上要る）")
    f0 = FOCAL_GUESS_PER_WIDTH * size[0]
    K0 = np.array([[f0, 0, size[0] / 2], [0, f0, size[1] / 2], [0, 0, 1]])
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, size, K0, np.zeros(5), flags=SIMPLE_LENS_FLAGS)
    errs = [math.nan] * len(images)
    for k, o, i, r, t in zip(used, objp, imgp, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, r, t, K, dist)
        errs[k] = float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1))))
    return Intrinsics(K, dist.ravel(), size, float(rms)), errs


@dataclass(frozen=True)
class CameraModel:
    intr: Intrinsics
    rvec: np.ndarray         # 床 → カメラの回転（Rodrigues）
    tvec: np.ndarray
    reproj_px: float         # 床のマーカーの角の再投影誤差の平均
    work_area_size_m: tuple[float, float]   # 作業領域の幅・奥行き（4 隅のマーカーの置き方から）
    rect_deviation_m: float  # 作業領域の角（マーカーから出したもの）の長方形からのずれの最大。長方形を仮定したときは nan

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    def position(self) -> np.ndarray:
        """カメラの位置（床の座標、m）。"""
        return (-self.R.T @ self.tvec.reshape(3)).astype(float)

    def to_px(self, xy: np.ndarray, z: float = 0.0) -> np.ndarray:
        """床の点（[N, 2]、高さ z の水平面）→ 画素 [N, 2]。"""
        xy = np.asarray(xy, float).reshape(-1, 2)
        pts = np.hstack([xy, np.full((len(xy), 1), z)])
        p, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.intr.K, self.intr.dist)
        return p.reshape(-1, 2)

    def to_floor(self, uv: np.ndarray, z: float = 0.0) -> np.ndarray:
        """画素 [N, 2] → 高さ z の水平面との交点（床の座標 [N, 2]）。"""
        uv = np.asarray(uv, float).reshape(-1, 1, 2)
        n = cv2.undistortPoints(uv, self.intr.K, self.intr.dist).reshape(-1, 2)
        # 床の座標の向き [3, N]。@（BLAS）は macOS（Accelerate）で、入力も結果も有限なのに「ゼロ除算・あふれ」の警告を出した
        # （2026-10-22、画像全体の画素で。einsum と結果は一致）。3×3 と 3×N の小さな積なので BLAS を通さない einsum にする
        rays = np.einsum("ij,jk->ik", self.R.T, np.hstack([n, np.ones((len(n), 1))]).T)
        c = self.position()
        s = (z - c[2]) / rays[2]
        if np.any(s <= 0):
            raise CalibrationError("画素の向きが平面と交わらない（地平線より上）")
        return (c[:2, None] + rays[:2] * s).T

    def to_world(self, uv: np.ndarray) -> np.ndarray:
        """床（高さ 0）への変換。perception.Homography と同じ名前（現場のランタイムがどちらでも使えるように）。"""
        return self.to_floor(uv, 0.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({
            "image_size": list(self.intr.size), "K": self.intr.K.tolist(), "dist": self.intr.dist.tolist(),
            "intrinsics_rms_px": self.intr.rms_px, "rvec": self.rvec.ravel().tolist(), "tvec": self.tvec.ravel().tolist(),
            "floor_reproj_px": self.reproj_px, "work_area_size_m": list(self.work_area_size_m),
            "rect_deviation_m": self.rect_deviation_m,
            "camera_position_m": self.position().tolist()}, sort_keys=False))

    @staticmethod
    def load(path: Path) -> "CameraModel":
        d = yaml.safe_load(path.read_text())
        intr = Intrinsics(np.array(d["K"]), np.array(d["dist"]), tuple(d["image_size"]), d["intrinsics_rms_px"])
        return CameraModel(intr, np.array(d["rvec"]), np.array(d["tvec"]), d["floor_reproj_px"],
                           tuple(d["work_area_size_m"]), float(d["rect_deviation_m"]))


def detect_field_markers(img: np.ndarray, m: MarkersCfg, refinement: int = FIELD_CORNER_REFINEMENT) -> dict[int, np.ndarray]:
    """床（とロボット）の ArUco を見つける。ID → 4 つの角の画素 [4, 2]。refinement は角の合わせ方（cv2.aruco.CORNER_REFINE_*）。"""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = refinement
    det = cv2.aruco.ArucoDetector(aruco_dictionary(m.field.dictionary), params)
    corners, ids, _ = det.detectMarkers(img)
    if ids is None:
        return {}
    out: dict[int, np.ndarray] = {}
    for c, i in zip(corners, ids.ravel()):
        if int(i) in out:
            raise CalibrationError(f"ID {int(i)} のマーカーが 2 つ見つかった")
        out[int(i)] = c.reshape(4, 2)
    return out


def estimate_extrinsics(img: np.ndarray, intr: Intrinsics, m: MarkersCfg, sim: SimCfg,
                        wall_thickness_m: float) -> CameraModel:
    """床の写真（BGR）の 4 隅のマーカーから、床に対するカメラの姿勢を求める（wall_thickness_m は markers.corner_marker_centers）。

    マーカーの向きは問わない（置く人が回してよい）ので、まず 4 つの中心で姿勢を求め、それで各マーカーの黒い四角の
    角を投影して、近い検出の角と対応させ、16 点で求め直す。
    """
    if (img.shape[1], img.shape[0]) != intr.size:
        raise CalibrationError(f"床の写真 {img.shape[1]}x{img.shape[0]} と内部パラメータの {intr.size} が違う")
    found = detect_field_markers(img, m)
    want = sorted(m.field.corner_ids.values())
    missing = [i for i in want if i not in found]
    if missing:
        raise CalibrationError(f"床の 4 隅のマーカーが見えない: ID {missing}（見えた {sorted(found)}）")
    centers = corner_marker_centers(m, sim, wall_thickness_m)
    obj = np.array([[*centers[i], 0.0] for i in want])
    pix = np.array([found[i].mean(axis=0) for i in want])
    ok, rvec, tvec = cv2.solvePnP(obj, pix, intr.K, intr.dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise CalibrationError("マーカーの中心から姿勢を求められない")
    model = CameraModel(intr, rvec, tvec, math.nan, tuple(sim.work_area.size_m), math.nan)
    obj16, pix16 = [], []
    for i in want:
        sq = corner_marker_square(m, sim, wall_thickness_m, i)
        proj = model.to_px(sq)
        for j, p in enumerate(found[i]):
            k = int(np.argmin(np.linalg.norm(proj - p, axis=1)))
            obj16.append([*sq[k], 0.0])
            pix16.append(p)
    obj16, pix16 = np.array(obj16), np.array(pix16)
    if len({tuple(o) for o in obj16}) != len(obj16):
        raise CalibrationError("マーカーの角の対応が決まらない（中心で求めた姿勢が大きくずれている）")
    ok, rvec, tvec = cv2.solvePnP(obj16, pix16, intr.K, intr.dist, rvec, tvec, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise CalibrationError("16 点で姿勢を求め直せない")
    p, _ = cv2.projectPoints(obj16, rvec, tvec, intr.K, intr.dist)
    err = float(np.mean(np.linalg.norm(p.reshape(-1, 2) - pix16, axis=1)))
    return CameraModel(intr, rvec, tvec, err, tuple(sim.work_area.size_m), math.nan)


def sim_with_area(sim: SimCfg, size_m: tuple[float, float]) -> SimCfg:
    """作業領域の大きさだけを差し替えた SimCfg（マーカーの配置・確かめの画像を、校正で求めた大きさで計算する）。"""
    return dataclasses.replace(sim, work_area=dataclasses.replace(sim.work_area, size_m=tuple(size_m)))


def estimate_extrinsics_free_rect(img: np.ndarray, intr: Intrinsics, m: MarkersCfg, sim: SimCfg,
                                  wall_thickness_m: float) -> CameraModel:
    """4 隅のマーカーを手で長方形の角あたりに置いた床の写真（BGR）から、カメラの姿勢と作業領域（長方形）を求める
    （2026-10-22、マーカーを自由に置いた長方形に対応する）。置き方の規則（奥は内側の角、手前は外側）は estimate_extrinsics と同じだが、
    手で置いたマーカーは長方形からずれ、回る（実機で 4 cm・7°）ので、マーカーごとの床の上の位置と向きを未知数にする。
    1. マーカー 1 枚ずつの姿勢（黒い四角の大きさが分かっている）から、床の平面と各マーカーの位置・向きの初期値を作る。
    2. カメラの姿勢と各マーカーの位置・向きで、16 点の再投影誤差を最小にする（床の座標の原点と向きは手前・左のマーカーで
       仮に決める）。マーカーの角の並びはマーカー自身の向きで決まるので、対応に迷いはない。
    3. 各マーカーから規則どおりの作業領域の角を出し、近い長方形（向き・中心・幅・奥行き）を当てはめ、その中心を原点、
       幅の向きを x にした床の座標に直す。角の長方形からのずれの最大を rect_deviation_m に入れる。"""
    from scipy.optimize import least_squares
    if (img.shape[1], img.shape[0]) != intr.size:
        raise CalibrationError(f"床の写真 {img.shape[1]}x{img.shape[0]} と内部パラメータの {intr.size} が違う")
    if wall_thickness_m < 0:
        raise ValueError(f"壁の厚みが負: {wall_thickness_m}")
    found = detect_field_markers(img, m)
    ids = m.field.corner_ids
    missing = [i for i in sorted(ids.values()) if i not in found]
    if missing:
        raise CalibrationError(f"床の 4 隅のマーカーが見えない: ID {missing}（見えた {sorted(found)}）")
    order = ["near_left", "near_right", "far_right", "far_left"]
    h = m.field.corner_marker_size_m / 2
    local = np.array([[-h, h], [h, h], [h, -h], [-h, -h]])   # ArUco の角の並び（マーカーの座標）
    sq3 = np.hstack([local, np.zeros((4, 1))])
    P, X = {}, {}
    for name in order:
        ok, r, t = cv2.solvePnP(sq3, found[ids[name]], intr.K, intr.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            raise CalibrationError(f"ID {ids[name]} のマーカー 1 枚の姿勢を求められない")
        P[name], X[name] = t.ravel(), cv2.Rodrigues(r)[0]
    # 仮の床の座標: 原点は手前・左のマーカーの中心、x は手前・右のマーカーへ、z は床の上（カメラの側）
    z = np.mean([X[n][:, 2] for n in order], axis=0)
    z /= np.linalg.norm(z)
    if -P["near_left"] @ z < 0:
        z = -z
    x = P["near_right"] - P["near_left"]
    x -= (x @ z) * z
    x /= np.linalg.norm(x)
    R0 = np.column_stack([x, np.cross(z, x), z])             # 仮の床 → カメラ
    t0 = P["near_left"]
    init = {}
    for n in order:
        c = R0.T @ (P[n] - t0)
        a = R0.T @ X[n][:, 0]
        init[n] = (c[0], c[1], math.atan2(a[1], a[0]))
    pix = np.vstack([found[ids[n]] for n in order])

    def unpack(v):
        """v: rvec 3、tvec 3、手前・左の向き、手前・右の x と向き、奥の 2 つの x・y・向き。"""
        poses = {"near_left": (0.0, 0.0, v[6]), "near_right": (v[7], 0.0, v[8]),
                 "far_right": tuple(v[9:12]), "far_left": tuple(v[12:15])}
        return v[:3], v[3:6], poses

    def floor_corners(poses):
        out = []
        for n in order:
            cx, cy, th = poses[n]
            c, s_ = math.cos(th), math.sin(th)
            out.append(local @ np.array([[c, s_], [-s_, c]]) + (cx, cy))
        return np.vstack(out)

    def resid(v):
        rvec, tvec, poses = unpack(v)
        q = np.hstack([floor_corners(poses), np.zeros((16, 1))])
        p, _ = cv2.projectPoints(q, rvec, tvec, intr.K, intr.dist)
        return (p.reshape(-1, 2) - pix).ravel()

    v0 = np.concatenate([cv2.Rodrigues(R0)[0].ravel(), t0, [init["near_left"][2]], [init["near_right"][0], init["near_right"][2]],
                         init["far_right"], init["far_left"]])
    r = least_squares(resid, v0, x_scale="jac")
    if not r.success:
        raise CalibrationError(f"カメラの姿勢とマーカーの位置を求められない: {r.message}")
    err = float(np.mean(np.linalg.norm(resid(r.x).reshape(-1, 2), axis=1)))
    rvec, tvec, poses = unpack(r.x)
    cen = {n: np.array(poses[n][:2]) for n in order}
    # 長方形の向き: 幅の向き（手前・奥の左 → 右）と、奥行きの向き（左・右の手前 → 奥）を 90° 回したものの平均
    unit = lambda d: d / np.linalg.norm(d)
    dirs = [unit(cen["near_right"] - cen["near_left"]), unit(cen["far_right"] - cen["far_left"])]
    dirs += [unit(np.array([d[1], -d[0]])) for d in (cen["far_left"] - cen["near_left"], cen["far_right"] - cen["near_right"])]
    mean = np.sum(dirs, axis=0)
    theta = math.atan2(mean[1], mean[0])
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])   # 長方形 → 仮の床
    half, t = m.field.paper_margin_m + h, wall_thickness_m
    q = {}
    for n in order:
        sx, sy = CORNER_SIGNS[n]
        off = (sx * half, half) if sy > 0 else (sx * (half - t), half + t)   # マーカーの中心 → 作業領域の角
        q[n] = rot.T @ cen[n] + off
    cq = np.mean([q[n] for n in order], axis=0)
    w = ((q["near_right"][0] - q["near_left"][0]) + (q["far_right"][0] - q["far_left"][0])) / 2
    d = ((q["far_left"][1] - q["near_left"][1]) + (q["far_right"][1] - q["near_right"][1])) / 2
    if min(w, d) <= 4 * half:
        raise CalibrationError(f"作業領域 {w:.3f} × {d:.3f} m が小さすぎる（マーカーが重なる）")
    dev = max(float(np.linalg.norm(q[n] - (cq + (CORNER_SIGNS[n][0] * w / 2, CORNER_SIGNS[n][1] * d / 2)))) for n in order)
    # 作業領域の座標（中心が原点、x は幅の向き）→ 仮の床: X_tmp = Rz (X + c)
    Rz = np.eye(3)
    Rz[:2, :2] = rot
    c3 = np.array([*(rot @ cq), 0.0])
    R = cv2.Rodrigues(rvec)[0]
    return CameraModel(intr, cv2.Rodrigues(R @ Rz)[0], (R @ c3 + tvec).reshape(3, 1), err, (float(w), float(d)), dev)


def draw_check(img: np.ndarray, cam: CameraModel, sim: SimCfg, goals: dict) -> np.ndarray:
    """確かめの画像: 作業領域の枠（赤）、10 cm の格子（白）、ゴールの枠（緑）を床の写真に重ねる。"""
    out = img.copy()
    hx, hy = sim.work_area.size_m[0] / 2, sim.work_area.size_m[1] / 2

    def line(a, b, color, w):
        pts = cam.to_px(np.linspace(a, b, 20))
        cv2.polylines(out, [np.int32(np.round(pts))], False, color, w, cv2.LINE_AA)
    for x in np.arange(-hx, hx + 1e-9, 0.1):
        line((x, -hy), (x, hy), (255, 255, 255), 1)
    for y in np.arange(-hy, hy + 1e-9, 0.1):
        line((-hx, y), (hx, y), (255, 255, 255), 1)
    for g in goals.values():
        (cx, cy), (sx, sy) = g.center, g.size
        c = [(cx - sx / 2, cy - sy / 2), (cx + sx / 2, cy - sy / 2), (cx + sx / 2, cy + sy / 2), (cx - sx / 2, cy + sy / 2)]
        for a, b in zip(c, c[1:] + c[:1]):
            line(a, b, (60, 180, 60), 2)
    c = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    for a, b in zip(c, c[1:] + c[:1]):
        line(a, b, (0, 0, 255), 2)
    return out
