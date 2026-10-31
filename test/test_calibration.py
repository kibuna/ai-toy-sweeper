"""カメラの校正（toy_sweeper.calibration）の試験。カメラの姿勢が分かっている合成画像で、求め直せるかを確かめる。"""
import math

import cv2
import numpy as np
import pytest

import dataclasses

from toy_sweeper.calibration import (CalibrationError, CameraModel, calibrate_intrinsics, charuco_board,
                                     estimate_extrinsics, estimate_extrinsics_free_rect, intrinsics_from_fov)
from toy_sweeper.config import load_all
from toy_sweeper.markers import corner_marker_centers, load_markers_cfg, marker_image

PX_PER_M = 2000          # 床の模様の細かさ（合成画像を作るためだけ）
WALL = 0.01              # 手前の壁の厚み（壁あり）
FLOOR_HALF = 0.5


@pytest.fixture(scope="module")
def cfg():
    return load_all()


@pytest.fixture(scope="module")
def m():
    return load_markers_cfg()


def _look_at(cam_pos, target):
    """床 → カメラの回転と並進（カメラの z が target を向き、画像の上が奥）。"""
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return cv2.Rodrigues(R)[0], -R @ cam_pos


def _sim_with_area(cfg, size_m):
    return dataclasses.replace(cfg.sim, work_area=dataclasses.replace(cfg.sim.work_area, size_m=tuple(size_m)))


def _floor_texture(m, cfg, rotations, size_m=None):
    n = int(2 * FLOOR_HALF * PX_PER_M)
    tex = np.full((n, n), 200, np.uint8)
    s = m.field.corner_marker_size_m
    sim = cfg.sim if size_m is None else _sim_with_area(cfg, size_m)
    for (mid, (x, y)), rot in zip(corner_marker_centers(m, sim, WALL).items(), rotations):
        img = np.rot90(marker_image(m, mid, int(s * PX_PER_M), int(m.field.paper_margin_m * PX_PER_M)), rot)
        u, v = int((x + FLOOR_HALF) * PX_PER_M), int((FLOOR_HALF - y) * PX_PER_M)
        h = img.shape[0] // 2
        tex[v - h:v - h + img.shape[0], u - h:u - h + img.shape[1]] = img
    return tex


def _render(tex, intr, rvec, tvec):
    """床の模様（上が奥）を、カメラから見た画像にする（z = 0 の平面の射影変換）。"""
    R = cv2.Rodrigues(rvec)[0]
    H_floor = intr.K @ np.column_stack([R[:, 0], R[:, 1], tvec.ravel()])          # 床 (x, y, 1) → 画素
    T = np.array([[1 / PX_PER_M, 0, -FLOOR_HALF], [0, -1 / PX_PER_M, FLOOR_HALF], [0, 0, 1]])   # 模様の画素 → 床
    img = cv2.warpPerspective(tex, H_floor @ T, intr.size, flags=cv2.INTER_AREA, borderValue=120)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_extrinsics_recovers_oblique_camera(cfg, m):
    intr = intrinsics_from_fov((1280, 720), 1.22)
    cam = np.array([0.05, -0.85, 0.8])
    rvec, tvec = _look_at(cam, (0.0, -0.08, 0.0))
    img = _render(_floor_texture(m, cfg, rotations=(0, 1, 2, 3)), intr, rvec, tvec)   # マーカーは 4 つとも違う向き
    model = estimate_extrinsics(img, intr, m, cfg.sim, WALL)
    assert np.linalg.norm(model.position() - cam) < 0.005
    assert model.reproj_px < 1.0
    # 床の点 → 画素 → 床の点（高さ 3 cm の平面でも）
    pts = np.array([[-0.3, -0.3], [0.2, 0.35], [0.0, 0.0]])
    for z in (0.0, 0.03):
        assert np.allclose(model.to_floor(model.to_px(pts, z), z), pts, atol=1e-6)


def test_extrinsics_requires_all_four_corners(cfg, m):
    intr = intrinsics_from_fov((1280, 720), 1.22)
    rvec, tvec = _look_at(np.array([0.0, -0.85, 0.8]), (0.0, -0.08, 0.0))
    tex = _floor_texture(m, cfg, rotations=(0, 0, 0, 0))
    tex[: tex.shape[0] // 3, : tex.shape[1] // 3] = 200    # 奥・左（ID 3）を消す
    with pytest.raises(CalibrationError, match="ID \\[3\\]"):
        estimate_extrinsics(_render(tex, intr, rvec, tvec), intr, m, cfg.sim, WALL)


@pytest.mark.parametrize("size_m", [(0.75, 0.59), (0.8, 0.8), (0.6, 0.7)])
def test_free_rect_recovers_area_size(cfg, m, size_m):
    """4 隅のマーカーを自由な大きさの長方形に置いても（2026-10-22）、幅・奥行きとカメラの
    姿勢を求め直せる。"""
    intr = intrinsics_from_fov((1920, 1080), 1.22)
    cam = np.array([0.05, -0.85, 0.8])
    rvec, tvec = _look_at(cam, (0.0, -0.08, 0.0))
    img = _render(_floor_texture(m, cfg, rotations=(0, 1, 2, 3), size_m=size_m), intr, rvec, tvec)
    model = estimate_extrinsics_free_rect(img, intr, m, cfg.sim, WALL)
    assert np.allclose(model.work_area_size_m, size_m, atol=0.003)
    assert np.linalg.norm(model.position() - cam) < 0.005
    assert model.reproj_px < 1.0


def _free_texture(m, cfg, size_m, moves):
    """自由な長方形の置き方から、マーカーごとにずらして回した床の模様（moves: ID → (dx m, dy m, 回転 deg)）。"""
    n = int(2 * FLOOR_HALF * PX_PER_M)
    tex = np.full((n, n), 200, np.uint8)
    side = int(m.field.corner_marker_size_m * PX_PER_M)
    margin = int(m.field.paper_margin_m * PX_PER_M)
    for mid, (x, y) in corner_marker_centers(m, _sim_with_area(cfg, size_m), WALL).items():
        dx, dy, deg = moves.get(mid, (0.0, 0.0, 0.0))
        img = marker_image(m, mid, side, margin)
        k = int(img.shape[0] * 1.5)
        canvas = np.full((k, k), 200, np.uint8)
        o = (k - img.shape[0]) // 2
        canvas[o:o + img.shape[0], o:o + img.shape[0]] = img
        canvas = cv2.warpAffine(canvas, cv2.getRotationMatrix2D((k / 2, k / 2), deg, 1.0), (k, k),
                                flags=cv2.INTER_LINEAR, borderValue=200)
        u, v = int((x + dx + FLOOR_HALF) * PX_PER_M), int((FLOOR_HALF - y - dy) * PX_PER_M)
        tex[v - k // 2:v - k // 2 + k, u - k // 2:u - k // 2 + k] = canvas
    return tex


def test_free_rect_tolerates_hand_placement(cfg, m):
    """手で置いたマーカーは長方形からずれる（2026-10-22 の実機: 奥の右が 4 cm 奥・7° 回っていた）。マーカーごとの位置と
    向きを求めるので再投影誤差は小さいまま、近い長方形を作業領域にし、角の長方形からのずれを返す。"""
    intr = intrinsics_from_fov((1920, 1080), 1.22)
    cam = np.array([0.05, -0.85, 0.8])
    rvec, tvec = _look_at(cam, (0.0, -0.08, 0.0))
    far_right = m.field.corner_ids["far_right"]
    img = _render(_free_texture(m, cfg, (0.75, 0.6), {far_right: (0.0, 0.04, 7.0)}), intr, rvec, tvec)
    model = estimate_extrinsics_free_rect(img, intr, m, cfg.sim, WALL)
    assert model.reproj_px < 1.0
    assert 0.015 < model.rect_deviation_m < 0.03          # 4 cm のずれを長方形の 4 つの角で分け合う
    assert np.allclose(model.work_area_size_m, (0.75, 0.62), atol=0.01)
    assert np.linalg.norm(model.position()[2] - cam[2]) < 0.01   # 高さ（床の平面）は正しい


def test_save_and_load(tmp_path, cfg, m):
    intr = intrinsics_from_fov((1280, 720), 1.22)
    rvec, tvec = _look_at(np.array([0.0, -0.85, 0.8]), (0.0, -0.08, 0.0))
    model = CameraModel(intr, rvec, tvec, 0.3, (0.75, 0.59), 0.012)
    model.save(tmp_path / "camera.yaml")
    back = CameraModel.load(tmp_path / "camera.yaml")
    assert np.allclose(back.position(), model.position()) and np.allclose(back.intr.K, intr.K)
    assert back.work_area_size_m == (0.75, 0.59) and back.rect_deviation_m == 0.012


def test_intrinsics_from_board_photos(m):
    """ボードをいろいろな向きから撮った合成画像（歪みなし）から、焦点距離と画像の中心を求め直せる。"""
    true = intrinsics_from_fov((1280, 720), 1.22)
    board = charuco_board(m)
    bw, bh = m.intrinsics.squares[0] * m.intrinsics.square_size_m, m.intrinsics.squares[1] * m.intrinsics.square_size_m
    ppm = 4000
    bimg = board.generateImage((int(bw * ppm), int(bh * ppm)), marginSize=0, borderBits=1)
    bimg = cv2.copyMakeBorder(bimg, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=255)
    T = np.array([[1 / ppm, 0, -200 / ppm], [0, 1 / ppm, -200 / ppm], [0, 0, 1]])   # 模様の画素 → ボードの座標
    rng = np.random.default_rng(0)
    images = []
    for _ in range(14):
        ang = rng.uniform(-0.5, 0.5, 3)
        R = cv2.Rodrigues(ang)[0]
        t = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.05, 0.05), rng.uniform(0.35, 0.55)]) - R @ [bw / 2, bh / 2, 0]
        H = true.K @ np.column_stack([R[:, 0], R[:, 1], t])
        img = cv2.warpPerspective(bimg, H @ T, true.size, flags=cv2.INTER_AREA, borderValue=128)
        images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    intr, errs = calibrate_intrinsics(images, m, min_corners=12)
    assert abs(intr.K[0, 0] - true.K[0, 0]) / true.K[0, 0] < 0.02
    assert abs(intr.K[0, 2] - 640) < 15 and abs(intr.K[1, 2] - 360) < 15
    assert intr.rms_px < 1.0


def test_intrinsics_simple_lens_from_few_photos(m):
    """ボードの写真が 5 枚でも、画像の中心を固定し fx = fy・接線 0・k3 0 の簡単なレンズのモデルなら、焦点距離が安定する
    （2026-10-22、C920 の 5 枚で全部を求めると fx 1800 / fy 4846 に壊れた）。"""
    true = intrinsics_from_fov((1920, 1080), 1.22)
    board = charuco_board(m)
    bw, bh = m.intrinsics.squares[0] * m.intrinsics.square_size_m, m.intrinsics.squares[1] * m.intrinsics.square_size_m
    ppm = 4000
    bimg = board.generateImage((int(bw * ppm), int(bh * ppm)), marginSize=0, borderBits=1)
    bimg = cv2.copyMakeBorder(bimg, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=255)
    T = np.array([[1 / ppm, 0, -200 / ppm], [0, 1 / ppm, -200 / ppm], [0, 0, 1]])
    rng = np.random.default_rng(1)
    images = []
    for _ in range(5):
        R = cv2.Rodrigues(rng.uniform(-0.3, 0.3, 3))[0]
        t = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.03, 0.03), rng.uniform(0.4, 0.6)]) - R @ [bw / 2, bh / 2, 0]
        H = true.K @ np.column_stack([R[:, 0], R[:, 1], t])
        images.append(cv2.cvtColor(cv2.warpPerspective(bimg, H @ T, true.size, flags=cv2.INTER_AREA, borderValue=128),
                                   cv2.COLOR_GRAY2BGR))
    intr, _ = calibrate_intrinsics(images, m, min_corners=12)
    assert intr.K[0, 0] == intr.K[1, 1]
    assert abs(intr.K[0, 0] - true.K[0, 0]) / true.K[0, 0] < 0.02
    assert (intr.K[0, 2], intr.K[1, 2]) == (960, 540)


def test_camera_from_gazebo_pose():
    """Gazebo のカメラの姿勢（[x, y, z, roll, pitch, yaw]、カメラは自分の +x を向く）と画角から CameraModel を作る
    （学習する検出器のデータの床の位置の誤差を測るため、2026-10-25）。向いた点は画像の中心、世界の −x は画像の左。"""
    from toy_sweeper.calibration import camera_from_gazebo_pose
    from toy_sweeper.seg_data import look_at_pose
    pose = look_at_pose((0.1, -0.8, 0.7), (0.05, 0.1, 0.0), 0.0)
    cam = camera_from_gazebo_pose(pose, 1.2, (1920, 1080))
    assert np.allclose(cam.position(), [0.1, -0.8, 0.7])
    assert np.allclose(cam.to_px(np.array([[0.05, 0.1]]))[0], [960, 540], atol=1e-6)
    left, right = cam.to_px(np.array([[-0.2, 0.1], [0.3, 0.1]]))
    assert left[0] < 960 < right[0]
    near, far = cam.to_px(np.array([[0.05, -0.2], [0.05, 0.4]]))
    assert far[1] < 540 < near[1]                     # 奥は画像の上
    assert np.allclose(cam.to_floor(cam.to_px(np.array([[0.2, -0.1]]), 0.01), 0.01), [[0.2, -0.1]])
