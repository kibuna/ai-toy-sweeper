"""ロボットの上の ArUco（ID 4）から車軸中点の姿勢を出す（toy_sweeper.field.hardware.ArucoRobotPose）。斜め上のカメラで、
高さ 40 mm・車軸から後ろへ 15 mm に貼ったマーカーを合成画像に描き、姿勢を求め直せるかを確かめる。"""
import math

import cv2
import numpy as np
import pytest

from test_calibration import _look_at
from toy_sweeper.calibration import CameraModel, intrinsics_from_fov
from toy_sweeper.field.hardware import ArucoRobotPose, RobotMarkerError, load_robot_marker_yaw, save_robot_marker_yaw
from toy_sweeper.field.interfaces import Frame
from toy_sweeper.geometry import Pose2D
from toy_sweeper.markers import load_markers_cfg, marker_image

INTR = intrinsics_from_fov((1920, 1080), 1.22)


@pytest.fixture(scope="module")
def m():
    return load_markers_cfg()


@pytest.fixture(scope="module")
def cam():
    rvec, tvec = _look_at(np.array([0.05, -0.85, 0.66]), (0.0, -0.05, 0.0))
    return CameraModel(INTR, rvec, tvec, 0.5, (0.75, 0.83), 0.01)


def _frame(m, cam, robot: Pose2D, yaw_offset: float) -> Frame:
    """ロボットの姿勢 robot のとき、マーカーを描いた画像（床は灰色）。マーカーの x 軸の床の向き = robot.yaw + yaw_offset。"""
    f = m.field
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    ox, oy = f.robot_marker_offset_m
    center = np.array([robot.x + c * ox - s * oy, robot.y + s * ox + c * oy])
    my = robot.yaw + yaw_offset
    ux, uy = np.array([math.cos(my), math.sin(my)]), np.array([-math.sin(my), math.cos(my)])
    h = f.robot_marker_size_m / 2
    quad = np.array([center - h * ux + h * uy, center + h * ux + h * uy, center + h * ux - h * uy, center - h * ux - h * uy])
    px = cam.to_px(quad, f.robot_marker_height_m).astype(np.float32)
    side = 400
    img = marker_image(m, f.robot_id, side, 0)
    src = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    out = np.full((1080, 1920), 120, np.uint8)
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, px), (1920, 1080), flags=cv2.INTER_AREA,
                                 borderValue=0)
    mask = cv2.warpPerspective(np.full_like(img, 255), cv2.getPerspectiveTransform(src, px), (1920, 1080))
    # 白い余白（紙）の代わりに、マーカーのまわりを白く塗る
    pad = cv2.dilate(mask, np.ones((25, 25), np.uint8))
    out[pad > 0] = 255
    out[mask > 0] = warped[mask > 0]
    return Frame(cv2.cvtColor(out, cv2.COLOR_GRAY2RGB), 0.0, 1)


@pytest.mark.parametrize("robot", [Pose2D(0.0, 0.0, 0.0), Pose2D(0.2, 0.25, 2.0), Pose2D(-0.3, -0.3, -2.5)])
def test_estimate_recovers_axle_pose(m, cam, robot):
    yaw_offset = 1.3   # 貼った向きは任意
    est = ArucoRobotPose(cam, m, yaw_offset).estimate(_frame(m, cam, robot, yaw_offset))
    assert math.hypot(est.x - robot.x, est.y - robot.y) < 0.002
    assert abs(math.remainder(est.yaw - robot.yaw, math.tau)) < math.radians(1.0)


def test_height_matters(m, cam):
    """高さを 0 とみなすと、斜め上のカメラでは位置が 1 cm 以上ずれる（高さを使う理由）。"""
    import dataclasses
    robot = Pose2D(0.2, 0.25, 0.5)
    frame = _frame(m, cam, robot, 0.0)
    flat = dataclasses.replace(m, field=dataclasses.replace(m.field, robot_marker_height_m=0.0))
    est = ArucoRobotPose(cam, flat, 0.0).estimate(frame)
    assert math.hypot(est.x - robot.x, est.y - robot.y) > 0.01


def test_missing_marker_raises(m, cam):
    frame = Frame(np.full((1080, 1920, 3), 120, np.uint8), 0.0, 1)
    with pytest.raises(RobotMarkerError, match="ID 4"):
        ArucoRobotPose(cam, m, 0.0).estimate(frame)


def test_yaw_file_roundtrip(tmp_path):
    p = tmp_path / "robot_marker.yaml"
    with pytest.raises(RobotMarkerError, match="向き"):
        load_robot_marker_yaw(p)
    save_robot_marker_yaw(p, 1.25, {"n": 3})
    assert load_robot_marker_yaw(p) == 1.25


def test_rereads_around_last_pose_when_full_frame_fails(m, cam, monkeypatch):
    """画像全体で読めないときは、最後の姿勢のまわりを 2 倍に拡大して読み直す（2026-10-26、実機の読めなかった 42 枚で 2 → 8 枚）。
    座標は画像全体で読んだときと同じになる。"""
    yaw_offset = 0.3
    robot = Pose2D(0.05, -0.04, 1.0)
    frame = _frame(m, cam, robot, yaw_offset)
    pose = ArucoRobotPose(cam, m, yaw_offset)
    full = pose.estimate(frame)
    real = pose._det
    h, w = frame.image.shape[:2]

    class FailOnFull:   # 画像全体では読めない（実機: 遠くて小さい・ぶれた）。切り出しなら読める
        def detectMarkers(self, img):
            return ((), None, ()) if img.shape[:2] == (h, w) else real.detectMarkers(img)
    pose._det = FailOnFull()
    got = pose.estimate(frame)
    assert pose.reread_hits == 1
    assert math.dist(got.xy, full.xy) < 0.003 and abs(got.yaw - full.yaw) < 0.03
