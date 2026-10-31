"""Gazebo（ROS 2 経由）の結合テスト。pixi 環境で実行する:

    ~/.pixi/bin/pixi run pytest test/test_gazebo.py

uv 環境には rclpy がないので skip になる（理由は表示される）。
"""
import math
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy", reason="Gazebo テストは pixi 環境で実行する")

from toy_sweeper.config import load_all  # noqa: E402
from toy_sweeper.geometry import Pose2D  # noqa: E402
from toy_sweeper.robot_model import gripper_center_x  # noqa: E402
from toy_sweeper.sim2d import ObjectSpec  # noqa: E402
from toy_sweeper.spaces import GripperState, ObjectType  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def backend():
    from toy_sweeper.gazebo.backend import GazeboBackend
    from toy_sweeper.gazebo.launcher import running_simulation
    with running_simulation():
        b = GazeboBackend(load_all())
        yield b
        b.close()


@pytest.fixture(scope="module")
def cfg():
    return load_all()


def test_forward_speed(backend, cfg):
    backend.reset(Pose2D(0.0, 0.0, 0.0), [], GripperState.OPEN)
    p0 = backend.robot_pose()
    assert abs(p0.x) < 0.005 and abs(p0.y) < 0.005
    backend.drive(cfg.sim.robot.linear_speed_mps, 0.0, 1.0)
    p = backend.robot_pose()
    # 指令 0.08 m/s × 1 s。加速と接地の誤差で 15% までは許す
    assert p.x == pytest.approx(cfg.sim.robot.linear_speed_mps, rel=0.15), p
    assert abs(p.y) < 0.01 and abs(p.yaw) < 0.1


def test_turn_in_place(backend, cfg):
    backend.reset(Pose2D(0.0, 0.0, 0.0), [], GripperState.OPEN)
    backend.drive(0.0, cfg.sim.robot.angular_speed_radps, 1.0)
    p = backend.robot_pose()
    assert p.yaw == pytest.approx(cfg.sim.robot.angular_speed_radps, rel=0.15), p
    assert math.hypot(p.x, p.y) < 0.01


def test_push_lego(backend, cfg):
    start = Pose2D(gripper_center_x(cfg.sim), 0.0, math.pi / 2)
    backend.reset(Pose2D(0.0, 0.0, 0.0), [ObjectSpec("obj0", ObjectType.LEGO, start, "red")], GripperState.OPEN)
    backend.drive(cfg.sim.robot.linear_speed_mps, 0.0, 2.0)
    o = backend.object_pose("obj0")
    assert o.x > start.x + 0.08, o
    assert abs(o.y) < 0.03
    backend.drive(0.0, 0.0, 0.5)
    before = backend.object_pose("obj0")
    backend.drive(0.0, 0.0, 0.5)
    after = backend.object_pose("obj0")
    assert math.hypot(after.x - before.x, after.y - before.y) < 0.002   # 止まると物体も止まる


def test_untouched_lego_stays(backend):
    backend.reset(Pose2D(0.0, 0.0, 0.0), [ObjectSpec("obj0", ObjectType.LEGO, Pose2D(0.2, 0.2, 0.3), "red"),
                                          ObjectSpec("obj3", ObjectType.LEGO, Pose2D(-0.2, 0.2, 0.0), "blue")],
                  GripperState.OPEN)
    backend.drive(0.0, 0.0, 1.0)
    o = backend.object_pose("obj0")
    assert math.hypot(o.x - 0.2, o.y - 0.2) < 0.002 and abs(o.yaw - 0.3) < 0.02



def test_grasp_and_carry(backend, cfg):
    # 爪の間の LEGO を閉じて挟み、後退しても付いてくる（Gazebo では摩擦で保持する）
    x0 = gripper_center_x(cfg.sim)
    backend.reset(Pose2D(0.0, 0.0, 0.0), [ObjectSpec("obj0", ObjectType.LEGO, Pose2D(x0, 0.0, 0.0), "red")],
                  GripperState.OPEN)
    backend.actuate_gripper(close=True)
    assert backend.gripper_state() == GripperState.CLOSED
    backend.drive(-cfg.sim.robot.linear_speed_mps, 0.0, 1.0)
    o = backend.object_pose("obj0")
    r = backend.robot_pose()
    assert math.hypot(o.x - (r.x + x0), o.y - r.y) < 0.02, (o, r)


def test_toy_car_rolls_along_slides_across(backend, cfg):
    # 4 輪のミニカー: 前後に押して止まると転がり続け、横に押して止まるとすぐ止まる。
    # 開いた爪は V 字でミニカー（幅 30 mm）を挟み込んでしまうので、閉じた爪の先端で押す
    from toy_sweeper.robot_model import front_x
    tip = front_x(cfg.sim) + cfg.sim.gripper.pivot_x_from_front_m + cfg.sim.gripper.jaw_length_m
    sx, sy, _ = cfg.sim.objects["toy_car"].size_m
    coast = {}
    for name, yaw, half in (("along", 0.0, sx / 2), ("across", math.pi / 2, sy / 2)):
        car = ObjectSpec("obj0", ObjectType.TOY_CAR, Pose2D(tip + half + 0.005, 0.0, yaw), "blue")
        backend.reset(Pose2D(0.0, 0.0, 0.0), [car], GripperState.CLOSED)
        backend.drive(cfg.sim.robot.linear_speed_mps, 0.0, 1.0)
        a = backend.object_pose("obj0")
        backend.drive(-cfg.sim.robot.linear_speed_mps, 0.0, 0.5)   # ロボットは下がって離れる
        b = backend.object_pose("obj0")
        coast[name] = b.x - a.x
    assert coast["along"] > 3 * max(coast["across"], 0.001), coast



def test_wall_stops_robot(backend, cfg):
    half = cfg.sim.work_area.size_m[0] / 2
    backend.reset(Pose2D(half - 0.15, 0.0, 0.0), [], GripperState.OPEN)
    backend.drive(cfg.sim.robot.linear_speed_mps, 0.0, 4.0)   # 0.32 m 分の前進指令
    r = backend.robot_pose()
    assert r.x < half - 0.05, r   # 車軸中点は爪の長さぶん壁の手前で止まる
