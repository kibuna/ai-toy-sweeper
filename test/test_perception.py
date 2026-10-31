import math

import numpy as np
import pytest

from toy_sweeper.config import load_all
from toy_sweeper.env import ToySweeperEnv
from toy_sweeper.geometry import Pose2D
from toy_sweeper.perception import Tracker, detect_objects, load_perception_cfg
from toy_sweeper.perception_env import PerceptionObsWrapper, robot_polygons, scene_homography
from toy_sweeper.render import draw_scene
from toy_sweeper.robot_model import gripper_center_x
from toy_sweeper.sim2d import ObjectSpec
from toy_sweeper.spaces import ObjectType


@pytest.fixture(scope="module")
def cfg():
    return load_all()


def test_detect_positions_and_types(cfg):
    pcfg = load_perception_cfg()
    objs = [ObjectSpec("a", ObjectType.LEGO, Pose2D(0.1, 0.2, 0.3), "black"),
            ObjectSpec("b", ObjectType.TOY_CAR, Pose2D(-0.2, -0.1, 1.2), "white"),
            ObjectSpec("c", ObjectType.LEGO, Pose2D(0.25, -0.25, -0.7), "yellow")]
    robot = Pose2D(-0.2, 0.2, 0.0)
    jaw = cfg.sim.gripper.jaw_open_angle_rad
    img, _ = draw_scene(cfg, robot, objs, {o.name: o.pose for o in objs}, jaw)
    dets = detect_objects(img, scene_homography(cfg), robot_polygons(cfg, robot, jaw), pcfg)
    assert len(dets) == 3
    for o in objs:
        d = min(dets, key=lambda d: math.dist(d.pose.xy, o.pose.xy))
        assert math.dist(d.pose.xy, o.pose.xy) < 0.003
        assert abs(math.remainder(d.pose.yaw - o.pose.yaw, math.pi)) < 0.1   # 180° の区別なし
        is_car = d.size_m[0] >= pcfg.toy_car_min_length_m
        assert is_car == (o.type == ObjectType.TOY_CAR)


def test_object_between_jaws_is_visible(cfg):
    # ロボットの部品だけを除くので、爪の間の物体は見える
    pcfg = load_perception_cfg()
    robot = Pose2D(0.0, 0.0, 0.0)
    o = ObjectSpec("a", ObjectType.LEGO, Pose2D(gripper_center_x(cfg.sim), 0.0, math.pi / 2), "red")
    jaw = cfg.sim.gripper.jaw_open_angle_rad
    img, _ = draw_scene(cfg, robot, [o], {"a": o.pose}, jaw)
    dets = detect_objects(img, scene_homography(cfg), robot_polygons(cfg, robot, jaw), pcfg)
    assert len(dets) == 1 and math.dist(dets[0].pose.xy, o.pose.xy) < 0.005


def test_tracker_keeps_ids_and_holds_missing(cfg):
    pcfg = load_perception_cfg()
    from toy_sweeper.perception import Detection
    tr = Tracker(pcfg, 0.0)
    d = lambda x, y: Detection(Pose2D(x, y, 0.0), (0.03, 0.016), (0, 0, 0), (0, 0, 0, 0))
    t0 = tr.update([d(0.0, 0.0), d(0.2, 0.0)])
    ids = {round(t.pose.x, 2): i for i, t in t0.items()}
    t1 = tr.update([d(0.21, 0.0), d(0.01, 0.0)])   # 順番が入れ替わっても同じ ID
    assert {round(t.pose.x, 2): i for i, t in t1.items()} == {0.01: ids[0.0], 0.21: ids[0.2]}
    for _ in range(pcfg.max_missed):
        t = tr.update([d(0.02, 0.0)])               # 片方が見えなくても max_missed までは保持
    assert len(t) == 2
    t = tr.update([d(0.02, 0.0)])
    assert len(t) == 1


def test_wrapper_runs_closed_loop(cfg):
    from toy_sweeper.policies import HeuristicPolicy
    env = PerceptionObsWrapper(ToySweeperEnv(cfg, stage=6), load_perception_cfg())
    pol = HeuristicPolicy(cfg)
    obs, _ = env.reset(seed=10000)
    for _ in range(20):
        obs, *_ , info = env.step(pol.predict(obs))
    assert np.nanmedian([m["target_err_m"] for m in env.metrics]) < 0.005


def test_tracker_chroma_gate():
    """色の門（2026-10-25、Gazebo のリハーサルの Stage 6 盤面 20008）: 前のフレームと色度が chroma_gate 以上違う検出には
    つながない（挟んだ緑の LEGO の追跡が、隣でくっついた黒・水色の LEGO の塊へ移り、黒い LEGO を対象として運んだ）。
    0 なら色を見ない（今まで）。明るさだけの変化（影）はつなぐ。"""
    from toy_sweeper.perception import Detection
    pcfg = load_perception_cfg()
    green, dark_green, gray_blue = (93, 174, 120), (62, 116, 80), (115, 140, 150)
    d = lambda x, rgb: Detection(Pose2D(x, 0.0, 0.0), (0.032, 0.016), rgb, (0, 0, 0, 0))
    for gate, follows in ((0.0, True), (14.0, False)):
        tr = Tracker(pcfg, gate)
        (tid,) = tr.update([d(0.0, green)])
        t = tr.update([d(0.01, gray_blue)])
        assert (t[tid].missed == 0 and t[tid].pose.x == 0.01) == follows
        assert len(t) == (1 if follows else 2)
    tr = Tracker(pcfg, 14.0)
    (tid,) = tr.update([d(0.0, green)])
    assert tr.update([d(0.01, dark_green)])[tid].missed == 0
