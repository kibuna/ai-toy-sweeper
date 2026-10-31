"""学習する検出器のデータ作り（toy_sweeper.seg_data）。"""
import math

import numpy as np
import pytest

from toy_sweeper.config import load_all
from toy_sweeper.geometry import Pose2D, box_corners
from toy_sweeper.perception_env import robot_polygons
from toy_sweeper.seg_data import (CONDITION_RANGES, convex_overlap, instances_from_labels, look_at_pose, sample_conditions,
                                  sample_layout)
from toy_sweeper.sim2d import ObjectSpec
from toy_sweeper.spaces import GripperState, ObjectType


def _forward(pose):
    """Gazebo のカメラの向き（自分の +x）を世界座標で。姿勢は roll-pitch-yaw（固定軸 x→y→z）。"""
    _, _, _, r, p, y = pose
    return np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)])


def test_look_at_pose():
    pose = look_at_pose((0.0, -0.85, 0.8), (0.0, 0.0, 0.0), 0.0)
    d = np.array([0.0, 0.85, -0.8])
    assert np.allclose(_forward(pose), d / np.linalg.norm(d))
    assert pose[:3] == [0.0, -0.85, 0.8] and abs(pose[5] - math.pi / 2) < 1e-9
    pose = look_at_pose((0.2, -0.7, 0.6), (-0.05, 0.1, 0.0), 0.02)
    d = np.array([-0.25, 0.8, -0.6])
    assert np.allclose(_forward(pose), d / np.linalg.norm(d)) and pose[3] == 0.02


def test_sample_conditions_in_range_and_reproducible():
    r = CONDITION_RANGES
    for seed in range(50):
        c = sample_conditions(np.random.default_rng(seed))
        g = c["sim"]["gazebo"]
        x, y, z = g["camera_pose"][:3]
        assert r["cam_x"][0] <= x <= r["cam_x"][1] and r["cam_y"][0] <= y <= r["cam_y"][1]
        assert r["cam_z"][0] <= z <= r["cam_z"][1] and r["hfov"][0] <= g["camera_hfov_rad"] <= r["hfov"][1]
        assert g["camera_image_px"] == [1920, 1080] and g["segmentation_camera"] is True
        assert r["sun_dir_z"][0] <= g["light_direction"][2] <= r["sun_dir_z"][1]
        assert all(0.0 <= v <= 1.0 for k in ("light_diffuse", "scene_ambient", "floor_rgb", "work_area_rgb") for v in g[k])
    assert sample_conditions(np.random.default_rng(3)) == sample_conditions(np.random.default_rng(3))


def test_convex_overlap():
    a = box_corners(Pose2D(0, 0, 0), (0.032, 0.016))
    assert convex_overlap(a, box_corners(Pose2D(0.03, 0, 0.3), (0.032, 0.016)))
    assert not convex_overlap(a, box_corners(Pose2D(0.033, 0, 0), (0.032, 0.016)))
    assert not convex_overlap(a, box_corners(Pose2D(0.0, 0.0165, 0), (0.032, 0.016)))


def _objects(n):
    kinds = [ObjectType.LEGO, ObjectType.TOY_CAR]
    return [ObjectSpec(f"obj{i}", kinds[i % 2], Pose2D(0, 0, 0), "red") for i in range(n)]


@pytest.mark.parametrize("seed", range(20))
def test_sample_layout_no_overlap(seed):
    cfg = load_all()
    lay = sample_layout(cfg, _objects(12), np.random.default_rng(seed))
    sizes = {o.name: cfg.sim.objects[o.type.name.lower()].size_m[:2] for o in lay.objects}
    boxes = {o.name: box_corners(o.pose, sizes[o.name]) for o in lay.objects}
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not convex_overlap(boxes[a], boxes[b]), (a, b)
    g = cfg.sim.gripper
    jaw = g.jaw_closed_angle_rad if lay.gripper == GripperState.CLOSED else g.jaw_open_angle_rad
    for n, box in boxes.items():
        if n == lay.held:
            continue
        assert not any(convex_overlap(box, p) for p in robot_polygons(cfg, lay.robot, jaw)), n
    assert 1 <= len(lay.objects) <= 12


def test_sample_layout_touching_and_held():
    """くっつきの割合 1 なら、どの物体も他の物体と 3 mm 以内（最初の 1 個を除く塊の一員）。爪の中の物体は爪の中心。"""
    cfg = load_all()
    from toy_sweeper.robot_model import gripper_center_x
    n_touch = n_held = 0
    for seed in range(30):
        lay = sample_layout(cfg, _objects(8), np.random.default_rng(seed), p_touch=1.0, p_held=1.0)
        if lay.held is not None:
            n_held += 1
            o = next(o for o in lay.objects if o.name == lay.held)
            gc = lay.robot.transform_point((gripper_center_x(cfg.sim), 0.0))
            assert math.dist(o.pose.xy, gc) < 1e-6
        n_touch += len(lay.touching)
        for a, b in lay.touching:
            assert a != b
    assert n_held >= 25 and n_touch >= 30


def test_instances_from_labels():
    lab = np.zeros((60, 80, 3), np.uint8)
    lab[10:20, 10:30] = (1, 0, 1)          # LEGO 1
    lab[30:50, 40:70] = (1, 0, 2)          # ミニカー 1（ロボットで 2 つに割れる: 大きい方だけ）
    lab[30:50, 52:54] = 0
    lab[5:7, 70:72] = (2, 0, 1)            # LEGO 2: 4 画素（min_pixels より少ない）
    inst = instances_from_labels(lab, min_pixels=10)
    assert sorted((c, n) for c, _, n in inst) == [(0, 200), (1, 16 * 20)]
    car = next(p for c, p, _ in inst if c == 1)
    xs, ys = car[:, 0], car[:, 1]
    assert xs.min() == 54 and xs.max() == 69 and ys.min() == 30 and ys.max() == 49   # 大きい方（54〜69 列）
