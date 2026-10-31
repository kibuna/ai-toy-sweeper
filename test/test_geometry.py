import math

import numpy as np
import pytest

from toy_sweeper.geometry import Pose2D, Rect, box_corners, wrap_angle


def test_wrap_angle():
    assert math.isclose(wrap_angle(3 * math.pi), math.pi) or math.isclose(wrap_angle(3 * math.pi), -math.pi)
    assert math.isclose(wrap_angle(0.5), 0.5)
    assert math.isclose(wrap_angle(-0.5 - 2 * math.pi), -0.5)


def test_rect_contains_and_corners():
    r = Rect(center=(0.3, 0.0), size=(0.2, 0.3))
    assert r.contains((0.3, 0.1))
    assert not r.contains((0.19, 0.0))
    c = box_corners(Pose2D(0.3, 0.0, 0.0), (0.04, 0.02))
    assert r.contains_all(c)
    c_out = box_corners(Pose2D(0.21, 0.0, 0.0), (0.04, 0.02))
    assert not r.contains_all(c_out)


def test_box_corners_rotated():
    c = box_corners(Pose2D(0.0, 0.0, math.pi / 2), (0.04, 0.02))
    xs, ys = np.array(c)[:, 0], np.array(c)[:, 1]
    assert np.allclose(sorted(set(np.round(np.abs(xs), 6))), [0.01])
    assert np.allclose(sorted(set(np.round(np.abs(ys), 6))), [0.02])


def test_pose_transform():
    p = Pose2D(1.0, 2.0, math.pi / 2)
    assert np.allclose(p.transform_point((1.0, 0.0)), (1.0, 3.0))


def test_signed_distance_to_rect():
    from toy_sweeper.geometry import signed_distance_to_rect
    r = Rect(center=(0.0, 0.0), size=(0.2, 0.2))
    assert signed_distance_to_rect((0.0, 0.0), r) == pytest.approx(-0.1)
    assert signed_distance_to_rect((0.15, 0.0), r) == pytest.approx(0.05)
    assert signed_distance_to_rect((0.13, 0.14), r) == pytest.approx(math.hypot(0.03, 0.04))
    assert signed_distance_to_rect((0.09, 0.0), r) == pytest.approx(-0.01)
