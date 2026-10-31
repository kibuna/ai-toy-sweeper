"""Rust 版（native/）の幾何が Python 版と同じ値（ビット単位）になる。"""
import math

import numpy as np
import pytest

import toy_sweeper_native as nat
from toy_sweeper import robot_model as rm
from toy_sweeper.geometry import wrap_angle
from toy_sweeper.config import load_all
from toy_sweeper.native import cfg_json


@pytest.fixture
def cfg():
    return load_all()


def _values():
    r = np.random.default_rng(0)
    v = list(r.normal(size=2000) * 10.0 ** r.integers(-12, 4, size=2000))
    return v + [0.0, -0.0, 1.0, math.pi, -math.pi, 1e-310, 5e-324, 1e300]


def test_hypot_matches_cpython():
    v = _values()
    for a, b in zip(v, reversed(v)):
        assert nat._hypot(a, b) == math.hypot(a, b), (a, b)


def test_wrap_and_remainder_match_python():
    for a in _values() + [3 * math.pi, -3 * math.pi, 2 * math.pi]:
        assert nat._wrap_angle(a) == wrap_angle(a), a
        assert nat._remainder(a, 2 * math.pi) == math.remainder(a, 2 * math.pi), a


def test_unknown_config_field_is_rejected(cfg):
    import json
    d = json.loads(cfg_json(cfg))
    d["task"]["reward"]["new_term"] = 1.0
    with pytest.raises(ValueError, match="unknown field"):
        nat._robot_model(json.dumps(d), 0.3, 0.05, 0.007)


@pytest.mark.parametrize("angle", [0.0, 0.2, 0.716])
def test_robot_model_matches(cfg, angle):
    s = cfg.sim
    for x in (0.03, 0.05, 0.06, 0.08):
        for half in (0.005, 0.0079, 0.015):
            got = nat._robot_model(cfg_json(cfg), angle, x, half)
            want = [rm.body_center_x(s), rm.front_x(s), rm.gripper_center_x(s), rm.robot_radius(s),
                    rm.jaw_inner_half_gap(s, angle, x), rm.angle_touching(s, x, half)]
            want += [c for side in rm.SIDES for poly in rm.jaw_polygons(s, side, angle) for p in poly for c in p]
            assert got == want
