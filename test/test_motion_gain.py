"""走りながら速さの倍率を割り出す（field.motion_gain.OnlineMotionGain）の試験。"""
import math
import random

import pytest

from toy_sweeper.field.motion_gain import OnlineMotionGain
from toy_sweeper.field.runtime import MotionModel

BASE = MotionModel(0.075, 0.1, 1.10, 1.00)
V, W = 0.145, 2.75
TTL = 0.5


def _drive(true_lin: float, true_ang: float, seconds: float, seed: int = 0, yaw_flip_at: float | None = None):
    """8 Hz の指令で走る実機（遅れ・時定数は BASE、倍率は true_*）を 1 ms で積み、10 fps の姿勢と送った指令を時刻順に返す。"""
    rnd = random.Random(seed)
    acts = [(V, 0.0), (-V, 0.0), (0.0, W), (0.0, -W), (0.0, 0.0)]
    sends = [(k * 0.125, *rnd.choice(acts)) for k in range(int(seconds / 0.125))]
    dt, t, v, w, x, y, yaw = 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    a = dt / BASE.lag_s
    events, i, next_pose = [], 0, 0.05
    while t < seconds:
        while i < len(sends) and sends[i][0] <= t:
            events.append(("send", sends[i]))
            i += 1
        k = max((j for j in range(len(sends)) if sends[j][0] <= t - BASE.delay_s), default=None)
        cv, cw = (0.0, 0.0) if k is None else sends[k][1:]
        v += (true_lin * cv - v) * a
        w += (true_ang * cw - w) * a
        x, y, yaw = x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw + w * dt
        t += dt
        if t >= next_pose:
            flip = yaw_flip_at is not None and abs(t - yaw_flip_at) < 0.05
            events.append(("pose", (t, x + rnd.gauss(0, 0.001), y + rnd.gauss(0, 0.001),
                                    yaw + (math.pi if flip else rnd.gauss(0, 0.01)))))
            next_pose += 0.1
    return events


def _feed(est: OnlineMotionGain, events) -> None:
    for kind, e in events:
        if kind == "send":
            est.add_send(*e)
        else:
            est.add_pose(*e)


def _est(**kw):
    return OnlineMotionGain(BASE, window_pairs=160, min_pairs=20, command_ttl_s=TTL, max_linear_mps=V,
                            max_angular_radps=W, **kw)


@pytest.mark.parametrize("true_lin,true_ang", [(1.30, 1.25), (0.80, 0.65)])
def test_finds_gains_from_driving(true_lin, true_ang):
    """電池で速さが変わった実機（10/30 に測った 1.30・1.25 と 0.80・0.65）の倍率を、走りながら割り出す。"""
    est = _est()
    _feed(est, _drive(true_lin, true_ang, 20.0))
    m = est.model()
    assert m.gain_linear == pytest.approx(true_lin, abs=0.05)
    assert m.gain_angular == pytest.approx(true_ang, abs=0.05)
    assert (m.delay_s, m.lag_s) == (BASE.delay_s, BASE.lag_s)


def test_uses_configured_gains_until_enough_pairs():
    est = _est()
    _feed(est, _drive(1.3, 1.25, 1.0))
    assert est.model() == BASE


def test_ignores_marker_misread():
    """マーカーの向きの読み違い（180° 跳ぶ）の組は、1 歩では動けないので使わない。"""
    est = _est()
    _feed(est, _drive(1.0, 1.0, 20.0, yaw_flip_at=10.0))
    assert est.model().gain_angular == pytest.approx(1.0, abs=0.05)
    assert est.rejected > 0


def test_keeps_gains_across_idle_gap():
    """指示と指示の間（指令も姿勢も来ない）をまたいでも、割り出した倍率を持ち越す。"""
    est = _est()
    _feed(est, _drive(1.3, 1.25, 20.0))
    before = est.model()
    est.add_pose(100.0, 0.0, 0.0, 0.0)
    est.add_pose(100.1, 0.0, 0.0, 0.0)
    assert est.model() == before


def test_rejects_bad_settings():
    with pytest.raises(ValueError):
        OnlineMotionGain(BASE, window_pairs=10, min_pairs=20, command_ttl_s=TTL, max_linear_mps=V, max_angular_radps=W)
    with pytest.raises(ValueError):
        OnlineMotionGain(MotionModel(0.075, 0.0, 1.0, 1.0), window_pairs=80, min_pairs=20, command_ttl_s=TTL,
                         max_linear_mps=V, max_angular_radps=W)
