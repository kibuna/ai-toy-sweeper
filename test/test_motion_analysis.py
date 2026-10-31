"""動きの計測の解析（toy_sweeper.field.motion_analysis）。速さ・遅れ・止まるまでの距離・旋回の速さが分かっている
合成の軌跡（0.1 mm のノイズ、10 fps）から求め直せるかを確かめる。"""
import math

import numpy as np
import pytest

from toy_sweeper.field.motion_analysis import Sample, analyze_spin, analyze_straight, marker_yaw_offset

FPS = 10.0
NOISE_M = 0.0001   # 実機の止まったマーカーのばらつき（0.1 mm、2026-10-22）


def _straight(v, delay, stop_dist, heading, direction=1, t_cmd=1.0, t_stop=3.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    t_move, stop_time = t_cmd + delay, stop_dist / v * 2   # 止まるまでは一定の減速
    for k in range(int(5.0 * FPS)):
        t = k / FPS
        if t < t_move:
            s = 0.0
        elif t < t_stop + delay:
            s = v * (t - t_move)
        else:
            u = min(t - t_stop - delay, stop_time)
            s = v * (t_stop - t_cmd) + v * u - v / (2 * stop_time) * u * u
        s *= direction
        x, y = s * math.cos(heading) + rng.normal(0, NOISE_M), s * math.sin(heading) + rng.normal(0, NOISE_M)
        out.append(Sample(t, x, y, heading, heading + 0.7, 0.05))
    return out


@pytest.mark.parametrize("direction", [1, -1])
def test_straight(direction):
    r = analyze_straight(_straight(0.09, 0.15, 0.01, 0.4, direction), t_cmd=1.0, t_stop=3.0, direction=direction)
    assert abs(r["speed_mps"] - 0.09) < 0.002
    assert abs(r["start_delay_s"] - 0.15) < 0.1 + 1e-9          # 10 fps なので 1 枚分まで
    # S を送ってから止まるまで: 遅れの間は同じ速さで進み（0.09 × 0.15）、その後 10 mm で止まる
    assert abs(r["stop_distance_m"] - (0.09 * 0.15 + 0.01)) < 0.002
    assert abs(r["yaw_rate_radps"]) < 0.01


def test_straight_requires_motion():
    with pytest.raises(ValueError, match="動かなかった"):
        analyze_straight(_straight(1e-6, 0.1, 1e-7, 0.0), t_cmd=1.0, t_stop=3.0, direction=1)


def test_marker_yaw_offset():
    """マーカーの x 軸の向き − 進んだ向き。後ろ向きの走りは進んだ向きを 180° 回して数える。"""
    fwd = _straight(0.09, 0.1, 0.01, 0.4, 1)
    back = _straight(0.09, 0.1, 0.01, 0.4, -1)
    for samples, d in ((fwd, 1), (back, -1)):
        assert abs(math.remainder(marker_yaw_offset(samples, d) - 0.7, math.tau)) < math.radians(0.5)


def test_spin():
    rng = np.random.default_rng(1)
    w, delay, t_cmd, t_stop = 1.2, 0.2, 1.0, 3.0
    samples = []
    for k in range(int(5.0 * FPS)):
        t = k / FPS
        yaw = 0.3 + w * min(max(t - t_cmd - delay, 0.0), t_stop - t_cmd)
        samples.append(Sample(t, 0.1 + rng.normal(0, NOISE_M), -0.2 + rng.normal(0, NOISE_M),
                              math.remainder(yaw, math.tau), 0.0, 0.05))
    r = analyze_spin(samples, t_cmd=t_cmd, t_stop=t_stop)
    assert abs(r["yaw_rate_radps"] - w) < 0.02
    assert r["axle_wander_m"] < 0.001


def test_marker_yaw_offset_on_curved_path():
    """右に曲がりながら進んでも（実機: 前進で −0.2 rad/s）、前進と後退で同じ値になる。
    止まっている時間が前（0.5 s）と後（1.5 s）で違っても、前後の向きの平均（弧の真ん中の向き）と比べるので偏らない。"""
    out = []
    for direction in (1, -1):
        samples, x, y, yaw = [], 0.0, 0.0, 0.3
        for k in range(32):
            t = k / FPS
            moving = 0.5 <= t < 1.3
            if moving:
                v, w = 0.15 * direction, -0.2 * direction
                mid = yaw + w / (2 * FPS)   # 1 枚の間の真ん中の向きで進める（一定の曲がり方の弧）
                x += v * math.cos(mid) / FPS
                y += v * math.sin(mid) / FPS
                yaw += w / FPS
            samples.append(Sample(t, x, y, yaw, yaw + 0.7, 0.05))
        out.append(marker_yaw_offset(samples, direction))
    for o in out:
        assert abs(math.remainder(o - 0.7, math.tau)) < math.radians(0.3)
