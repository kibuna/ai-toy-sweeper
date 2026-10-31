"""scripts/fit_motion_from_telemetry.py（実機の走りの記録から、指令に対する遅れ・惰性・倍率を割り出す。2026-10-30）を、
答えの分かっている合成の記録で試す。"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "fit_motion", Path(__file__).resolve().parents[1] / "scripts/fit_motion_from_telemetry.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)


def _synthetic(tmp_path, d, tau, k, seed=0):
    """8 Hz で左右の旋回・前進・止まるを乱数で送り、10 fps で姿勢を撮った記録（d・τ・k の真値で動かす）。"""
    rng = np.random.default_rng(seed)
    t = 1000.0
    sends = []
    for i in range(240):
        a = rng.choice(["L", "R", "F", "S"])
        sends.append({"kind": "send", "t": t + 0.125 * i, "seq": i, "linear": 0.145 if a == "F" else 0.0,
                      "angular": {"L": 2.75, "R": -2.75}.get(a, 0.0)})
    times = np.array([s["t"] for s in sends])
    t0, t1 = times[0] - 1.0, times[-1] + 0.5
    ts, yaw = fm.simulate(times, np.array([s["angular"] for s in sends]), t0, t1, d, tau, k)
    _, dist = fm.simulate(times, np.array([s["linear"] for s in sends]), t0, t1, d, tau, k)
    x = y = 0.0
    poses = []
    prev = 0
    for st in np.arange(times[0] + 0.05, times[-1], 0.1):
        i = int(np.searchsorted(ts, st))
        # 前進は向きに沿って積む（細かい刻みで）
        for j in range(prev, i):
            step = dist[j + 1] - dist[j]
            x += step * math.cos(yaw[j]); y += step * math.sin(yaw[j])
        prev = i
        poses.append({"kind": "pose", "stamp": float(st), "seq": len(poses), "x": x, "y": y, "yaw": math.atan2(math.sin(yaw[i]), math.cos(yaw[i])),
                      "measured": True})
    (tmp_path / "telemetry.jsonl").write_text("".join(json.dumps(e) + "\n" for e in sends + poses))
    return tmp_path


def test_fit_recovers_delay_lag_and_gain(tmp_path):
    d0 = _synthetic(tmp_path, d=0.15, tau=0.06, k=0.9)
    segs = [s for sends, poses in fm.load([d0]) for s in fm.segments(sends, poses, 1.0)]
    best = min(((fm.residual(segs, "turn", d, tau, k)[0], d, tau, k)
                for d in (0.0, 0.1, 0.15, 0.2) for tau in (0.005, 0.06, 0.15) for k in (0.8, 0.9, 1.0)))
    assert best[1:] == (0.15, 0.06, 0.9)


def test_load_rejects_empty_record(tmp_path):
    (tmp_path / "telemetry.jsonl").write_text(json.dumps({"kind": "pose", "stamp": 0, "x": 0, "y": 0, "yaw": 0,
                                                           "measured": True}) + "\n")
    with pytest.raises(ValueError, match="割り出せない"):
        fm.load([tmp_path])
