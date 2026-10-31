"""動きの計測の解析（doc/real_robot_runbook.md）。scripts/measure_motion.py が
カメラで記録した姿勢の列（Sample）から、速さ・動き出しの遅れ・止まるまでの距離・旋回の速さを出す。
時刻はすべて time.monotonic の秒。フレームの時刻は撮影時刻の見積もり（Frame.stamp、config/field.yaml の capture_latency_s を
引いたもの。capture_latency_s は未測定なので、遅れの値はその分ずれうる）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 動き出したとみなす移動。止まったマーカーのばらつき（標準偏差 0.1 mm、2026-10-22 の実機）の 20 倍
MOVE_THRESHOLD_M = 0.002
# 動き出してから速さが一定になるまで除く時間。速さの当てはめに使わない（未検証の目安。1 回 1.0 s の走りで 0.5 s 分残る）
SETTLE_S = 0.2
# 走りの前後の止まった姿勢として使う枚数（指令の前 PRE_S 0.5 s = 10 fps で 5 枚。measure_motion.py）
EDGE_SAMPLES = 3
# 速さの当てはめに要るフレームの数の下限。止まったマーカーのばらつき 0.1 mm なら、10 fps の 3 枚（0.2 s）で速さの誤差は
# 約 0.7 mm/s（0.15 m/s の 0.5%）。5 枚にしていたら、動き出しが 0.33 s 遅れた回（指令 20、1.0 s）で足りなかった（2026-10-22）
MIN_FIT_SAMPLES = 3


class NoMotionError(ValueError):
    """指令を送っても動かなかった（計測の結果。指令が小さすぎるなど）。"""


@dataclass(frozen=True)
class Sample:
    t: float            # 撮影時刻
    x: float            # 車軸の中点（床の座標）
    y: float
    yaw: float          # ロボットの前の向き
    marker_yaw: float   # マーカーの x 軸の床の向き
    side_m: float       # マーカーの黒い四角の 1 辺（マーカーの高さの平面で）。印刷 50 mm と違えば、傾き・高さの誤り


def _arrays(samples: list[Sample]):
    if len(samples) < 2:
        raise ValueError(f"姿勢が {len(samples)} 個しかない")
    t = np.array([s.t for s in samples])
    xy = np.array([(s.x, s.y) for s in samples])
    return t, xy


def _fit_slope(t: np.ndarray, v: np.ndarray, t0: float, t1: float, what: str) -> float:
    k = (t >= t0) & (t <= t1)
    if k.sum() < MIN_FIT_SAMPLES:
        raise ValueError(f"{what}: {t0:.2f}〜{t1:.2f} s のフレームが {int(k.sum())} 枚（{MIN_FIT_SAMPLES} 枚以上要る）。走る時間を延ばす")
    return float(np.polyfit(t[k], v[k], 1)[0])


def analyze_straight(samples: list[Sample], t_cmd: float, t_stop: float, direction: int) -> dict:
    """前進（direction 1）・後退（−1）の 1 回。t_cmd: 最初の M を送った時刻、t_stop: S を送った時刻。
    進む向きは、指令の前の姿勢の向き（後退は逆）。stop_distance_m は S を送ってから止まるまでの距離（S が効くまでの遅れの
    間に進む分を含む）。"""
    t, xy = _arrays(samples)
    before = t < t_cmd
    if not before.any():
        raise ValueError("指令の前の姿勢がない（記録を先に始める）")
    p0 = xy[before].mean(axis=0)
    yaw0 = math.atan2(np.mean([math.sin(s.yaw) for s, b in zip(samples, before) if b]),
                      np.mean([math.cos(s.yaw) for s, b in zip(samples, before) if b]))
    u = direction * np.array([math.cos(yaw0), math.sin(yaw0)])
    along = (xy - p0) @ u
    moved = np.nonzero((t >= t_cmd) & (along > MOVE_THRESHOLD_M))[0]
    if len(moved) == 0:
        raise NoMotionError(f"動かなかった（{MOVE_THRESHOLD_M * 1000:.0f} mm 以上進んだフレームがない。"
                            f"最大 {float(np.max(np.abs(along))) * 1000:.1f} mm）")
    t_move = float(t[moved[0]])
    speed = _fit_slope(t, along, t_move + SETTLE_S, t_stop, "速さ")
    yaw = np.unwrap([s.yaw for s in samples])
    yaw_rate = _fit_slope(t, yaw, t_move + SETTLE_S, t_stop, "向きの変化")
    after = t >= t_stop
    if after.sum() < 2:
        raise ValueError("S の後の姿勢が足りない（止まるまで記録を続ける）")
    along_at_stop = float(np.interp(t_stop, t, along))
    return {"speed_mps": speed, "start_delay_s": t_move - t_cmd, "stop_distance_m": float(along[-1]) - along_at_stop,
            "yaw_rate_radps": yaw_rate, "distance_m": float(along[-1]),
            "lateral_m": float((xy[-1] - p0) @ np.array([-u[1], u[0]]))}


def marker_yaw_offset(samples: list[Sample], direction: int) -> float:
    """まっすぐ走った 1 回から、マーカーの x 軸の向き − ロボットの前の向き（rad）。走りの前後の止まった姿勢（最初と最後の
    EDGE_SAMPLES 枚）を使う。進んだ向き（前の位置 → 後の位置）を、前後のマーカーの向きの平均と比べる。右に曲がりながら進んでも、
    向きの変わり方が一定なら、弦の向きは弧の真ん中の向き（前後の向きの平均）と同じなので偏らない（2026-10-22、全体の平均と
    比べていたら、前の 0.5 s と後の 1.5 s の差で前進と後退が 1〜2° ずれた）。後退は進んだ向きを 180° 回す。"""
    if len(samples) < 2 * EDGE_SAMPLES:
        raise ValueError(f"姿勢が {len(samples)} 個（前後に {EDGE_SAMPLES} 枚ずつ要る）")
    _, xy = _arrays(samples)
    d = xy[-EDGE_SAMPLES:].mean(axis=0) - xy[:EDGE_SAMPLES].mean(axis=0)
    if np.linalg.norm(d) < 10 * MOVE_THRESHOLD_M:
        raise NoMotionError(f"進んだ距離 {np.linalg.norm(d) * 1000:.1f} mm が短すぎる（向きが決まらない）")
    heading = math.atan2(d[1], d[0]) + (0.0 if direction > 0 else math.pi)
    my = np.array([s.marker_yaw for s in samples])

    def circ(a):
        return math.atan2(np.sin(a).mean(), np.cos(a).mean())
    a0, a1 = circ(my[:EDGE_SAMPLES]), circ(my[-EDGE_SAMPLES:])
    mid = a0 + math.remainder(a1 - a0, math.tau) / 2
    return math.remainder(mid - heading, math.tau)


def analyze_spin(samples: list[Sample], t_cmd: float, t_stop: float) -> dict:
    """その場の旋回の 1 回。旋回の速さ（rad/s、左回りが正）と、車軸の中点の動き（取り付け位置が正しければ小さい）。"""
    t, xy = _arrays(samples)
    yaw = np.unwrap([s.yaw for s in samples])
    before = t < t_cmd
    if not before.any():
        raise ValueError("指令の前の姿勢がない（記録を先に始める）")
    turned = np.nonzero((t >= t_cmd) & (np.abs(yaw - yaw[before].mean()) > math.radians(1.0)))[0]
    if len(turned) == 0:
        raise NoMotionError("回らなかった（1° 以上向きが変わったフレームがない）")
    t_move = float(t[turned[0]])
    rate = _fit_slope(t, yaw, t_move + SETTLE_S, t_stop, "旋回の速さ")
    during = (t >= t_cmd) & (t <= t_stop)
    wander = float(np.max(np.linalg.norm(xy[during] - xy[during].mean(axis=0), axis=1)))
    return {"yaw_rate_radps": rate, "start_delay_s": t_move - t_cmd, "axle_wander_m": wander,
            "turned_rad": float(yaw[-1] - yaw[before].mean())}
