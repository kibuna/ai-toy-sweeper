"""走っている間に、送った走行の指令と読めた姿勢から、実機の前進・旋回の速さの倍率を割り出す（2026-10-30）。

実機の速さは電池で ±3 割変わる（10/30 の走りの記録: 減った電池で 旋回 0.65・前進 0.80、新しい電池で 1.25・1.30、昼は
1.00・1.10）。決めた倍率のままでは姿勢の先読み（runtime._predict_pose）がずれ、爪を閉じる位置がずれる。遅れと時定数は
config/field.yaml の real.motion_delay_s・motion_lag_s のまま（電池で変わらなかった: 70〜100 ms・60〜100 ms）にして、倍率だけを
読めた姿勢どうしの組から割り出す（scripts/fit_motion_from_telemetry.py と同じモデルで、倍率 1 で積んだ動きに対する実際の動きの
比の中央値）。組が min_pairs 未満の間は、設定の倍率を使う。指示をまたいで持ち越す（RealRig が 1 つ持つ）。

中央値にしたのは、10/30 の記録 3 つ（昼・電池が減った後・替えた後）で、走りながらの見積もりの 5〜95 % の幅を比べて、最小二乗
（窓 160 組、旋回 昼 0.78〜1.04・減った後 0.49〜0.89）より狭かった（0.91〜1.07・0.84〜0.95）から。物に引っかかって回れない間の組が
最小二乗を引っぱる。
"""
import bisect
import dataclasses
import math
import statistics
from collections import deque

from ..geometry import wrap_angle
from .runtime import MOTION_DT_S, MotionModel

# 読めた姿勢どうしの間がこれより長い組は使わない（マーカーが読めない間をまたぐ。fit_motion_from_telemetry と同じ）
MAX_PAIR_GAP_S = 0.3
# 組を使う、倍率 1 で積んだ動きの下限（1 組の間 約 0.1 s で、前進 14.5 mm・旋回 0.27 rad の半分くらい）。動き始め・止まり際の
# 小さい組は姿勢の誤差（2〜3 mm・0.04 rad）で比が大きく振れる。10/30 の記録 3 つで 2 mm・0.02 rad と比べ、旋回の見積もりの
# 5〜95 % の幅が狭かった（窓 160、中央値: 替えた後 0.65〜1.51 → 1.14〜1.51、減った後 0.39〜0.86 → 0.84〜0.95）
MIN_PRED_LINEAR_M = 0.008
MIN_PRED_ANGULAR_RAD = 0.15
# 1 組で動けるのは、指令の速さの 2 倍（倍率の上限の見込み）× 間 + 姿勢の誤差まで。超えたらマーカーの読み違い
MAX_GAIN_FOR_BOUND = 2.0
POSE_ERR_LINEAR_M = 0.005
POSE_ERR_ANGULAR_RAD = 0.05


class OnlineMotionGain:
    def __init__(self, base: MotionModel, *, window_pairs: int, min_pairs: int, command_ttl_s: float,
                 max_linear_mps: float, max_angular_radps: float):
        """base: 遅れ・時定数と、組が足りない間の倍率。window_pairs: 倍率を割り出す、新しい組の数（前進・旋回それぞれ）。"""
        if not (0 < min_pairs <= window_pairs):
            raise ValueError(f"0 < min_pairs {min_pairs} <= window_pairs {window_pairs}")
        if base.lag_s <= 0 or base.delay_s < 0:
            raise ValueError(f"時定数は正、遅れは 0 以上: {base}")
        self.base = base
        self.min_pairs = min_pairs
        self.ttl = command_ttl_s
        self.max_lin, self.max_ang = max_linear_mps, max_angular_radps
        self._times: list[float] = []
        self._cmds: list[tuple[float, float]] = []
        self._t: float | None = None           # 積んだ時刻（最後の姿勢の時刻）
        self._v = self._w = 0.0                # 倍率 1 での速さ（惰性）
        self._cum = (0.0, 0.0)                 # 倍率 1 で積んだ前進・旋回
        self._last: tuple | None = None        # (時刻, x, y, yaw, 積んだ前進, 積んだ旋回)
        self._pairs = {"linear": deque(maxlen=window_pairs), "angular": deque(maxlen=window_pairs)}
        self.rejected = 0                      # 1 組では動けない動き（読み違い）として捨てた組

    def add_send(self, t: float, linear: float, angular: float) -> None:
        if self._times and t < self._times[-1]:
            raise ValueError(f"指令の時刻が戻った: {t} < {self._times[-1]}")
        self._times.append(t)
        self._cmds.append((linear, angular))

    def _cmd(self, t: float) -> tuple[float, float]:
        i = bisect.bisect_right(self._times, t) - 1
        if i < 0:
            return 0.0, 0.0
        end = min(self._times[i + 1] if i + 1 < len(self._times) else math.inf, self._times[i] + self.ttl)
        return self._cmds[i] if t < end else (0.0, 0.0)

    def _advance(self, t1: float) -> None:
        """倍率 1 の速さを t1 まで積む（runtime._predict_pose と同じ刻み）。"""
        a = min(1.0, MOTION_DT_S / self.base.lag_s)
        t, (cl, ca) = self._t, self._cum
        while t < t1:
            dt = min(MOTION_DT_S, t1 - t)
            cv, cw = self._cmd(t - self.base.delay_s)
            self._v += (cv - self._v) * a
            self._w += (cw - self._w) * a
            cl, ca = cl + self._v * dt, ca + self._w * dt
            t += dt
        self._t, self._cum = t1, (cl, ca)
        # 積み終えた時刻より前の指令は要らない（効いている 1 つは残す）
        keep = bisect.bisect_right(self._times, t1 - self.base.delay_s - self.ttl) - 1
        if keep > 0:
            del self._times[:keep], self._cmds[:keep]

    def add_pose(self, stamp: float, x: float, y: float, yaw: float) -> None:
        """読めた姿勢（マーカーで測ったものだけ。見積もった姿勢は渡さない）。"""
        if self._t is None:
            self._t = stamp
        elif stamp <= self._t:
            raise ValueError(f"姿勢の時刻が戻った: {stamp} <= {self._t}")
        else:
            self._advance(stamp)
        cur = (stamp, x, y, yaw, *self._cum)
        prev, self._last = self._last, cur
        if prev is None or stamp - prev[0] > MAX_PAIR_GAP_S:
            return
        gap = stamp - prev[0]
        pl, pa = cur[4] - prev[4], cur[5] - prev[5]
        ml = math.cos(prev[3]) * (x - prev[1]) + math.sin(prev[3]) * (y - prev[2])
        ma = wrap_angle(yaw - prev[3])
        if (abs(ml) > MAX_GAIN_FOR_BOUND * self.max_lin * gap + POSE_ERR_LINEAR_M
                or abs(ma) > MAX_GAIN_FOR_BOUND * self.max_ang * gap + POSE_ERR_ANGULAR_RAD):
            self.rejected += 1
            return
        if abs(pl) >= MIN_PRED_LINEAR_M:
            self._pairs["linear"].append((pl, ml))
        if abs(pa) >= MIN_PRED_ANGULAR_RAD:
            self._pairs["angular"].append((pa, ma))

    def _gain(self, axis: str, default: float) -> float:
        pairs = self._pairs[axis]
        if len(pairs) < self.min_pairs:
            return default
        return statistics.median(m / p for p, m in pairs)

    def model(self) -> MotionModel:
        """今の動き方（遅れ・時定数は設定、倍率は割り出したもの）。"""
        return dataclasses.replace(self.base, gain_linear=self._gain("linear", self.base.gain_linear),
                                   gain_angular=self._gain("angular", self.base.gain_angular))

    def counts(self) -> tuple[int, int]:
        return len(self._pairs["linear"]), len(self._pairs["angular"])
