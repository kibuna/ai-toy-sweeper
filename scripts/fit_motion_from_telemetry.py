"""実機の走りの記録（dev_log/field_app/<日時>/telemetry.jsonl）から、指令に対するロボットの動き方を割り出す（2026-10-30）。

    uv run python scripts/fit_motion_from_telemetry.py dev_log/field_app/<日時> [ほかの記録 ...]

モデル（旋回と前進のそれぞれ）: 送った指令の速さ u(t)（送った時刻から有効、次の指令まで続く）が、遅れ d の後に倍率 k で
効き始め、実際の速さ v は時定数 τ で追いつく（dv/dt = (k·u(t − d) − v) / τ。止めた後も τ のあいだ惰性で動く）。
姿勢は撮った時刻（frame.stamp、アプリに届いた時刻）で比べるので、d にはカメラの中の遅れも入る。ランタイムの先読み
（runtime.to_now）が使う時刻と同じ基準なので、先読みとシミュレーションに入れる遅れはこの d でよい。

d・τ・k を格子で総当たりし、読めた姿勢どうしの間の向きの変化（旋回）・進んだ距離（前進）の二乗誤差が最小のものを出す。
出力は標準出力の表だけ（値を採用するときは、設定のコメントに根拠つきで書く）。
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

DT = 0.005   # 速さを積分する刻み（5 ms。遅れの格子 10 ms より細かく）


def load(dirs: list[Path]) -> list[tuple[list[dict], list[dict]]]:
    """記録ごとに（送った走行の指令、読めた姿勢）。走行の指令が無い・姿勢が読めていない記録は ValueError。"""
    out = []
    for d in dirs:
        path = d / "telemetry.jsonl"
        sends, poses = [], []
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e["kind"] == "send" and "linear" in e:
                sends.append(e)
            elif e["kind"] == "pose" and e["measured"]:
                poses.append(e)
        if not sends or len(poses) < 10:
            raise ValueError(f"{path}: 走行の指令 {len(sends)} 件・読めた姿勢 {len(poses)} 件では割り出せない")
        out.append((sends, poses))
    return out


def segments(sends: list[dict], poses: list[dict], gap_s: float) -> list[tuple[list[dict], list[dict]]]:
    """指令が gap_s 以上途切れた所（指示と指示の間）で分ける。姿勢は、その区間の指令の間に撮ったもの。"""
    segs, cur = [], [sends[0]]
    for a, b in zip(sends, sends[1:]):
        if b["t"] - a["t"] > gap_s:
            segs.append(cur)
            cur = []
        cur.append(b)
    segs.append(cur)
    out = []
    for s in segs:
        ps = [p for p in poses if s[0]["t"] <= p["stamp"] <= s[-1]["t"] + gap_s]
        if len(ps) >= 10:
            out.append((s, ps))
    return out


def simulate(times: np.ndarray, u: np.ndarray, t0: float, t1: float, d: float, tau: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """指令（時刻 times に値 u、次の指令まで続く）から、[t0, t1] の実際の速さを積分した位置（累積）を返す（刻みの時刻、累積）。"""
    ts = np.arange(t0, t1, DT)
    idx = np.searchsorted(times, ts - d, side="right") - 1
    cmd = np.where(idx >= 0, u[np.clip(idx, 0, None)], 0.0) * k
    a = min(1.0, DT / max(tau, DT))
    v = lfilter([a], [1.0, -(1.0 - a)], cmd)     # v[i] = v[i−1] + a (cmd[i] − v[i−1])
    return ts, np.concatenate([[0.0], np.cumsum(v[:-1]) * DT])


def residual(segs, which: str, d: float, tau: float, k: float) -> tuple[float, int]:
    """読めた姿勢どうしの間の、向きの変化（turn）または前に進んだ距離（forward）の二乗誤差の和と個数。"""
    err, n = 0.0, 0
    for sends, poses in segs:
        times = np.array([s["t"] for s in sends])
        u = np.array([s["angular"] if which == "turn" else s["linear"] for s in sends])
        t0, t1 = poses[0]["stamp"] - 1.0, poses[-1]["stamp"] + DT
        ts, cum = simulate(times, u, t0, t1, d, tau, k)
        at = lambda t: float(np.interp(t, ts, cum))
        for a, b in zip(poses, poses[1:]):
            if b["stamp"] - a["stamp"] > 0.3:     # 読めない間をまたぐ組は使わない
                continue
            if which == "turn":
                meas = math.atan2(math.sin(b["yaw"] - a["yaw"]), math.cos(b["yaw"] - a["yaw"]))
            else:
                meas = math.cos(a["yaw"]) * (b["x"] - a["x"]) + math.sin(a["yaw"]) * (b["y"] - a["y"])
            pred = at(b["stamp"]) - at(a["stamp"])
            err += (meas - pred) ** 2
            n += 1
    return err, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="dev_log/field_app/<日時>（telemetry.jsonl のある所）")
    ap.add_argument("--gap", type=float, default=1.0, help="これ以上指令が途切れたら別の区間にする（s）")
    a = ap.parse_args()
    segs = [s for sends, poses in load(a.dirs) for s in segments(sends, poses, a.gap)]
    print(f"区間 {len(segs)}、読めた姿勢 {sum(len(p) for _, p in segs)} 件")
    ds = np.arange(0.0, 0.41, 0.01)
    taus = np.array([0.005, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3])
    ks = np.arange(0.6, 1.45, 0.05)
    for which in ("turn", "forward"):
        best = None
        base, n = residual(segs, which, 0.0, DT, 1.0)
        for d in ds:
            for tau in taus:
                for k in ks:
                    e, _ = residual(segs, which, d, tau, k)
                    if best is None or e < best[0]:
                        best = (e, d, tau, k)
        e, d, tau, k = best
        unit = "°" if which == "turn" else " mm"
        scale = math.degrees(1) if which == "turn" else 1000
        print(f"{which}: 組 {n}、遅れ d {1000 * d:.0f} ms・時定数 τ {1000 * tau:.0f} ms・倍率 k {k:.2f}、"
              f"1 組の誤差 {scale * math.sqrt(e / n):.1f}{unit}（遅れ 0・惰性なし・倍率 1 なら {scale * math.sqrt(base / n):.1f}{unit}）")


if __name__ == "__main__":
    main()
