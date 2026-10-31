"""行動そのものを辺にした格子上の A*（state lattice）。

ノードはロボットの姿勢 (x, y, θ)。辺は 4 つの走行の行動（前進・後退 step_m、左右の旋回 turn_rad）で、
1 回 = コスト 1（時間）。向きは計画を始めた向き + k × turn_rad に限る（実際の行動で取れる向きだけ）。
重複判定は (x, y) を cell_m、θ を C-space の向きの刻みで丸める（hybrid A* と同じ）。
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..geometry import Pose2D
from ..spaces import Action
from .config import SearchCfg
from .cspace import CSpace

# 旋回の掃引を確かめる、旋回 1 回の中の分割数（C-space の向きの刻み 5° に対し 14.3° を 4 分割）
_TURN_SUBSTEPS = 4

GoalTest = Callable[[float, float, float], "list[Action] | None"]
Heuristic = Callable[[float, float, float], float]


@dataclass
class SearchResult:
    actions: list[Action] | None   # None は計画失敗
    expansions: int
    reason: str                    # "found" / "exhausted"（到達できる範囲を調べ尽くした）/ "limit"（展開数の上限）
    cost: float


def kinematic_step(p: Pose2D, a: Action, step_m: float, turn_rad: float) -> Pose2D:
    """行動 1 回の名目の移動（計画の予測と、計画を続けてよいかの判定に使う）。"""
    if a == Action.FORWARD:
        return Pose2D(p.x + step_m * math.cos(p.yaw), p.y + step_m * math.sin(p.yaw), p.yaw)
    if a == Action.BACKWARD:
        return Pose2D(p.x - step_m * math.cos(p.yaw), p.y - step_m * math.sin(p.yaw), p.yaw)
    if a == Action.TURN_LEFT:
        return Pose2D(p.x, p.y, p.yaw + turn_rad)
    if a == Action.TURN_RIGHT:
        return Pose2D(p.x, p.y, p.yaw - turn_rad)
    raise ValueError(f"走行の行動でない: {a!r}")


class SegmentHeuristic:
    """目標が「向き θ_k の直線上の線分（車軸中点の位置）」の集まりのときのヒューリスティック（行動の回数）。

    h = min_k（線分までの距離 / step_m + 向きの差 / turn_rad の切り捨て）。障害物は考えない。
    向きは旋回でしか変わらないので、向きの項は下限になっている。
    """

    def __init__(self, segments: list[tuple[float, float, float, float, float]], step_m: float, turn_rad: float):
        if not segments:
            raise ValueError("目標の線分が空")
        a = np.array(segments, dtype=float)
        self.ax, self.ay, self.bx, self.by, self.th = a.T
        self.dx, self.dy = self.bx - self.ax, self.by - self.ay
        self.l2 = np.maximum(self.dx ** 2 + self.dy ** 2, 1e-12)
        self.step, self.turn = step_m, turn_rad

    def __call__(self, x: float, y: float, th: float) -> float:
        t = np.clip(((x - self.ax) * self.dx + (y - self.ay) * self.dy) / self.l2, 0.0, 1.0)
        d = np.hypot(self.ax + t * self.dx - x, self.ay + t * self.dy - y)
        dth = np.abs((self.th - th + math.pi) % (2 * math.pi) - math.pi)
        return float(np.min(d / self.step + np.floor(dth / self.turn + 1e-6)))


def geodesic_field(free: np.ndarray, sources: list[tuple[int, int]], cell_m: float) -> np.ndarray:
    """free（True が通れる画素）の上で、sources からの 8 近傍の最短距離（m）。届かない画素は inf。

    numpy の繰り返しの緩和（Bellman-Ford の格子版）。収束まで最長経路の画素数ほど回る（1 cm 格子で 100 回前後）。
    """
    INF = np.float32(np.inf)
    d = np.full(free.shape, INF, np.float32)
    src = [(r, c) for r, c in sources if 0 <= r < free.shape[0] and 0 <= c < free.shape[1]]
    if not src:
        raise ValueError("距離場の出発点が格子の中に無い")
    rr, cc = zip(*src)
    d[list(rr), list(cc)] = 0.0
    passable = free.copy()
    passable[list(rr), list(cc)] = True
    diag = np.float32(math.sqrt(2) * cell_m)
    straight = np.float32(cell_m)
    for _ in range(4 * max(free.shape)):
        n = d.copy()
        np.minimum(n[1:, :], d[:-1, :] + straight, out=n[1:, :])
        np.minimum(n[:-1, :], d[1:, :] + straight, out=n[:-1, :])
        np.minimum(n[:, 1:], d[:, :-1] + straight, out=n[:, 1:])
        np.minimum(n[:, :-1], d[:, 1:] + straight, out=n[:, :-1])
        np.minimum(n[1:, 1:], d[:-1, :-1] + diag, out=n[1:, 1:])
        np.minimum(n[:-1, :-1], d[1:, 1:] + diag, out=n[:-1, :-1])
        np.minimum(n[1:, :-1], d[:-1, 1:] + diag, out=n[1:, :-1])
        np.minimum(n[:-1, 1:], d[1:, :-1] + diag, out=n[:-1, 1:])
        n[~passable] = INF
        if np.array_equal(n, d):
            return d
        d = n
    raise RuntimeError("距離場が収束しない（格子の大きさに対して反復が足りない不具合）")


class GeodesicHeuristic:
    """障害物を避けた 2D の距離（車軸中点、m）を行動の回数にしたもの。届かない位置は inf（探索で枝刈りされる）。"""

    def __init__(self, field: np.ndarray, x0: float, y0: float, cell_m: float, step_m: float, offset_m: float = 0.0):
        self.f, self.x0, self.y0, self.cell, self.step, self.offset = field, x0, y0, cell_m, step_m, offset_m

    def __call__(self, x: float, y: float, th: float) -> float:
        col, row = int((x - self.x0) / self.cell), int((y - self.y0) / self.cell)
        if not (0 <= row < self.f.shape[0] and 0 <= col < self.f.shape[1]):
            return math.inf
        return max(0.0, float(self.f[row, col]) - self.offset) / self.step


def lattice_astar(start: Pose2D, cs: CSpace, goal_test: GoalTest, heuristic: Heuristic, p: SearchCfg,
                  step_m: float, turn_rad: float, weight: float, max_expansions: int) -> SearchResult:
    """start から goal_test が行動列（残りの直進など）を返すノードまでの最短（重み付き）の行動列。

    start 自体の衝突は問わない（開始時に重なっている物体は overlap 層のコストで抜け出させる）。
    """
    n_bins = cs.n_bins
    two_pi = 2 * math.pi
    th0 = start.yaw
    trig: dict[int, tuple[float, float, float, int, tuple[int, ...]]] = {}

    def heading(k: int):
        t = trig.get(k)
        if t is None:
            th = th0 + k * turn_rad
            j = int(round(th / two_pi * n_bins)) % n_bins
            t = (th, math.cos(th), math.sin(th), j, ())
            trig[k] = t
        return t

    turn_bins: dict[tuple[int, int], tuple[int, ...]] = {}

    def sweep_bins(k: int, dk: int) -> tuple[int, ...]:
        key = (k, dk)
        b = turn_bins.get(key)
        if b is None:
            th = heading(k)[0]
            b = tuple(sorted({int(round((th + dk * turn_rad * i / _TURN_SUBSTEPS) / two_pi * n_bins)) % n_bins
                              for i in range(1, _TURN_SUBSTEPS + 1)}))
            turn_bins[key] = b
        return b

    cell = p.cell_m
    w = weight
    back_c = 1.0 + p.backward_extra_cost
    layer = cs.layer
    grid = cs.grid
    gx0, gy0, res, gn = grid.x0, grid.y0, grid.res, grid.n

    def cost_at(x: float, y: float, j: int) -> float:
        col, row = int((x - gx0) / res), int((y - gy0) / res)
        if not (0 <= row < gn and 0 <= col < gn) or x < gx0 or y < gy0:
            return math.inf
        return float(layer(j)[row, col])

    # ノード: (x, y, k)
    xs, ys, ks, parent, act, gs = [start.x], [start.y], [0], [-1], [None], [0.0]
    best: dict[tuple[int, int, int], float] = {}
    j0 = heading(0)[3]
    best[(int(round(start.x / cell)), int(round(start.y / cell)), j0)] = 0.0
    heap = [(w * heuristic(start.x, start.y, th0), 0.0, 0)]
    expansions = 0
    while heap:
        _, g, i = heapq.heappop(heap)
        x, y, k = xs[i], ys[i], ks[i]
        th, c, s, j, _ = heading(k)
        if g > best.get((int(round(x / cell)), int(round(y / cell)), j), math.inf):
            continue
        tail = goal_test(x, y, th)
        if tail is not None:
            path = []
            while i > 0:
                path.append(act[i])
                i = parent[i]
            path.reverse()
            return SearchResult(path + list(tail), expansions, "found", g)
        expansions += 1
        if expansions > max_expansions:
            return SearchResult(None, expansions, "limit", math.inf)
        succ = []
        for a, sign, base in ((Action.FORWARD, 1.0, 1.0), (Action.BACKWARD, -1.0, back_c)):
            nx, ny = x + sign * step_m * c, y + sign * step_m * s
            cm = cost_at(x + 0.5 * sign * step_m * c, y + 0.5 * sign * step_m * s, j)
            ce = cost_at(nx, ny, j)
            if cm == math.inf or ce == math.inf:
                continue
            succ.append((a, nx, ny, k, base + max(cm, ce)))
        for a, dk in ((Action.TURN_LEFT, 1), (Action.TURN_RIGHT, -1)):
            cmax = 0.0
            for jb in sweep_bins(k, dk):
                cb = cost_at(x, y, jb)
                if cb > cmax:
                    cmax = cb
                    if cb == math.inf:
                        break
            if cmax == math.inf:
                continue
            succ.append((a, x, y, k + dk, 1.0 + cmax))
        for a, nx, ny, nk, dc in succ:
            ng = g + dc
            nj = heading(nk)[3]
            key = (int(round(nx / cell)), int(round(ny / cell)), nj)
            if ng >= best.get(key, math.inf):
                continue
            hv = heuristic(nx, ny, heading(nk)[0])
            if hv == math.inf:  # 2D の距離場で目標に届かない位置は展開しない
                continue
            best[key] = ng
            xs.append(nx)
            ys.append(ny)
            ks.append(nk)
            parent.append(i)
            act.append(a)
            gs.append(ng)
            heapq.heappush(heap, (ng + w * hv, ng, len(xs) - 1))
    return SearchResult(None, expansions, "exhausted", math.inf)
