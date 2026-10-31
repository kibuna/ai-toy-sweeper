"""Rust の NpRng（native/src/rng.rs）が numpy の Generator(PCG64) と同じ乱数列を出すことの確かめ。

gymnasium の seeding.np_random(seed) は np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed))) を作る。
同じ状態から始めて、random / uniform / integers / normal / choice を混ぜて引き、値が == で一致し、
最後の状態（PCG64 の state・inc と、32 ビットの残り）も一致することを見る。
"""
import math
import random
import struct

import numpy as np
import pytest
from gymnasium.utils import seeding

from toy_sweeper_native import NpRng

M64 = (1 << 64) - 1
M128 = (1 << 128) - 1
MULT = (2549297995355413924 << 64) | 4865540595714422341   # numpy の pcg64.h の PCG_DEFAULT_MULTIPLIER_128
KI0 = 0x000EF33D8025EF6A   # numpy の ziggurat_constants.h の ki_double[0]（これ以上の rabs で idx 0 なら裾）

SEEDS = list(range(50)) + [2**32, 2**63 - 1, 2**64 + 12345, 123456789012345678901234567890]

# integers(n) の n。2**32 は next_uint32 をそのまま返す道、3 * 2**30 と 2**31 + 1 は 32 ビットの Lemire の
# 引き直しがよく起きる（確率 1/4、ほぼ 1/2）、2**63 + 1 に近い大きな n は 64 ビットの Lemire の引き直しがよく起きる
INT_NS = [1, 2, 3, 4, 7, 10, 1000, 2**16 + 1, 2**31, 2**31 + 1, 3 * 2**30, 2**32 - 1, 2**32, 2**32 + 1, 2**33,
          2**40 + 7, 2**62 + 2**61, 2**63 - 1]
INT_RANGES = [(-5, 5), (3, 4), (-2**40, 2**40), (-2**63, 2**63 - 1), (-2**63, -2**63 + 3), (10, 2**32 + 10)]
UNIFORM_RANGES = [(0.0, 1.0), (-1.0, 1.0), (-math.pi, math.pi), (-10.0, -3.5), (0.2, 0.8), (2.5, 2.5),
                  (-1e-3, 1e-3), (-1e300, 1e300), (0.0, 1e-300)]


def bits(x: float) -> bytes:
    return struct.pack("<d", x)   # -0.0 と 0.0 も区別する


def numpy_state(g) -> tuple[int, int, int | None]:
    s = g.bit_generator.state
    return s["state"]["state"], s["state"]["inc"], (s["uinteger"] if s["has_uint32"] else None)


def rust_from(g) -> NpRng:
    st, inc, buf = numpy_state(g)
    return NpRng(st, inc, buf)


def rust_state(r) -> tuple[int, int, int | None]:
    st, inc = r.state()
    return st, inc, r.uint32_buffer()


def steps_between(a: int, b: int, inc: int, limit: int = 1000) -> int:
    """PCG64 の状態 a から b まで何回進んだか（使った 64 ビットの数）。"""
    for k in range(limit):
        if a == b:
            return k
        a = (a * MULT + inc) & M128
    raise AssertionError("状態が届かない")


def first_output(s: int, inc: int) -> int:
    """状態 s から次に出る 64 ビット（pcg64.h の XSL-RR）。道の数え上げにだけ使う。"""
    s = (s * MULT + inc) & M128
    x = (s >> 64) ^ (s & M64)
    rot = s >> 122
    return ((x >> rot) | (x << ((64 - rot) & 63))) & M64


class Paths:
    """どの道を通ったかの数え上げ（引いた前後の状態から決める）。"""

    def __init__(self):
        self.n = {"normal_fast": 0, "normal_wedge": 0, "normal_tail": 0, "normal_retry": 0,
                  "int32_reject": 0, "int64_reject": 0, "int32_buffered": 0, "int_no_draw": 0}

    def normal(self, before, after):
        s0, inc, _ = before
        k = steps_between(s0, after[0], inc)
        r = first_output(s0, inc)
        idx, rabs = r & 0xFF, (r >> 9) & ((1 << 52) - 1)
        if idx == 0 and rabs >= KI0:
            self.n["normal_tail"] += 1
        elif k == 1:
            self.n["normal_fast"] += 1
        elif k == 2:
            self.n["normal_wedge"] += 1
        else:
            self.n["normal_retry"] += 1   # 楔で外れて引き直した

    def integers(self, n_range, before, after):
        k = steps_between(before[0], after[0], before[1])
        if n_range == 0:
            assert k == 0 and before[2] == after[2]
            self.n["int_no_draw"] += 1
        elif n_range <= 0xFFFFFFFF:
            # 32 ビットを何個使ったか: 64 ビット 1 個で 2 個、前の残りを使えば +1、残りが出れば -1
            used = 2 * k + (before[2] is not None) - (after[2] is not None)
            if before[2] is not None:
                self.n["int32_buffered"] += 1
            if used > 1:
                self.n["int32_reject"] += 1
        else:
            if k > 1:
                self.n["int64_reject"] += 1


def run_mixed(seed: int, n_ops: int, paths: Paths | None = None, weights=None):
    g, _ = seeding.np_random(seed)
    r = rust_from(g)
    assert numpy_state(g) == rust_state(r)
    drv = random.Random(seed * 7919 + 1)   # 何を引くかを決める乱数（試験の側）
    ops = ["random", "uniform", "uniform_star", "integers", "integers_range", "normal", "normal3", "raw", "choice"]
    w = weights or [3, 3, 1, 4, 2, 4, 1, 1, 1]
    for _ in range(n_ops):
        op = drv.choices(ops, w)[0]
        before = numpy_state(g)
        if op == "random":
            assert bits(g.random()) == bits(r.random())
        elif op == "uniform":
            lo, hi = drv.choice(UNIFORM_RANGES)
            assert bits(g.uniform(lo, hi)) == bits(r.uniform(lo, hi))
        elif op == "uniform_star":
            t = drv.choice(UNIFORM_RANGES)
            assert bits(float(g.uniform(*t))) == bits(r.uniform(*t))
        elif op == "integers":
            n = drv.choice(INT_NS)
            a, b = g.integers(n), r.integers(n)
            assert isinstance(a, np.int64) and int(a) == b
            if paths:
                paths.integers(n - 1, before, numpy_state(g))
        elif op == "integers_range":
            lo, hi = drv.choice(INT_RANGES)
            a, b = g.integers(lo, hi), r.integers_range(lo, hi)
            assert isinstance(a, np.int64) and int(a) == b
            if paths:
                paths.integers(hi - 1 - lo, before, numpy_state(g))
        elif op == "normal":
            assert bits(g.normal()) == bits(r.normal())
            if paths:
                paths.normal(before, numpy_state(g))
        elif op == "normal3":
            a = g.normal(size=3)
            assert [bits(x) for x in a.tolist()] == [bits(r.normal()) for _ in range(3)]
        elif op == "raw":
            assert int(g.bit_generator.random_raw()) == r.next_u64()
        elif op == "choice":
            pop = drv.choice([1, 2, 5, 9, 20, 100, 2**33])
            k = drv.randint(0, min(pop, 6))
            assert g.choice(pop, size=k, replace=False).tolist() == r.choice_no_replace(pop, k)
        assert numpy_state(g) == rust_state(r), op
    return g, r


@pytest.mark.parametrize("seed", SEEDS)
def test_mixed_sequences_match_numpy(seed):
    run_mixed(seed, 5000)


def test_all_paths_are_covered():
    """正規分布の裾・楔、整数の 32 / 64 ビットの引き直し、32 ビットの残りを使う道を、それぞれ何度も通る。"""
    paths = Paths()
    for seed in range(4):
        run_mixed(1000 + seed, 100000, paths, weights=[1, 1, 0, 3, 1, 8, 0, 0, 0])
    print(paths.n)
    for k, v in paths.n.items():
        assert v >= 20, (k, paths.n)


@pytest.mark.parametrize("pop,k", [(10001, 201), (20000, 20000), (12345, 3000), (10001, 200), (10000, 9000)])
def test_choice_large_population(pop, k):
    """母集団が 10000 を超え size が pop // 50 を超えると、numpy は Floyd 法でなく末尾の並べ替えを使う。両方の境目を見る。"""
    g, _ = seeding.np_random(pop + k)
    g.integers(10)   # 32 ビットの残りがある状態から
    r = rust_from(g)
    assert g.choice(pop, size=k, replace=False).tolist() == r.choice_no_replace(pop, k)
    assert numpy_state(g) == rust_state(r)


def test_seeding_is_gymnasium():
    """gymnasium の np_random は SeedSequence(seed) から PCG64 を作る（Rust 側はこの状態から始める）。"""
    g, _ = seeding.np_random(123)
    h = np.random.Generator(np.random.PCG64(np.random.SeedSequence(123)))
    assert g.bit_generator.state == h.bit_generator.state


def test_integers_one_draws_nothing():
    g, _ = seeding.np_random(5)
    g.integers(10)   # 32 ビットの残りを作っておく
    r = rust_from(g)
    before = numpy_state(g)
    assert g.integers(1) == 0 and r.integers(1) == 0
    assert numpy_state(g) == before == rust_state(r)


def test_normal_size3_is_three_scalar_draws():
    g, _ = seeding.np_random(9)
    h, _ = seeding.np_random(9)
    assert [bits(x) for x in g.normal(size=3).tolist()] == [bits(h.normal()) for _ in range(3)]


def test_invalid_arguments_raise():
    """numpy が ValueError / OverflowError を出す引数では、Rust も止まる（黙って何かを返さない）。"""
    g, _ = seeding.np_random(0)
    r = rust_from(g)
    for n in [0, -1]:
        with pytest.raises(ValueError):
            g.integers(n)
        with pytest.raises(BaseException):
            r.integers(n)
    with pytest.raises(BaseException):
        r.integers_range(3, 3)
    for lo, hi in [(1.0, 0.0), (0.0, -0.0), (-1e308, 1e308), (0.0, math.inf), (math.nan, 1.0)]:
        with pytest.raises((ValueError, OverflowError)):
            g.uniform(lo, hi)
        with pytest.raises(BaseException):
            r.uniform(lo, hi)
    with pytest.raises(BaseException):
        r.choice_no_replace(3, 4)
    with pytest.raises(BaseException):
        NpRng(2, 2, None)   # inc は奇数
    assert numpy_state(g) == rust_state(r)   # 止まった呼び出しは乱数を使わない
