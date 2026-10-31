"""Rust の TaskSampler（native/src/sampler.rs）が Python の toy_sweeper.task.TaskSampler と同じ盤面を引くことの確かめ。

同じ乱数の状態から 1 つの Generator で続けて引き（32 ビットの残りも次の盤面に持ち越す）、盤面のすべての値が
ビット単位で一致し、引いた後の乱数の状態（state・inc・32 ビットの残り）も一致することを見る。
"""
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils import seeding

import toy_sweeper_native as nat
from toy_sweeper.config import load_all
from toy_sweeper.geometry import Pose2D
from toy_sweeper.native import cfg_json
from toy_sweeper.task import STAGES, TaskSampler, approach_point, gripper_point

TEST_CFG = Path(__file__).resolve().parent / "configs"   # テストだけで使う設定の差し替え

EXP = Path(__file__).resolve().parents[1] / "config" / "experiments"
# 盤面の引き方を変える設定: ゴールの決め方（random / zones）、名前付きゴールの位置、作業領域の大きさ。
# imperfect と no_walls は盤面の引き方に関わらない設定だが、Rust 側が設定を読めて同じ盤面になることを見る
CONFIGS = {
    "default": (),
    "zones": (TEST_CFG / "goals_zones.yaml",),
    "imperfect": (EXP / "imperfect.yaml",),
    "imperfect_zones": (EXP / "imperfect.yaml", TEST_CFG / "goals_zones.yaml"),
    "no_walls": (TEST_CFG / "no_walls.yaml",),
    "edges_zones": (TEST_CFG / "goals_edges.yaml", TEST_CFG / "goals_zones.yaml"),
    "field_1m": (TEST_CFG / "field_1m.yaml",),
    "field_1m_zones": (TEST_CFG / "field_1m.yaml", TEST_CFG / "goals_zones.yaml"),
    "multi_toy_car": (EXP / "multi_toy_car.yaml",),
    "multi_toy_car_zones": (EXP / "multi_toy_car.yaml", TEST_CFG / "goals_zones.yaml"),
    "target_group": (EXP / "multi_toy_car.yaml", TEST_CFG / "target_group.yaml"),
    "target_group_zones": (TEST_CFG / "target_group.yaml", TEST_CFG / "goals_zones.yaml"),
    # v30: 盤面ごとの作業領域
    "free_area": (EXP / "free_area.yaml", EXP / "multi_toy_car.yaml"),
    "free_area_zones": (EXP / "free_area.yaml", TEST_CFG / "goals_zones.yaml"),
    # v31: 実測した車体（大きさ・車軸の位置）と物体
    "hw_1022": (EXP / "free_area.yaml", EXP / "multi_toy_car.yaml", EXP / "robot_8hz.yaml", EXP / "hw_1022.yaml"),
}
N_BOARDS = 300


def numpy_state(g):
    s = g.bit_generator.state
    return s["state"]["state"], s["state"]["inc"], (s["uinteger"] if s["has_uint32"] else None)


def canon(x):
    """比べる形: float は hex（-0.0 と 0.0 も区別する）、tuple と list は同じに扱う。"""
    if isinstance(x, float):
        return ("f", x.hex())
    if isinstance(x, (tuple, list)):
        return tuple(canon(v) for v in x)
    if isinstance(x, dict):
        return {k: canon(v) for k, v in x.items()}
    assert x is None or isinstance(x, (int, str)), type(x)
    return x


def rect(r):
    return (tuple(r.center), tuple(r.size))


def py_dict(ep):
    return {
        "stage": ep.stage,
        "robot": (ep.robot.x, ep.robot.y, ep.robot.yaw),
        "objects": [(o.name, int(o.type), o.pose.x, o.pose.y, o.pose.yaw, o.color) for o in ep.objects],
        "target_names": list(ep.target_names),
        "reach_point": ep.reach_point,
        "goal_name": ep.goal_name,
        "goal": rect(ep.goal),
        "zones": [rect(z) for z in ep.zones],
        "home_overrides": [tuple(x) for x in ep.home_overrides],
        "area_size": ep.area_size,
    }


_CACHE = {}


def setup(name):
    if name not in _CACHE:
        cfg = load_all(overrides=CONFIGS[name])
        _CACHE[name] = (cfg, TaskSampler(cfg.sim, cfg.task), nat.TaskSampler(cfg_json(cfg)))
    return _CACHE[name]


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_boards_match_python(name, stage):
    _, py, rs = setup(name)
    g, _ = seeding.np_random(1000 * stage + len(name))
    g.integers(10)   # 32 ビットの残りがある状態から始める
    st = numpy_state(g)
    n_obj = set()
    n_over = 0
    for i in range(N_BOARDS):
        want = py_dict(py.sample(stage, g))
        n_over += bool(want["home_overrides"])
        got, st = rs.sample(stage, st)
        assert canon(got) == canon(want), (name, stage, i)
        assert st == numpy_state(g), (name, stage, i)
        n_obj.add(len(want["objects"]))
    # 物体の数がばらつく Stage は、いろいろな数の盤面を見た
    if stage in (6, 7, 8, 9):
        assert len(n_obj) >= 5, n_obj
    if stage == 9:   # やり残しのある盤面も、ない盤面も見た
        assert 0 < n_over < N_BOARDS, n_over


@pytest.mark.parametrize("mode", ["zones", "random"])
@pytest.mark.parametrize("delivered, wedged", [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)])
def test_stage9_mistake_boards_match_python(mode, delivered, wedged):
    """Stage 9 のやり残しを必ず加える設定（対象を全部ゴールに置き直す確率 0 と 1、挟まった対象外 v35 の確率 0・0.5・1）でも、
    盤面と乱数の状態が一致する。"""
    cfg = load_all(overrides=CONFIGS["zones"] if mode == "zones" else ())
    sp = dataclasses.replace(cfg.task.sampling, stage9_mistake_probability=1.0,
                             stage9_all_delivered_probability=delivered, stage9_wedged_probability=wedged)
    cfg = dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, sampling=sp))
    py, rs = TaskSampler(cfg.sim, cfg.task), nat.TaskSampler(cfg_json(cfg))
    g, _ = seeding.np_random(4242 + int(delivered) + int(10 * wedged))
    st = numpy_state(g)
    kinds = set()
    for i in range(N_BOARDS):
        ep = py.sample(9, g)
        got, st = rs.sample(9, st)
        assert canon(got) == canon(py_dict(ep)), (mode, delivered, i)
        assert st == numpy_state(g), (mode, delivered, i)
        over = dict(ep.home_overrides)
        assert bool(set(over) & set(ep.target_names)) == bool(delivered)
        kinds |= {"in" if v == -1 else "out" for k, v in over.items() if k not in ep.target_names}
    assert kinds == {"in", "out"}


def test_sample_count_matches_chained_python():
    """_sample_count（n 枚続けて引き、乱数の状態だけ返す）も Python と同じところへ進む。"""
    _, py, rs = setup("default")
    for stage in STAGES:
        g, _ = seeding.np_random(77 + stage)
        st = numpy_state(g)
        for _ in range(50):
            py.sample(stage, g)
        assert rs._sample_count(stage, st, 50) == numpy_state(g)


def test_unknown_stage_raises():
    _, py, rs = setup("default")
    g, _ = seeding.np_random(0)
    st = numpy_state(g)
    for stage in (-1, 10, 100):
        with pytest.raises(ValueError, match="未定義の Stage"):
            py.sample(stage, g)
        with pytest.raises(ValueError, match="未定義の Stage"):
            rs.sample(stage, st)
    assert numpy_state(g) == st


def test_max_tries_exhausted_raises():
    """作業領域いっぱいのゴールでは対象をゴールの外に置けない。Python は RuntimeError、Rust も止まる。"""
    cfg = load_all()
    sp = dataclasses.replace(cfg.task.sampling, random_goal_size_m=[0.8, 0.8], max_tries=50)
    task = dataclasses.replace(cfg.task, sampling=sp)
    g, _ = seeding.np_random(0)
    st = numpy_state(g)
    with pytest.raises(RuntimeError, match="有効な盤面が引けない"):
        TaskSampler(cfg.sim, task).sample(1, g)
    d = json.loads(cfg_json(cfg))
    d["task"]["sampling"]["random_goal_size_m"] = [0.8, 0.8]
    d["task"]["sampling"]["max_tries"] = 50
    with pytest.raises(BaseException, match="有効な盤面が引けない"):
        nat.TaskSampler(json.dumps(d)).sample(1, st)


def test_bad_rng_state_raises():
    _, _, rs = setup("default")
    with pytest.raises(BaseException):
        rs.sample(0, (1, 2, None))   # inc は奇数


def test_gripper_and_approach_point_match():
    cfg = load_all()
    j = cfg_json(cfg)
    r = np.random.default_rng(0)
    for _ in range(500):
        x, y, yaw = r.uniform(-0.4, 0.4), r.uniform(-0.4, 0.4), r.uniform(-4, 4)
        assert nat._gripper_point(j, x, y, yaw) == gripper_point(cfg.sim, Pose2D(x, y, yaw))
        o, gl, off = tuple(r.uniform(-0.4, 0.4, 2)), tuple(r.uniform(-0.4, 0.4, 2)), r.uniform(0, 0.1)
        assert nat._approach_point(o, gl, off) == approach_point(o, gl, off)
    with pytest.raises(ValueError):
        approach_point((0.1, 0.2), (0.1, 0.2), 0.05)
    with pytest.raises(BaseException, match="ゴール中心と一致"):
        nat._approach_point((0.1, 0.2), (0.1, 0.2), 0.05)


def test_speed_vs_python():
    """1 スレッドでの盤面/秒（Python と Rust）。Rust は Python から呼ぶ形（辞書を作る）と、Rust の中だけの形。"""
    _, py, rs = setup("default")
    rows = []
    for stage in STAGES:
        g, _ = seeding.np_random(stage)
        st0 = numpy_state(g)
        n_py = 200
        t = time.perf_counter()
        for _ in range(n_py):
            py.sample(stage, g)
        v_py = n_py / (time.perf_counter() - t)
        n_rs = 2000
        st = st0
        t = time.perf_counter()
        for _ in range(n_rs):
            _, st = rs.sample(stage, st)
        v_call = n_rs / (time.perf_counter() - t)
        t = time.perf_counter()
        rs._sample_count(stage, st0, n_rs)
        v_rs = n_rs / (time.perf_counter() - t)
        rows.append((stage, v_py, v_call, v_rs))
    print("\nstage  python/s  rust(call)/s  rust(inner)/s  倍率(inner)")
    for stage, a, b, c in rows:
        print(f"{stage:5d} {a:9.0f} {b:13.0f} {c:14.0f} {c / a:10.1f}")
    assert all(c > a for _, a, _, c in rows)
