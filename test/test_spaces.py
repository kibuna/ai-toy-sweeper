import math
from pathlib import Path

import numpy as np
import pytest

from toy_sweeper.config import load_all
from toy_sweeper.geometry import Pose2D, Rect
from toy_sweeper.spaces import (BASE_DIM, EXTRA_DIM, SLOT_DIM, Action, GripperState, ObservationBuilder, ObjectType,
                                SlotObject, TaskMode, WorldSnapshot)

TEST_CFG = Path(__file__).resolve().parent / "configs"   # テストだけで使う設定の差し替え


@pytest.fixture
def builder():
    cfg = load_all()
    return ObservationBuilder(cfg.sim, cfg.task)


def _snap(**kw):
    base = dict(
        robot=Pose2D(0.0, 0.0, 0.0),
        target=Pose2D(0.1, 0.0, 0.0),
        target_type=ObjectType.LEGO,
        goal=Rect(center=(0.3, 0.0), size=(0.2, 0.3)),
        gripper=GripperState.OPEN,
        prev_action=Action.STOP,
        obs_age_s=0.0,
        other_targets=(),
        obstacles=(),
        n_remaining=1,
        robot_motion=(0.0, 0.0, 0.0),
        time_left=1.0,
        target_held=False,
        task_mode=TaskMode.DELIVER,
        area_size=(0.8, 0.8),
    )
    base.update(kw)
    base.setdefault("destination", base["goal"])
    return WorldSnapshot(**base)


def test_shape_and_dtype(builder):
    o = builder.build(_snap())
    assert o.shape == (builder.dim,) and o.dtype == np.float32
    assert builder.dim == BASE_DIM + EXTRA_DIM + SLOT_DIM * (builder.n_target_slots + builder.n_obstacle_slots)
    assert builder.observation_space.contains(o)


def test_layout_values(builder):
    o = builder.build(_snap(prev_action=Action.TURN_LEFT, gripper=GripperState.CLOSED))
    d = builder.decode(o)
    assert np.allclose(d["robot_xy"], (0.0, 0.0))
    assert np.allclose(d["target_xy"], (0.1, 0.0), atol=1e-6)
    assert np.allclose(d["goal_center"], (0.3, 0.0), atol=1e-6)
    assert d["prev_action"] == Action.TURN_LEFT
    assert d["gripper"] == GripperState.CLOSED
    # ロボット正面に対象物: 方位 0 → sin 0, cos 1
    assert np.allclose(d["robot_to_target_bearing"], (0.0, 1.0), atol=1e-6)


def test_target_heading_is_180_symmetric(builder):
    a = builder.build(_snap(target=Pose2D(0.1, 0.0, 0.3)))
    b = builder.build(_snap(target=Pose2D(0.1, 0.0, 0.3 + math.pi)))
    assert np.allclose(a, b, atol=1e-6)


def test_no_target_type_for_reach(builder):
    d = builder.decode(builder.build(_snap(target_type=None)))
    assert d["target_type"] is None


def test_out_of_bounds_raises(builder):
    with pytest.raises(ValueError):
        builder.build(_snap(robot=Pose2D(5.0, 0.0, 0.0)))


def test_obstacles_sorted_and_masked(builder):
    far = SlotObject(Pose2D(0.3, 0.3, 0.0), ObjectType.LEGO, (0.0, 0.0), None)
    near = SlotObject(Pose2D(0.0, 0.1, 0.0), ObjectType.TOY_CAR, (0.0, 0.0), None)  # ロボット（yaw 0）の左 0.1 m
    d = builder.decode(builder.build(_snap(obstacles=(far, near))))
    obs = d["obstacles"]
    assert len(obs) == 2
    assert obs[0]["type"] == ObjectType.TOY_CAR and np.allclose(obs[0]["local_xy"], (0.0, 0.1), atol=1e-6)
    assert obs[1]["type"] == ObjectType.LEGO


def test_obstacles_truncated_to_slots(builder):
    many = tuple(SlotObject(Pose2D(-0.3 + 0.05 * i, -0.3, 0.0), ObjectType.LEGO, (0.0, 0.0), None) for i in range(builder.n_obstacle_slots + 3))
    d = builder.decode(builder.build(_snap(obstacles=many)))
    assert len(d["obstacles"]) == builder.n_obstacle_slots


def test_obstacle_in_robot_frame(builder):
    # ロボットが左（yaw = pi/2）を向くと、世界座標で左にある物体はロボット座標で前方になる
    ob = SlotObject(Pose2D(0.0, 0.1, 0.0), ObjectType.LEGO, (0.0, 0.0), None)
    d = builder.decode(builder.build(_snap(robot=Pose2D(0.0, 0.0, math.pi / 2), obstacles=(ob,))))
    assert np.allclose(d["obstacles"][0]["local_xy"], (0.1, 0.0), atol=1e-6)


def test_other_targets_flagged(builder):
    t = (SlotObject(Pose2D(0.2, 0.0, 0.0), ObjectType.LEGO, (0.0, 0.0), None),)
    o = (SlotObject(Pose2D(-0.2, 0.0, 0.0), ObjectType.LEGO, (0.0, 0.0), None),)
    d = builder.decode(builder.build(_snap(other_targets=t, obstacles=o, n_remaining=2)))
    assert d["n_remaining"] == 2
    assert [x["is_target"] for x in d["other_targets"]] == [True]
    assert [x["is_target"] for x in d["obstacles"]] == [False]


def test_v9_fields(builder):
    ob = SlotObject(Pose2D(0.0, 0.1, 0.0), ObjectType.LEGO, (0.05, 0.0), Rect(center=(0.3, 0.0), size=(0.2, 0.3)))   # 開始時から x に 5 cm 動いた。置き場は今のゴール
    s = _snap(robot=Pose2D(0.0, 0.0, math.pi / 2), obstacles=(ob,), robot_motion=(0.02, 0.0, 0.1), time_left=0.25,
              target_held=True)
    d = builder.decode(builder.build(s))
    assert np.allclose(d["robot_motion"], (0.02, 0.0, 0.1), atol=1e-6)
    assert d["time_left"] == pytest.approx(0.25) and d["target_held"]
    o = d["obstacles"][0]
    # ロボットが左（+y）を向いているので、世界の +x の変位はロボット座標で −y
    assert np.allclose(o["displacement_local"], (0.0, -0.05), atol=1e-6) and o["has_home"]
    # 主対象（0.1, 0）からゴール（x 0.2〜0.4）の縁まで 0.1 m（外は正）
    assert d["target_goal_sdist"] == pytest.approx(0.1, abs=1e-6)


def test_home_zone_distance(builder):
    """置き場を持つ物体のスロットは、置き場の縁までの距離と「置き場を持つ」旗（v18）。"""
    home = Rect(center=(-0.3, 0.3), size=(0.2, 0.2))
    ob = SlotObject(Pose2D(-0.3, 0.3, 0.0), ObjectType.LEGO, (0.0, 0.0), home)
    free = SlotObject(Pose2D(-0.1, -0.2, 0.0), ObjectType.LEGO, (0.0, 0.0), None)
    d = builder.decode(builder.build(_snap(obstacles=(ob, free))))
    by_x = {round(o["local_xy"][0], 3): o for o in d["obstacles"]}
    assert by_x[-0.3]["has_home"] and by_x[-0.3]["goal_sdist"] == pytest.approx(-0.1, abs=1e-6)
    # 置き場の無い物体は今のゴール（x 0.2〜0.4、y −0.15〜0.15）の縁まで: x 方向 0.3、y 方向 0.05 → 角までの距離
    assert not by_x[-0.1]["has_home"] and by_x[-0.1]["goal_sdist"] == pytest.approx(math.hypot(0.3, 0.05), abs=1e-6)


def _corner_builder():
    import dataclasses
    cfg = load_all()
    ob = dataclasses.replace(cfg.task.observation, edge_distance="corners")
    return ObservationBuilder(cfg.sim, dataclasses.replace(cfg.task, observation=ob))


def test_edge_distance_corners():
    """edge_distance: corners（v25）: 置き場を持つ物体・対象は、外形の 4 隅のうちいちばん外の角の符号付き距離。
    置き場を持たない対象外は中心のまま（ゴールの中の対象外の判定が中心なので）。"""
    b = _corner_builder()
    home = Rect(center=(-0.3, 0.3), size=(0.2, 0.2))   # x −0.4〜−0.2
    # LEGO（31.8 × 15.8 mm）の中心は置き場の縁の 10 mm 内側、+x の角は 15.9 − 10 = 5.9 mm 外
    ob = SlotObject(Pose2D(-0.21, 0.3, 0.0), ObjectType.LEGO, (0.0, 0.0), home)
    free = SlotObject(Pose2D(-0.1, -0.2, 0.0), ObjectType.LEGO, (0.0, 0.0), None)
    other_t = (SlotObject(Pose2D(0.3, 0.1, 0.0), ObjectType.LEGO, (0.0, 0.0), None),)   # ゴールの中（y 縁まで 50 mm）
    s = _snap(obstacles=(ob, free), other_targets=other_t)
    d = b.decode(b.build(s))
    by_x = {round(o["local_xy"][0], 3): o for o in d["obstacles"]}
    assert by_x[-0.21]["goal_sdist"] == pytest.approx(0.0059, abs=1e-6)
    assert by_x[-0.1]["goal_sdist"] == pytest.approx(math.hypot(0.3, 0.05), abs=1e-6)
    # 対象（ゴールの中）: いちばん外の角は y の縁から 0.05 − 0.0079 = 0.0421 内側
    assert d["other_targets"][0]["goal_sdist"] == pytest.approx(-0.0421, abs=1e-6)
    # 主対象（0.1, 0）: −x の角が x 0.0841 → ゴール（x 0.2〜）の縁まで 0.1159
    assert d["target_goal_sdist"] == pytest.approx(0.1159, abs=1e-6)
    # center（既定）はこれまでどおり中心
    dc = ObservationBuilder(load_all().sim, load_all().task)
    d0 = dc.decode(dc.build(s))
    assert {round(o["local_xy"][0], 3): o for o in d0["obstacles"]}[-0.21]["goal_sdist"] == pytest.approx(-0.01, abs=1e-6)
    assert d0["target_goal_sdist"] == pytest.approx(0.1, abs=1e-6)


def test_edge_distance_corners_sign_matches_inside_goal():
    """corners の値が 0 以下 ⟺ 4 隅が中（成功の判定 inside_goal と同じ）。向きと位置を変えて確かめる。"""
    from toy_sweeper.snapshot import inside_goal
    b = _corner_builder()
    cfg = load_all()
    home = Rect(center=(-0.3, 0.3), size=(0.2, 0.2))
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = Pose2D(-0.3 + rng.uniform(-0.13, 0.13), 0.3 + rng.uniform(-0.13, 0.13), rng.uniform(-math.pi, math.pi))
        t = ObjectType(int(rng.integers(2)))
        d = b.decode(b.build(_snap(obstacles=(SlotObject(p, t, (0.0, 0.0), home),))))
        assert (d["obstacles"][0]["goal_sdist"] <= 0) == inside_goal(cfg.sim, home, t, p)


def test_cleanup_primary_observation():
    """cleanup_primary（v26）: 頭に「いまの仕事」の one-hot 3 次元（残りの対象の数の後）。頭のゴールの欄は行き先、
    スロットの縁までの距離は今のゴール（置き場を持たない物体）。無効なら次元も値も今までと同じ。"""
    import dataclasses
    from toy_sweeper.spaces import MODE_DIM, TaskMode
    cfg = load_all()
    ob = dataclasses.replace(cfg.task.observation, cleanup_primary=True)
    b = ObservationBuilder(cfg.sim, dataclasses.replace(cfg.task, observation=ob))
    b0 = ObservationBuilder(cfg.sim, cfg.task)
    assert b.dim == b0.dim + MODE_DIM and b.head_dim == BASE_DIM + EXTRA_DIM + MODE_DIM and b0.head_dim == BASE_DIM + EXTRA_DIM
    zone = Rect(center=(-0.3, 0.3), size=(0.2, 0.2))
    free = SlotObject(Pose2D(-0.1, -0.2, 0.0), ObjectType.LEGO, (0.0, 0.0), None)
    s = _snap(task_mode=TaskMode.RETURN, destination=zone, target=Pose2D(-0.12, 0.3, 0.0), obstacles=(free,))
    o = b.build(s)
    assert list(o[BASE_DIM + EXTRA_DIM:BASE_DIM + EXTRA_DIM + MODE_DIM]) == [0.0, 1.0, 0.0]
    d = b.decode(o)
    assert d["task_mode"] == TaskMode.RETURN and np.allclose(d["goal_center"], (-0.3, 0.3), atol=1e-6)
    assert d["target_goal_sdist"] == pytest.approx(0.08, abs=1e-6)     # 中心 −0.12 → 置き場の縁 −0.2 まで
    # 置き場を持たない物体のスロットは今のゴール（x 0.2〜0.4、y −0.15〜0.15）まで
    assert d["obstacles"][0]["goal_sdist"] == pytest.approx(math.hypot(0.3, 0.05), abs=1e-6)
    # 無効: 運ぶ以外の仕事は作られない
    o0 = b0.build(_snap())
    assert o0.shape == (b0.dim,) and np.array_equal(o0, b0.build(_snap(task_mode=TaskMode.DELIVER)))
    with pytest.raises(ValueError, match="cleanup_primary"):
        b0.build(s)


def test_area_size_observation():
    """v30: area_size が真なら、頭の終わり（仕事の 3 次元の後）に作業領域の半分の幅・奥行きを位置の尺度で割った 2 次元。
    位置の尺度は position_scale_m で、sim.work_area（実機のランタイムが校正の大きさで置き換える）によらない。"""
    import dataclasses
    from pathlib import Path
    from toy_sweeper.spaces import AREA_DIM, MODE_DIM
    exp = Path(__file__).resolve().parents[1] / "config" / "experiments"
    cfg = load_all(overrides=(exp / "cleanup_guidance.yaml", exp / "free_area.yaml"))
    b = ObservationBuilder(cfg.sim, cfg.task)
    assert AREA_DIM == 2 and b.head_dim == BASE_DIM + EXTRA_DIM + MODE_DIM + AREA_DIM
    o = b.build(_snap(area_size=(0.6, 1.0), robot=Pose2D(0.2, -0.1, 0.0)))
    assert list(o[b.head_dim - AREA_DIM:b.head_dim]) == pytest.approx([0.75, 1.25])
    assert b.decode(o)["area_size"] == pytest.approx((0.6, 1.0))
    # sim.work_area を変えても尺度は 0.4 m のまま
    sim2 = dataclasses.replace(cfg.sim, work_area=dataclasses.replace(cfg.sim.work_area, size_m=(1.0, 0.7)))
    o2 = ObservationBuilder(sim2, cfg.task).build(_snap(area_size=(0.6, 1.0), robot=Pose2D(0.2, -0.1, 0.0)))
    assert np.array_equal(o, o2) and o[0] == pytest.approx(0.5) and o[1] == pytest.approx(-0.25)
    # 無効なら次元は今までと同じ
    ob = dataclasses.replace(cfg.task.observation, area_size=False)
    assert ObservationBuilder(cfg.sim, dataclasses.replace(cfg.task, observation=ob)).dim == b.dim - AREA_DIM


def test_obs_bound_covers_largest_free_area():
    """v30: 値域は ±3（0.8 m 四方・尺度 0.4 m で決めた）を、いちばん広い作業領域（壁なしなら外へ出てよい分を足す）と
    尺度の比で広げる。1.0 m 四方の外へ 0.10 m 出たロボットから、反対の角の外の物体までの観測が値域に入る。今までの
    設定では ±3 のまま。"""
    from pathlib import Path
    from toy_sweeper.spaces import OBS_BOUND, obs_bound
    exp = Path(__file__).resolve().parents[1] / "config" / "experiments"
    base = load_all()
    assert obs_bound(base.sim, base.task) == OBS_BOUND == 3.0
    nw = load_all(overrides=(TEST_CFG / "no_walls.yaml",))
    assert obs_bound(nw.sim, nw.task) == 3.0
    cfg = load_all(overrides=(exp / "free_area.yaml",))
    assert obs_bound(cfg.sim, cfg.task) == pytest.approx(4.5)
    b = ObservationBuilder(cfg.sim, cfg.task)
    assert b.observation_space.high[0] == pytest.approx(4.5)
    far = SlotObject(Pose2D(0.6, 0.6, 0.0), ObjectType.LEGO, (0.0, 0.0), None)
    o = b.build(_snap(area_size=(1.0, 1.0), robot=Pose2D(-0.6, -0.6, 0.3), obstacles=(far,)))
    assert np.abs(o).max() > 3.0
