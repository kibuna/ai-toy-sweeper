"""幾何計画の基準 Policy（toy_sweeper.planner）のテスト。"""
import math
import shutil

import cv2
import numpy as np
import pytest
import yaml

from toy_sweeper.config import ConfigError, load_all
from toy_sweeper.env import ToySweeperEnv
from toy_sweeper.evaluate import evaluate, run_episode, summarize
from toy_sweeper.geometry import Pose2D, Rect
from toy_sweeper.planner import PlannerPolicy, load_planner_cfg, world_state_from_env
from toy_sweeper.planner.config import CONFIG_DIR
from toy_sweeper.planner.cspace import Footprint, Grid, KernelCache, CSpace, box_polygon, convex_overlap, transform
from toy_sweeper.planner.search import GeodesicHeuristic, geodesic_field, kinematic_step, lattice_astar
from toy_sweeper.planner.world import Geometry, ObjState, WorldState
from toy_sweeper.sim2d import ObjectSpec, Sim2D
from toy_sweeper.spaces import Action, GripperState, ObjectType
from toy_sweeper.task import EpisodeSpec


@pytest.fixture(scope="module")
def cfg():
    return load_all()


@pytest.fixture(scope="module")
def pcfg():
    return load_planner_cfg()


# ---------- 設定 ----------
def test_planner_config_is_strict(tmp_path):
    shutil.copy(CONFIG_DIR / "planner.yaml", tmp_path / "planner.yaml")
    load_planner_cfg(tmp_path)  # そのままなら読める
    d = yaml.safe_load((tmp_path / "planner.yaml").read_text())
    d["search"]["unknown_key"] = 1
    (tmp_path / "planner.yaml").write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError):
        load_planner_cfg(tmp_path)
    d["search"].pop("unknown_key")
    d["search"].pop("cell_m")
    (tmp_path / "planner.yaml").write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError):
        load_planner_cfg(tmp_path)


def test_relocation_must_be_below_failure_threshold(cfg, pcfg):
    import dataclasses
    cfg = dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, reward=dataclasses.replace(
        cfg.task.reward, non_target_failure=True)))   # 失敗の判定がある（v18 まで）ときだけの制約
    bad = dataclasses.replace(pcfg, foreign=dataclasses.replace(pcfg.foreign, max_relocation_m=0.25))
    with pytest.raises(ConfigError):
        PlannerPolicy(cfg, bad)


# ---------- 幾何 ----------
def test_convex_overlap():
    a = box_polygon(Pose2D(0, 0, 0), (0.1, 0.1))
    assert convex_overlap(a, box_polygon(Pose2D(0.09, 0, 0.3), (0.1, 0.02)))
    assert not convex_overlap(a, box_polygon(Pose2D(0.2, 0, 0.0), (0.1, 0.1)))
    assert not convex_overlap(a, box_polygon(Pose2D(0.1, 0.1, math.pi / 4), (0.02, 0.02)))


def _cspace_with_lego(cfg, pcfg, lego: Pose2D, margin: float):
    pol = PlannerPolicy(cfg, pcfg)
    area = Rect((0.0, 0.0), (0.8, 0.8))
    grid = Grid(area, pol.pad, pcfg.grid.resolution_m)
    size = cfg.sim.objects["lego"].size_m[:2]
    hard = grid.raster([box_polygon(lego, size, margin)])
    cs = CSpace(grid, KernelCache(pcfg.grid.resolution_m, pcfg.grid.heading_bins), pol.open_fp, hard, None, None,
                0.0, 0.0)
    return pol, cs, size


def test_cspace_agrees_with_polygons(cfg, pcfg):
    """C-space が「重なる」を見逃さない（3 mm 以上めり込む姿勢は必ず衝突）、かつ 2 cm 離れた姿勢は通す。"""
    lego = Pose2D(0.05, -0.03, 0.4)
    m = pcfg.margins.hard_margin_m
    pol, cs, size = _cspace_with_lego(cfg, pcfg, lego, m)
    rng = np.random.default_rng(0)
    deep = box_polygon(lego, (size[0] - 0.006, size[1] - 0.006))
    far = box_polygon(lego, size, 0.02)
    n_deep = n_far = 0
    for _ in range(3000):
        p = Pose2D(float(rng.uniform(-0.15, 0.25)), float(rng.uniform(-0.2, 0.15)), float(rng.uniform(-math.pi, math.pi)))
        robot = transform(list(pol.open_fp.polys), p)
        c = cs.cost_at(p.x, p.y, p.yaw)
        if any(convex_overlap(q, deep) for q in robot):
            n_deep += 1
            assert c == math.inf, p
        elif not any(convex_overlap(q, far) for q in robot):
            n_far += 1
            assert c < math.inf, p
    assert n_deep > 50 and n_far > 500


def test_geodesic_field_goes_around_wall():
    free = np.ones((20, 20), bool)
    free[5, 0:15] = False  # 横の壁（右端だけ開いている）
    d = geodesic_field(free, [(0, 0)], 0.01)
    assert d[0, 0] == 0
    assert math.isinf(d[5, 3])
    # 壁の真下へは、壁の右端を回る
    assert d[10, 0] > 0.2


def test_lattice_astar_avoids_obstacles(cfg, pcfg):
    """壁のように並べた LEGO の向こうへ行く経路が、どの姿勢でも障害物と重ならない。"""
    pol = PlannerPolicy(cfg, pcfg)
    area = Rect((0.0, 0.0), (0.8, 0.8))
    grid = Grid(area, pol.pad, pcfg.grid.resolution_m)
    size = cfg.sim.objects["lego"].size_m[:2]
    legos = [Pose2D(0.0, y, math.pi / 2) for y in np.arange(-0.4, 0.15, 0.03)]
    polys = [box_polygon(p, size, pcfg.margins.hard_margin_m) for p in legos]
    hard = grid.raster(polys) | grid.outside_area()
    cs = CSpace(grid, KernelCache(pcfg.grid.resolution_m, pcfg.grid.heading_bins), pol.open_fp, hard, None, None, 0, 0)
    start = Pose2D(-0.2, -0.2, 0.0)
    goal = (0.2, -0.2)

    def goal_test(x, y, th):
        return [] if math.hypot(x - goal[0], y - goal[1]) < 0.03 else None

    # 障害物を避けた 2D の距離（計画器と同じ作り方: 1 cm 格子、壁と障害物から heuristic_clearance_m）
    d = cv2.distanceTransform((1 - hard).astype(np.uint8), cv2.DIST_L2, 5) * grid.res
    free = (d > pcfg.search.heuristic_clearance_m)[::2, ::2]
    cell = 2 * grid.res
    field = geodesic_field(free, [(int((goal[1] - grid.y0) / cell), int((goal[0] - grid.x0) / cell))], cell)
    h = GeodesicHeuristic(field, grid.x0, grid.y0, cell, pol.step_m)
    res = lattice_astar(start, cs, goal_test, h, pcfg.search, pol.step_m, pol.turn_rad,
                        pcfg.search.heuristic_weights[0], pcfg.search.max_expansions[0])
    assert res.actions is not None, res
    p = start
    exact = [box_polygon(q, size) for q in legos]
    max_y = p.y
    for a in res.actions:
        p = kinematic_step(p, a, pol.step_m, pol.turn_rad)
        max_y = max(max_y, p.y)
        robot = transform(list(pol.open_fp.polys), p)
        assert not any(convex_overlap(r, e) for r in robot for e in exact), p
    assert math.hypot(p.x - goal[0], p.y - goal[1]) < 0.03
    assert max_y > 0.15  # 壁の上の開いた側を回る


def test_held_object_is_geometric(cfg, pcfg):
    geo = Geometry(cfg.sim, pcfg.held.lateral_tol_m)
    robot = Pose2D(0.0, 0.0, 0.3)
    near = robot.transform_point((0.088, 0.003))
    o = ObjState("a", ObjectType.LEGO, Pose2D(near[0], near[1], 0.0), Pose2D(0.2, 0.2, 0.0), True)
    goal = Rect((0.3, 0.3), (0.2, 0.2))
    area = Rect((0, 0), (0.8, 0.8))
    w = WorldState(robot, GripperState.CLOSED, (o,), goal, area)
    assert geo.held_object(w) == o
    assert geo.held_object(WorldState(robot, GripperState.OPEN, (o,), goal, area)) is None
    far = robot.transform_point((0.2, 0.0))
    o2 = ObjState("a", ObjectType.LEGO, Pose2D(far[0], far[1], 0.0), o.initial, True)
    assert geo.held_object(WorldState(robot, GripperState.CLOSED, (o2,), goal, area)) is None


# ---------- 状態の読み方 ----------
class _NoInternalsSim(Sim2D):
    """計画器がシミュレータの内部（把持状態・爪の角度）を読まないことを確かめる。"""

    def grasped_object(self):
        raise AssertionError("計画器が grasped_object を読んだ")

    def jaw_angle(self):
        raise AssertionError("計画器が jaw_angle を読んだ")


def test_planner_reads_only_observable_state(cfg, pcfg):
    env = ToySweeperEnv(cfg, stage=5, backend=_NoInternalsSim(cfg.sim))
    pol = PlannerPolicy(cfg, pcfg).bind(env)
    r = run_episode(env, pol, 50000)
    assert r.success, r


def test_world_state_uses_observed_poses(cfg):
    env = ToySweeperEnv(cfg, stage=6)
    env.reset(seed=3)
    w = world_state_from_env(env)
    poses, robot = env.observed_poses()
    assert w.robot == robot
    assert {o.name: o.pose for o in w.objects} == poses
    assert {o.name for o in w.objects if o.is_target} == set(env.episode.target_names)
    assert all(o.initial == next(s.pose for s in env.episode.objects if s.name == o.name) for o in w.objects)


def test_predict_requires_bind_and_reset(cfg, pcfg):
    env = ToySweeperEnv(cfg, stage=2)
    obs, _ = env.reset(seed=1)
    with pytest.raises(RuntimeError):
        PlannerPolicy(cfg, pcfg).predict(obs)
    pol = PlannerPolicy(cfg, pcfg).bind(env)
    with pytest.raises(RuntimeError):
        pol.predict(obs)
    pol.reset()
    pol.predict(obs)
    env.reset(seed=2)
    with pytest.raises(RuntimeError):  # env だけ reset して policy.reset() を忘れた
        pol.predict(obs)


def test_toy_car_cannot_be_grasped_in_2d(cfg):
    """計画器がミニカーを押す前提: 2D ではどの向きで差し込んでもミニカーの中心が開いた爪の梁の間に入らず、挟めない。

    （2026-10-17、挟む案の検討。98 通り試して全滅。中心は前方 99〜112 mm で止まり、梁の範囲は
    75.7〜92.9 mm）。爪や把持の判定を変えたらこのテストが落ちるので、ミニカーを挟む方式を見直す。
    """
    sim = Sim2D(cfg.sim)
    for phi in np.linspace(0, math.pi / 2, 5):
        for n in range(2, 7):
            sim.reset(Pose2D(0.0, 0.0, 0.0), [ObjectSpec("car", ObjectType.TOY_CAR, Pose2D(0.16, 0.0, float(phi)), "red")],
                      GripperState.OPEN)
            for _ in range(n):
                sim.drive(cfg.sim.robot.linear_speed_mps, 0.0, cfg.task.control.action_duration_s)
            sim.actuate_gripper(close=True)
            assert sim.grasped_object() is None, (phi, n)


# ---------- 盤面を解く ----------
def test_planner_grasps_lego_next_to_wall(cfg, pcfg):
    """壁から 2 cm の LEGO（押すだけでは運べない）を挟んで運ぶ。"""
    goal = Rect((0.0, 0.0), (0.2, 0.2))
    ep = EpisodeSpec(5, Pose2D(0.0, -0.2, -0.3), (ObjectSpec("t", ObjectType.LEGO, Pose2D(0.3, -0.38, 0.3), "red"),),
                     ("t",), None, "test", goal)
    env = ToySweeperEnv(cfg, stage=5)
    pol = PlannerPolicy(cfg, pcfg).bind(env)
    obs, _ = env.reset(seed=0, options={"episode": ep})
    pol.reset()
    for _ in range(env.max_steps()):
        obs, _, term, trunc, info = env.step(pol.predict(obs))
        if term or trunc:
            break
    assert info["is_success"], info


def test_planner_pushes_toy_car_sideways(cfg, pcfg):
    goal = Rect((0.2, 0.0), (0.2, 0.2))
    ep = EpisodeSpec(4, Pose2D(-0.25, 0.0, 0.0), (ObjectSpec("t", ObjectType.TOY_CAR, Pose2D(-0.05, 0.02, 1.4), "red"),),
                     ("t",), None, "test", goal)
    env = ToySweeperEnv(cfg, stage=4)
    pol = PlannerPolicy(cfg, pcfg).bind(env)
    obs, _ = env.reset(seed=0, options={"episode": ep})
    pol.reset()
    for _ in range(env.max_steps()):
        obs, _, term, trunc, info = env.step(pol.predict(obs))
        if term or trunc:
            break
    assert info["is_success"], info


def test_planner_rotates_car_aligned_with_goal(cfg, pcfg):
    """車軸の向きにゴールがある（横向きに押す直線が無い）ミニカーを、端を押して回してから押し込む。"""
    goal = Rect((-0.26, -0.04), (0.15, 0.25))
    ep = EpisodeSpec(4, Pose2D(0.22, -0.17, 0.7),
                     (ObjectSpec("t", ObjectType.TOY_CAR, Pose2D(0.05, -0.08, 0.0), "red"),), ("t",), None, "test", goal)
    import dataclasses
    no_via = dataclasses.replace(pcfg, push=dataclasses.replace(pcfg.push, max_via=0))  # 2 段の押しを使わせない
    env = ToySweeperEnv(cfg, stage=4)
    pol = PlannerPolicy(cfg, no_via).bind(env)
    obs, _ = env.reset(seed=0, options={"episode": ep})
    pol.reset()
    pivoted = False
    for _ in range(env.max_steps()):
        obs, _, term, trunc, info = env.step(pol.predict(obs))
        pivoted |= pol.task is not None and pol.task.n_pivot > 0
        if term or trunc:
            break
    assert pivoted
    assert info["is_success"], info


# 開発用 Seed（10000〜）とも最終評価用（20000〜）とも別の Seed 50000〜。実測（2026-10-17）から下げた下限で、デグレを検出する
PLANNER_FLOOR = {1: (5, 5), 2: (5, 5), 5: (5, 5), 7: (5, 3)}


@pytest.mark.parametrize("stage", sorted(PLANNER_FLOOR))
def test_planner_solves_boards(cfg, pcfg, stage):
    n, floor = PLANNER_FLOOR[stage]
    env = ToySweeperEnv(cfg, stage=stage)
    pol = PlannerPolicy(cfg, pcfg).bind(env)
    s = summarize(evaluate(env, pol, range(50000, 50000 + n)))
    assert s["success_rate"] * n >= floor, s
    assert not s["failures"], s  # 対象外を動かしすぎない


def test_planner_rejects_stage9(cfg, pcfg):
    """計画器は置き場の指定（Stage 9、v24）を読まないので、Stage 9 の盤面は例外にする。"""
    env = ToySweeperEnv(cfg, stage=9)
    env.reset(seed=0)
    pol = PlannerPolicy(cfg, pcfg).bind(env)
    with pytest.raises(ValueError, match="Stage 9"):
        pol.reset()
