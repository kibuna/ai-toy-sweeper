"""Rust 版の Sim2D（native/src/sim2d.rs）が Python 版（toy_sweeper.sim2d、pymunk）とビット単位で同じ軌跡になる。

盤面は TaskSampler（Stage 0〜8、壁あり・なし）。行動は、物体へ向かって閉じる・運ぶ・放す・押すを多めにした乱数の列
（つかむ、壁や物体に当たる場面を通すため）。env と同じく、遅延の間は前の指令で走る分割した drive も混ぜる。
毎歩、ロボット・全物体の姿勢、爪の角度、グリッパーの状態、つかんだ物体、車体に触れた物体を == で比べる。
"""
import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import pytest

import toy_sweeper_native as nat
from toy_sweeper.config import load_all
from toy_sweeper.native import cfg_json
from toy_sweeper.robot_model import jaw_inner_half_gap, robot_radius
from toy_sweeper.sim2d import ObjectSpec, Sim2D
from toy_sweeper.spaces import DRIVE_ACTIONS, Action, GripperState, ObjectType, drive_velocity
from toy_sweeper.task import STAGES, TaskSampler
from toy_sweeper.geometry import Pose2D

ROOT = Path(__file__).resolve().parents[1]
IMPERFECT = ROOT / "config" / "experiments" / "imperfect.yaml"
# 通した場面の数（つかむ、挟み損ね、車体に触れた物体、壁に届く位置にいた歩）
_COVER = {"episodes": 0, "steps": 0, "grasps": 0, "grasp_fails": 0, "chassis_contacts": 0, "near_wall": 0}


def _with_walls(cfg, enabled: bool):
    return dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, walls=dataclasses.replace(cfg.sim.walls,
                                                                                               enabled=enabled)))


def _pair(cfg, robot, objects, gripper=GripperState.OPEN):
    py = Sim2D(cfg.sim)
    rs = nat.Sim2D(cfg_json(cfg))
    py.reset(robot, list(objects), gripper)
    rs.reset((robot.x, robot.y, robot.yaw), [(o.name, int(o.type), o.pose.x, o.pose.y, o.pose.yaw) for o in objects],
             int(gripper))
    return py, rs


def _assert_same(py, rs, names, where):
    p = py.robot_pose()
    assert rs.robot_pose() == (p.x, p.y, p.yaw), where
    for n in names:
        q = py.object_pose(n)
        assert rs.object_pose(n) == (q.x, q.y, q.yaw), (where, n)
    assert rs.jaw_angle() == py.jaw_angle(), where
    assert rs.gripper_state() == int(py.gripper_state()), where
    assert rs.grasped_object() == py.grasped_object(), where
    assert set(rs.chassis_contacts()) == py.chassis_contacts(), where
    assert rs.time_s == py.time_s, where


def _heuristic(cfg, py, rng) -> Action:
    """物体へ向かって閉じる・運ぶ・放す・押すを多めにした乱数の行動。"""
    if rng.random() < 0.25:
        return Action(int(rng.integers(len(Action))))
    if py.grasped_object() is not None:
        # 運ぶ（壁や他の物体に当てる）。ときどき放す
        return Action.OPEN_GRIPPER if rng.random() < 0.08 else Action(int(rng.choice([1, 1, 1, 3, 4, 2])))
    if py.gripper_state() == GripperState.CLOSED:
        return Action.OPEN_GRIPPER
    names = py.object_names()
    if not names:
        return Action(int(rng.choice([int(a) for a in DRIVE_ACTIONS])))
    rp = py.robot_pose()
    c, s = math.cos(rp.yaw), math.sin(rp.yaw)
    best = None
    for n in names:
        o = py.object_pose(n)
        dx, dy = o.x - rp.x, o.y - rp.y
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        d = math.hypot(lx, ly)
        if best is None or d < best[0]:
            best = (d, lx, ly)
    _, lx, ly = best
    if abs(ly) < jaw_inner_half_gap(cfg.sim, py.jaw_angle(), lx):
        return Action.CLOSE_GRIPPER
    a = math.atan2(ly, lx)
    if a > 0.2:
        return Action.TURN_LEFT
    if a < -0.2:
        return Action.TURN_RIGHT
    return Action.FORWARD


def _run_episode(cfg, py, rs, names, rng, n_steps, latency, gain, where, check_rng=None):
    T = cfg.task.control.action_duration_s
    applied = (0.0, 0.0)
    for k in range(n_steps):
        a = _heuristic(cfg, py, rng)
        if a in (Action.OPEN_GRIPPER, Action.CLOSE_GRIPPER):
            close = a == Action.CLOSE_GRIPPER
            if latency > 0 and applied != (0.0, 0.0):
                py.drive(*applied, latency)
                rs.drive(*applied, latency)
            before = py.grasped_object()
            target = py._object_between_jaws() if close and before is None else None
            py.actuate_gripper(close=close)
            rs.actuate_gripper(close)
            if target is not None:
                _COVER["grasps" if py.grasped_object() is not None else "grasp_fails"] += 1
            applied = (0.0, 0.0)
        else:
            v, w = drive_velocity(cfg.sim.robot, a)
            cmd = (v * gain[0], w * gain[1])
            if latency > 0:
                py.drive(*applied, latency)
                rs.drive(*applied, latency)
            py.drive(*cmd, T - latency)
            rs.drive(*cmd, T - latency)
            applied = cmd
        _COVER["chassis_contacts"] += len(py.chassis_contacts())
        p = py.robot_pose()
        hx, hy = cfg.sim.work_area.size_m[0] / 2, cfg.sim.work_area.size_m[1] / 2
        if cfg.sim.walls.enabled and max(abs(p.x) - hx, abs(p.y) - hy) > -robot_radius(cfg.sim):
            _COVER["near_wall"] += 1
        _assert_same(py, rs, names, (*where, k, a))
        if check_rng is not None:
            check_rng(k)
        _COVER["steps"] += 1
    _COVER["episodes"] += 1


@pytest.fixture(scope="module")
def cfg():
    return load_all()


@pytest.mark.parametrize("walls", [True, False])
@pytest.mark.parametrize("stage", STAGES)
def test_trajectory_matches_pymunk(cfg, stage, walls):
    c = _with_walls(cfg, walls)
    sampler = TaskSampler(c.sim, c.task)
    for seed in range(3):
        ep = sampler.sample(stage, np.random.default_rng(1000 * stage + seed))
        rng = np.random.default_rng(seed + 77 * stage + (5000 if walls else 0))
        latency = (0.0, 0.025, 0.05)[seed]
        gain = (float(rng.uniform(0.9, 1.1)), float(rng.uniform(0.9, 1.1)))
        py, rs = _pair(c, ep.robot, ep.objects)
        names = [o.name for o in ep.objects]
        _assert_same(py, rs, names, (stage, walls, seed, "reset"))
        _run_episode(c, py, rs, names, rng, int(rng.integers(100, 301)), latency, gain, (stage, walls, seed))


def test_grasp_and_contact_coverage(cfg):
    """上の一致の試験が、つかむ・車体の接触の場面を実際に通している（STAGES の試験の後に走る）。"""
    print(f"\n一致: {_COVER}")
    if _COVER["episodes"] == 0:
        pytest.skip("軌跡の試験が走っていない")
    assert _COVER["grasps"] >= 20 and _COVER["chassis_contacts"] >= 20 and _COVER["near_wall"] >= 100, _COVER



@pytest.mark.parametrize("stage", (1, 3, 6, 8))
def test_motor_lag_matches(cfg, stage):
    """v36: 旋回の惰性（motor_lag_s、刻みごとに v += (v_cmd − v) × dt / τ）ありでも軌跡がビット単位で同じ。"""
    sampler = TaskSampler(cfg.sim, cfg.task)
    for seed, tau in enumerate((0.05, 0.1, 0.15)):
        ep = sampler.sample(stage, np.random.default_rng(700 + 10 * stage + seed))
        py, rs = _pair(cfg, ep.robot, ep.objects)
        py.motor_lag_s = tau
        rs.set_motor_lag(tau)
        rng = np.random.default_rng(seed + 31 * stage)
        names = [o.name for o in ep.objects]
        _run_episode(cfg, py, rs, names, rng, 200, (0.0, 0.025, 0.05)[seed], (0.95, 1.05), ("lag", stage, seed))

def test_grasp_fail_matches_with_same_rng():
    """挟み損ね（v21）あり: 同じ乱数の状態から、軌跡も、引いた後の乱数の状態も同じ。"""
    cfg = load_all(overrides=(IMPERFECT,))
    assert cfg.sim.sim2d.grasp_fail.enabled
    before = dict(_COVER)
    for stage in (1, 3, 6, 8):
        for seed in range(3):
            ep = TaskSampler(cfg.sim, cfg.task).sample(stage, np.random.default_rng(300 + 10 * stage + seed))
            py, rs = _pair(cfg, ep.robot, ep.objects)
            py.rng = np.random.Generator(np.random.PCG64(seed + 11 * stage))
            st = py.rng.bit_generator.state["state"]
            rs.set_rng(st["state"], st["inc"])
            names = [o.name for o in ep.objects]
            rng = np.random.default_rng(seed)

            def check_rng(k):
                # 両方から 1 つ引いて比べる（引く数が同じなので、以後も同じ流れのまま）
                assert rs.rng_random() == py.rng.random(), (stage, seed, k)
            _run_episode(cfg, py, rs, names, rng, 150, 0.025, (1.0, 1.0), ("fail", stage, seed), check_rng)
    print(f"\n挟み損ねあり: { {k: _COVER[k] - before[k] for k in _COVER} }")


def test_grasp_fail_is_exercised():
    """挟み損ねが実際に起きる盤面で、乱数の引き方（判定・弾く向き・速さ）まで同じ。"""
    cfg = load_all(overrides=(IMPERFECT,))
    from toy_sweeper.robot_model import gripper_center_x
    x0 = gripper_center_x(cfg.sim)
    n_fail = n_grasp = 0
    for k in range(60):
        r = np.random.default_rng(k)
        # 爪の間で横にずらした LEGO（ずれが大きいほど挟み損ねやすい）
        ly = float(r.uniform(-0.012, 0.012))
        objs = [ObjectSpec("lego0", ObjectType.LEGO, Pose2D(x0, ly, float(r.uniform(-0.3, 0.3))), "red"),
                ObjectSpec("car0", ObjectType.TOY_CAR, Pose2D(x0 + 0.12, -0.05, 0.4), "blue")]
        py, rs = _pair(cfg, Pose2D(0.0, 0.0, 0.0), objs)
        py.rng = np.random.Generator(np.random.PCG64(k))
        st = py.rng.bit_generator.state["state"]
        rs.set_rng(st["state"], st["inc"])
        py.actuate_gripper(close=True)
        rs.actuate_gripper(True)
        if py.grasped_object() is None:
            n_fail += 1
        else:
            n_grasp += 1
        py.drive(0.0, 0.0, 0.5)
        rs.drive(0.0, 0.0, 0.5)
        _assert_same(py, rs, ["lego0", "car0"], k)
        assert rs.rng_random() == py.rng.random(), k
    print(f"\n挟み損ね {n_fail} 回、つかんだ {n_grasp} 回")
    assert n_fail >= 5 and n_grasp >= 5


def test_grasp_fail_without_rng_raises():
    cfg = load_all(overrides=(IMPERFECT,))
    from toy_sweeper.robot_model import gripper_center_x
    objs = [ObjectSpec("lego0", ObjectType.LEGO, Pose2D(gripper_center_x(cfg.sim), 0.0, 0.0), "red")]
    _, rs = _pair(cfg, Pose2D(0.0, 0.0, 0.0), objs)
    with pytest.raises(RuntimeError, match="乱数"):
        rs.actuate_gripper(True)


def test_jaw_motion_per_substep_matches(cfg):
    """爪がサーボの速さで回る途中も同じ（SimWorld と同じく set_gripper の後に 1 刻みずつ drive）。回っている途中の
    開閉の指令（閉じながら開く、開きながら閉じる、対象へ閉じている途中の閉じる）も混ぜ、1 刻みごとに比べる。"""
    from toy_sweeper.robot_model import gripper_center_x
    dt = cfg.sim.sim2d.physics_dt_s
    x0 = gripper_center_x(cfg.sim)
    n_grasp = 0
    for k in range(12):
        r = np.random.default_rng(900 + k)
        # 爪の間に LEGO（ずらす）、周りにミニカーと LEGO
        objs = [ObjectSpec("a", ObjectType.LEGO, Pose2D(x0 + float(r.uniform(-0.01, 0.01)), float(r.uniform(-0.008, 0.008)),
                                                        float(r.uniform(-0.5, 0.5))), "red"),
                ObjectSpec("b", ObjectType.TOY_CAR, Pose2D(x0 + 0.02, 0.05, float(r.uniform(-3, 3))), "blue"),
                ObjectSpec("c", ObjectType.LEGO, Pose2D(x0 - 0.01, -0.035, float(r.uniform(-3, 3))), "green")]
        py, rs = _pair(cfg, Pose2D(0.0, 0.0, 0.0), objs)
        for t in range(120):
            if r.random() < 0.08:
                close = bool(r.random() < 0.6)
                py.set_gripper(close)
                rs.set_gripper(close)
            v, w = (0.0, 0.0) if r.random() < 0.6 else (float(r.uniform(-0.1, 0.1)), float(r.uniform(-1, 1)))
            py.drive(v, w, dt)
            rs.drive(v, w, dt)
            _assert_same(py, rs, ["a", "b", "c"], (k, t))
            n_grasp += py.grasped_object() is not None
    assert n_grasp > 50


def test_slow_jaws_raise(cfg):
    """actuation_time_s の間に回り切らない速さは、Rust 版も例外（設定の検査を通さずに作った設定で）。"""
    slow = dataclasses.replace(cfg, sim=dataclasses.replace(
        cfg.sim, gripper=dataclasses.replace(cfg.sim.gripper, jaw_speed_radps=1.0)))
    rs = nat.Sim2D(cfg_json(slow))
    rs.reset((0.0, 0.0, 0.0), [], int(GripperState.CLOSED))
    with pytest.raises(RuntimeError, match="actuation_time_s"):
        rs.actuate_gripper(False)


def test_errors(cfg):
    rs = nat.Sim2D(cfg_json(cfg))
    with pytest.raises(RuntimeError, match="reset"):
        rs.drive(0.1, 0.0, 0.25)
    with pytest.raises(ValueError, match="重複"):
        rs.reset((0.0, 0.0, 0.0), [("a", 0, 0.2, 0.0, 0.0), ("a", 1, -0.2, 0.0, 0.0)], int(GripperState.OPEN))
    with pytest.raises(ValueError, match="OPEN か CLOSED"):
        rs.reset((0.0, 0.0, 0.0), [], int(GripperState.UNKNOWN))
    with pytest.raises(ValueError, match="OPEN か CLOSED"):
        rs.reset((0.0, 0.0, 0.0), [], 7)
    with pytest.raises(ValueError, match="物体の種類"):
        rs.reset((0.0, 0.0, 0.0), [("a", 5, 0.2, 0.0, 0.0)], int(GripperState.OPEN))
    rs.reset((0.0, 0.0, 0.0), [("a", 0, 0.2, 0.0, 0.0)], int(GripperState.OPEN))
    with pytest.raises(ValueError, match="整数倍"):
        rs.drive(0.1, 0.0, 0.001)
    with pytest.raises(ValueError, match="整数倍"):
        rs.drive(0.1, 0.0, 0.0)
    with pytest.raises(KeyError):
        rs.object_pose("b")


def test_closed_start_and_names(cfg):
    py, rs = _pair(cfg, Pose2D(0.1, -0.2, 0.7), [ObjectSpec("x", ObjectType.TOY_CAR, Pose2D(0.3, 0.1, 1.0), "red")],
                   GripperState.CLOSED)
    _assert_same(py, rs, ["x"], "closed")
    assert rs.object_names() == ["x"]
    py.drive(0.1, 0.5, 1.0)
    rs.drive(0.1, 0.5, 1.0)
    _assert_same(py, rs, ["x"], "closed-drive")


def test_reset_again_matches(cfg):
    """つかんだまま reset し直しても（前の世界を捨てて）、新しい Python 版と同じ軌跡になる。"""
    from toy_sweeper.robot_model import gripper_center_x
    x0 = gripper_center_x(cfg.sim)
    py, rs = _pair(cfg, Pose2D(0.0, 0.0, 0.0), [ObjectSpec("a", ObjectType.LEGO, Pose2D(x0, 0.0, 0.0), "red")])
    py.actuate_gripper(close=True)
    rs.actuate_gripper(True)
    assert rs.grasped_object() == py.grasped_object() == "a"
    ep = TaskSampler(cfg.sim, cfg.task).sample(8, np.random.default_rng(5))
    objs = [(o.name, int(o.type), o.pose.x, o.pose.y, o.pose.yaw) for o in ep.objects]
    rs.reset((ep.robot.x, ep.robot.y, ep.robot.yaw), objs, int(GripperState.CLOSED))
    py = Sim2D(cfg.sim)
    py.reset(ep.robot, list(ep.objects), GripperState.CLOSED)
    names = [o.name for o in ep.objects]
    _assert_same(py, rs, names, "reset")
    _run_episode(cfg, py, rs, names, np.random.default_rng(1), 100, 0.0, (1.0, 1.0), ("again",))


def test_speed(cfg):
    """Stage 8 の盤面で 1 歩（action_duration_s の drive）の速さを比べる（1 スレッド）。"""
    ep = TaskSampler(cfg.sim, cfg.task).sample(8, np.random.default_rng(0))
    T = cfg.task.control.action_duration_s
    out = {}
    for label in ("python", "rust"):
        py, rs = _pair(cfg, ep.robot, ep.objects)
        sim = py if label == "python" else rs
        acts = [(0.1, 0.0), (0.0, 1.0), (0.1, 0.0), (0.0, -1.0), (-0.1, 0.0)]
        n = 100 if label == "python" else 2000
        t0 = time.perf_counter()
        for k in range(n):
            sim.drive(*acts[k % len(acts)], T)
        out[label] = n / (time.perf_counter() - t0)
    print(f"\nSim2D の 1 歩/s（Stage 8、{len(ep.objects)} 物体）: Python {out['python']:.0f}、Rust {out['rust']:.0f}"
          f"（{out['rust'] / out['python']:.1f} 倍）")
    assert out["rust"] > out["python"]
