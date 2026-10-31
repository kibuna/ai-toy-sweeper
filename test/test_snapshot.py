"""snapshot.build_snapshot（観測の入力を作る純粋関数）の試験（v9）。

v7 から v8 のリファクタで使った「切り出す前の実装の写し」との比較は、v9 で観測の定義を変えたので外した
（git 履歴の test/test_snapshot.py に残る）。
"""
import math

import numpy as np
import pytest

from toy_sweeper.config import load_all
from toy_sweeper.env import ToySweeperEnv
from toy_sweeper.geometry import Pose2D
from toy_sweeper.robot_model import gripper_center_x
from toy_sweeper.snapshot import build_snapshot, target_held
from toy_sweeper.spaces import Action, GripperState


@pytest.fixture(scope="module")
def cfg():
    return load_all()


def _kw():
    return dict(extra_obstacles=(), obs_age_s=0.0, robot_motion=(0.0, 0.0, 0.0), time_left=1.0)


@pytest.mark.parametrize("stage", [1, 6, 7])
def test_env_uses_build_snapshot(cfg, stage):
    """env の観測は、同じ入力の build_snapshot と同じ（主対象の持ち越しを含む）。"""
    env = ToySweeperEnv(cfg, stage=stage)
    rng = np.random.default_rng(stage)
    obs, _ = env.reset(seed=stage)
    for _ in range(30):
        b = env.backend
        poses = {o.name: b.object_pose(o.name) for o in env.episode.objects}
        snap, _ = build_snapshot(env.episode, cfg.sim, cfg.task, poses, b.robot_pose(), b.gripper_state(),
                                 env._prev_action, env._last_primary, **{**_kw(), "robot_motion": (0.0, 0.0, 0.0),
                                                                         "time_left": (env.max_steps() - env._steps) / env.max_steps()})
        assert snap.n_remaining == env.obs_builder.decode(obs)["n_remaining"]
        obs, _, term, trunc, _ = env.step(int(rng.integers(7)))
        if term or trunc:
            break


def test_displacement_and_started_in_goal(cfg):
    env = ToySweeperEnv(cfg, stage=6)
    env.reset(seed=10)
    ep = env.episode
    poses = {o.name: o.pose for o in ep.objects}
    moved = ep.objects[1].name
    p = poses[moved]
    poses[moved] = Pose2D(p.x + 0.03, p.y, p.yaw)
    snap, _ = build_snapshot(ep, cfg.sim, cfg.task, poses, ep.robot, GripperState.OPEN, Action.STOP, ep.target_names[0],
                             **_kw())
    got = [o for o in snap.obstacles if o.pose == poses[moved]]
    assert len(got) == 1 and got[0].displacement == pytest.approx((0.03, 0.0))


def test_target_held(cfg):
    r = Pose2D(0.0, 0.0, 0.0)
    t = Pose2D(gripper_center_x(cfg.sim), 0.0, 0.0)
    assert target_held(cfg.sim, cfg.task, r, GripperState.CLOSED, t)
    assert not target_held(cfg.sim, cfg.task, r, GripperState.OPEN, t)
    assert not target_held(cfg.sim, cfg.task, r, GripperState.CLOSED, Pose2D(t.x, 0.05, 0.0))


def test_build_snapshot_rejects_bad_input(cfg):
    env = ToySweeperEnv(cfg, stage=7)
    env.reset(seed=0)
    ep = env.episode
    poses = {o.name: o.pose for o in ep.objects}
    missing = {k: v for k, v in poses.items() if k != ep.target_names[0]}
    with pytest.raises(KeyError):
        build_snapshot(ep, cfg.sim, cfg.task, missing, ep.robot, GripperState.OPEN, Action.STOP, ep.target_names[0], **_kw())
    with pytest.raises(ValueError):
        build_snapshot(ep, cfg.sim, cfg.task, poses, ep.robot, GripperState.OPEN, Action.STOP, "not_a_target", **_kw())
    with pytest.raises(ValueError):
        build_snapshot(ep, cfg.sim, cfg.task, poses, ep.robot, GripperState.OPEN, Action.STOP, ep.target_names[0],
                       **{**_kw(), "time_left": 1.5})


@pytest.mark.parametrize("stage", [6, 8, 9])
def test_pending_fixes_match_env_counts(stage):
    """pending_fixes（v26、直すべき物体）の数は env の _n_home_out と _n_foreign_in_goal と同じ。物体が両方に当たる
    （置き場の物体がゴールの中に入った）ときは「戻す」の 1 個（env では両方に数える）。ランダムな行動で盤面を崩して確かめる。"""
    import dataclasses
    import numpy as np
    from toy_sweeper.config import load_all
    from toy_sweeper.env import ToySweeperEnv
    from toy_sweeper.snapshot import pending_fixes
    cfg = load_all()
    if stage == 9:
        sp = dataclasses.replace(cfg.task.sampling, stage9_mistake_probability=1.0)
        cfg = dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, sampling=sp))
    env = ToySweeperEnv(cfg, stage)
    rng = np.random.default_rng(stage)
    seen = 0
    for seed in range(6):
        env.reset(seed=seed)
        ep = env.episode
        for _ in range(60):
            _, _, term, trunc, _ = env.step(int(rng.integers(5)))
            poses = {o.name: env.backend.object_pose(o.name) for o in ep.objects}
            ret, rem = pending_fixes(ep, cfg.sim, poses)
            both = sum(n in ret for n in rem)
            assert not both
            n_ret_and_foreign = sum(1 for n in ret if ep.goal.contains(poses[n].xy)
                                    and not env._started_in_goal(next(o for o in ep.objects if o.name == n)))
            assert len(ret) == env._n_home_out()
            assert len(rem) + n_ret_and_foreign == env._n_foreign_in_goal()
            seen += len(ret) + len(rem)
            if term or trunc:
                break
    if stage == 9:
        assert seen > 0


def _cleanup_cfg(on=True):
    import dataclasses
    cfg = load_all()
    ob = dataclasses.replace(cfg.task.observation, cleanup_primary=on)
    return dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, observation=ob))


def _cleanup_board(cfg):
    from toy_sweeper.sim2d import ObjectSpec
    from toy_sweeper.spaces import ObjectType
    from toy_sweeper.task import EpisodeSpec, goal_rects
    g = goal_rects(cfg.sim)
    objs = (ObjectSpec("t", ObjectType.LEGO, Pose2D(0.0, 0.0, 0.0), "red"),
            ObjectSpec("h", ObjectType.LEGO, Pose2D(-0.3, 0.3, 0.0), "blue"),     # top_left の中から（置き場あり）
            ObjectSpec("f", ObjectType.LEGO, Pose2D(0.0, -0.2, 0.0), "green"))    # どこにも入っていない（置き場なし）
    return EpisodeSpec(6, Pose2D(0.0, 0.0, 0.0), objs, ("t",), None, "top_right", g["top_right"], (g["top_left"],)), g


def test_cleanup_primary_switches_to_pending_fix():
    """cleanup_primary（v26）: 全対象が入っていて直すべき物体があれば、主対象をそれにし、行き先を置き場（戻す）か
    今のゴール（出す）にする。返す主対象（次の last_primary）は対象のまま。"""
    from toy_sweeper.spaces import TaskMode
    cfg = _cleanup_cfg()
    ep, g = _cleanup_board(cfg)
    t_in = Pose2D(0.3, 0.3, 0.0)
    snap_args = (ep, cfg.sim, cfg.task)
    # 戻す: h が置き場の外
    poses = {"t": t_in, "h": Pose2D(-0.12, 0.3, 0.0), "f": Pose2D(0.0, -0.2, 0.0)}
    s, p = build_snapshot(*snap_args, poses, Pose2D(0.0, 0.0, 0.0), GripperState.OPEN, Action.STOP, "t", **_kw())
    assert p == "t" and s.task_mode == TaskMode.RETURN and s.target == poses["h"] and s.destination == g["top_left"]
    assert s.goal == g["top_right"] and [o.pose for o in s.other_targets] == [t_in]
    assert all(o.pose != poses["h"] for o in s.obstacles) and len(s.obstacles) == 1
    # 出す: f がゴールの中
    poses = {"t": t_in, "h": Pose2D(-0.3, 0.3, 0.0), "f": Pose2D(0.35, 0.3, 0.0)}
    s, p = build_snapshot(*snap_args, poses, Pose2D(0.0, 0.0, 0.0), GripperState.OPEN, Action.STOP, "t", **_kw())
    assert p == "t" and s.task_mode == TaskMode.REMOVE and s.target == poses["f"] and s.destination == g["top_right"]
    # 両方: 接近点の近い方（ロボットを h の右に置く。h の接近点は置き場の中心と反対側 = 右）
    poses = {"t": t_in, "h": Pose2D(-0.12, 0.3, 0.0), "f": Pose2D(0.35, 0.3, 0.0)}
    s, _ = build_snapshot(*snap_args, poses, Pose2D(-0.12, 0.25, math.pi), GripperState.OPEN, Action.STOP, "t", **_kw())
    assert s.target == poses["h"]
    s, _ = build_snapshot(*snap_args, poses, Pose2D(0.3, 0.1, 0.0), GripperState.OPEN, Action.STOP, "t", **_kw())
    assert s.target == poses["f"]
    # 直すものが無い: 前の主対象（対象）を持ち越して運ぶ
    poses = {"t": t_in, "h": Pose2D(-0.3, 0.3, 0.0), "f": Pose2D(0.0, -0.2, 0.0)}
    s, p = build_snapshot(*snap_args, poses, Pose2D(0.0, 0.0, 0.0), GripperState.OPEN, Action.STOP, "t", **_kw())
    assert p == "t" and s.task_mode == TaskMode.DELIVER and s.target == t_in and s.destination == g["top_right"]
    # 未配達の対象があれば、直すものがあっても運ぶ
    poses = {"t": Pose2D(0.0, 0.0, 0.0), "h": Pose2D(-0.12, 0.3, 0.0), "f": Pose2D(0.0, -0.2, 0.0)}
    s, _ = build_snapshot(*snap_args, poses, Pose2D(0.0, 0.0, 0.0), GripperState.OPEN, Action.STOP, "t", **_kw())
    assert s.task_mode == TaskMode.DELIVER and s.target == poses["t"]
    # 無効なら、直すものがあっても運ぶ（今までと同じ）
    off = _cleanup_cfg(False)
    poses = {"t": t_in, "h": Pose2D(-0.12, 0.3, 0.0), "f": Pose2D(0.0, -0.2, 0.0)}
    s, p = build_snapshot(ep, off.sim, off.task, poses, Pose2D(0.0, 0.0, 0.0), GripperState.OPEN, Action.STOP, "t", **_kw())
    assert p == "t" and s.task_mode == TaskMode.DELIVER and s.target == t_in and s.destination == s.goal
