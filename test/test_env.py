import math

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from toy_sweeper.config import load_all
from toy_sweeper.env import ToySweeperEnv
from toy_sweeper.geometry import Pose2D
from toy_sweeper.robot_model import front_x, gripper_center_x
from toy_sweeper.sim2d import ObjectSpec
from toy_sweeper.spaces import Action, GripperState, ObjectType
from toy_sweeper.task import EpisodeSpec


@pytest.fixture
def cfg():
    return load_all()


def _ep(cfg, robot, obj_pose=None, stage=2, reach=None, others=()):
    from toy_sweeper.geometry import Rect
    goal = Rect((0.3, 0.0), (0.2, 0.3))   # テスト用のゴール（右端の中央）
    if stage == 0:
        return EpisodeSpec(0, robot, (), (), reach, "test", goal)
    obj = ObjectSpec("target", ObjectType.LEGO, obj_pose, "red")
    extra = tuple(ObjectSpec(f"o{i}", ObjectType.LEGO, p, "blue") for i, p in enumerate(others))
    return EpisodeSpec(stage, robot, (obj, *extra), ("target",), None, "test", goal)


@pytest.mark.parametrize("stage", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_check_env(cfg, stage):
    check_env(ToySweeperEnv(cfg, stage=stage), warn=True)


def test_seed_reproducible(cfg):
    env = ToySweeperEnv(cfg, stage=6)
    o1, _ = env.reset(seed=123)
    acts = [1, 3, 1, 1, 4, 1]
    r1 = [env.step(a)[1] for a in acts]
    o2, _ = env.reset(seed=123)
    r2 = [env.step(a)[1] for a in acts]
    assert np.allclose(o1, o2) and np.allclose(r1, r2)


def test_success_inside_goal(cfg):
    env = ToySweeperEnv(cfg, stage=2)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(0.0, 0.0, 0.0), Pose2D(0.3, 0.0, 0.0))})
    _, r, term, trunc, info = env.step(Action.STOP)
    assert term and not trunc and info["is_success"]
    # v9: 形のつく項は γΦ(s′) − Φ(s)、終端で Φ(s′) = 0。開始時から配達済みの分（+delivery_bonus）は終端で差し引かれる
    rw = cfg.task.reward
    assert r == pytest.approx(rw.success_bonus - rw.time_penalty - rw.delivery_bonus, abs=1e-3)


def test_push_toward_goal_gives_progress(cfg):
    env = ToySweeperEnv(cfg, stage=1)
    x0 = gripper_center_x(cfg.sim)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(-0.1, 0.0, 0.0), Pose2D(-0.1 + x0, 0.0, math.pi / 2), stage=1)})
    total = 0.0
    for _ in range(10):
        _, r, term, _, info = env.step(Action.FORWARD)
        total += info["reward_terms"]["progress"]
    assert total > 0.5


def test_wall_stops_robot_and_object(cfg):
    # 作業領域の端は壁。壁に向かって進み続けても、ロボットも押した LEGO も外に出ない
    env = ToySweeperEnv(cfg, stage=2)
    half = cfg.sim.work_area.size_m[0] / 2
    x0 = gripper_center_x(cfg.sim)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(-0.2, 0.0, math.pi), Pose2D(-0.2 - x0, 0.0, 0.0))})
    for _ in range(40):   # 0.8 m 分の前進指令
        _, _, term, _, info = env.step(Action.FORWARD)
        assert not term, info
    r, o = env.backend.robot_pose(), env.backend.object_pose("target")
    assert -half < o.x < -half + 0.03 and -half < r.x
    assert abs(r.yaw - math.pi) < 0.3


def test_redundant_gripper_penalized(cfg):
    env = ToySweeperEnv(cfg, stage=2)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(-0.2, 0.0, 0.0), Pose2D(0.0, 0.2, 0.0))})
    _, _, _, _, info = env.step(Action.OPEN_GRIPPER)  # 初期状態は OPEN
    assert info["reward_terms"]["redundant_gripper"] == pytest.approx(-cfg.task.reward.redundant_gripper_penalty)
    _, _, _, _, info = env.step(Action.CLOSE_GRIPPER)
    assert info["reward_terms"]["redundant_gripper"] == 0.0


def test_reach_stage_success(cfg):
    env = ToySweeperEnv(cfg, stage=0)
    gx = gripper_center_x(cfg.sim)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(0.0, 0.0, 0.0), stage=0, reach=(gx + 0.01, 0.0))})
    _, _, term, _, info = env.step(Action.STOP)
    assert term and info["is_success"]


def test_truncation(cfg):
    env = ToySweeperEnv(cfg, stage=2)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(-0.2, 0.0, 0.0), Pose2D(0.0, 0.2, 0.0))})
    for i in range(cfg.task.episode.max_steps):
        _, _, term, trunc, _ = env.step(Action.STOP)
        assert not term
    assert trunc


def test_render_rgb(cfg):
    env = ToySweeperEnv(cfg, stage=2, render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8


def test_set_stage(cfg):
    env = ToySweeperEnv(cfg, stage=0)
    env.set_stage(2)
    _, info = env.reset(seed=0)
    assert info["stage"] == 2
    with pytest.raises(ValueError):
        env.set_stage(10)


def test_non_target_moved_fails(cfg):
    # non_target_failure を有効にした（v18 まで）とき、対象外の LEGO を正面に置いて押し続けると、許容値を超えた時点で失敗
    import dataclasses
    cfg = dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, reward=dataclasses.replace(
        cfg.task.reward, non_target_failure=True)))
    env = ToySweeperEnv(cfg, stage=6)
    x0 = gripper_center_x(cfg.sim)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(-0.2, -0.2, 0.0), Pose2D(0.0, 0.2, 0.0), stage=6,
                                              others=(Pose2D(-0.2 + x0, -0.2, 0.0),))})
    penalized = False
    for _ in range(40):
        _, r, term, _, info = env.step(Action.FORWARD)
        if term:
            break
        penalized |= info["reward_terms"]["non_target"] < 0   # 途中では移動の減点が出る（終端では Φ を 0 にするので戻る）
    assert term and info["failure"] == "non_target_moved"
    assert env.backend.object_pose("o0").x - (-0.2 + x0) > cfg.task.reward.non_target_tolerance_m
    assert penalized


def test_observation_includes_obstacles(cfg):
    env = ToySweeperEnv(cfg, stage=6)
    obs, _ = env.reset(seed=0)
    d = env.obs_builder.decode(obs)
    assert len(d["obstacles"]) == min(len(env.episode.objects) - 1, cfg.task.observation.n_obstacle_slots)


def test_object_already_in_goal_may_move_inside_goal(cfg):
    # ゴール（x 0.2〜0.4）に入っている対象外 LEGO は、ゴール内で押しても移動距離 0 とみなす（減点も失敗もしない）
    env = ToySweeperEnv(cfg, stage=6)
    x0 = gripper_center_x(cfg.sim)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(0.23 - x0, 0.0, 0.0), Pose2D(-0.2, 0.25, 0.0), stage=6,
                                              others=(Pose2D(0.23, 0.0, 0.0),))})
    for _ in range(6):   # 12 cm 分
        _, _, term, _, info = env.step(Action.FORWARD)
        assert not term, info
        assert info["reward_terms"]["non_target"] == 0.0
    assert env.backend.object_pose("o0").x > 0.23 + 0.05


def _with_effects(cfg, **kw):
    import dataclasses
    rwe = dataclasses.replace(cfg.sim.real_world_effects, **kw)
    return dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, real_world_effects=rwe))


def test_command_latency_continues_previous_command(cfg):
    c = _with_effects(cfg, command_latency_s=(0.1, 0.1))
    env = ToySweeperEnv(c, stage=2)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(-0.2, -0.2, 0.0), Pose2D(0.0, 0.2, 0.0))})
    assert env.latency_s == pytest.approx(0.1)
    env.step(Action.FORWARD)   # 最初の 0.1 s は停止のまま
    x1 = env.backend.robot_pose().x
    # 発進・停止の加速時間（駆動力の上限による約 8 ms）の分だけ 1 mm 未満ずれる
    assert x1 - (-0.2) == pytest.approx(cfg.sim.robot.linear_speed_mps * 0.15, abs=1e-3)
    env.step(Action.STOP)      # 停止が届くまで 0.1 s 進む
    assert env.backend.robot_pose().x - x1 == pytest.approx(cfg.sim.robot.linear_speed_mps * 0.1, abs=1e-3)


def test_wheel_asymmetry_curves(cfg):
    c = _with_effects(cfg, wheel_gain=(0.9, 0.9))
    env = ToySweeperEnv(c, stage=2)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(-0.2, -0.2, 0.0), Pose2D(0.0, 0.2, 0.0))})
    env.wheel_gain = (0.9, 1.1)  # 右が速い → 左へ曲がる
    for _ in range(4):
        env.step(Action.FORWARD)
    assert env.backend.robot_pose().yaw > 0.05


def test_obs_noise_does_not_touch_reward(cfg):
    c = _with_effects(cfg, obs_position_noise_m=0.01, obs_yaw_noise_rad=0.1)
    ep = _ep(cfg, Pose2D(-0.2, -0.2, 0.0), Pose2D(0.0, 0.2, 0.0))
    noisy, clean = ToySweeperEnv(c, stage=2), ToySweeperEnv(cfg, stage=2)
    o1, _ = noisy.reset(seed=0, options={"episode": ep})
    o2, _ = clean.reset(seed=0, options={"episode": ep})
    assert not np.allclose(o1, o2)
    r1 = noisy.step(Action.FORWARD)[1]
    r2 = clean.step(Action.FORWARD)[1]
    assert r1 == pytest.approx(r2)


def test_replay_wall_penetration_case(cfg):
    # 2026-10-16 に学習中の例外（LEGO が壁の外へ）を起こした盤面と行動列の再生。test/data/wall_penetration_repro.txt
    import ast
    from pathlib import Path
    from toy_sweeper.geometry import Rect  # noqa: F401（eval で使う）
    txt = (Path(__file__).parent / "data" / "wall_penetration_repro.txt").read_text()
    ep = eval(txt.split("\n")[0][len("episode: "):].replace("<ObjectType.LEGO: 0>", "ObjectType.LEGO"))
    acts = ast.literal_eval(txt.split("actions: ")[1].strip())
    env = ToySweeperEnv(cfg, stage=ep.stage)
    env.reset(seed=0, options={"episode": ep})
    half = cfg.sim.work_area.size_m[0] / 2
    for a in acts:
        env.step(a)
        o = env.backend.object_pose("obj0")
        assert max(abs(o.x), abs(o.y)) + cfg.sim.objects["lego"].size_m[1] / 2 < half + 0.003


def test_non_target_pushed_into_goal_is_penalized_not_failed(cfg):
    # v9: 対象外をゴール（x 0.2〜0.4）へ押し込んでも失敗にはならない。入ったら減点（取り出せば戻る）
    env = ToySweeperEnv(cfg, stage=6)
    x0 = gripper_center_x(cfg.sim)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(0.16 - x0, 0.0, 0.0), Pose2D(-0.2, -0.25, 0.0), stage=6,
                                              others=(Pose2D(0.16, 0.0, 0.0),))})
    entered = False
    for _ in range(10):
        _, _, term, _, info = env.step(Action.FORWARD)
        assert not term
        if info["n_foreign_in_goal"] == 1 and not entered:
            entered = True
            assert info["reward_terms"]["foreign_in_goal"] < 0
    assert entered


def test_target_with_non_target_in_goal_is_not_success(cfg):
    # v9: 対象と一緒に対象外も運び込んだら、対象外を出すまで成功にならない（エピソードは続く）
    env = ToySweeperEnv(cfg, stage=6)
    x0 = gripper_center_x(cfg.sim)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(0.17 - x0, 0.0, 0.0), Pose2D(0.17, 0.0, math.pi / 2), stage=6,
                                              others=(Pose2D(0.17 + 0.0158 + 0.002, 0.0, math.pi / 2),))})
    for _ in range(15):
        _, _, term, _, info = env.step(Action.FORWARD)
        if info["n_delivered"] == 1 and info["n_foreign_in_goal"] == 1:
            break
    assert info["n_delivered"] == 1 and info["n_foreign_in_goal"] == 1
    assert not term and not info["is_success"]


def _multi_ep(cfg, robot, target_poses, others=()):
    from toy_sweeper.geometry import Rect
    goal = Rect((0.3, 0.0), (0.2, 0.3))
    objs = tuple(ObjectSpec(f"t{i}", ObjectType.LEGO, p, "red") for i, p in enumerate(target_poses))
    objs += tuple(ObjectSpec(f"o{i}", ObjectType.LEGO, p, "blue") for i, p in enumerate(others))
    return EpisodeSpec(7, robot, objs, tuple(f"t{i}" for i in range(len(target_poses))), None, "test", goal)


def test_push_two_targets_together(cfg):
    # 並んだ赤い LEGO 2 個をまとめてゴールへ押すと、1 個入るごとに delivery、全部入って成功
    env = ToySweeperEnv(cfg, stage=7)
    x0 = gripper_center_x(cfg.sim)
    t = (Pose2D(0.1, 0.0, math.pi / 2), Pose2D(0.1 + 0.0158 + 0.001, 0.0, math.pi / 2))
    env.reset(seed=0, options={"episode": _multi_ep(cfg, Pose2D(0.1 - x0, 0.0, 0.0), t)})
    seen = set()
    for _ in range(20):
        _, _, term, _, info = env.step(Action.FORWARD)
        seen.add(info["n_delivered"])
        if term:
            break
    assert term and info["is_success"] and info["n_delivered"] == 2


def test_multi_target_obs_has_other_targets(cfg):
    env = ToySweeperEnv(cfg, stage=7)
    obs, _ = env.reset(seed=3)
    d = env.obs_builder.decode(obs)
    n_t = len(env.episode.target_names)
    assert d["n_remaining"] == n_t
    assert len(d["other_targets"]) == min(n_t - 1, cfg.task.observation.n_target_slots)
    assert env.max_steps() == cfg.task.episode.max_steps + cfg.task.episode.max_steps_per_extra_target * (n_t - 1)


def test_rehearsal_mixes_previous_stages(cfg):
    env = ToySweeperEnv(cfg, stage=0)
    env.set_curriculum(7, (1, 2), 0.5)
    stages = [env.reset(seed=s)[1]["stage"] for s in range(200)]
    assert set(stages) == {1, 2, 7} and 0.3 < stages.count(7) / 200 < 0.7
    env.set_stage(2)
    assert {env.reset(seed=s)[1]["stage"] for s in range(20)} == {2}
    with pytest.raises(ValueError):
        env.set_curriculum(7, (), 0.3)


def test_replay_grasped_object_squeeze_case(cfg):
    # 2026-10-16 の 2 件目: 挟んだ LEGO が隣の LEGO を壁に押し付け、弾き出して壁をすり抜けさせた盤面と行動列の再生
    import ast
    from pathlib import Path
    from toy_sweeper.geometry import Rect  # noqa: F401（eval で使う）
    txt = (Path(__file__).parent / "data" / "wall_penetration_repro2.txt").read_text()
    spec = txt.split("\n")[0][len("episode: "):].replace("<ObjectType.LEGO: 0>", "ObjectType.LEGO").replace(
        "<ObjectType.TOY_CAR: 1>", "ObjectType.TOY_CAR")
    ep = eval(spec)
    acts = ast.literal_eval(txt.split("actions: ")[1].strip())
    env = ToySweeperEnv(cfg, stage=ep.stage)
    env.reset(seed=0, options={"episode": ep})
    for a in acts:
        _, _, term, trunc, _ = env.step(a)   # 壁の外に出れば RuntimeError
        if term or trunc:
            break


def test_near_goal_shaping_penalizes_pushing_non_target_toward_goal(cfg):
    import dataclasses
    rw = dataclasses.replace(cfg.task.reward, non_target_near_goal_per_m=20.0)
    c = dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, reward=rw))
    env = ToySweeperEnv(c, stage=6)
    x0 = gripper_center_x(cfg.sim)
    # 対象外の LEGO（x 0.13）をゴール（x 0.2 から）へ押す。帯（10 cm）の中なので減点が出る
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(0.13 - x0, 0.0, 0.0), Pose2D(-0.2, -0.25, 0.0), stage=6,
                                              others=(Pose2D(0.13, 0.0, 0.0),))})
    _, _, _, _, info = env.step(Action.FORWARD)
    assert info["reward_terms"]["non_target_near_goal"] < 0


def test_robot_motion_observation(cfg):
    # v9: 直前 1 ステップの移動。何もない所で前進すれば前に約 2 cm、旋回なら向きだけ変わる
    env = ToySweeperEnv(cfg, stage=2)
    env.reset(seed=0, options={"episode": _ep(cfg, Pose2D(-0.2, -0.2, 0.7), Pose2D(0.0, 0.2, 0.0))})
    obs, *_ = env.step(Action.FORWARD)
    m = env.obs_builder.decode(obs)["robot_motion"]
    step = cfg.sim.robot.linear_speed_mps * cfg.task.control.action_duration_s
    assert m[0] == pytest.approx(step, rel=0.05) and abs(m[1]) < 1e-3 and abs(m[2]) < 1e-3
    obs, *_ = env.step(Action.TURN_LEFT)
    m = env.obs_builder.decode(obs)["robot_motion"]
    assert abs(m[0]) < 2e-3 and m[2] == pytest.approx(cfg.sim.robot.angular_speed_radps * cfg.task.control.action_duration_s, rel=0.05)
    obs, *_ = env.step(Action.STOP)
    # 旋回の直後の停止は、減速（駆動トルクの上限による約 8 ms）の間に少し回る
    assert np.allclose(env.obs_builder.decode(obs)["robot_motion"], 0.0, atol=0.02)


def test_observation_error_carries_episode_and_actions(cfg, monkeypatch):
    """Observation の値域の例外に、再生に要る盤面と行動の列を付ける（2026-10-17、v16 s0 の例外は付いておらず再現できなかった）。"""
    env = ToySweeperEnv(cfg, stage=6)
    env.reset(seed=5)
    env.step(int(Action.FORWARD))

    def bad(snap):
        raise ValueError("Observation が値域 ±3.0 の外: test")
    monkeypatch.setattr(env.obs_builder, "build", bad)
    with pytest.raises(ValueError, match=r"(?s)値域.*episode: EpisodeSpec.*actions: \[1, 3\]"):
        env.step(int(Action.TURN_LEFT))


def test_standing_still_costs_only_time_penalty(cfg):
    """何も変わらない 1 歩の報酬は時間の罰だけ（v17）。γΦ(s′) − Φ(s) だと (1 − γ)|Φ| が足され、ゴールから遠い
    対象が多いほど止まっているだけで正の報酬になった（2026-10-18、Stage 7 で失敗した Seed の報酬が成功の 2 倍）。"""
    env = ToySweeperEnv(cfg, stage=7)
    env.reset(seed=3)
    assert env._phi["progress"] < -1.0   # ゴールから遠い盤面
    _, r, term, trunc, info = env.step(int(Action.STOP))
    assert not (term or trunc)
    assert r == pytest.approx(-cfg.task.reward.time_penalty, abs=1e-9)


def test_pushing_object_out_of_home_zone_is_penalized_and_blocks_success(cfg):
    """置き場（他のゴール）に入っていた物体を外へ出すと、ポテンシャルの減点が付き、成功にならない（v18）。"""
    from toy_sweeper.geometry import Rect
    goal = Rect((0.3, 0.0), (0.2, 0.3))
    home = Rect((0.0, 0.0), (0.12, 0.12))
    # ロボットは置き場の物体の真後ろで +x を向く。前進すると物体を置き場から押し出す
    robot = Pose2D(-0.12, 0.0, 0.0)
    sorted_ = ObjectSpec("s", ObjectType.LEGO, Pose2D(0.0, 0.0, math.pi / 2), "blue")
    target = ObjectSpec("t", ObjectType.LEGO, Pose2D(0.3, -0.25, 0.0), "red")
    ep = EpisodeSpec(8, robot, (target, sorted_), ("t",), None, "test", goal, zones=(home,))
    env = ToySweeperEnv(cfg, stage=8)
    env.reset(seed=0, options={"episode": ep})
    assert env._n_home_out() == 0
    for _ in range(12):
        _, r, term, trunc, info = env.step(int(Action.FORWARD))
        if env._n_home_out() == 1:
            break
    assert env._n_home_out() == 1 and info["n_home_out"] == 1
    assert info["reward_terms"]["home_out"] == pytest.approx(-cfg.task.reward.home_out_penalty)
    # 対象を直接ゴールに入れても、置き場から出た物体があるので成功にしない
    env.backend._objects["t"][0].position = (0.3, 0.0)
    _, _, term, _, info = env.step(int(Action.STOP))
    assert info["n_delivered"] == 1 and not info["is_success"]


def test_slot_uses_home_zone(cfg):
    """置き場に入っていた物体のスロットは、置き場の縁までの距離を持つ（v18）。"""
    from toy_sweeper.geometry import Rect
    goal = Rect((0.3, 0.0), (0.2, 0.3))
    home = Rect((-0.3, 0.3), (0.2, 0.2))
    sorted_ = ObjectSpec("s", ObjectType.LEGO, Pose2D(-0.3, 0.3, 0.0), "blue")
    target = ObjectSpec("t", ObjectType.LEGO, Pose2D(0.0, -0.2, 0.0), "red")
    ep = EpisodeSpec(8, Pose2D(0.0, 0.0, 0.0), (target, sorted_), ("t",), None, "test", goal, zones=(home,))
    env = ToySweeperEnv(cfg, stage=8)
    obs, _ = env.reset(seed=0, options={"episode": ep})
    o = env.obs_builder.decode(obs)["obstacles"][0]
    assert o["has_home"] and o["goal_sdist"] == pytest.approx(-0.1, abs=1e-6)


def _stage9_cfg(cfg, p=1.0):
    import dataclasses
    sp = dataclasses.replace(cfg.task.sampling, stage9_mistake_probability=p)
    return dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, sampling=sp))


def test_stage9_mistakes_counted_at_start(cfg):
    """Stage 9（v24）: 入れてしまった物体はゴールの対象外、出してしまった物体は置き場から出た物体として開始時に数え、
    開始時は成功の条件を満たさない。置き場の旗も指定に従う（入れてしまった物体と、置き直した対象は置き場なし）。"""
    from toy_sweeper.snapshot import home_zone
    c9 = _stage9_cfg(cfg)
    env = ToySweeperEnv(c9, stage=9)
    seen = {"in": 0, "out": 0, "delivered": 0}
    for seed in range(60):
        env.reset(seed=seed)
        ep = env.episode
        over = dict(ep.home_overrides)
        n_in = sum(v == -1 for k, v in over.items() if k not in ep.target_names)
        n_out = sum(v >= 1 for v in over.values())
        assert n_in + n_out >= 1
        assert env._n_foreign_in_goal() == n_in and env._n_home_out() == n_out
        delivered = bool(set(over) & set(ep.target_names))
        assert env._n_delivered() == (len(ep.target_names) if delivered else 0)
        assert not env._is_success()
        for o in ep.objects:
            if o.name in over:
                want = None if over[o.name] == -1 else (ep.goal, *ep.zones)[over[o.name]]
                assert home_zone(c9.sim, ep, o) == want
        _, _, term, trunc, info = env.step(int(Action.STOP))
        assert info["n_foreign_in_goal"] == n_in and info["n_home_out"] == n_out and not info["is_success"]
        seen["in"] += n_in
        seen["out"] += n_out
        seen["delivered"] += delivered
    assert min(seen.values()) >= 5, seen


def test_stage9_taking_foreign_out_succeeds(cfg):
    """入れてしまった物体（指定: 置き場なし）をゴールから出せば、全対象が入っていれば成功になる（もともと外の扱い）。"""
    from toy_sweeper.geometry import Rect
    goal = Rect((0.3, 0.0), (0.2, 0.3))
    target = ObjectSpec("t", ObjectType.LEGO, Pose2D(0.3, 0.1, 0.0), "red")
    foreign = ObjectSpec("f", ObjectType.LEGO, Pose2D(0.3, -0.05, 0.0), "blue")
    ep = EpisodeSpec(9, Pose2D(-0.2, 0.0, 0.0), (target, foreign), ("t",), None, "test", goal,
                     home_overrides=(("t", -1), ("f", -1)))
    env = ToySweeperEnv(cfg, stage=9)
    env.reset(seed=0, options={"episode": ep})
    assert env._n_foreign_in_goal() == 1 and not env._is_success()
    env.backend._objects["f"][0].position = (0.0, -0.3)
    _, _, term, _, info = env.step(int(Action.STOP))
    assert term and info["is_success"] and info["n_foreign_in_goal"] == 0


def test_started_in_goal_is_home_zone_goal(cfg):
    """「もともとゴールの中」（ゴールの対象外の判定）は「置き場が今のゴール」と同じ（Stage 0〜8 で、指定がないとき）。"""
    from toy_sweeper.snapshot import inside_goal
    for stage in (6, 7, 8):
        env = ToySweeperEnv(cfg, stage=stage)
        n_in = 0
        for seed in range(40):
            env.reset(seed=seed)
            for o in env.episode.objects:
                if o.name not in env.episode.target_names:
                    old = inside_goal(cfg.sim, env.episode.goal, o.type, o.pose)
                    assert env._started_in_goal(o) == old
                    n_in += old
        assert n_in > 0, stage


def test_bad_home_overrides_raise(cfg):
    from toy_sweeper.geometry import Rect
    goal = Rect((0.3, 0.0), (0.2, 0.3))
    target = ObjectSpec("t", ObjectType.LEGO, Pose2D(0.0, 0.1, 0.0), "red")
    env = ToySweeperEnv(cfg, stage=9)
    for over, msg in (((("x", -1),), "盤面の物体にない"), ((("t", -1), ("t", -1)), "重複"), ((("t", 1),), "置き場の番号")):
        ep = EpisodeSpec(9, Pose2D(-0.2, 0.0, 0.0), (target,), ("t",), None, "test", goal, home_overrides=over)
        with pytest.raises(ValueError, match=msg):
            env.reset(seed=0, options={"episode": ep})


def test_chassis_contact_penalty_for_targets_only(cfg):
    """対象が車体の形（アームは除く）に触れた Action には減点（v19）。対象外が触れても減点しない。"""
    from toy_sweeper.geometry import Rect
    from toy_sweeper.robot_model import front_x
    goal = Rect((0.3, 0.0), (0.2, 0.3))
    half_w = cfg.sim.robot.body_size_m[1] / 2
    car2 = ObjectSpec("t", ObjectType.TOY_CAR, Pose2D(-0.2 + front_x(cfg.sim) + 0.02, -(half_w - 0.008), 0.0), "red")
    oth2 = ObjectSpec("o", ObjectType.TOY_CAR, Pose2D(-0.2 + front_x(cfg.sim) + 0.02, half_w - 0.008, 0.0), "blue")
    ep = EpisodeSpec(7, Pose2D(-0.2, 0.0, 0.0), (car2, oth2), ("t",), None, "test", goal)
    env = ToySweeperEnv(cfg, stage=7)
    env.reset(seed=0, options={"episode": ep})
    total = 0
    for _ in range(3):
        _, _, _, _, info = env.step(int(Action.FORWARD))
        assert info["reward_terms"]["chassis_contact"] == pytest.approx(-cfg.task.reward.chassis_contact_penalty * info["n_chassis_contact"])
        total += info["n_chassis_contact"]
    assert total >= 2   # 対象（右の角の前）だけが数えられ、1 Action に最大 1


def test_non_target_may_move_far_without_failure(cfg):
    """対象外を 20 cm 超動かしても失敗にしない（v19、「最後に全部片付けばよい」）。"""
    assert not cfg.task.reward.non_target_failure
    env = ToySweeperEnv(cfg, stage=7)
    env.reset(seed=4)
    n = next(o.name for o in env.episode.objects if o.name not in env.episode.target_names)
    p = env.backend.object_pose(n)
    env.backend._objects[n][0].position = (p.x + (0.3 if p.x < 0 else -0.3), p.y)
    _, _, term, _, info = env.step(int(Action.STOP))
    assert info["failure"] is None


def test_no_walls_out_of_area_is_failure(cfg):
    """壁なし（v19nw）: 壁が無く、ロボットが作業領域の外に出たら失敗（例外にしない）。"""
    import dataclasses
    from toy_sweeper.geometry import Rect
    c = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, walls=dataclasses.replace(cfg.sim.walls, enabled=False)))
    goal = Rect((-0.3, 0.0), (0.2, 0.3))
    t = ObjectSpec("t", ObjectType.LEGO, Pose2D(-0.1, 0.2, 0.0), "red")
    ep = EpisodeSpec(2, Pose2D(0.33, 0.0, 0.0), (t,), ("t",), None, "test", goal)   # 右の端の近くで右を向く
    env = ToySweeperEnv(c, stage=2)
    env.reset(seed=0, options={"episode": ep})
    for _ in range(20):
        _, r, term, _, info = env.step(int(Action.FORWARD))
        if term:
            break
    assert term and info["failure"] == "out_of_area"
    assert env.backend.robot_pose().x > c.sim.work_area.size_m[0] / 2


@pytest.mark.parametrize("x, failed", [(0.30 + 0.05, False), (0.30 + 0.12, True)])
def test_free_area_out_of_area_margin(x, failed):
    """v30: 壁なしでは、ロボットか物体の中心が盤面の作業領域（EpisodeSpec.area_size）の縁から out_of_area_margin_m
    を超えて外に出たら失敗。その内側なら失敗にしない。"""
    from pathlib import Path
    from toy_sweeper.geometry import Rect
    c = load_all(overrides=(Path(__file__).resolve().parents[1] / "config" / "experiments" / "free_area.yaml",))
    assert c.sim.walls.out_of_area_margin_m == pytest.approx(0.10)
    goal = Rect((-0.15, 0.0), (0.2, 0.2))
    t = ObjectSpec("t", ObjectType.LEGO, Pose2D(-0.1, 0.25, 0.0), "red")
    o = ObjectSpec("o", ObjectType.LEGO, Pose2D(x, 0.0, 0.0), "blue")    # 0.6 m 四方の右の縁 0.30 の外
    ep = EpisodeSpec(7, Pose2D(0.0, -0.2, 0.0), (t, o), ("t",), None, "test", goal, area_size=(0.6, 0.6))
    env = ToySweeperEnv(c, stage=7)
    env.reset(seed=0, options={"episode": ep})
    _, _, term, _, info = env.step(int(Action.STOP))
    assert info["failure"] == ("out_of_area" if failed else None) and term == failed
    # 観測に大きさを入れる設定で、盤面に大きさが無いのは例外（sim.work_area に黙って落とさない）
    import dataclasses
    with pytest.raises(ValueError, match="area_size"):
        env.reset(seed=0, options={"episode": dataclasses.replace(ep, area_size=None)})


def test_occluded_object_keeps_last_seen_pose(cfg):
    """隠れ（v21）: 仮想のカメラ（手前・斜め上）から見て車体の陰（奥）に入った物体は、動いても最後に見えた位置で観測する。"""
    import dataclasses
    from toy_sweeper.geometry import Rect
    ob = dataclasses.replace(cfg.task.observation, occlusion=dataclasses.replace(cfg.task.observation.occlusion, enabled=True))
    c = dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, observation=ob))
    goal = Rect((0.3, -0.3), (0.2, 0.2))
    t = ObjectSpec("t", ObjectType.LEGO, Pose2D(-0.3, -0.3, 0.0), "red")
    hidden = ObjectSpec("h", ObjectType.LEGO, Pose2D(0.0, 0.07, 0.0), "blue")     # ロボットのすぐ奥（カメラから見て陰）
    ep = EpisodeSpec(7, Pose2D(0.0, 0.0, math.pi / 2), (t, hidden), ("t",), None, "test", goal)
    env = ToySweeperEnv(c, stage=7)
    env.reset(seed=0, options={"episode": ep})
    assert env._observed[0]["h"].xy == pytest.approx((0.0, 0.07))
    env.backend._objects["h"][0].position = (0.02, 0.09)       # 陰の中で動かす
    env.step(int(Action.STOP))
    assert env._observed[0]["h"].xy == pytest.approx((0.0, 0.07))   # 最後に見えた位置のまま
    env.backend._objects["h"][0].position = (0.25, 0.2)        # 陰の外へ
    env.step(int(Action.STOP))
    assert env._observed[0]["h"].xy == pytest.approx((0.25, 0.2), abs=1e-3)
    assert env._observed[0]["t"].xy == pytest.approx((-0.3, -0.3), abs=1e-3)   # 見えている物体はそのまま


def test_cleanup_shaping_potentials():
    """cleanup_shaping（v26）: 全対象が入った後の近づく項は直すべき対象外の接近点まで、運ぶ項に対象外の直すべき距離
    （置き場の物体は角のはみ出し、置き場の無い物体はゴールの中への深さ）を足す。無効なら今までと同じ。"""
    import dataclasses
    from toy_sweeper.snapshot import fix_approach_distance, home_zone, pending_fixes
    base = load_all()
    sp = dataclasses.replace(base.task.sampling, stage9_mistake_probability=1.0, stage9_all_delivered_probability=1.0)
    off = dataclasses.replace(base, task=dataclasses.replace(base.task, sampling=sp))
    on = dataclasses.replace(off, task=dataclasses.replace(off.task, reward=dataclasses.replace(off.task.reward,
                                                                                              cleanup_shaping=True)))
    for seed in range(5):
        e0, e1 = ToySweeperEnv(off, 9), ToySweeperEnv(on, 9)
        e0.reset(seed=seed)
        e1.reset(seed=seed)
        ep = e1.episode
        poses = {o.name: e1.backend.object_pose(o.name) for o in ep.objects}
        assert e1._n_delivered() == len(ep.target_names)
        ret, rem = pending_fixes(ep, on.sim, poses)
        assert ret or rem
        assert e0._approach_distance() == 0.0
        robot = e1.backend.robot_pose()
        specs = {o.name: o for o in ep.objects}
        want = min([fix_approach_distance(on.sim, on.task, ep.goal, home_zone(on.sim, ep, specs[n]), robot, poses[n])
                    for n in ret] + [fix_approach_distance(on.sim, on.task, ep.goal, None, robot, poses[n]) for n in rem])
        assert e1._approach_distance() == pytest.approx(want)
        extra = e1._progress_distance() - e0._progress_distance()
        assert extra > 0
        assert extra == pytest.approx(sum(e1._fix_distance(specs[n], poses[n]) for n in ret + rem))


def test_gripper_toggle_penalty(cfg):
    """v28（効率の検証）: 実際に状態の変わる開閉 1 回あたりの減点 gripper_toggle_penalty（既定 0）。状態の変わらない開閉は
    今までどおり redundant_gripper だけ。"""
    import dataclasses
    on = dataclasses.replace(cfg, task=dataclasses.replace(
        cfg.task, reward=dataclasses.replace(cfg.task.reward, gripper_toggle_penalty=0.05)))
    env = ToySweeperEnv(on, stage=1)
    env.reset(seed=0)
    _, _, _, _, info = env.step(Action.CLOSE_GRIPPER)
    assert info["reward_terms"]["gripper_toggle"] == pytest.approx(-0.05)
    _, _, _, _, info = env.step(Action.CLOSE_GRIPPER)        # もう閉じている
    assert info["reward_terms"]["gripper_toggle"] == 0.0
    _, _, _, _, info = env.step(Action.FORWARD)
    assert info["reward_terms"]["gripper_toggle"] == 0.0
    env = ToySweeperEnv(cfg, stage=1)
    env.reset(seed=0)
    _, _, _, _, info = env.step(Action.CLOSE_GRIPPER)
    assert info["reward_terms"]["gripper_toggle"] == 0.0


def test_speed_gain_scales_both_wheels():
    """v31: real_world_effects.speed_gain は左右共通の速さの倍率（電池の減りなど）。0.5 なら前進 1 歩の距離が半分。
    幅 0（下限 == 上限）なら乱数を引かないので、既定 [1, 1] では盤面と乱数の列は今までと同じ。"""
    import dataclasses
    from toy_sweeper.geometry import Rect
    cfg = load_all()
    goal = Rect((0.3, 0.3), (0.2, 0.2))
    t = ObjectSpec("t", ObjectType.LEGO, Pose2D(-0.3, 0.3, 0.0), "red")
    ep = EpisodeSpec(2, Pose2D(-0.2, -0.2, 0.0), (t,), ("t",), None, "test", goal)

    def forward(c):
        env = ToySweeperEnv(c, stage=2)
        env.reset(seed=0, options={"episode": ep})
        env.step(int(Action.FORWARD))
        return env.backend.robot_pose().x + 0.2, env.np_random.random()
    half = dataclasses.replace(cfg.sim, real_world_effects=dataclasses.replace(cfg.sim.real_world_effects, speed_gain=(0.5, 0.5)))
    d1, r1 = forward(cfg)
    d2, r2 = forward(dataclasses.replace(cfg, sim=half))
    assert d2 == pytest.approx(d1 / 2, rel=0.05) and r1 == r2


def test_targets_only_success_ignores_cleanup(cfg):
    """success.require_cleanup: false（2026-10-22、時間切れの 8 割が直しで止まる）: 全対象が入れば、ゴールに対象外が
    入っていても、置き場から出た対象外があっても成功。true（既定）では今までどおり成功にしない。"""
    import dataclasses
    from toy_sweeper.geometry import Rect
    goal = Rect((0.3, 0.0), (0.2, 0.3))
    target = ObjectSpec("t", ObjectType.LEGO, Pose2D(0.3, 0.1, 0.0), "red")
    foreign = ObjectSpec("f", ObjectType.LEGO, Pose2D(0.3, -0.05, 0.0), "blue")
    ep = EpisodeSpec(9, Pose2D(-0.2, 0.0, 0.0), (target, foreign), ("t",), None, "test", goal,
                     home_overrides=(("t", -1), ("f", -1)))
    assert cfg.task.success.require_cleanup
    loose = dataclasses.replace(cfg, task=dataclasses.replace(
        cfg.task, success=dataclasses.replace(cfg.task.success, require_cleanup=False)))
    for c, want in ((cfg, False), (loose, True)):
        env = ToySweeperEnv(c, stage=9)
        env.reset(seed=0, options={"episode": ep})
        _, _, term, _, info = env.step(int(Action.STOP))
        assert info["n_foreign_in_goal"] == 1 and info["is_success"] == want and term == want


def test_leftover_penalty_at_end(cfg):
    """reward.leftover_penalty（v34）: 終わったときにゴールの中の対象外・置き場から出た対象外 1 個あたりの減点。ポテンシャルの
    foreign_in_goal は終端で戻る（Φ(s′) = 0）ので、全対象が入れば成功（require_cleanup: false）では押し込んでも損がない。
    その分を終わりに残す。終わらない歩と、0（既定）では 0。"""
    import dataclasses
    from toy_sweeper.geometry import Rect
    goal = Rect((0.3, 0.0), (0.2, 0.3))
    target = ObjectSpec("t", ObjectType.LEGO, Pose2D(0.3, 0.1, 0.0), "red")
    foreign = ObjectSpec("f", ObjectType.LEGO, Pose2D(0.3, -0.05, 0.0), "blue")
    ep = EpisodeSpec(9, Pose2D(-0.2, 0.0, 0.0), (target, foreign), ("t",), None, "test", goal,
                     home_overrides=(("t", -1), ("f", -1)))
    assert cfg.task.reward.leftover_penalty == 0.0

    def with_(require_cleanup, penalty):
        t = cfg.task
        return dataclasses.replace(cfg, task=dataclasses.replace(
            t, success=dataclasses.replace(t.success, require_cleanup=require_cleanup),
            reward=dataclasses.replace(t.reward, leftover_penalty=penalty)))
    for c, want in ((with_(False, 10.0), -10.0), (with_(False, 0.0), 0.0)):
        env = ToySweeperEnv(c, stage=9)
        env.reset(seed=0, options={"episode": ep})
        _, r, term, _, info = env.step(int(Action.STOP))
        assert term and info["is_success"] and info["reward_terms"]["leftover"] == want
        assert r == sum(info["reward_terms"].values())
    env = ToySweeperEnv(with_(True, 10.0), stage=9)     # 直すまで終わらない: 歩ごとには減点しない
    env.reset(seed=0, options={"episode": ep})
    _, _, term, _, info = env.step(int(Action.STOP))
    assert not term and info["reward_terms"]["leftover"] == 0.0


def test_motor_lag_drawn_per_episode(cfg):
    """v36: real_world_effects.motor_lag_s をエピソードごとに一様に引き、Sim2D の時定数にする。幅 0 なら乱数を引かない
    （既定 [0, 0] では乱数の列は今までと同じ）。"""
    from toy_sweeper.geometry import Rect
    from toy_sweeper.task import EpisodeSpec
    t = ObjectSpec("t", ObjectType.LEGO, Pose2D(-0.3, 0.3, 0.0), "red")
    ep = EpisodeSpec(2, Pose2D(-0.2, -0.2, 0.0), (t,), ("t",), None, "test", Rect((0.3, 0.3), (0.2, 0.2)))

    def run(c):
        env = ToySweeperEnv(c, stage=2)
        env.reset(seed=0, options={"episode": ep})
        tau = env.backend.motor_lag_s
        env.step(int(Action.FORWARD))
        return tau, env.backend.robot_pose().x + 0.2, env.np_random.random()
    tau0, d0, r0 = run(cfg)
    tau1, d1, r1 = run(_with_effects(cfg, motor_lag_s=(0.1, 0.1)))
    taus = {run(_with_effects(cfg, motor_lag_s=(0.05, 0.15)))[0]}
    assert tau0 == 0.0 and tau1 == 0.1 and r0 == r1 and d1 < 0.8 * d0
    env = ToySweeperEnv(_with_effects(cfg, motor_lag_s=(0.05, 0.15)), stage=2)
    for s in range(20):
        env.reset(seed=s, options={"episode": ep})
        taus.add(env.backend.motor_lag_s)
    assert len(taus) > 10 and all(0.05 <= x <= 0.15 for x in taus)


def test_motor_lag_experiment_only_adds_lag():
    """config/experiments/motor_lag.yaml（v36）は motor_lag_s だけを変える。"""
    import yaml
    from toy_sweeper.config import CONFIG_DIR
    d = yaml.safe_load((CONFIG_DIR / "experiments" / "motor_lag.yaml").read_text())
    assert d == {"sim": {"real_world_effects": {"motor_lag_s": [0.05, 0.15]}}}
