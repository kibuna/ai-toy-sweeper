import pytest

from toy_sweeper.config import ConfigError, load_all, strict_dataclass_from_dict
from dataclasses import dataclass
from pathlib import Path

TEST_CFG = Path(__file__).resolve().parent / "configs"   # テストだけで使う設定の差し替え


@dataclass(frozen=True)
class _Inner:
    a: float
    b: list[float]


@dataclass(frozen=True)
class _Outer:
    inner: _Inner
    n: int


def test_strict_loader_builds_nested():
    o = strict_dataclass_from_dict(_Outer, {"inner": {"a": 1.0, "b": [1, 2]}, "n": 3})
    assert o.inner.a == 1.0 and o.inner.b == (1.0, 2.0) and o.n == 3


def test_missing_key_raises():
    with pytest.raises(ConfigError, match="missing"):
        strict_dataclass_from_dict(_Outer, {"inner": {"a": 1.0, "b": [1]}})


def test_unknown_key_raises():
    with pytest.raises(ConfigError, match="unknown"):
        strict_dataclass_from_dict(_Outer, {"inner": {"a": 1.0, "b": [1]}, "n": 1, "typo": 2})


def test_wrong_type_raises():
    with pytest.raises(ConfigError):
        strict_dataclass_from_dict(_Outer, {"inner": {"a": "x", "b": [1]}, "n": 1})


def test_project_configs_load():
    cfg = load_all()
    assert cfg.sim.work_area.size_m == (0.8, 0.8)
    assert cfg.task.control.action_duration_s == 0.25
    assert cfg.rl.ppo.n_steps == 2048
    assert len(cfg.rl.curriculum.stages) == len(cfg.rl.curriculum.promote_success_rate)


def test_skirt_must_be_lower_than_objects():
    import dataclasses
    from toy_sweeper.config import validate
    cfg = load_all()
    bad_robot = dataclasses.replace(cfg.sim.robot, skirt_bottom_z_m=0.012)
    bad = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, robot=bad_robot))
    with pytest.raises(ConfigError, match="潜る"):
        validate(bad)


def test_override_merges_and_rejects_typos(tmp_path):
    ok = tmp_path / "ok.yaml"
    ok.write_text("rl:\n  ppo:\n    net_arch: [256, 256]\n")
    cfg = load_all(overrides=(ok,))
    assert cfg.rl.ppo.net_arch == (256, 256) and cfg.rl.ppo.n_steps == 2048
    bad = tmp_path / "bad.yaml"
    bad.write_text("rl:\n  ppo:\n    net_archh: [256, 256]\n")
    with pytest.raises(ConfigError, match="net_archh"):
        load_all(overrides=(bad,))


def test_experiment_overrides_load():
    from toy_sweeper.config import CONFIG_DIR
    for f in sorted([*(CONFIG_DIR / "experiments").glob("*.yaml"), *TEST_CFG.glob("*.yaml")]):
        load_all(overrides=(f,))


def test_checkpoint_every_steps_must_cover_one_vec_step():
    """rl.training.checkpoint_every_steps は n_envs 以上（SB3 の save_freq は VecEnv の 1 歩 = n_envs ステップ単位）。"""
    import dataclasses
    from toy_sweeper.config import validate
    cfg = load_all()
    bad = dataclasses.replace(cfg, rl=dataclasses.replace(cfg.rl, training=dataclasses.replace(
        cfg.rl.training, checkpoint_every_steps=cfg.rl.training.n_envs - 1)))
    with pytest.raises(ConfigError, match="checkpoint_every_steps"):
        validate(bad)


@pytest.mark.parametrize("field, value", [
    ("stage9_mistake_probability", -0.1), ("stage9_mistake_probability", 1.5),
    ("stage9_all_delivered_probability", -0.1), ("stage9_all_delivered_probability", 1.01),
    ("stage9_home_out_m", (0.0, 0.1)), ("stage9_home_out_m", (0.1, 0.05)), ("stage9_home_out_m", (0.01,)),
    ("stage9_wedged_probability", -0.1), ("stage9_wedged_probability", 1.01),
])
def test_stage9_sampling_ranges(field, value):
    """Stage 9（v24）の設定の範囲: 確率は 0〜1、置き場の外の距離は 0 < 下限 <= 上限。"""
    import dataclasses
    from toy_sweeper.config import validate
    cfg = load_all()
    sp = dataclasses.replace(cfg.task.sampling, **{field: value})
    bad = dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, sampling=sp))
    with pytest.raises(ConfigError, match=field):
        validate(bad)


def test_cleanup_experiment_curriculum():
    """config/experiments/cleanup.yaml（v24）: Curriculum の最後が Stage 9。"""
    from toy_sweeper.config import CONFIG_DIR
    cfg = load_all(overrides=(CONFIG_DIR / "experiments" / "cleanup.yaml",))
    cc = cfg.rl.curriculum
    assert cc.stages == (0, 1, 2, 4, 5, 6, 7, 8, 9) and len(cc.promote_success_rate) == len(cc.stages)
    base = load_all().rl.curriculum
    assert cc.promote_success_rate[:-1] == base.promote_success_rate
    # train_ppo.py --start-stage 9: Curriculum の最後から始め、復習は Stage 0〜8（ToySweeperEnv が受け付ける）
    from toy_sweeper.env import ToySweeperEnv
    from toy_sweeper.train import start_index
    i = start_index(cc.stages, 9)
    assert i == len(cc.stages) - 1
    env = ToySweeperEnv(cfg, stage=9)
    env.set_curriculum(cc.stages[i], cc.stages[:i], cc.rehearsal_probability)
    stages = {env.reset(seed=s)[1]["stage"] for s in range(60)}
    assert 9 in stages and len(stages) >= 3, stages


def test_edge_distance_must_be_center_or_corners(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("task:\n  observation:\n    edge_distance: corner\n")
    with pytest.raises((ConfigError, ValueError), match="edge_distance"):
        load_all(overrides=(bad,))
    assert load_all().task.observation.edge_distance == "center"


def test_free_area_validation(tmp_path):
    """v30: work_area_size_range_m は [] か [[幅の下限, 上限], [奥行きの下限, 上限]]。壁ありとは一緒に使えない。
    乱数のゴールの上限は、いちばん狭い作業領域より小さい。"""
    import dataclasses
    import pytest
    from toy_sweeper.config import ConfigError, validate
    from pathlib import Path
    cfg = load_all(overrides=(Path(__file__).resolve().parents[1] / "config" / "experiments" / "free_area.yaml",))
    validate(cfg)

    def with_(sim=None, **sp):
        c = dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, sampling=dataclasses.replace(cfg.task.sampling, **sp)))
        return dataclasses.replace(c, sim=sim) if sim is not None else c
    walls_on = dataclasses.replace(cfg.sim, walls=dataclasses.replace(cfg.sim.walls, enabled=True))
    for bad in (with_(sim=walls_on), with_(work_area_size_range_m=((0.6, 1.0),)),
                with_(work_area_size_range_m=((1.0, 0.6), (0.6, 1.0))),
                with_(work_area_size_range_m=((0.2, 1.0), (0.6, 1.0)))):
        with pytest.raises(ConfigError):
            validate(bad)
    neg = dataclasses.replace(cfg.sim, walls=dataclasses.replace(cfg.sim.walls, out_of_area_margin_m=-0.1))
    with pytest.raises(ConfigError):
        validate(with_(sim=neg))


def test_failure_must_not_beat_staying():
    """v30 の失敗から: 壁なし（外へ出ると失敗）で failure_penalty が止まり続けの価値 time_penalty / (1 − γ) より軽いと、
    失敗で終わらせる方が得になる。その組み合わせは設定の確かめで止める。壁ありで失敗に届かないなら問わない。"""
    import dataclasses
    from pathlib import Path
    from toy_sweeper.config import validate
    exp = Path(__file__).resolve().parents[1] / "config" / "experiments"
    ok = load_all(overrides=(exp / "efficient.yaml", exp / "free_area.yaml"))
    validate(ok)
    v30 = dataclasses.replace(ok, task=dataclasses.replace(ok.task, reward=dataclasses.replace(ok.task.reward, failure_penalty=5.0)))
    with pytest.raises(ConfigError, match="failure_penalty"):
        validate(v30)
    walls = load_all(overrides=(exp / "efficient.yaml",))
    assert walls.task.reward.failure_penalty == 5.0
    validate(walls)


def test_steps_x2_restores_time_budget():
    """v33: 8 Hz の歩数の上限は 4 Hz の基準（300 + 150 / 個 × 0.25 s = 75 s + 37.5 s / 個）と同じ時間になる 2 倍。"""
    from pathlib import Path
    from toy_sweeper.config import validate
    exp = Path(__file__).resolve().parents[1] / "config" / "experiments"
    base = load_all()
    cfg = load_all(overrides=(exp / "robot_8hz.yaml", exp / "steps_x2.yaml"))
    validate(cfg)
    for k in ("max_steps", "max_steps_per_extra_target"):
        assert getattr(cfg.task.episode, k) * cfg.task.control.action_duration_s == \
            getattr(base.task.episode, k) * base.task.control.action_duration_s


def test_leftover_penalty_must_not_beat_finishing():
    """v34: 3 個残した成功（success_bonus − 3 × leftover_penalty）が止まり続けの価値より低い減点は止める。"""
    import dataclasses
    from pathlib import Path
    from toy_sweeper.config import validate
    exp = Path(__file__).resolve().parents[1] / "config" / "experiments"
    ok = load_all(overrides=(exp / "efficient.yaml", TEST_CFG / "no_push_in.yaml"))
    validate(ok)
    for bad in (10.0, -1.0):
        c = dataclasses.replace(ok, task=dataclasses.replace(ok.task, reward=dataclasses.replace(ok.task.reward, leftover_penalty=bad)))
        with pytest.raises(ConfigError, match="leftover_penalty"):
            validate(c)


@pytest.mark.parametrize("value", [(-0.01, 0.1), (0.1, 0.05), (0.004, 0.1), (0.1,)])
def test_motor_lag_range(value):
    """v36: motor_lag_s は [下限, 上限] で 0 <= 下限 <= 上限、上限 > 0 なら下限 >= physics_dt_s（dt / τ > 1 で振れる）。"""
    import dataclasses
    from toy_sweeper.config import validate
    cfg = load_all()
    rwe = dataclasses.replace(cfg.sim.real_world_effects, motor_lag_s=value)
    bad = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, real_world_effects=rwe))
    with pytest.raises(ConfigError, match="motor_lag_s"):
        validate(bad)


def test_snapshots_of_used_policies_load():
    """鍵を足したら、実機で使う方策の config_snapshot が読める。"""
    from toy_sweeper.config import CONFIG_DIR
    snap = CONFIG_DIR.parent / "checkpoints" / "ppo_v32_s0_x2" / "config_snapshot"
    assert load_all(config_dir=snap).sim.real_world_effects.motor_lag_s == (0.0, 0.0)
