from stable_baselines3.common.vec_env import DummyVecEnv

from toy_sweeper.config import load_all
from toy_sweeper.env import ToySweeperEnv
from toy_sweeper.train import ProgressLogCallback, build_model

from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "config" / "experiments"


def test_progress_counts_from_resume_point():
    """--resume では learn が再開時点から total を足して学習するので、進捗の目標と速さも再開時点から数える。"""
    cfg = load_all()
    model = build_model(cfg, DummyVecEnv([lambda: ToySweeperEnv(cfg, stage=1)]))
    model.num_timesteps = 20_000_000
    cb = ProgressLogCallback(1000)
    cb.init_callback(model)
    cb.on_training_start({}, {})
    assert cb.start == 20_000_000
    assert cb.start + cb.total == 20_001_000


def test_init_model_copies_weights_and_uses_current_config(tmp_path):
    """--init: 重みは渡したモデルから、ハイパーパラメータは今の設定から（v18）。"""
    import dataclasses
    import numpy as np
    import pytest
    from toy_sweeper.train import init_model
    cfg = load_all()
    venv = DummyVecEnv([lambda: ToySweeperEnv(cfg, stage=1)])
    src = build_model(cfg, venv)
    src.save(tmp_path / "src.zip")
    cfg2 = dataclasses.replace(cfg, rl=dataclasses.replace(cfg.rl, ppo=dataclasses.replace(cfg.rl.ppo, ent_coef=0.02)))
    m = init_model(cfg2, venv, tmp_path / "src.zip")
    assert m.ent_coef == 0.02
    obs = venv.reset()
    assert np.allclose(m.policy.predict_values(m.policy.obs_to_tensor(obs)[0]).detach().numpy(),
                       src.policy.predict_values(src.policy.obs_to_tensor(obs)[0]).detach().numpy())
    # ネットワークの形が違えば例外
    cfg3 = dataclasses.replace(cfg, rl=dataclasses.replace(cfg.rl, ppo=dataclasses.replace(cfg.rl.ppo, net_arch=(32, 32))))
    with pytest.raises(Exception):
        init_model(cfg3, venv, tmp_path / "src.zip")


def test_start_index():
    import pytest
    from toy_sweeper.train import start_index
    assert start_index((0, 1, 2, 4, 5, 6, 7, 8), 8) == 7
    with pytest.raises(ValueError):
        start_index((0, 1, 2), 3)


def test_lr_schedule_linear_decays_to_zero():
    """rl.ppo.lr_schedule: linear は学習率を learning_rate から 0 へ直線に下げる（SB3 の progress_remaining 1 → 0）。"""
    import dataclasses
    import pytest
    cfg = load_all()
    venv = DummyVecEnv([lambda: ToySweeperEnv(cfg, stage=1)])
    assert build_model(cfg, venv).lr_schedule(0.5) == cfg.rl.ppo.learning_rate  # 既定は constant
    cfg2 = dataclasses.replace(cfg, rl=dataclasses.replace(cfg.rl, ppo=dataclasses.replace(cfg.rl.ppo, lr_schedule="linear")))
    m = build_model(cfg2, venv)
    assert m.lr_schedule(1.0) == pytest.approx(cfg.rl.ppo.learning_rate)
    assert m.lr_schedule(0.25) == pytest.approx(cfg.rl.ppo.learning_rate * 0.25)
    assert m.lr_schedule(0.0) == 0.0
    with pytest.raises(ValueError):
        dataclasses.replace(cfg.rl.ppo, lr_schedule="cosine")


def test_init_model_widens_head_for_cleanup_primary(tmp_path):
    """--init で cleanup_primary（v26、頭に 3 次元）を有効にしたとき: 方策と価値の最初の全結合に 0 の列を足して
    読み込み、読み込んだ直後は元のモデルと同じ出力（新しい 3 次元の値によらない）。Adam の状態も同じ形で広げる。"""
    import dataclasses
    import numpy as np
    import torch
    from toy_sweeper.spaces import BASE_DIM, EXTRA_DIM, MODE_DIM
    from toy_sweeper.train import init_model
    for exp in ((), (EXP / "set_encoder.yaml",)):
        cfg = load_all(overrides=exp)
        cfg = dataclasses.replace(cfg, rl=dataclasses.replace(cfg.rl, ppo=dataclasses.replace(
            cfg.rl.ppo, n_steps=64, batch_size=64, n_epochs=1)))
        venv = DummyVecEnv([lambda: ToySweeperEnv(cfg, stage=6)])
        src = build_model(cfg, venv)
        src.learn(64)          # Adam の状態を作る
        src.save(tmp_path / "src.zip")
        ob = dataclasses.replace(cfg.task.observation, cleanup_primary=True)
        cfg2 = dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, observation=ob))
        venv2 = DummyVecEnv([lambda: ToySweeperEnv(cfg2, stage=6)])
        m = init_model(cfg2, venv2, tmp_path / "src.zip")
        obs = venv.reset()
        at = BASE_DIM + EXTRA_DIM
        for mode in ([0, 0, 0], [0, 1, 0], [1, 0, 1]):
            o2 = np.concatenate([obs[:, :at], np.array([mode], dtype=np.float32), obs[:, at:]], axis=1)
            t1, t2 = src.policy.obs_to_tensor(obs)[0], m.policy.obs_to_tensor(o2)[0]
            assert torch.allclose(src.policy.predict_values(t1), m.policy.predict_values(t2), atol=1e-6)
            assert torch.allclose(src.policy.get_distribution(t1).distribution.logits,
                                  m.policy.get_distribution(t2).distribution.logits, atol=1e-6)
        st = m.policy.optimizer.state_dict()["state"]
        for p_, s_ in zip(m.policy.parameters(), (st[i] for i in range(len(list(m.policy.parameters()))))):
            assert s_["exp_avg"].shape == p_.shape
        # 3 次元以外の違いは例外のまま
        assert MODE_DIM == 3


def test_init_model_widens_head_for_area_size(tmp_path):
    """--init で area_size（v30、頭の終わりに 2 次元）を有効にしたとき: 仕事の 3 次元のある v26 型のモデルから（+2）と、
    どちらも無いモデルから（+3 +2 が続いて入る）、0 の列を足して読み込み、直後は元と同じ出力。他の差は例外。"""
    import dataclasses
    import numpy as np
    import pytest
    import torch
    from toy_sweeper.spaces import AREA_DIM, BASE_DIM, EXTRA_DIM, MODE_DIM
    from toy_sweeper.train import init_model
    small = dict(n_steps=64, batch_size=64, n_epochs=1)
    for src_exp, head_old in (((EXP / "cleanup_guidance.yaml",), BASE_DIM + EXTRA_DIM + MODE_DIM), ((), BASE_DIM + EXTRA_DIM)):
        cfg = load_all(overrides=(EXP / "set_encoder.yaml", *src_exp))
        cfg = dataclasses.replace(cfg, rl=dataclasses.replace(cfg.rl, ppo=dataclasses.replace(cfg.rl.ppo, **small)))
        venv = DummyVecEnv([lambda: ToySweeperEnv(cfg, stage=6)])
        src = build_model(cfg, venv)
        src.learn(64)
        src.save(tmp_path / "src.zip")
        cfg2 = load_all(overrides=(EXP / "set_encoder.yaml", EXP / "cleanup_guidance.yaml", EXP / "free_area.yaml"))
        cfg2 = dataclasses.replace(cfg2, rl=dataclasses.replace(cfg2.rl, ppo=dataclasses.replace(cfg2.rl.ppo, **small)))
        venv2 = DummyVecEnv([lambda: ToySweeperEnv(cfg2, stage=6)])
        m = init_model(cfg2, venv2, tmp_path / "src.zip")
        obs = venv.reset()
        n_new = BASE_DIM + EXTRA_DIM + MODE_DIM + AREA_DIM - head_old
        for fill in (0.0, 1.3):
            o2 = np.concatenate([obs[:, :head_old], np.full((1, n_new), fill, dtype=np.float32), obs[:, head_old:]], axis=1)
            t1, t2 = src.policy.obs_to_tensor(obs)[0], m.policy.obs_to_tensor(o2)[0]
            assert torch.allclose(src.policy.predict_values(t1), m.policy.predict_values(t2), atol=1e-6)
            assert torch.allclose(src.policy.get_distribution(t1).distribution.logits,
                                  m.policy.get_distribution(t2).distribution.logits, atol=1e-6)
    # area_size を足さない設定へ、area_size のあるモデルから読むのは例外（狭める変換は無い）
    m.save(tmp_path / "area.zip")
    cfg3 = load_all(overrides=(EXP / "set_encoder.yaml", EXP / "cleanup_guidance.yaml"))
    with pytest.raises(ValueError):
        init_model(cfg3, DummyVecEnv([lambda: ToySweeperEnv(cfg3, stage=6)]), tmp_path / "area.zip")
