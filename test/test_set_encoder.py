import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from toy_sweeper.config import load_all
from toy_sweeper.env import ToySweeperEnv
from toy_sweeper.set_encoder import SlotSetExtractor
from toy_sweeper.spaces import BASE_DIM, EXTRA_DIM, SLOT_DIM
from toy_sweeper.train import build_model

HEAD = BASE_DIM + EXTRA_DIM


@pytest.fixture(scope="module")
def setup():
    cfg = load_all()
    env = ToySweeperEnv(cfg, stage=7)
    obs, _ = env.reset(seed=10006)   # 対象 5 個（対象のスロット 4 つが全部埋まる）
    ext = SlotSetExtractor(env.observation_space, n_target_slots=cfg.task.observation.n_target_slots,
                           n_obstacle_slots=cfg.task.observation.n_obstacle_slots, slot_hidden=(32, 32))
    return cfg, env, torch.as_tensor(obs[None]), ext


def _slot(o, k):
    return slice(HEAD + SLOT_DIM * k, HEAD + SLOT_DIM * (k + 1))


def test_shape(setup):
    _, _, obs, ext = setup
    assert ext(obs).shape == (1, ext.features_dim)


def test_order_of_target_slots_does_not_matter(setup):
    _, _, obs, ext = setup
    o = obs.clone()
    o[0, _slot(o, 0)], o[0, _slot(o, 3)] = obs[0, _slot(obs, 3)], obs[0, _slot(obs, 0)]
    assert torch.allclose(ext(o), ext(obs), atol=1e-6)


def test_empty_slots_are_ignored(setup):
    """存在マスクが 0 のスロットは、中身が何でも特徴に効かない。"""
    _, _, obs, ext = setup
    o = obs.clone()
    o[0, _slot(o, 3)] = 0.0
    o2 = o.clone()
    o2[0, _slot(o2, 3).start:_slot(o2, 3).stop - 1] = 2.5   # マスク（最後の要素）は 0 のまま
    assert torch.allclose(ext(o), ext(o2), atol=1e-6)


def test_targets_and_obstacles_are_pooled_separately(setup):
    """対象のスロットと対象外のスロットを入れ替えると特徴が変わる（同じ集合に混ぜない）。"""
    cfg, _, obs, ext = setup
    nt = cfg.task.observation.n_target_slots
    o = obs.clone()
    o[0, _slot(o, 0)], o[0, _slot(o, nt)] = obs[0, _slot(obs, nt)], obs[0, _slot(obs, 0)]
    assert not torch.allclose(ext(o), ext(obs), atol=1e-4)


def test_build_model_uses_set_extractor(tmp_path):
    p = tmp_path / "set.yaml"
    p.write_text("rl:\n  ppo:\n    features: set\n")
    cfg = load_all(overrides=(p,))
    m = build_model(cfg, DummyVecEnv([lambda: ToySweeperEnv(cfg, stage=7)]))
    assert isinstance(m.policy.features_extractor, SlotSetExtractor)
    m.save(tmp_path / "m.zip")
    from stable_baselines3 import PPO
    obs, _ = ToySweeperEnv(cfg, stage=7).reset(seed=1)
    assert np.array_equal(PPO.load(tmp_path / "m.zip", device="cpu").predict(obs, deterministic=True)[0],
                          m.predict(obs, deterministic=True)[0])
