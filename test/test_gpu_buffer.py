"""GpuRolloutBuffer と分布の検査を外すことが、学習の結果（重み）を変えない。"""
import dataclasses

import pytest
import torch

from toy_sweeper.config import load_all
from toy_sweeper.native import NativeVecEnv
from toy_sweeper.train import build_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU が要る")


def _train(fast: bool) -> dict:
    cfg = load_all()
    rl = cfg.rl
    cfg = dataclasses.replace(cfg, rl=dataclasses.replace(
        rl, ppo=dataclasses.replace(rl.ppo, n_steps=16, batch_size=512, features="set"),
        training=dataclasses.replace(rl.training, n_envs=64, device="cuda", env_backend="native", env_threads=4,
                                     fast_ppo=fast)))
    torch.distributions.Distribution.set_default_validate_args(not fast)
    try:
        model = build_model(cfg, NativeVecEnv(cfg, 64, 0, 4, 7))
        model.learn(64 * 16 * 3)
    finally:
        torch.distributions.Distribution.set_default_validate_args(True)
    return {k: v.detach().cpu() for k, v in model.policy.state_dict().items()}


def test_fast_ppo_gives_identical_weights():
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        a, b = _train(False), _train(True)
    finally:
        torch.use_deterministic_algorithms(False)
    assert all(torch.equal(a[k], b[k]) for k in a)
