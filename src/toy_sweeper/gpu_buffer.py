"""PPO の更新のときに、rollout を 1 回だけまとめて GPU に送る RolloutBuffer。

SB3 の RolloutBuffer は、ミニバッチごとに CPU で添字を引き、GPU へ送る（10 エポック × 4 ミニバッチ = 40 回）。
ここでは、ミニバッチの並び（np.random.permutation）とデータの値は SB3 と同じまま、平らにした全体を 1 回だけ GPU に置き、
添字を GPU で引く。学習の結果は変わらない（test/test_gpu_buffer.py）。
"""
from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples

_NAMES = ("observations", "actions", "values", "log_probs", "advantages", "returns")


class GpuRolloutBuffer(RolloutBuffer):
    def reset(self) -> None:
        super().reset()
        self._on_device: dict[str, th.Tensor] | None = None

    def get(self, batch_size: int | None = None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)   # SB3 と同じ乱数の引き方
        if not self.generator_ready:
            for name in _NAMES:
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True
        if self._on_device is None:
            self._on_device = {name: th.as_tensor(self.__dict__[name], device=self.device) for name in _NAMES}
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        idx_all = th.as_tensor(indices, device=self.device)
        start = 0
        while start < self.buffer_size * self.n_envs:
            yield self._gpu_samples(idx_all[start:start + batch_size])
            start += batch_size

    def _gpu_samples(self, idx: th.Tensor) -> RolloutBufferSamples:
        d = self._on_device
        return RolloutBufferSamples(d["observations"][idx], d["actions"][idx], d["values"][idx].flatten(),
                                    d["log_probs"][idx].flatten(), d["advantages"][idx].flatten(),
                                    d["returns"][idx].flatten())
