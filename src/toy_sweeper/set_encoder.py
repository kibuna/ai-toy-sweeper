"""物体のスロットを共通の重みで符号化してからまとめる特徴抽出（DeepSets、Zaheer et al. 2017。v16）。

全結合（MlpPolicy）はスロットごとに別の重みを持つので、まれにしか埋まらないスロット（対象 5 個のときだけ埋まる
4 つめの対象スロット）の重みが育たず、そこに値が入ると方策が壊れた（2026-10-17）。
ここでは全スロットに同じ MLP φ を掛け、存在マスクの立ったスロットだけを対象・対象外の組ごとに平均と最大でまとめる。
出力は [主対象・ロボット・ゴールなどの特徴（そのまま）, 対象の平均, 対象の最大, 対象外の平均, 対象外の最大]。
"""
from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .spaces import AREA_DIM, BASE_DIM, EXTRA_DIM, MODE_DIM, SLOT_DIM

HEAD_DIM = BASE_DIM + EXTRA_DIM
# 頭の次元: v26 の cleanup_primary では仕事の 3 次元、v30 の area_size では作業領域の 2 次元が足される
# （spaces.ObservationBuilder.head_dim）
HEAD_DIMS = (HEAD_DIM, HEAD_DIM + MODE_DIM, HEAD_DIM + AREA_DIM, HEAD_DIM + MODE_DIM + AREA_DIM)


class SlotSetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, n_target_slots: int, n_obstacle_slots: int,
                 slot_hidden: tuple[int, ...]):
        # 頭の次元は Observation の次元から決める（保存したモデルの引数を変えずに読めるように）
        head = observation_space.shape[0] - SLOT_DIM * (n_target_slots + n_obstacle_slots)
        assert head in HEAD_DIMS, f"Observation の次元 {observation_space.shape} の頭 {head} が {HEAD_DIMS} のどれでもない"
        assert len(slot_hidden) >= 1
        super().__init__(observation_space, head + 4 * slot_hidden[-1])
        self.head = head
        self.n_t, self.n_o = n_target_slots, n_obstacle_slots
        layers, d = [], SLOT_DIM
        for h in slot_hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.phi = nn.Sequential(*layers)

    @staticmethod
    def _pool(e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """e [B, K, H]、mask [B, K]（0/1）。物体が 1 つも無い組は 0。"""
        m = mask.unsqueeze(-1)
        cnt = m.sum(1)
        mean = (e * m).sum(1) / cnt.clamp(min=1.0)
        mx = e.masked_fill(m == 0, float("-inf")).max(1).values
        mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
        return torch.cat([mean, mx], dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        slots = obs[:, self.head:].reshape(b, self.n_t + self.n_o, SLOT_DIM)
        mask = slots[:, :, -1]   # 存在マスク（spaces.ObservationBuilder._slot_features の最後の要素）
        e = self.phi(slots)
        return torch.cat([obs[:, :self.head], self._pool(e[:, :self.n_t], mask[:, :self.n_t]),
                          self._pool(e[:, self.n_t:], mask[:, self.n_t:])], dim=1)
