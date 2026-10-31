"""評価の指標（toy_sweeper.evaluate）。"""
import math

import numpy as np
import pytest

from toy_sweeper.config import load_all
from toy_sweeper.env import ToySweeperEnv
from toy_sweeper.evaluate import TURN_REVERSAL_MIN_RAD, count_turn_reversals, run_episode


def test_turn_reversals_count_sign_changes_of_large_yaw_steps():
    """v36: 1 歩ごとの向きの変化のうち |Δψ| ≥ 0.1 rad の歩だけを並べ、符号が前と変わった回数。"""
    assert TURN_REVERSAL_MIN_RAD == 0.1
    assert count_turn_reversals([0.0, 0.3, 0.6, 0.3, 0.32, 0.0, 0.3]) == 2   # +, +, −, (小), −, + → 2 回
    assert count_turn_reversals([0.0, 0.05, 0.0, 0.05, 0.0]) == 0             # 小さな揺れは数えない
    # ±π をまたいでも向きの差で見る
    assert count_turn_reversals([math.pi - 0.1, -math.pi + 0.1, math.pi - 0.1]) == 1
    assert count_turn_reversals([0.0]) == 0


def test_run_episode_counts_turn_reversals():
    class Wiggle:
        def __init__(self):
            self.k = 0

        def reset(self):
            self.k = 0

        def predict(self, obs):
            self.k += 1
            return 3 if self.k % 4 < 2 else 4    # 左 2 歩、右 2 歩を繰り返す
    cfg = load_all()
    env = ToySweeperEnv(cfg, stage=1)
    r = run_episode(env, Wiggle(), seed=0)
    assert r.steps >= 8 and r.turn_reversals >= r.steps // 2 - 2
