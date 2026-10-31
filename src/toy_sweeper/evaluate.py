"""固定 Seed での Policy 評価。"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

import numpy as np

from .env import ToySweeperEnv

log = logging.getLogger("toy_sweeper.evaluate")

# 旋回の切り返しに数える 1 歩の向きの変化の下限（v36「旋回の惰性」）。前進 1 歩で左右の利得の差から出る
# 向きの変化（約 0.044 rad）より大きく、旋回 1 歩（約 0.27〜0.34 rad）より小さい。他の値と比べていない（未検証）
TURN_REVERSAL_MIN_RAD = 0.1


def count_turn_reversals(yaws: list[float], min_rad: float = TURN_REVERSAL_MIN_RAD) -> int:
    """1 歩ごとの真の向きの列から、|Δψ| ≥ min_rad の歩だけを並べ、Δψ の符号が前と変わった回数。"""
    n, last = 0, 0
    for a, b in zip(yaws, yaws[1:]):
        d = math.remainder(b - a, 2 * math.pi)
        if abs(d) < min_rad:
            continue
        s = 1 if d > 0 else -1
        n += int(last != 0 and s != last)
        last = s
    return n


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    success: bool
    failure: str | None
    truncated: bool
    steps: int
    episode_return: float
    final_goal_distance_m: float
    gripper_toggles: int
    chassis_contact_steps: int    # 対象が車体の形（アームは除く）に触れた Action の数（v19。減点が 0 の設定では 0）
    final_foreign_in_goal: int    # 終わったときのゴールの中の対象外（もともと外）の数（v34）
    final_home_out: int           # 終わったときの置き場から出た対象外の数（v34）
    turn_reversals: int           # 旋回の切り返しの回数（v36、count_turn_reversals）


def run_episode(env: ToySweeperEnv, policy, seed: int, frames: list | None = None) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    policy.reset()
    ret, toggles, chassis = 0.0, 0, 0
    yaws = [env.true_robot_yaw()]
    if frames is not None:
        frames.append(env.render())
    while True:
        a = policy.predict(obs)
        toggles += int(a in (5, 6))
        obs, r, term, trunc, info = env.step(a)
        ret += r
        chassis += int(info["n_chassis_contact"] > 0)
        yaws.append(env.true_robot_yaw())
        if frames is not None:
            frames.append(env.render())
        if term or trunc:
            return EpisodeResult(seed, bool(info["is_success"]), info["failure"], bool(trunc), info["steps"],
                                 float(ret), float(env._progress_distance()), toggles, chassis,
                                 int(info["n_foreign_in_goal"]), int(info["n_home_out"]), count_turn_reversals(yaws))


def evaluate(env: ToySweeperEnv, policy, seeds) -> list[EpisodeResult]:
    out = []
    for s in seeds:
        r = run_episode(env, policy, int(s))
        log.debug("seed %d success=%s steps=%d failure=%s", r.seed, r.success, r.steps, r.failure)
        out.append(r)
    return out


def summarize(results: list[EpisodeResult]) -> dict:
    if not results:
        raise ValueError("評価結果が空")
    succ = np.array([r.success for r in results])
    n = len(results)
    p = succ.mean()
    return {
        "n": n,
        "success_rate": float(p),
        "success_rate_se": float(np.sqrt(p * (1 - p) / n)),
        "mean_steps_success": float(np.mean([r.steps for r in results if r.success])) if succ.any() else float("nan"),
        "mean_return": float(np.mean([r.episode_return for r in results])),
        "mean_final_goal_distance_m": float(np.mean([r.final_goal_distance_m for r in results])),
        "failures": {k: sum(1 for r in results if r.failure == k) for k in sorted({r.failure for r in results if r.failure})},
        "truncated": int(sum(r.truncated for r in results)),
        "mean_gripper_toggles": float(np.mean([r.gripper_toggles for r in results])),
        # 旋回の切り返し（v36）: 1 回あたりと 100 歩あたり
        "mean_turn_reversals": float(np.mean([r.turn_reversals for r in results])),
        "turn_reversals_per_100_steps": float(100 * sum(r.turn_reversals for r in results) / max(1, sum(r.steps for r in results))),
        # 車体で押した割合: 対象が車体に触れた Action / 全 Action（v19）
        "chassis_contact_rate": float(sum(r.chassis_contact_steps for r in results) / max(1, sum(r.steps for r in results))),
        # 成功のうち、ゴールの中の対象外か置き場から出た対象外が残っていた割合（v34。require_cleanup: true では 0）
        "success_with_leftover_rate": (float(np.mean([r.final_foreign_in_goal + r.final_home_out > 0
                                                      for r in results if r.success])) if succ.any() else float("nan")),
    }


def results_to_rows(results: list[EpisodeResult]) -> list[dict]:
    return [asdict(r) for r in results]
