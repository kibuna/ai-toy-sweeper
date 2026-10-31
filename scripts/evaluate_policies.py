"""固定テスト Seed（config/rl.yaml evaluation）で Policy を評価し、表を保存する。

使い方:
    # 2D シミュレータ
    uv run python scripts/evaluate_policies.py --backend sim2d --stages 1 2 3 --policies random heuristic ppo \
        --model checkpoints/ppo_v36_s0_x2/final.zip --out data/evaluations/my_sim2d
    # 幾何計画の基準。--jobs で Seed を分けて並列に評価する（結果は 1 プロセスと同じ）
    uv run python scripts/evaluate_policies.py --backend sim2d --stages 1 2 4 5 6 7 --policies planner mixed --n 100 \
        --jobs 4 --out data/evaluations/planner_dev
    # env を Rust 版にして速く評価する（$OV は方策の run_info.json の overrides。README の「学習と評価」）。結果は Python 版の env と同じ（Seed・Policy・設定が同じなら
    # episodes.csv は同じ、summary.json は meta.env と wall_s のほかは同じ）。sim2d だけ、planner は不可
    uv run python scripts/evaluate_policies.py --backend sim2d --env native --stages 4 6 7 8 --policies ppo --stochastic \
        --model checkpoints/ppo_v36_s0_x2/final.zip --override $OV --n 100 --jobs 4 \
        --out data/evaluations/my_native_dev
    # Gazebo（pixi 環境で実行。--launch-sim なしなら先に bash scripts/start_simulation.sh を起動しておく）
    ~/.pixi/bin/pixi run python scripts/evaluate_policies.py --backend gazebo --launch-sim --stages 2 --policies heuristic \
        --n 20 --out data/evaluations/heuristic_gazebo
出力: <out>/episodes.csv（1 行 1 エピソード）、<out>/summary.json、<out>/summary.md、
      planner なら <out>/planner_stats.csv（エピソードごとの計画の回数・失敗の理由・1 ステップの計算時間）
"""
import argparse
import csv
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from toy_sweeper.config import load_all
from toy_sweeper.env import ToySweeperEnv
from toy_sweeper.evaluate import evaluate, results_to_rows, summarize
from toy_sweeper.native import NativeEnv
from toy_sweeper.planner import PlannerPolicy, load_planner_cfg
from toy_sweeper.policies import CarryHeuristicPolicy, HeuristicPolicy, MixedHeuristicPolicy, PPOPolicy, RandomPolicy

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ["random", "heuristic", "carry", "mixed", "ppo", "planner"]
ENVS = ["python", "native"]
# env の中身（盤面・観測した姿勢）を読む Policy。NativeEnv は観測しか出さないので評価できない
NEEDS_PYTHON_ENV = {"planner"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["sim2d", "gazebo"])
    ap.add_argument("--stages", type=int, nargs="+", required=True)
    ap.add_argument("--policies", nargs="+", required=True, choices=POLICIES)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--n", type=int, help="エピソード数（既定は rl.yaml evaluation.n_episodes / test_n_episodes）")
    ap.add_argument("--test", action="store_true", help="最終評価用の Seed（rl.yaml evaluation.test_*）を使う。比較・選択には使わない")
    ap.add_argument("--stochastic", action="store_true", help="PPO の行動を確率的に選ぶ（既定は決定的）")
    ap.add_argument("--env", default="python", choices=ENVS,
                    help="sim2d の env。native は Rust 版で、結果は python と同じ（planner は不可）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--launch-sim", action="store_true", help="Gazebo をこのスクリプトから起動・終了する")
    ap.add_argument("--verbose", action="store_true", help="エピソードごとの結果を表示")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Seed を分けて並列に評価するプロセス数（sim2d のみ。random は Seed をまたいで乱数を使うので不可）")
    ap.add_argument("--override", type=Path, nargs="*", default=[],
                    help="config/experiments/*.yaml（例: 実機誤差を入れた環境で評価する）")
    ap.add_argument("--planner-override", type=Path, nargs="*", default=[],
                    help="計画器の設定の差分（config/experiments/planner_*.yaml、トップレベル planner:）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if "ppo" in a.policies and a.model is None:
        ap.error("ppo を評価するには --model が必要")
    if a.env == "native" and a.backend != "sim2d":
        ap.error("--env native は --backend sim2d のときだけ")
    if a.env == "native" and NEEDS_PYTHON_ENV & set(a.policies):
        ap.error(f"--env native では {sorted(NEEDS_PYTHON_ENV & set(a.policies))} を評価できない（env の中身を読む）")
    if a.jobs < 1:
        ap.error("--jobs は 1 以上")
    if a.jobs > 1 and (a.backend != "sim2d" or "random" in a.policies):
        ap.error("--jobs > 1 は sim2d で、random 以外の Policy のときだけ（結果が分け方で変わらないもの）")
    cfg = load_all(overrides=tuple(a.override))
    ev = cfg.rl.evaluation
    start, limit = (ev.test_seed_start, ev.test_n_episodes) if a.test else (ev.seed_start, ev.n_episodes)
    n = a.n if a.n is not None else limit
    if n > limit:
        ap.error(f"--n は評価 Seed 範囲 {limit} 件以下")
    seeds = range(start, start + n)

    if a.launch_sim and a.backend != "gazebo":
        ap.error("--launch-sim は --backend gazebo のときだけ")
    if a.launch_sim:
        from toy_sweeper.gazebo.launcher import running_simulation
        out_dir = a.out if a.out.is_absolute() else ROOT / a.out
        out_dir.mkdir(parents=True, exist_ok=True)
        with running_simulation(out_dir / "gazebo.log"):
            run(a, cfg, n, seeds)
    else:
        run(a, cfg, n, seeds)


def make_policy(name: str, cfg, env, model: Path | None, stochastic: bool, planner_overrides=()):
    return {"random": lambda: RandomPolicy(0), "heuristic": lambda: HeuristicPolicy(cfg),
            "carry": lambda: CarryHeuristicPolicy(cfg), "mixed": lambda: MixedHeuristicPolicy(cfg),
            "ppo": lambda: PPOPolicy(model, deterministic=not stochastic, cfg=cfg),
            "planner": lambda: PlannerPolicy(cfg, load_planner_cfg(overrides=tuple(planner_overrides))).bind(env)}[name]()


def eval_chunk(job: tuple) -> tuple[list, list]:
    """1 つの (Stage, Policy, Seed の範囲) を評価する（並列のワーカーでも本体でも使う）。"""
    stage, name, seeds, model, stochastic, overrides, planner_overrides, backend, env_kind = job
    if env_kind not in ENVS:
        raise ValueError(f"未知の env: {env_kind}")
    if env_kind == "native" and backend is not None:
        raise ValueError("env native は sim2d だけ（Gazebo のバックエンドは使えない）")
    if env_kind == "native" and name in NEEDS_PYTHON_ENV:
        raise ValueError(f"env native では {name} を評価できない（env の中身を読む）")
    import torch
    # 1 歩ずつ 1 件の推論なので 1 スレッドで足りる。2026-10-18、既定のままでは --jobs 4 の各ワーカーが 4 コアずつ使い、
    # 並行する学習の速さが 1/3 に落ちた
    torch.set_num_threads(1)
    cfg = load_all(overrides=tuple(overrides))
    env = NativeEnv(cfg, stage) if env_kind == "native" else ToySweeperEnv(cfg, stage=stage, backend=backend)
    pol = make_policy(name, cfg, env, model, stochastic, planner_overrides)
    res = evaluate(env, pol, seeds)
    stats = []
    if isinstance(pol, PlannerPolicy):
        pol.finish()
        stats = [{"seed": int(s), **row} for s, row in zip(seeds, pol.episode_log)]
    return res, stats


def run(a, cfg, n, seeds):
    if a.backend == "gazebo":
        from toy_sweeper.gazebo.backend import GazeboBackend
        backend = GazeboBackend(cfg)
    else:
        backend = None  # ToySweeperEnv が Sim2D を作る

    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows, summary, pstats = [], {}, []
    pool = ProcessPoolExecutor(max_workers=a.jobs) if a.jobs > 1 else None
    for stage in a.stages:
        for name in a.policies:
            t0 = time.time()
            if pool is None:
                res, stats = eval_chunk((stage, name, list(seeds), a.model, a.stochastic, a.override,
                                         a.planner_override, backend, a.env))
            else:
                chunks = [list(map(int, c)) for c in np.array_split(np.array(seeds), a.jobs) if len(c)]
                res, stats = [], []
                for r, st in pool.map(eval_chunk, [(stage, name, c, a.model, a.stochastic, a.override,
                                                    a.planner_override, None, a.env) for c in chunks]):
                    res += r
                    stats += st
            s = summarize(res)
            s["wall_s"] = time.time() - t0
            if stats:
                t = [st["step_time_mean_ms"] * st["steps"] for st in stats]
                s["planner"] = {"step_time_mean_ms": float(sum(t) / max(1, sum(st["steps"] for st in stats))),
                                "step_time_max_ms": float(max(st["step_time_max_ms"] for st in stats)),
                                "plan_failures": int(sum(st["plan_failures"] for st in stats)),
                                "no_task_steps": int(sum(st["no_task_steps"] for st in stats))}
                pstats += [{"stage": stage, **st} for st in stats]
            summary[f"stage{stage}/{name}"] = s
            print(f"stage {stage} {name:9s} success {s['success_rate']:.2f}±{s['success_rate_se']:.2f} "
                  f"steps {s['mean_steps_success']:.1f} fail {s['failures']} trunc {s['truncated']} chassis {s['chassis_contact_rate']:.3f} ({s['wall_s']:.0f} s)",
                  flush=True)
            rows += [{"stage": stage, "policy": name, **r} for r in results_to_rows(res)]
    if pool is not None:
        pool.shutdown()

    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if pstats:
        with (out / "planner_stats.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pstats[0]))
            w.writeheader()
            w.writerows(pstats)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    meta = {"backend": a.backend, "env": a.env, "model": str(a.model) if a.model else None, "overrides": [str(o) for o in a.override],
            "planner_overrides": [str(o) for o in a.planner_override],
            "test_set": a.test, "stochastic": a.stochastic, "seeds": [seeds.start, seeds.stop - 1],
            "git_commit": commit, "git_dirty": dirty, "jobs": a.jobs, "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "results": summary}, indent=2, ensure_ascii=False))
    lines = [f"backend: {a.backend} / env: {a.env} / model: {meta['model']} / seeds {seeds.start}-{seeds.stop - 1} / commit {commit}"
             + (" (未コミットの変更あり)" if dirty else ""), "",
             "| stage | policy | 成功率 | 標準誤差 | 成功時の平均ステップ | 失敗内訳 | 打ち切り |", "|---|---|---:|---:|---:|---|---:|"]
    for k, s in summary.items():
        st, pol = k.split("/")
        lines.append(f"| {st[5:]} | {pol} | {s['success_rate']:.2f} | {s['success_rate_se']:.2f} | "
                     f"{s['mean_steps_success']:.1f} | {s['failures'] or '-'} | {s['truncated']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
