"""PPO を 2D シミュレータで学習する（Curriculum: config/rl.yaml）。

使い方:
    uv run python scripts/train_ppo.py --run ppo_v1                       # 本番（rl.yaml の total_timesteps）
    uv run python scripts/train_ppo.py --run smoke --timesteps 50000       # 時間計測用の短い実行
    uv run python scripts/train_ppo.py --run ppo_v1 --resume checkpoints/ppo_v1/ckpt_400000_steps.zip
    uv run python scripts/train_ppo.py --run set_encoder --override config/experiments/set_encoder.yaml
    uv run python scripts/train_ppo.py --run my_ft --init checkpoints/ppo_v32_s0_x2/final.zip --start-stage 9  # 重みだけ引き継ぐ
出力: checkpoints/<run>/（モデル、設定スナップショット、Curriculum 状態）、dev_log/<run>/（train.log, progress.csv, TensorBoard）
"""
import argparse
from pathlib import Path

from toy_sweeper.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--timesteps", type=int, help="rl.yaml の total_timesteps を上書き（計測用）")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--override", type=Path, nargs="*", default=[],
                    help="config/experiments/*.yaml（設定の差分）。書いた順に重ねる")
    ap.add_argument("--bc", type=Path, help="模倣による初期化の手本（.npz）")
    ap.add_argument("--init", type=Path, help="重みを引き継ぐモデル。ハイパーパラメータは今の設定（--start-stage と一緒に）")
    ap.add_argument("--start-stage", type=int, help="--init のとき Curriculum を始める Stage")
    a = ap.parse_args()
    train(a.run, tuple(a.override), a.timesteps, a.resume, a.bc, a.init, a.start_stage)


if __name__ == "__main__":
    main()
