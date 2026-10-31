# 設定の差し替え（config/experiments/）

`config/sim.yaml`・`task.yaml`・`rl.yaml` に、差分だけを書いた YAML を順に重ねる（`--override a.yaml b.yaml ...`、後のものが勝つ）。
各ファイルの頭に、その値である理由を書いてある。今使うものだけを置く（2026-10-31 に整理した。テストだけで使うものは
`test/configs/`）。

## 実機の方策（ppo_v36_s0_x2）を学習・評価した設定

この順に重ねる（`checkpoints/ppo_v36_s0_x2/run_info.json` の `overrides` と同じ）。学習は `steps_x2.yaml` を除いた 13 個で、
評価と実機は歩数の上限 2 倍の `steps_x2.yaml` まで重ねる。

| 順 | ファイル | 中身 |
|---|---|---|
| 1 | set_encoder.yaml | 物の集合を共通の重みで読んでまとめる（DeepSets、v16） |
| 2 | chassis03.yaml | 車体で運ぶ物に触れる減点 0.3 / 個（v20） |
| 3 | imperfect.yaml | 挟み損ね・位置の誤差・車体の陰で隠れる（v21） |
| 4 | gpu_ppo.yaml | Rust 版の環境で 1,024 並列、PPO を GPU で |
| 5 | cleanup.yaml | やり残しから始まる盤面（Stage 9）を段階に足す（v24） |
| 6 | corner_edge.yaml | 縁までの距離を物の 4 隅で測る（v25） |
| 7 | cleanup_guidance.yaml | 後片付けの手がかり（いまの仕事と行き先を観測に。v26） |
| 8 | multi_toy_car.yaml | Stage 7・8 の運ぶ物の半分ほどをミニカーに（v28） |
| 9 | efficient.yaml | 時間と爪の開閉の減点（v28） |
| 10 | free_area.yaml | 作業領域を 0.6〜1.0 m の自由な長方形に、壁なし、失敗の減点 20（v30） |
| 11 | robot_8hz.yaml | 1 秒 8 回の行動（v30） |
| 12 | hw_1022.yaml | 実機で測った車体・爪・速さ（v31） |
| 13 | motor_lag.yaml | 車輪の速さの惰性、時定数 0.05〜0.15 秒（v36） |
| 14 | steps_x2.yaml | 歩数の上限 2 倍（評価と実機。学習はしない） |

## 評価・Gazebo・学習で足すもの

| ファイル | 使うところ |
|---|---|
| area_real.yaml | 2D の評価で、作業領域を実機の大きさ（74.8 × 82.6 cm）に固定する |
| dev300.yaml | 2D の評価を 300 回に（開発用の Seed 10000〜10299） |
| seed1.yaml・seed2.yaml | 同じ設定で学習の Seed を変える（下の学習の例の s1・s2） |
| gazebo_oblique.yaml | Gazebo のカメラを斜め上に置く（リハーサル・話しかける画面の Gazebo・検出器のデータ作り） |
| gazebo_area08.yaml | Gazebo のリハーサルで、作業領域を Gazebo の世界と同じ 0.8 m 四方に固定する |
| no_chassis_penalty.yaml | Gazebo では車体の接触を数えないので、その減点を 0 にする |

## 例

学習（ppo_v36_s0。v32 の重みから、Stage 9 から始める。s1・s2 は `--override` の最後に `config/experiments/seed1.yaml`・
`seed2.yaml` を足し、`--run` を `ppo_v36_s1`・`ppo_v36_s2` にする）:

```bash
uv run python scripts/train_ppo.py --run ppo_v36_s0 --init checkpoints/ppo_v32_s0_x2/final.zip --start-stage 9 \
    --timesteps 50000000 --override config/experiments/set_encoder.yaml config/experiments/chassis03.yaml \
    config/experiments/imperfect.yaml config/experiments/gpu_ppo.yaml config/experiments/cleanup.yaml \
    config/experiments/corner_edge.yaml config/experiments/cleanup_guidance.yaml config/experiments/multi_toy_car.yaml \
    config/experiments/efficient.yaml config/experiments/free_area.yaml config/experiments/robot_8hz.yaml \
    config/experiments/hw_1022.yaml config/experiments/motor_lag.yaml
```

評価（2D、実機の大きさ、300 回）:

```bash
OV=$(python3 -c "import json; print(' '.join(json.load(open('checkpoints/ppo_v36_s0_x2/run_info.json'))['overrides']))")
uv run python scripts/evaluate_policies.py --backend sim2d --stages 7 8 9 --policies ppo --stochastic \
    --model checkpoints/ppo_v36_s0_x2/final.zip --n 300 --override $OV config/experiments/area_real.yaml \
    --out data/evaluations/<名前>
```
