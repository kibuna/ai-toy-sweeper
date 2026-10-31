# ai-toy-sweeper

話しかけると、床に散らかったおもちゃ（ブロックとミニカー）を片付ける小型ロボット。子どもの声を Whisper で文字にし、VLM
（Qwen3-VL-4B）で「何をどこへ」に直す。どう動くかは、自作のシミュレータで強化学習（PPO）した方策が決める。2D シミュレーション・
Gazebo（ROS 2）・実機（Cutebot V3 ＋ サーボの爪、web カメラ 1 台）で同じ方策を動かす。

- レポート: [doc/report.md](doc/report.md)（背景・しくみ・工夫・結果。開発中の試行錯誤はここにまとめた）
- 動かし方: この README の「動かし方」、実機の準備は [doc/real_robot_runbook.md](doc/real_robot_runbook.md) と
  [doc/camera_setup.md](doc/camera_setup.md)

本文の「ブロック」はおもちゃの組み立てブロック（LEGO® など）を指す。LEGO は LEGO Group の商標。

## 必要なもの

| 用途 | 機材・ソフト |
|---|---|
| 学習・Gazebo・音声サーバー | Linux（x86_64）、NVIDIA の GPU（開発では RTX 3090、WSL2 の Ubuntu）、uv、Rust（学習環境の Rust 版のビルド） |
| 実機の現場 | macOS（開発では MacBook）、uv |
| ロボット | ELECFREAKS Cutebot V3、micro:bit V2、サーボの爪（M5Stack Unit Catch 相当）、単 4 電池 3 本 |
| カメラ | USB の web カメラ（開発では Logitech C920、1920×1080・10 fps）、床に置く ArUco マーカー（`scripts/make_markers_pdf.py` で印刷） |

## 環境構築

```bash
# Linux（学習・Gazebo・音声サーバー）。lock ファイルどおりに入れる（sudo 不要）
bash scripts/setup_env.sh        # uv sync --frozen と pixi install --locked、ROS 2 のパッケージのビルド
bash scripts/check_env.sh        # 動作確認

# macOS（実機の現場）
uv sync
```

- Python の環境は uv（`pyproject.toml`、`uv.lock`）。学習環境の Rust 版（`native/`）は Linux で uv が maturin でビルドする。
- ROS 2 Jazzy・Gazebo Harmonic と VLM・音声の環境は pixi（`pixi.toml`、`pixi.lock`）。`pixi run`（既定の環境）が Gazebo、
  `pixi run -e vlm` が音声サーバー。
- モデルの重み（Qwen/Qwen3-VL-4B-Instruct、openai/whisper-large-v3）は初めて起動したときに Hugging Face から取ってくる。
- 実機で使う方策は `checkpoints/ppo_v36_s0_x2/` に入れてある（`final.zip`、学習の設定 `config_snapshot/`、`run_info.json`）。
  v36（旋回の惰性を入れて学習し直した版）の重みを、歩数の上限だけ 2 倍にして使う。学習を始めた重み（v32）は
  `checkpoints/ppo_v32_s0_x2/`。

```bash
uv run pytest                    # テスト（Gazebo と GPU のテストは、その環境が無ければ飛ばす）
```

## 動かし方

### 音声サーバー（Linux、GPU）

```bash
~/.pixi/bin/pixi run -e vlm python scripts/voice_server.py --host 127.0.0.1 --port 8765
# 別の PC から使うときは ssh でつなぐ: ssh -N -L 8765:localhost:8765 <ユーザー>@<GPU の PC>
```

### 話しかける画面（2D シミュレーション・実機・Gazebo）

```bash
uv run python scripts/field_app.py --model checkpoints/ppo_v36_s0_x2/final.zip \
    --voice-server http://127.0.0.1:8765 --port 8080 --world sim --gazebo-ssh none
```

ブラウザで http://localhost:8080/ を開き、録音ボタンを押して「あかいレゴをみぎうえにかたづけて」などと話す。`/admin` が管理の画面、
`/calibrate` がカメラの校正の画面。カメラ・校正・ロボットが揃っていれば、画面の「おとなよう」の欄で実機に切り替えられる。
`--gazebo-ssh <ユーザー>@<GPU の PC>` を付けると、選んだときだけ GPU の PC で Gazebo を起動して中継する。

### 学習と評価（Linux）

```bash
# 学習。設定の差し替えは config/experiments/ から重ねる（一覧は config/experiments/README.md）。実機の方策（v36）の 5,000 万歩は
# v32 の重みから Stage 9 で始め、run_info.json の overrides から steps_x2.yaml を除いた 13 個を重ねた
uv run python scripts/train_ppo.py --run <run名> --init checkpoints/ppo_v32_s0_x2/final.zip --start-stage 9 \
    --timesteps 50000000 --override <run_info.json の overrides から steps_x2.yaml を除いたもの>

# 2D の評価（固定の Seed、300 回）。方策を学習したときの設定の差し替えを同じ順で付ける（一覧は run_info.json の overrides）
OV=$(python3 -c "import json; print(' '.join(json.load(open('checkpoints/ppo_v36_s0_x2/run_info.json'))['overrides']))")
uv run python scripts/evaluate_policies.py --backend sim2d --stages 7 8 9 --policies ppo --stochastic \
    --model checkpoints/ppo_v36_s0_x2/final.zip --n 300 --override $OV --out data/evaluations/<名前>

# Gazebo のリハーサル（物理とカメラの画像、実機と同じ知覚のプログラム）。上の $OV に Gazebo の作業領域の差し替えを足す
~/.pixi/bin/pixi run python scripts/eval_rehearsal.py --model checkpoints/ppo_v36_s0_x2/final.zip --stage 7 --seed 20000 \
    --boards 20 --perception camera --identify truth --rules rehearsal --target-type lego --detector background \
    --override $OV config/experiments/gazebo_area08.yaml --out data/evaluations/<名前>
```

実機の方策は 2,000 万歩から継ぎ足して合計 6.1 億歩学習したもので、版ごとの変更と結果は [doc/report.md](doc/report.md) の付録にある。

### 実機

1. micro:bit に [microbit/main.ts](microbit/main.ts) を書き込む（手順は [microbit/README.md](microbit/README.md)）。
2. 床の 4 隅に ArUco マーカーを置き、カメラを斜め上に固定する（[doc/camera_setup.md](doc/camera_setup.md)）。
3. 話しかける画面のアプリを起動し、`/calibrate` で校正して背景を撮る。
4. 話しかける画面で実機に切り替えて話しかける。

手順の全体と確かめ方は [doc/real_robot_runbook.md](doc/real_robot_runbook.md)。

物の検出は背景との差（`config/field.yaml` の `real.detector: background`）を使う。学習した検出器（`--detector learned`、
`checkpoints/seg_v1/best.onnx`）の重みは同梱していない（Gazebo の画像だけで学習したもので、実機では背景との差の方が良かった）。

### 自分の環境に合わせて変えるところ

開発した環境（家の床、Logitech C920、この車体と爪）で決めた値がそのまま入っている。別の環境で実機を動かすときは次を合わせる。

| 場所 | 中身 | 合わせ方 |
|---|---|---|
| `config/field.yaml` の `camera.device_name` | 開くカメラの名前（`HD Pro Webcam C920`）。名前が合わないと「開けない」で止まる | 使うカメラの名前に変える |
| `config/field.yaml` の `camera.width_px`・`height_px`・`fps` | 1920×1080、10 fps。カメラが受け付けた値を読み返し、違えば止まる | カメラが出せる値に変える（変えたら校正し直す） |
| `data/calibration/camera.yaml`・`background.yaml` | この部屋・このカメラの校正と、背景を撮ったときの記録。背景の画像そのもの（`data/camera/background.png`）は同梱しない | `/calibrate` で校正し、背景を撮る（[doc/camera_setup.md](doc/camera_setup.md)） |
| `data/calibration/robot_marker.yaml` | ロボットの上のマーカーの向きのずれ | 貼り直したら `uv run python scripts/measure_motion.py yaw` で測り直す |
| `microbit/main.ts` の `OPEN_ANGLE`・`CLOSE_ANGLE` | この爪で決めたサーボの角度（20・60） | 別の爪なら、閉じきって鳴らない角度に合わせる |
| `--voice-server` と ssh のトンネル | 音声サーバーの場所（例の `127.0.0.1:8765` はトンネル越し） | 自分の GPU の PC に向ける |
| `scripts/field_app.py` の `keep_awake()` | スリープさせないため、起動時に macOS の `caffeinate` を必ず呼ぶ。無ければ止まる | macOS で動かす。ほかの OS で動かすなら、この関数の中身を外すか、その OS のスリープ止めに書き換える |
| `scripts/field_app.py` の `--gazebo-ssh` の既定値と `GAZEBO_REMOTE_DIR`・`GAZEBO_PIXI` | 開発に使った GPU の PC の宛先とパス。画面で Gazebo を選んだときだけ使う | 実機と 2D だけなら `--gazebo-ssh none`。Gazebo も使うなら、この 3 つを自分の PC に合わせる |

## 構成

| パス | 内容 |
|---|---|
| `src/toy_sweeper/` | 本体。`sim2d.py`（2D の物理）、`env.py`（学習環境）、`train.py`、`policies.py`、`spaces.py`・`snapshot.py`（観測）、`perception*.py`（検出と追跡）、`vlm.py`、`voice/`（音声）、`field/`（現場のランタイム・画面・実機）、`gazebo/`、`planner/`（幾何計画の基準の方策） |
| `native/` | 学習環境の Rust 版（物理は Chipmunk2D） |
| `ros2_ws/src/toy_sweeper_gazebo/` | Gazebo と ros_gz_bridge の launch |
| `microbit/` | ロボットの micro:bit のプログラム |
| `checkpoints/ppo_v36_s0_x2/` | 実機で使う方策（`ppo_v32_s0_x2/` はその学習を始めた重み） |
| `config/` | 設定。各値にその値である理由を書いてある（`sim.yaml` 寸法と物理、`task.yaml` 盤面と報酬、`rl.yaml` 学習、`field.yaml` 現場、`voice.yaml` 音声、`vlm.yaml` と `prompts/` VLM、`experiments/` 実機の方策と評価・Gazebo で重ねる差し替え（一覧は experiments/README.md）） |
| `scripts/` | 起動するプログラム（学習・評価・アプリ・Gazebo・校正・図の作成） |
| `test/` | テスト |
| `doc/` | レポートと図、環境構築・カメラの準備・実機の手順 |
| `data/` | 指示の例、VLM の評価の指示、今の校正の値 |

## 参考

- ELECFREAKS Smart Cutebot: https://wiki.elecfreaks.com/en/microbit/microbit-smart-car/microbit-smart-cutebot/cutebot_car
- M5Stack Unit Catch（グリッパー）: https://docs.m5stack.com/en/unit/catch
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- RoboStack: https://robostack.github.io/
- Chipmunk2D: https://chipmunk-physics.net/
