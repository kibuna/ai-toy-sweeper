# 環境構築

二台で動かす。GPU の PC（Linux）が音声サーバー（Whisper と VLM）・学習・Gazebo を、MacBook が実機の現場（カメラ・BLE・
話しかける画面・方策）を受け持つ。実機を動かすだけなら、GPU の PC は音声サーバーだけでよい。

## GPU の PC（Linux）

前提は Ubuntu 24.04（WSL2 可）、x86_64、NVIDIA の GPU、`uv` が入っていること。学習環境の Rust 版（`native/`）のビルドに
Rust も要る。

```bash
bash scripts/setup_env.sh   # uv sync --frozen と pixi install --locked、ROS 2 のパッケージのビルド（sudo 不要）
bash scripts/check_env.sh   # 動作確認。1 つでも失敗すれば非 0 で終わる
```

| 用途 | 管理ツール | 固定のしかた |
|---|---|---|
| 学習、2D シミュレータ、テスト（gymnasium、SB3、torch、pymunk、OpenCV） | uv | `uv.lock`（`uv sync --frozen`） |
| ROS 2 Jazzy、Gazebo Harmonic（gz-sim 8）、ros_gz、colcon | pixi（RoboStack） | `pixi.lock`（`pixi install --locked`） |
| 音声サーバー（Whisper large-v3、Qwen3-VL-4B） | pixi の `vlm` 環境 | `pixi.lock` |

- Python は `uv run ...`、ROS 2・Gazebo は `~/.pixi/bin/pixi run ...`、音声サーバーは `~/.pixi/bin/pixi run -e vlm ...` で動かす。
- ROS 2 は sudo なしで入れるため、apt ではなく RoboStack（conda-forge）で入れる。
- pixi の環境は起動時に `scripts/pixi_activate.sh` で `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` にする。WSL2 では既定の
  Fast DDS が別のプロセスを見つけられなかった。
- モデルの重み（Qwen/Qwen3-VL-4B-Instruct、openai/whisper-large-v3）は、音声サーバーを初めて起動したときに Hugging Face から
  取ってくる。

## MacBook（実機の現場）

```bash
git clone <リポジトリ> && cd ai-toy-sweeper
uv sync
```

- macOS では、ターミナルにカメラと Bluetooth の許可が要る（初めて使うときにダイアログが出る）。
- 実機で使う方策（`checkpoints/ppo_v36_s0_x2/`）はリポジトリに入っている。
- 音声サーバーへは ssh のトンネルでつなぐ（家の LAN に開けない）:

  ```bash
  ssh -N -L 8765:localhost:8765 <ユーザー>@<GPU の PC>
  ```

## テスト

```bash
uv run pytest   # Gazebo と GPU のテストは、その環境が無ければ飛ばす
```
