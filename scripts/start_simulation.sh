#!/usr/bin/env bash
# Gazebo（一時停止）と ros_gz_bridge を起動する。Ctrl-C で終了。
#   bash scripts/start_simulation.sh            # ヘッドレス
#   bash scripts/start_simulation.sh gui:=true  # GUI 付き
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "$HOME/.pixi/bin/pixi" run ros2 launch toy_sweeper_gazebo sim.launch.py "$@"
