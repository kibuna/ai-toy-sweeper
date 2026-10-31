#!/usr/bin/env bash
# 環境が使える状態かを検証する。どれか一つでも失敗すれば非 0 で終了する。
#
#   bash scripts/check_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXI="$HOME/.pixi/bin/pixi"
cd "$ROOT"

echo "[1/4] uv 環境の import"
uv run python -c "import gymnasium, stable_baselines3, pymunk, torch, cv2; print('OK: python libs', gymnasium.__version__, stable_baselines3.__version__, pymunk.version, torch.__version__)"

echo "[2/4] pixi 環境の ROS 2 / Gazebo の版"
"$PIXI" run bash -c '[ "$ROS_DISTRO" = jazzy ] || { echo "ERROR: ROS_DISTRO=$ROS_DISTRO"; exit 1; }
v=$(gz sim --versions); case "$v" in 8.*) echo "OK: ROS 2 $ROS_DISTRO, gz-sim $v, RMW=$RMW_IMPLEMENTATION";; *) echo "ERROR: gz-sim $v は Harmonic(8.x) でない"; exit 1;; esac'

echo "[3/4] ROS 2 のプロセス間 pub/sub"
"$PIXI" run bash -c 'python scripts/check_ros.py talker & t=$!; sleep 1; python scripts/check_ros.py listener; wait $t'

echo "[4/4] Gazebo のヘッドレス実行"
"$PIXI" run bash -c 'timeout 60 gz sim -s -r --iterations 1000 empty.sdf -v 1' && echo "OK: gz sim headless"

echo "すべて OK"
