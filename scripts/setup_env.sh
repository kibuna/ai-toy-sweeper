#!/usr/bin/env bash
# 開発環境を lock ファイルどおりに再現する。sudo 不要。
#
#   bash scripts/setup_env.sh
#
# - Python（学習・2D シミュレータ）: uv + uv.lock（--frozen: lock と食い違えば失敗）
# - ROS 2 Jazzy + Gazebo Harmonic: pixi + pixi.lock（--locked: lock と食い違えば失敗）
# - pixi 本体は PIXI_VERSION に固定して ~/.pixi/bin に入れる
# 終わったら scripts/check_env.sh で動作確認する。
set -euo pipefail

PIXI_VERSION="0.82.0"  # 2026-10-16 にこの版で pixi.lock を作成
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXI="$HOME/.pixi/bin/pixi"

command -v uv >/dev/null || { echo "ERROR: uv がない。https://docs.astral.sh/uv/ を参照して入れること" >&2; exit 1; }

if [ ! -x "$PIXI" ] || [ "$("$PIXI" --version)" != "pixi $PIXI_VERSION" ]; then
    echo "pixi $PIXI_VERSION を $HOME/.pixi/bin に導入する"
    mkdir -p "$HOME/.pixi/bin"
    tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/pixi.tar.gz" \
        "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz"
    tar xzf "$tmp/pixi.tar.gz" -C "$tmp"
    mv "$tmp/pixi" "$PIXI"
    rm -rf "$tmp"
fi
[ "$("$PIXI" --version)" = "pixi $PIXI_VERSION" ] || { echo "ERROR: pixi の版が $PIXI_VERSION でない" >&2; exit 1; }

cd "$ROOT"
uv sync --frozen
"$PIXI" install --locked

if [ -d "$ROOT/ros2_ws/src" ]; then
    "$PIXI" run bash -c "cd ros2_ws && colcon build"  # --symlink-install は新しい setuptools で失敗するため使わない
fi
echo "完了。次に: bash scripts/check_env.sh"
