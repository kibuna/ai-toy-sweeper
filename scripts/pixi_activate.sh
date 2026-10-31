# pixi 環境の起動時に読み込まれる（pixi.toml [activation]）。
#
# RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
#   理由: 2026-10-16、この開発機（WSL2 mirrored network, Ubuntu 24.04, RoboStack Jazzy）で
#   既定の Fast DDS は別プロセス間の発見に失敗し、CycloneDDS では 0.2 s で通った
#   （送る側と受ける側の 2 つのノードで確認）。実機デモも同じ PC で行うため本番でも成り立つ。
#   CycloneDDS は Jazzy の Tier 1 RMW。Fast DDS 側の設定で直す案との比較は未実施。
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# WSL2 で Gazebo の描画（GUI、カメラセンサー）を GPU で行う設定。
#   2026-10-16、この開発機では /dev/dri の権限がなく EGL のヘッドレス描画は失敗した。
#   WSLg の X（DISPLAY=:0）と Mesa の d3d12 ドライバー（/dev/dxg 経由）なら描画できた。
if [ -e /dev/dxg ]; then
    export GALLIUM_DRIVER=d3d12
    export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    export DISPLAY="${DISPLAY:-:0}"
fi

# colcon でビルドした ROS 2 パッケージを読み込む（未ビルドなら何もしない）
if [ -f "$PIXI_PROJECT_ROOT/ros2_ws/install/local_setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$PIXI_PROJECT_ROOT/ros2_ws/install/local_setup.bash"
fi
