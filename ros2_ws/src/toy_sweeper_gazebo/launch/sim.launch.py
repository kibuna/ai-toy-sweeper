"""Gazebo（一時停止状態）と ros_gz_bridge を起動する。

    ros2 launch toy_sweeper_gazebo sim.launch.py            # ヘッドレス
    ros2 launch toy_sweeper_gazebo sim.launch.py gui:=true  # GUI も開く（WSLg）
    ros2 launch toy_sweeper_gazebo sim.launch.py camera:=true  # 俯瞰カメラ（動画記録用）を入れる
    ros2 launch toy_sweeper_gazebo sim.launch.py camera:=true overrides:=config/experiments/gazebo_oblique.yaml
    ros2 launch toy_sweeper_gazebo sim.launch.py camera:=true markers:=true board:=<枠→{kind, color} の JSON>

World は起動時に config/sim.yaml から生成する（toy_sweeper.gazebo.sdf_gen）。
Gazebo は一時停止で立ち上がり、sim 時間は GazeboBackend が ControlWorld で進める。
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import json
from pathlib import Path

from toy_sweeper.config import CONFIG_DIR, load_all
from toy_sweeper.gazebo.sdf_gen import bridge_args, echo_bridge_args, write_world


def _setup(context):
    ov = LaunchConfiguration("overrides").perform(context)
    cfg = load_all(overrides=tuple(Path(p) for p in ov.split(",") if p))   # 設定の差し替え（カンマ区切り）
    camera = LaunchConfiguration("camera").perform(context) == "true"
    # 盤面の JSON（枠 → {"kind", "color"}）。渡したら、その物体のモデルだけを、盤面の色で置く
    bpath = LaunchConfiguration("board").perform(context)
    board = json.loads(Path(bpath).read_text()) if bpath else None
    colors = {k: v["color"] for k, v in board.items()} if board else None
    only = {k: v["kind"] for k, v in board.items()} if board else None
    markers = LaunchConfiguration("markers").perform(context) == "true"
    rate = LaunchConfiguration("camera_rate").perform(context)   # カメラの周期（sim 時間、Hz）。空なら 1 Action に 1 枚
    world = write_world(cfg, CONFIG_DIR.parent / "data" / "generated" / "gazebo", camera, colors, markers,
                        float(rate) if rate else None, only)
    actions = [
        ExecuteProcess(cmd=["gz", "sim", "-s", "-v", "2", str(world)], output="screen"),
        Node(package="ros_gz_bridge", executable="parameter_bridge", arguments=bridge_args(cfg, camera), output="screen"),
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="cmd_echo_bridge",
             arguments=echo_bridge_args()[0], remappings=echo_bridge_args()[1], output="screen"),
    ]
    if LaunchConfiguration("gui").perform(context) == "true":
        actions.append(ExecuteProcess(cmd=["gz", "sim", "-g", "-v", "2"], output="screen"))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("camera", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("overrides", default_value=""),
        DeclareLaunchArgument("board", default_value=""),
        DeclareLaunchArgument("markers", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("camera_rate", default_value=""),
        OpaqueFunction(function=_setup),
    ])
