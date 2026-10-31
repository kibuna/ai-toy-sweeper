"""Gazebo + ros_gz_bridge を子プロセスとして起動し、終了時に確実に止める。"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
from pathlib import Path

from ..config import CONFIG_DIR

ROOT = CONFIG_DIR.parent


@contextlib.contextmanager
def running_simulation(log_path: Path | None = None, gui: bool = False, camera: bool = False,
                       overrides: tuple[Path, ...] = (), board: list | None = None, markers: bool = False,
                       camera_rate_hz: float | None = None):
    """overrides: World を作る設定の差し替え（config/experiments/*.yaml）。GazeboBackend にも同じ設定を渡すこと。
    board: 盤面の物体（ObjectSpec のリスト）。渡すと、その物体のモデルだけを盤面の色で置く（GazeboBackend にも
    board_models(board) を渡すこと）。markers: 床の 4 隅に ArUco マーカーを描く（config/markers.yaml）。"""
    cpath = ""
    if board:
        cpath = str((ROOT / "data" / "generated" / "gazebo" / "board.json").resolve())
        Path(cpath).parent.mkdir(parents=True, exist_ok=True)
        Path(cpath).write_text(json.dumps({o.name: {"kind": o.type.name.lower(), "color": o.color} for o in board}))
    log = open(log_path, "w") if log_path else subprocess.DEVNULL
    ov = ",".join(str(Path(p).resolve()) for p in overrides)
    if "," in "".join(str(p) for p in overrides):
        raise ValueError(f"設定のパスにカンマがある: {overrides}")
    # ros2 launch は空の値（name:=）を受け付けないので、値のある引数だけ渡す
    opt = {"overrides": ov, "board": cpath, "camera_rate": str(camera_rate_hz) if camera_rate_hz else ""}
    proc = subprocess.Popen(["ros2", "launch", "toy_sweeper_gazebo", "sim.launch.py", f"gui:={'true' if gui else 'false'}",
                             f"camera:={'true' if camera else 'false'}", f"markers:={'true' if markers else 'false'}",
                             *(f"{k}:={v}" for k, v in opt.items() if v)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        yield proc
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        if log_path:
            log.close()
