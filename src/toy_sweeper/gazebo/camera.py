"""Gazebo の俯瞰カメラ画像を受け取る（動画記録用。Policy の入力には使わない）。"""
from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image

from .backend import GazeboBackend, _ns
from .sdf_gen import CAMERA_TOPIC


class OverheadCamera:
    def __init__(self, backend: GazeboBackend):
        self.b = backend
        self.latest: tuple[int, np.ndarray] | None = None
        backend.node.create_subscription(Image, CAMERA_TOPIC, self._cb, 2)

    def _cb(self, msg: Image) -> None:
        if msg.encoding != "rgb8":
            raise ValueError(f"想定外の画像形式: {msg.encoding}")
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        self.latest = (_ns(msg.header.stamp), img.copy())

    def frame(self) -> np.ndarray:
        """現在の sim 時刻以降に撮られた画像を待って返す。"""
        t = self.b.clock_ns
        self.b._wait(lambda: self.latest is not None and self.latest[0] >= t - 1_000_000, "カメラ画像")
        return self.latest[1]
