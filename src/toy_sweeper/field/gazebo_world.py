"""Gazebo を実時間で進める世界とカメラ（現場のランタイム用）。pixi 環境（rclpy）でのみ使える。

- GazeboWorld: 専用のスレッドで、壁時計に合わせて Gazebo を step_s ずつ進める（GazeboBackend の同期実行を使う）。
  指令の扱い（届くまでの遅れ、期限切れで止まるウォッチドッグ、爪の開閉の間は止まる）は SimWorld と同じ。
- GazeboCamera: Gazebo のカメラ画像（斜め上など、gazebo.camera_pose）を Latest[Frame] に流す。Frame の truth には
  撮影を受け取った時点の真値を入れる（真値を使う部品、例えば SimTruthIdentifier や ArUco の代わりのため）。

ROS のコールバック（姿勢、カメラ）は世界のスレッドが GazeboBackend を回す間に処理される。他のスレッドは
ロックの下のコピー（_truth、_image）だけを読む。
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
import time

import numpy as np

from ..config import AllCfg
from ..geometry import Pose2D
from ..gazebo.backend import GazeboBackend
from ..sim2d import ObjectSpec
from ..spaces import GripperState
from ..task import EpisodeSpec
from .config import SimWorldCfg
from .interfaces import Command, DriveCommand, Frame, GripperCommand, SimTruth, StopCommand
from .latest import Latest
from .rig import SimRig

log = logging.getLogger(__name__)


class GazeboWorld:
    def __init__(self, cfg: AllCfg, fsim: SimWorldCfg, board: EpisodeSpec, backend: GazeboBackend, step_s: float):
        self.cfg, self.fsim, self.b, self.step_s = cfg, fsim, backend, step_s
        self.board = board
        self._lock = threading.Lock()
        self._pending: list[tuple[float, Command]] = []        # (効く時刻, 指令)
        self._drive: DriveCommand | None = None
        self._deadline: float | None = None
        self._hold_until = -math.inf
        self._sent: tuple[float, float] | None = None          # Gazebo に出している速度
        self._gripper = GripperState.OPEN
        self._reset_to: EpisodeSpec | None = None
        self._stop = threading.Event()
        self.time: float | None = None
        self._truth: SimTruth | None = None
        self.dt = step_s                 # SimRig.running() が停止の確かめに使う
        self.watchdog_stops = 0
        self.max_lag_s = 0.0
        self.step_wall_s: list[float] = []
        self.b.reset(board.robot, list(board.objects), GripperState.OPEN)
        self._snapshot()

    # ---------- ランタイム側から ----------
    def submit(self, cmd: Command) -> None:
        with self._lock:
            self._pending.append((time.monotonic() + self.fsim.command_latency_s, cmd))

    def truth(self) -> tuple[SimTruth, float]:
        with self._lock:
            if self.time is None or self._truth is None:
                raise RuntimeError("GazeboWorld がまだ動いていない（run() が先）")
            return self._truth, self.time

    def robot_pose(self) -> Pose2D:
        return self.truth()[0].robot

    def object_poses(self) -> dict[str, Pose2D]:
        return dict(self.truth()[0].poses)

    def reset_board(self, board: EpisodeSpec) -> None:
        """盤面を置き直す（世界のスレッドで行い、終わるまで待つ）。"""
        with self._lock:
            self._reset_to = board
        while True:
            with self._lock:
                if self._reset_to is None:
                    return
            time.sleep(0.01)

    # ---------- 世界のスレッド ----------
    def _snapshot(self) -> None:
        g = self.cfg.sim.gripper
        b = self.b
        t = SimTruth(robot=b.robot_pose(), objects=self.board.objects,
                     poses={o.name: b.object_pose(o.name) for o in self.board.objects},
                     jaw_angle=g.jaw_closed_angle_rad if self._gripper == GripperState.CLOSED else g.jaw_open_angle_rad,
                     gripper=self._gripper)
        with self._lock:
            self._truth = t

    def _apply_due(self, now: float) -> None:
        with self._lock:
            due = [c for at, c in self._pending if at <= now]
            self._pending = [(at, c) for at, c in self._pending if at > now]
        for cmd in due:
            if isinstance(cmd, StopCommand):
                self._drive, self._deadline = None, None
            elif isinstance(cmd, DriveCommand):
                self._drive, self._deadline = cmd, now + cmd.ttl_s
            elif isinstance(cmd, GripperCommand):
                self._set_velocity(0.0, 0.0)
                self.b._set_jaws(cmd.close)
                self._gripper = GripperState.CLOSED if cmd.close else GripperState.OPEN
                self._hold_until = now + self.cfg.sim.gripper.actuation_time_s
                self._deadline = now + cmd.ttl_s
            else:
                raise TypeError(f"未知の指令: {cmd!r}")

    def _set_velocity(self, v: float, w: float) -> None:
        if self._sent != (v, w):
            self.b._publish_cmd(v, w)
            self._sent = (v, w)

    def _velocity(self, now: float) -> tuple[float, float]:
        if self._deadline is not None and now > self._deadline:
            if self._drive is not None:
                log.warning("GazeboWorld: 指令 seq=%d の期限が切れたので止まる（ウォッチドッグ）", self._drive.seq)
                self.watchdog_stops += 1
            self._drive, self._deadline = None, None
        if self._drive is None or now < self._hold_until:
            return 0.0, 0.0
        return self._drive.linear, self._drive.angular

    def run_blocking(self) -> None:
        t0 = time.monotonic()
        n = 0
        with self._lock:
            self.time = t0
        while not self._stop.is_set():
            with self._lock:
                reset = self._reset_to
            if reset is not None:
                self._set_velocity(0.0, 0.0)
                self.board = reset
                self.b.reset(reset.robot, list(reset.objects), GripperState.OPEN)
                self._drive, self._deadline, self._hold_until, self._gripper = None, None, -math.inf, GripperState.OPEN
                with self._lock:
                    self._pending.clear()
                self._snapshot()
                t0, n = time.monotonic(), 0
                with self._lock:
                    self._reset_to = None
                continue
            due = t0 + (n + 1) * self.step_s
            now = time.monotonic()
            if now < due:
                time.sleep(due - now)
                continue
            lag = now - due
            self.max_lag_s = max(self.max_lag_s, lag)
            if lag > self.fsim.max_realtime_lag_s:
                raise RuntimeError(f"GazeboWorld: Gazebo が実時間から {lag:.3f} s 遅れた（上限 {self.fsim.max_realtime_lag_s} s）")
            self._apply_due(now)
            self._set_velocity(*self._velocity(now))
            t = time.perf_counter()
            self.b._step(self.step_s)
            self.step_wall_s.append(time.perf_counter() - t)
            n += 1
            self._snapshot()
            with self._lock:
                self.time = t0 + n * self.step_s

    async def run(self) -> None:
        self._stop.clear()
        fut = asyncio.ensure_future(asyncio.to_thread(self.run_blocking))
        try:
            await asyncio.shield(fut)
        except asyncio.CancelledError:
            self._stop.set()
            await fut
            raise


class GazeboCamera:
    """Gazebo のカメラ画像を fps で Latest[Frame] に流す（CameraSource）。"""

    def __init__(self, world: GazeboWorld, fps: float):
        from sensor_msgs.msg import Image
        from ..gazebo.sdf_gen import CAMERA_TOPIC
        self.world, self.fps = world, fps
        self.frames: Latest[Frame] = Latest("camera")
        self.missed_slots = 0            # SimRig.truth_report の項目（Gazebo のカメラは数えない）
        self.render_s: list[float] = []
        self._img: tuple[int, np.ndarray, float] | None = None   # (番号, 画像, 受け取った時刻)
        self._n = 0
        self._lock = threading.Lock()
        world.b.node.create_subscription(Image, CAMERA_TOPIC, self._cb, 2)

    def _cb(self, msg) -> None:
        if msg.encoding != "rgb8":
            raise ValueError(f"想定外の画像形式: {msg.encoding}")
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
        with self._lock:
            self._n += 1
            self._img = (self._n, img, time.monotonic())

    async def run(self) -> None:
        seq_seen, seq = 0, 0
        period = 1.0 / self.fps
        while True:
            await asyncio.sleep(period / 4)
            with self._lock:
                item = self._img
            if item is None or item[0] == seq_seen:
                continue
            seq_seen = item[0]
            truth, _ = self.world.truth()
            seq += 1
            self.frames.set(Frame(item[1], item[2], seq, truth), item[2])


class GazeboRig(SimRig):
    """Gazebo の世界・カメラ・リンク・ArUco の代わり・真値の同定と、起動時のカメラの校正（床の 4 隅のマーカー）。
    SimRig と同じ使い方（running / run_instruction / make_runtime / reset_board / use_truth_perception）。"""

    def __init__(self, cfg: AllCfg, f, pcfg, board: EpisodeSpec, seed: int, backend: GazeboBackend, step_s: float):
        from .sim import SimArucoStandIn, SimRobotLink, SimTruthIdentifier
        self.cfg, self.f, self.pcfg, self.board = cfg, f, pcfg, board
        self.world = GazeboWorld(cfg, f.sim, board, backend, step_s)
        self.camera = GazeboCamera(self.world, f.sim.camera_fps)
        self.link = SimRobotLink(self.world)
        self.pose = SimArucoStandIn(f.sim, seed)
        self.identifier = SimTruthIdentifier(f.perception.association_tolerance_m)
        self.hom = None              # calibrate() で床のマーカーから求める
        self.drift_after_stop_m = None
        self.detector = None
        self.tracker_factory = None

    def calibrate(self, image: np.ndarray):
        """Gazebo のカメラ画像（RGB）の床の 4 隅のマーカーで、カメラの姿勢を求める（実機の「校正する」と同じ）。"""
        import cv2
        from ..calibration import estimate_extrinsics, intrinsics_from_fov
        from ..markers import load_markers_cfg
        gz = self.cfg.sim.gazebo
        intr = intrinsics_from_fov(tuple(gz.camera_image_px), gz.camera_hfov_rad)
        cam = estimate_extrinsics(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), intr, load_markers_cfg(), self.cfg.sim,
                                  self.cfg.sim.walls.thickness_m)
        log.info("カメラの校正: 位置 %s m、再投影誤差 %.2f px", np.round(cam.position(), 3).tolist(), cam.reproj_px)
        self.hom = cam
        return cam
