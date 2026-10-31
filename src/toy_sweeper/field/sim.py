"""実機の代わりの実時間シミュレーション。

- SimWorld: Sim2D を壁時計に合わせて別スレッドで進める（ランタイムのイベントループが止まっても世界は進む）。
  受けた指令は command_latency_s 後に効き、最後に効いた指令から ttl_s を過ぎたら止まる（micro:bit のウォッチドッグ、
  安全の 3 層目の代わり）。グリッパーの指令は Sim2D と同じく actuation_time_s の間ロボットを止める。
- SimCamera: SimWorld の状態を draw_scene で描き、camera_fps ごとに capture_latency_s 遅れて渡す。
- SimRobotLink: 指令を SimWorld に渡す。disconnect() で切断を模す（試験用）。
- SimArucoStandIn: ArUco はシミュレーションしない。撮影時刻の真の姿勢に正規ノイズを足したものを返す。
- SimTruthIdentifier: 開始時の物体の同定（実機では VLM の役）を、真の物体の一覧との位置の照合で行う
  （perception_env.PerceptionObsWrapper.reset と同じ）。その後は追跡の ID だけで追う。
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..config import AllCfg
from ..geometry import Pose2D
from ..render import draw_scene
from ..sim2d import ObjectSpec, Sim2D
from ..spaces import GripperState
from ..task import EpisodeSpec
from .config import SimWorldCfg
from .interfaces import (Command, DriveCommand, FieldStop, Frame, GripperCommand, LinkError, LinkStatus, SimTruth, StopCommand,
                         TrackView, check_not_expired)
from .latest import Latest

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppliedCommand:
    """SimWorld の記録: 指令が届いた時刻と効き始めた時刻。"""
    kind: str
    seq: int
    received: float
    applied: float


# 止めている間に、再開を見に行く間隔（物理の刻みより長くてよい。未検証）
PAUSE_POLL_S = 0.05


class SimWorld:
    def __init__(self, cfg: AllCfg, fsim: SimWorldCfg, board: EpisodeSpec, seed: int):
        """seed: 物理の乱数（挟み損ね sim2d.grasp_fail。学習の env.reset が渡すものの代わり。盤面の Seed を使う）。"""
        self.cfg = cfg
        self.fsim = fsim
        self.board = board
        self.sim = Sim2D(cfg.sim)
        # 2026-10-22、挟み損ねの有効な設定（v31〜、管理のアプリの --world sim）で、乱数が無く閉じたときに例外になった
        self.sim.rng = np.random.default_rng(seed)
        # 止めている（管理のアプリで実機を動かしている間。物理を進めない）
        self.paused = False
        self.sim.reset(board.robot, list(board.objects), GripperState.OPEN)
        self.dt = self.sim.dt
        self._lock = threading.Lock()
        self._pending: list[tuple[float, float, Command]] = []   # (効く時刻, 届いた時刻, 指令)
        self._drive: DriveCommand | None = None      # 今効いている走行の指令
        self._deadline: float | None = None           # ウォッチドッグの期限（最後に効いた指令 + ttl）
        self._hold_until = -math.inf                  # グリッパーの開閉で止まっている間
        self._stop = threading.Event()
        self.time: float | None = None                # 最後に進めた物理の刻みの壁時計の時刻
        self.applied: list[AppliedCommand] = []
        self.watchdog_stops = 0
        self.max_lag_s = 0.0
        self.idle_resyncs = 0   # 止まっている間の遅れを数え直した回数
        self.last_moving: float | None = None         # 最後に速度の指令が 0 でなかった刻みの時刻

    # ---------- ランタイム側から（どのスレッドからでもよい） ----------
    def submit(self, cmd: Command) -> None:
        now = time.monotonic()
        with self._lock:
            self._pending.append((now + self.fsim.command_latency_s, now, cmd))

    def truth(self) -> tuple[SimTruth, float]:
        """今の状態（真値）と、その状態の時刻。"""
        with self._lock:
            if self.time is None:
                raise RuntimeError("SimWorld がまだ動いていない（run() が先）")
            s = self.sim
            t = SimTruth(robot=s.robot_pose(), objects=self.board.objects,
                         poses={o.name: s.object_pose(o.name) for o in self.board.objects},
                         jaw_angle=s.jaw_angle(), gripper=s.gripper_state())
            return t, self.time

    def robot_pose(self) -> Pose2D:
        with self._lock:
            return self.sim.robot_pose()

    def reset_board(self, board: EpisodeSpec) -> None:
        """盤面を置き直す（talk_sim のリセット）。届いていない指令と今の指令は捨てる。"""
        with self._lock:
            self.board = board
            self.sim.reset(board.robot, list(board.objects), GripperState.OPEN)
            self._pending.clear()
            self._drive, self._deadline, self._hold_until = None, None, -math.inf

    def object_poses(self) -> dict[str, Pose2D]:
        with self._lock:
            return {o.name: self.sim.object_pose(o.name) for o in self.board.objects}

    # ---------- 世界のスレッド ----------
    def _apply_due(self, t: float) -> None:
        due = [p for p in self._pending if p[0] <= t]
        if not due:
            return
        self._pending = [p for p in self._pending if p[0] > t]
        for at, received, cmd in due:
            self.applied.append(AppliedCommand(type(cmd).__name__, cmd.seq, received, t))
            if isinstance(cmd, StopCommand):
                self._drive, self._deadline = None, None
            elif isinstance(cmd, DriveCommand):
                self._drive, self._deadline = cmd, t + cmd.ttl_s
            elif isinstance(cmd, GripperCommand):
                self.sim.set_gripper(cmd.close)
                self._hold_until = t + self.cfg.sim.gripper.actuation_time_s
                self._deadline = t + cmd.ttl_s
            else:
                raise TypeError(f"未知の指令: {cmd!r}")

    def _velocity(self, t: float) -> tuple[float, float]:
        if self._deadline is not None and t > self._deadline:
            if self._drive is not None:
                log.warning("SimWorld: 指令 seq=%d の期限が切れたので止まる（ウォッチドッグ）", self._drive.seq)
                self.watchdog_stops += 1
            self._drive, self._deadline = None, None
        if self._drive is None or t < self._hold_until:
            return 0.0, 0.0
        return self._drive.linear, self._drive.angular

    def run_blocking(self) -> None:
        t0 = time.monotonic()
        with self._lock:
            self.time = t0
        n = 0
        while not self._stop.is_set():
            if self.paused:   # 止めている間は進めず、再開したら今から数え直す（遅れとみなさない）
                time.sleep(PAUSE_POLL_S)
                t0 = time.monotonic() - n * self.dt
                continue
            step_start = t0 + n * self.dt
            now = time.monotonic()
            if now < step_start + self.dt:
                time.sleep(step_start + self.dt - now)
                continue
            lag = now - (step_start + self.dt)
            self.max_lag_s = max(self.max_lag_s, lag)
            if lag > self.fsim.max_realtime_lag_s and self._drive is None:
                # 動いていない間の遅れ（スリープからの復帰、ほかの処理に押された）は、動きに影響しない。今から数え直す
                log.warning("SimWorld: 止まっている間に実時間から %.3f s 遅れた。数え直す", lag)
                self.idle_resyncs += 1
                t0 = now - n * self.dt
                continue
            if lag > self.fsim.max_realtime_lag_s:
                raise RuntimeError(f"SimWorld: 物理が実時間から {lag:.3f} s 遅れた（上限 {self.fsim.max_realtime_lag_s} s）")
            with self._lock:
                if self.paused:   # 確かめた後に止められた（pause は錠を取るので、戻った後は 1 歩も進めない）
                    continue
                self._apply_due(step_start)
                v, w = self._velocity(step_start)
                self.sim.drive(v, w, self.dt)
                n += 1
                self.time = t0 + n * self.dt
                if v != 0.0 or w != 0.0:
                    self.last_moving = self.time

    def pause(self) -> None:
        with self._lock:
            self.paused = True

    def resume(self) -> None:
        self.paused = False

    async def run(self) -> None:
        """止める（キャンセル）まで世界を進める。物理が実時間に追いつかなければ例外。"""
        self._stop.clear()
        fut = asyncio.ensure_future(asyncio.to_thread(self.run_blocking))
        try:
            await asyncio.shield(fut)
        except asyncio.CancelledError:
            self._stop.set()
            await fut
            raise


class SimCamera:
    """SimWorld の俯瞰画像（draw_scene）。freeze() でフレームが来なくなる（USB カメラが固まった、の試験用）。"""

    def __init__(self, cfg: AllCfg, fsim: SimWorldCfg, world: SimWorld):
        self.cfg = cfg
        self.fps = fsim.camera_fps
        self.latency = fsim.capture_latency_s
        self.world = world
        self.frames: Latest[Frame] = Latest("camera")
        self.frozen = False
        self.render_s: list[float] = []
        self.missed_slots = 0

    def freeze(self) -> None:
        log.warning("SimCamera: フレームを止める（試験）")
        self.frozen = True

    def _render(self, truth: SimTruth) -> np.ndarray:
        t = time.perf_counter()
        img, _ = draw_scene(self.cfg, truth.robot, truth.objects, truth.poses, truth.jaw_angle)
        self.render_s.append(time.perf_counter() - t)
        return img

    async def run(self) -> None:
        period = 1.0 / self.fps
        t0 = time.monotonic()
        k, seq = 0, 0
        while True:
            k += 1
            await asyncio.sleep(max(0.0, t0 + k * period - time.monotonic()))
            late = time.monotonic() - (t0 + k * period)
            if late > period:  # 描画が遅れて撮影の時刻を飛ばした。数えて記録する
                skip = int(late // period)
                self.missed_slots += skip
                k += skip
            if self.frozen:
                continue
            truth, stamp = self.world.truth()
            img = await asyncio.to_thread(self._render, truth)
            await asyncio.sleep(max(0.0, stamp + self.latency - time.monotonic()))
            if self.frozen:
                continue
            seq += 1
            self.frames.set(Frame(img, stamp, seq, truth), stamp)


class SimRobotLink:
    def __init__(self, world: SimWorld):
        self.world = world
        self._connected = False
        self._detail = "未接続"
        self.n_sent = 0
        self.last_sent: float | None = None
        self.sent: list[Command] = []

    async def connect(self) -> None:
        self._connected, self._detail = True, "シミュレーション"

    def disconnect(self) -> None:
        """切断を模す（試験用）。以後の送信は LinkError。"""
        log.warning("SimRobotLink: 切断する（試験）")
        self._connected, self._detail = False, "切断（試験）"

    async def send(self, cmd: Command) -> None:
        if not self._connected:
            raise LinkError(f"SimRobotLink: 接続していない（{self._detail}）")
        check_not_expired(cmd)
        self.world.submit(cmd)
        self.sent.append(cmd)
        self.n_sent += 1
        self.last_sent = time.monotonic()

    def status(self) -> LinkStatus:
        return LinkStatus(self._connected, self._detail, self.n_sent, self.last_sent)

    async def close(self) -> None:
        self._connected, self._detail = False, "閉じた"


class SimArucoStandIn:
    """ArUco の代わり: 撮影時刻の真の姿勢 + 正規ノイズ（標準偏差は field.yaml の sim.robot_*_noise）。"""

    def __init__(self, fsim: SimWorldCfg, seed: int):
        self.pos_sd = fsim.robot_pose_noise_m
        self.yaw_sd = fsim.robot_yaw_noise_rad
        self.rng = np.random.default_rng(seed)

    def estimate(self, frame: Frame) -> Pose2D:
        if frame.truth is None:
            raise ValueError("SimArucoStandIn にはシミュレーションのフレーム（真値つき）が必要")
        r = frame.truth.robot
        n = self.rng.normal(size=3)
        return Pose2D(r.x + self.pos_sd * n[0], r.y + self.pos_sd * n[1], r.yaw + self.yaw_sd * n[2])


class SimTruthIdentifier:
    """開始時の同定: 真の物体ごとに、tolerance_m 以内で最も近い追跡を結び付ける。見つからない物体があれば例外。
    allow_missing（Gazebo のリハーサルの実機の決まり、2026-10-23）: 例外にせず、その物体は同定しないで missing に残す
    （実機の VLM は見えた塊だけを同定して始める。くっついた 2 個は 1 個として扱われる）。"""

    def __init__(self, tolerance_m: float, allow_missing: bool = False):
        self.tol = tolerance_m
        self.allow_missing = allow_missing
        self.missing: list[str] = []   # 直前の identify で同定しなかった物体の名前（allow_missing のとき）
        self.last = None   # VLM の問い合わせ（実機の VlmObjectIdentifier.last）は無い。管理の画面が読む

    def identify(self, frame: Frame, tracks: dict[int, TrackView]) -> dict[int, ObjectSpec]:
        if frame.truth is None:
            raise ValueError("SimTruthIdentifier にはシミュレーションのフレーム（真値つき）が必要")
        out: dict[int, ObjectSpec] = {}
        self.missing = []
        for o in frame.truth.objects:
            p = frame.truth.poses[o.name]
            free = [t for t in tracks.values() if t.id not in out]
            near = min(free, key=lambda t: math.dist(t.pose.xy, p.xy), default=None)
            if near is None or math.dist(near.pose.xy, p.xy) > self.tol:
                if self.allow_missing:
                    self.missing.append(o.name)
                    continue
                raise FieldStop("start_undetected", f"開始時に物体 {o.name}（{o.color} {o.type.name}）を検出できない"
                                f"（{self.tol} m 以内に追跡がない。盤面か検出の設定の問題）")
            out[near.id] = ObjectSpec(o.name, o.type, near.pose, o.color)
        return out


class TruthDetector:
    """シミュレーションの真値から検出を作る（talk_sim.py --perception truth）。カメラ画像は使わない。

    カメラ画像からの検出は、接した物体が 1 つの塊に見えて追跡が入れ替わる（Stage 6 の盤面で、押した LEGO が隣の LEGO と
    くっついて止まった、2026-10-18）。言葉で動かす流れを確かめるときに、認識の問題を切り離すために使う。
    検出の形（中心、向き、大きさ、平均色、外接矩形）はカメラからの検出と同じ。平均色は描画の色。
    """

    def __init__(self, cfg: AllCfg, hom):
        from ..render import object_rgb
        self.cfg, self.hom, self.rgb = cfg, hom, object_rgb

    def __call__(self, frame: Frame, robot: Pose2D, jaw_angle: float) -> list:
        from ..geometry import box_corners
        from ..perception import Detection
        from ..spaces import OBJECT_TYPE_NAMES
        t = frame.truth
        if t is None:
            raise ValueError("TruthDetector はシミュレーションのフレーム（truth あり）にしか使えない")
        out = []
        for o in t.objects:
            p = t.poses[o.name]
            sx, sy = self.cfg.sim.objects[OBJECT_TYPE_NAMES[o.type]].size_m[:2]
            px = self.hom.to_px(np.array(box_corners(p, (sx, sy))))
            x0, y0 = np.floor(px.min(axis=0)).astype(int)
            x1, y1 = np.ceil(px.max(axis=0)).astype(int)
            out.append(Detection(p, (max(sx, sy), min(sx, sy)), tuple(float(c) for c in self.rgb(o.color)),
                                 (int(x0), int(y0), int(x1), int(y1))))
        return out


class TruthTracker:
    """TruthDetector の検出（物体の並びは毎回同じ）を、並びの番号を ID にした追跡にする。近い順の対応付けは
    接した物体の ID を入れ替えることがある（2026-10-18、真値の検出でも黄色い LEGO の追跡が隣の黒い LEGO に移った）。"""

    def __init__(self):
        self.n: int | None = None

    def update(self, dets: list) -> dict:
        from ..perception import Track
        if self.n is None:
            self.n = len(dets)
        if len(dets) != self.n:
            raise ValueError(f"真値の検出の数が変わった: {self.n} → {len(dets)}")
        return {i: Track(i, d.pose, d.size_m, d.mean_rgb) for i, d in enumerate(dets)}
