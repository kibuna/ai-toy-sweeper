"""現場のランタイムと機器（カメラ・ロボットとの通信・姿勢の推定・物体の同定）の約束。

実機の実装は field/hardware.py、シミュレーションの実装は field/sim.py。
時刻はすべて time.monotonic() の秒（同じ機械の中だけで比べる）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..geometry import Pose2D
from ..sim2d import ObjectSpec
from ..spaces import GripperState
from .latest import Latest


# ---------- 止める理由 ----------
class FieldStop(RuntimeError):
    """ロボットを止めて終わった（安全監視・見失い・停止の要求など）。ロボットに停止を送ってから投げる。

    reason は機械が読む短い名前（stale_camera, target_lost, link_lost, stop_request, ...）。
    """

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.report: dict | None = None   # ランタイムが終了時の記録（時間の統計など）を付ける


class FrameUnusable(RuntimeError):
    """このフレームは知覚に使えない（実機: ロボットのマーカーが一瞬見えない、手や足が入って背景との差の塊が大きすぎる）。
    ランタイムはそのフレームを飛ばす。続けば観測が古くなり、止まって待ち、さらに続けば止める（safety.max_obs_age_s・
    stale_obs_hold_s）。2026-10-22、1 フレームで片付け全体が止まったため。"""


class PoseLost(FrameUnusable):
    """このフレームではロボットの姿勢が取れない。"""


class LinkError(RuntimeError):
    """ロボットとの通信の失敗（切断、書き込みの失敗）。"""


class CommandExpired(LinkError):
    """有効期限を過ぎた指令を送ろうとした（安全の 2 層目: 古い指令は送らない）。"""


# ---------- カメラ ----------
@dataclass(frozen=True)
class SimTruth:
    """シミュレーションだけが持つ、撮影時刻の真値（ArUco の代わりと、開始時の物体の同定に使う）。"""
    robot: Pose2D
    objects: tuple[ObjectSpec, ...]   # 種類・色（姿勢は poses）
    poses: dict[str, Pose2D]
    jaw_angle: float
    gripper: GripperState


@dataclass(frozen=True)
class Frame:
    image: np.ndarray          # RGB、uint8
    stamp: float               # 撮影時刻（monotonic）
    seq: int                   # カメラの通し番号
    truth: SimTruth | None = None  # 実カメラでは None


class CameraSource(Protocol):
    fps: float
    frames: Latest[Frame]

    async def run(self) -> None:
        """止める（キャンセル）までフレームを frames に置き続ける。失敗は例外。"""


class RobotPoseEstimator(Protocol):
    def estimate(self, frame: Frame) -> Pose2D:
        """フレームからロボットの姿勢（車軸中点、Pose2D）を出す。重い処理なのでスレッドで呼ばれる。見つからなければ例外。"""


@dataclass(frozen=True)
class TrackView:
    """追跡 1 本の、ある時点の写し（perception.Track は更新されるので写して渡す）。"""
    id: int
    pose: Pose2D
    size_m: tuple[float, float]
    mean_rgb: tuple[float, float, float]
    missed: int
    hits: int = 1    # 対応が付いたフレームの数（perception.Track.hits）


class ObjectIdentifier(Protocol):
    def identify(self, frame: Frame, tracks: dict[int, TrackView]) -> dict[int, ObjectSpec]:
        """開始時に、追跡ごとの物体（名前・種類・色）を決める。決められない物体があれば例外。"""


# ---------- ロボットへの指令 ----------
@dataclass(frozen=True)
class DriveCommand:
    seq: int
    stamp: float      # 作った時刻
    ttl_s: float      # 有効期限（作ってから）
    linear: float     # m/s（車軸中点）
    angular: float    # rad/s（左回りが正）


@dataclass(frozen=True)
class GripperCommand:
    seq: int
    stamp: float
    ttl_s: float
    close: bool


@dataclass(frozen=True)
class StopCommand:
    seq: int
    stamp: float


Command = DriveCommand | GripperCommand | StopCommand


def check_not_expired(cmd: Command, now: float | None = None) -> None:
    """期限切れの指令なら CommandExpired。停止の指令はいつでも送れる。"""
    if isinstance(cmd, StopCommand):
        return
    age = (time.monotonic() if now is None else now) - cmd.stamp
    if age > cmd.ttl_s:
        raise CommandExpired(f"指令 seq={cmd.seq} は作ってから {age:.3f} s（期限 {cmd.ttl_s} s）")


@dataclass(frozen=True)
class LinkStatus:
    connected: bool
    detail: str
    n_sent: int = 0
    last_sent: float | None = None   # 最後に送り終えた時刻
    extra: dict = field(default_factory=dict)


class RobotLink(Protocol):
    async def connect(self) -> None: ...

    async def send(self, cmd: Command) -> None:
        """指令を送る。期限切れなら CommandExpired、通信の失敗は LinkError。"""

    def status(self) -> LinkStatus: ...

    async def close(self) -> None: ...
