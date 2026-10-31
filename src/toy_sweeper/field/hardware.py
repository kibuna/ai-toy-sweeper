"""実機の機器（USB カメラ、micro:bit との BLE UART）。2026-10-17 時点で機器は未着で、一度も動かしていない（未検証）。

このモジュールは機器がなくても import できる（bleak は接続するときにだけ import する）。
- OpenCVCamera: cv2.VideoCapture を別スレッドで読み、最新のフレームだけを渡す。
- BleRobotLink: 指令（`M,L,R,seq` / `G,1,seq` / `S,seq`、改行で区切る）を micro:bit の UART に書く。
  書式・符号・範囲・改行の規則は、Cutebot の MakeCode 拡張と micro:bit のプログラムを作ってから確定する（未検証）。
  micro:bit 側のプログラムは、最後の指令から control.command_ttl_s（0.5 s）で止まるウォッチドッグを持つこと（§15.2 の 3 層目）。
- ArucoRobotPose: ロボットの上のマーカーから車軸の中点の姿勢（2026-10-22）。
- real_homography / VlmObjectIdentifier: 未実装（呼ぶと NotImplementedError）。
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..calibration import CameraModel
from ..config import RobotCfg
from ..geometry import Pose2D
from ..markers import MarkersCfg, aruco_dictionary
from ..spaces import DRIVE_ACTIONS, Action, drive_velocity
from ..perception import Homography
from ..sim2d import ObjectSpec
from .config import BleCfg, CameraCfg
from .interfaces import (Command, DriveCommand, Frame, GripperCommand, LinkError, LinkStatus, PoseLost, StopCommand,
                         TrackView, check_not_expired)
from .latest import Latest

log = logging.getLogger(__name__)


# ---------- カメラ ----------
# カメラから読めないとき、開き直す間隔と、あきらめるまでの時間（USB の一時的な途切れを越える程度。未検証）
CAMERA_REOPEN_WAIT_S = 0.5
CAMERA_REOPEN_GIVE_UP_S = 10.0
# 開いてから読み捨てるフレームの数（露出が落ち着くまで。最初の数枚は暗い・明るすぎる。未検証）
CAMERA_WARMUP_FRAMES = 10


def list_cameras() -> list[tuple[str, str]]:
    """macOS のカメラの一覧（固有の ID、名前）。どのカメラも開かない（点かない）。OpenCV の AVFoundation と同じ種類
    （内蔵の広角・外付け。外付けに iPhone の連係カメラが入る）を引く。"""
    import AVFoundation as AV
    s = AV.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        [AV.AVCaptureDeviceTypeBuiltInWideAngleCamera, AV.AVCaptureDeviceTypeExternal], AV.AVMediaTypeVideo, 0)
    return [(str(d.uniqueID()), str(d.localizedName())) for d in s.devices()]


def camera_index(name: str, devices: list[tuple[str, str]] | None = None) -> int:
    """名前のカメラの cv2.VideoCapture の番号。OpenCV の番号は一覧を固有の ID で並べた順（2026-10-27 実測、
    test_camera_index_by_name）。無い・2 台あれば RuntimeError。"""
    devices = list_cameras() if devices is None else devices
    order = sorted(devices)
    hits = [i for i, (_, n) in enumerate(order) if n == name]
    if not hits:
        raise RuntimeError(f"カメラ「{name}」が見つからない（つながっているのは {[n for _, n in order]}）。USB を差し直す")
    if len(hits) > 1:
        raise RuntimeError(f"カメラ「{name}」が {len(hits)} 台ある（どれか決められない）")
    return hits[0]


class OpenCVCamera:
    def __init__(self, c: CameraCfg):
        self.c = c
        self.fps = c.fps
        self.frames: Latest[Frame] = Latest("camera")
        self._stop = threading.Event()

    def _open(self):
        # 番号は開くたびに名前から引く（抜き差し・iPhone の有無で変わる。開き直しでも追いつく）
        idx = camera_index(self.c.device_name)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise RuntimeError(f"カメラ「{self.c.device_name}」（{idx} 番）を開けない")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.c.width_px)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.c.height_px)
        cap.set(cv2.CAP_PROP_FPS, self.c.fps)
        got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS))
        if got[:2] != (self.c.width_px, self.c.height_px) or abs(got[2] - self.c.fps) > 0.5:
            cap.release()
            raise RuntimeError(f"カメラが設定を受け付けない: 要求 {self.c.width_px}x{self.c.height_px} {self.c.fps} fps、"
                               f"実際 {got[0]}x{got[1]} {got[2]} fps（黙って違う設定で使わない）")
        return cap

    def _capture_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        cap = self._open()
        seq = 0
        failing_since: float | None = None
        try:
            while not self._stop.is_set():
                ok, bgr = cap.read()
                t = time.monotonic() - self.c.capture_latency_s
                if not ok:
                    # 一時的に読めない（2026-10-22、起動から 13 s で 1 回読めずアプリごと止まった）: 開き直して続ける。
                    # CAMERA_REOPEN_GIVE_UP_S 読めなければ止める（抜けた）
                    now = time.monotonic()
                    failing_since = now if failing_since is None else failing_since
                    if now - failing_since > CAMERA_REOPEN_GIVE_UP_S:
                        raise RuntimeError(f"カメラから {now - failing_since:.0f} s 読めない（抜けた・止まった）")
                    log.warning("カメラから読めない。開き直す")
                    cap.release()
                    time.sleep(CAMERA_REOPEN_WAIT_S)
                    try:
                        cap = self._open()
                    except RuntimeError as e:
                        log.warning("カメラを開き直せない: %s", e)
                    continue
                if failing_since is not None:
                    log.info("カメラを開き直した（%.1f s 読めなかった）", time.monotonic() - failing_since)
                    failing_since = None
                seq += 1
                frame = Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t, seq)
                loop.call_soon_threadsafe(self.frames.set, frame, t)
        finally:
            cap.release()

    async def run(self) -> None:
        self._stop.clear()
        fut = asyncio.ensure_future(asyncio.to_thread(self._capture_loop, asyncio.get_running_loop()))
        try:
            await asyncio.shield(fut)
        except asyncio.CancelledError:
            self._stop.set()
            await fut
            raise


# ロボットのマーカーの角の合わせ方。毎フレーム（10 fps）読むので、1920×1080 で 1 枚 8 ms の SUBPIX（APRILTAG は 111 ms で
# 間に合わず、実機の記録が 5.6 fps に落ちた、2026-10-22）。合成画像での誤差は SUBPIX 0.24 px・APRILTAG 0.07 px
ROBOT_CORNER_REFINEMENT = cv2.aruco.CORNER_REFINE_SUBPIX
# ロボットのマーカーのビットの読み方（OpenCV の既定 0.6 / 0.13 / 4）。斜め上のカメラから浅い角度で見たマーカーは縦に潰れ、
# 既定では止まっていても読めない向きがあった（2026-10-22、旋回の後に止まったまま見失った）。見失ったフレーム 23 枚で比べ、
# 3 つとも変えると 18/23 枚で読め（既定 3/23）、床の 4 隅以外の ID は出なかった。多角形の近似もゆるめると 19/23 だが、
# 無い ID（8、17）が出たので使わない。角の位置（姿勢の精度）には効かない
ROBOT_BIT_PARAMS = {"errorCorrectionRate": 1.0, "perspectiveRemoveIgnoredMarginPerCell": 0.33,
                    "perspectiveRemovePixelPerCell": 8,
                    # 2 値化の窓を広げる（OpenCV の既定は 23 まで 10 刻み）。2026-10-22、実機で奥・暗い所で読めなかった 17 枚（多くは
                    # 手が写ったもの）で、既定 2/17 → 4/17、無い ID は出なかった（しきい値の定数を変えると無い ID が出た）
                    "adaptiveThreshWinSizeMax": 53, "adaptiveThreshWinSizeStep": 4}


# ロボットのマーカーを画像全体で読めないとき、最後の位置のまわり ±REREAD_HALF_PX を REREAD_SCALE 倍にして読み直す。2026-10-26、
# 実機で読めなかった 42 枚（最後の位置のまわり 600×400 の切り出し）で、そのまま 2 / 42、2 倍 8 / 42、2 倍 + 鮮鋭化 9 / 42、
# 誤り訂正を緩める 2 / 42。±200 px は 1 秒ほど読めない間に動く量（約 15 cm、奥で約 150 px）より広い。どれも未検証
REREAD_HALF_PX = 200
REREAD_SCALE = 2.0


class RobotMarkerError(RuntimeError):
    pass


class MarkerNotVisible(RobotMarkerError, PoseLost):
    """ロボットのマーカーがこのフレームで見えない（一時的。ランタイムはフレームを飛ばす）。"""


def load_robot_marker_yaw(path: Path) -> float:
    """マーカーの向きのずれ（マーカーの x 軸の床の向き − ロボットの前の向き、rad）。scripts/measure_motion.py が測って置く。"""
    if not path.exists():
        raise RobotMarkerError(f"ロボットのマーカーの向き（{path}）がない。scripts/measure_motion.py の yaw で、ロボットを"
                               "まっすぐ走らせて測る")
    return float(yaml.safe_load(path.read_text())["yaw_offset_rad"])


def save_robot_marker_yaw(path: Path, yaw_offset_rad: float, detail: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"yaw_offset_rad": float(yaw_offset_rad), **detail}, sort_keys=False, allow_unicode=True))


def pick_camera_by_background(frames: dict[int, np.ndarray | None], background_rgb: np.ndarray,
                              max_mean_diff: float) -> int:
    """カメラの番号ごとの画像（開けない・設定を受け付けないものは None）から、背景（校正の空の床）にいちばん似たものを選ぶ。
    似ている度合いは、縮めた画像の画素の差の平均（0〜255）。いちばん小さくても max_mean_diff を超えたら RuntimeError。"""
    small = lambda im: cv2.resize(im, (192, 108), interpolation=cv2.INTER_AREA).astype(np.float32)
    b = small(background_rgb)
    diffs = {i: float(np.abs(small(f) - b).mean()) for i, f in frames.items() if f is not None}
    log.info("カメラの番号ごとの背景との差: %s", {i: round(d, 1) for i, d in diffs.items()})
    if not diffs or min(diffs.values()) > max_mean_diff:
        raise RuntimeError(f"背景（校正の空の床）に似たカメラが無い: 差 {diffs}（上限 {max_mean_diff}）。カメラがつながっているか、"
                           "動いていないかを確かめ、動いたなら校正し直す")
    return min(diffs, key=diffs.get)


def verify_camera(c: CameraCfg) -> None:
    """名前のカメラだけを開き、設定（解像度・fps）を受け付けるか確かめて閉じる。ほかのカメラは開かない（2026-10-26、起動のたびに
    全部の番号を開いて iPhone・MacBook のカメラが点いた）。C920 以外（MacBook・iPhone）は 10 fps を
    受け付けないので、名前から引いた番号が違っていれば（OpenCV の並べ方が変わった）ここで RuntimeError になる。"""
    try:
        OpenCVCamera(c)._open().release()
    except RuntimeError as e:
        raise RuntimeError(f"カメラ「{c.device_name}」: {e}") from e


class ArucoRobotPose:
    """ロボットの上の ArUco マーカー（config/markers.yaml の robot_id）から、車軸の中点の姿勢を出す。
    マーカーの 4 つの角を、カメラの校正（CameraModel）でマーカーの高さの水平面に戻し、中心と向きを出す。向きは
    yaw_offset_rad（貼った向きのずれ）を引いてロボットの前の向きにし、中心から取り付け位置（robot_marker_offset_m）を
    引いて車軸の中点にする。見つからなければ RobotMarkerError。"""

    def __init__(self, cam: CameraModel, m: MarkersCfg, yaw_offset_rad: float):
        self.cam, self.m, self.yaw_offset = cam, m, yaw_offset_rad
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = ROBOT_CORNER_REFINEMENT
        for k, v in ROBOT_BIT_PARAMS.items():
            setattr(params, k, v)
        self._det = cv2.aruco.ArucoDetector(aruco_dictionary(m.field.dictionary), params)
        self.last_pose: Pose2D | None = None   # 最後に読めた姿勢（読めないときに、そのまわりを拡大して読み直す）
        self.reread_hits = 0                   # 拡大して読み直して読めたフレームの数

    def observe(self, bgr: np.ndarray, hint_xy=None) -> tuple[np.ndarray, float, float]:
        """マーカーの中心（床の座標）、マーカーの x 軸の床の向き（rad）、黒い四角の 1 辺の長さ（m、確かめ用）。
        hint_xy（最後に見えたロボットの位置）があれば、画像全体で読めないとき、そのまわりを拡大して読み直す。"""
        f = self.m.field
        corners, ids, _ = self._det.detectMarkers(bgr)
        found = [] if ids is None else ids.ravel().tolist()
        if found.count(f.robot_id) != 1 and hint_xy is not None:
            got = self._reread(bgr, hint_xy)
            if got is not None:
                self.reread_hits += 1
                corners, found = [got], [f.robot_id]
        if found.count(f.robot_id) != 1:
            raise MarkerNotVisible(f"ロボットのマーカー ID {f.robot_id} が 1 つ見えない（見えた {sorted(found)}）")
        q = self.cam.to_floor(corners[found.index(f.robot_id)].reshape(4, 2), f.robot_marker_height_m)   # ArUco の角の並び: 左上・右上・右下・左下
        ex = (q[1] - q[0]) + (q[2] - q[3])
        side = float(np.mean([np.linalg.norm(q[k] - q[(k + 1) % 4]) for k in range(4)]))
        return q.mean(axis=0), math.atan2(ex[1], ex[0]), side

    def _reread(self, bgr: np.ndarray, hint_xy) -> np.ndarray | None:
        """最後の位置のまわり（REREAD_HALF_PX）を REREAD_SCALE 倍に拡大して読む。読めたら画像全体の座標の角（4, 2）。"""
        f = self.m.field
        u, v = (int(c) for c in self.cam.to_px(np.array([hint_xy]), f.robot_marker_height_m)[0])
        h, w = bgr.shape[:2]
        x0, y0 = max(0, u - REREAD_HALF_PX), max(0, v - REREAD_HALF_PX)
        x1, y1 = min(w, u + REREAD_HALF_PX), min(h, v + REREAD_HALF_PX)
        if x1 - x0 < 16 or y1 - y0 < 16:   # 最後の位置が画像の外
            return None
        crop = cv2.resize(bgr[y0:y1, x0:x1], None, fx=REREAD_SCALE, fy=REREAD_SCALE, interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = self._det.detectMarkers(crop)
        found = [] if ids is None else ids.ravel().tolist()
        if found.count(f.robot_id) != 1:
            return None
        return corners[found.index(f.robot_id)].reshape(4, 2) / REREAD_SCALE + np.array([x0, y0], np.float32)

    def pose_from(self, center: np.ndarray, marker_yaw: float) -> Pose2D:
        yaw = math.remainder(marker_yaw - self.yaw_offset, math.tau)
        ox, oy = self.m.field.robot_marker_offset_m
        c, s = math.cos(yaw), math.sin(yaw)
        return Pose2D(float(center[0] - (c * ox - s * oy)), float(center[1] - (s * ox + c * oy)), yaw)

    def estimate(self, frame: Frame) -> Pose2D:
        last = self.last_pose
        center, marker_yaw, _ = self.observe(cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR),
                                             None if last is None else last.xy)
        self.last_pose = self.pose_from(center, marker_yaw)
        return self.last_pose


def real_homography() -> Homography:
    """実カメラの画素 → 床面の射影変換（キャリブレーション）。カメラ到着後に実装する。"""
    raise NotImplementedError("実カメラのキャリブレーションは未実装（作業領域の四隅のマーカーを撮って求める）")


class VlmObjectIdentifier:
    """開始時の物体の同定（追跡ごとに切り出して 3090 の VLM で種類・色を分類）。カメラ到着後に実装する。"""

    def identify(self, frame: Frame, tracks: dict[int, TrackView]) -> dict[int, ObjectSpec]:
        raise NotImplementedError("VLM による物体の同定は未実装（scripts/eval_vlm.py の分類をサーバーに載せる）")


# ---------- BLE ----------
def wheel_motor_values(linear: float, angular: float, robot: RobotCfg, ble: BleCfg) -> tuple[int, int]:
    """走行の指令 (v, ω) を左右のモーター指令（Cutebot の -100〜100 の整数）に直す。(v, ω) は学習の Action の速さ
    （spaces.drive_velocity）のどれかでなければならず、Action ごとの実測の表（ble.action_motor）を引く。速さに比例させない
    （実機は動き出す境目があり比例しない。動きの計測）。表に無い速さは ValueError。"""
    for a in DRIVE_ACTIONS:
        if drive_velocity(robot, a) == (linear, angular):
            return (0, 0) if a == Action.STOP else tuple(ble.action_motor[a.name])
    raise ValueError(f"走行の指令 (v={linear}, ω={angular}) が学習の Action の速さ（{robot.linear_speed_mps} m/s・"
                     f"{robot.angular_speed_radps} rad/s）のどれでもない")


def encode_command(cmd: Command, robot: RobotCfg, ble: BleCfg) -> bytes:
    """指令の書式（未検証）。改行で終わる ASCII。max_payload_bytes を超えたら ValueError。"""
    if isinstance(cmd, DriveCommand):
        left, right = wheel_motor_values(cmd.linear, cmd.angular, robot, ble)
        text = f"M,{left},{right},{cmd.seq}"
    elif isinstance(cmd, GripperCommand):
        text = f"G,{0 if cmd.close else 1},{cmd.seq}"   # G,1 = 開く、G,0 = 閉じる
    elif isinstance(cmd, StopCommand):
        text = f"S,{cmd.seq}"
    else:
        raise TypeError(f"未知の指令: {cmd!r}")
    data = (text + "\n").encode("ascii")
    if len(data) > ble.max_payload_bytes:
        raise ValueError(f"指令 {text!r} が {len(data)} バイトで、1 回の書き込みの上限 {ble.max_payload_bytes} を超える")
    return data


class BleRobotLink:
    def __init__(self, ble: BleCfg, robot: RobotCfg):
        self.ble = ble
        self.robot = robot
        self._client = None
        self._lost: str | None = None
        self.n_sent = 0
        self.last_sent: float | None = None
        self._lock = asyncio.Lock()

    def _on_disconnect(self, client) -> None:
        log.error("BLE: 切断された（%s）", client.address)
        self._lost = "切断された"

    async def connect(self) -> None:
        """つながっていれば何もしない（ランタイムは指示ごとに呼ぶ。micro:bit は 1 つしか接続を受けない）。"""
        if self._client is not None and self._lost is None and self._client.is_connected:
            return
        from bleak import BleakClient, BleakScanner
        b = self.ble
        log.info("BLE: 「%s」で始まる機器を探す（%.0f s まで）", b.device_name_prefix, b.scan_timeout_s)
        dev = await BleakScanner.find_device_by_filter(
            lambda d, adv: (d.name or "").startswith(b.device_name_prefix), timeout=b.scan_timeout_s)
        if dev is None:
            raise LinkError(f"BLE: 「{b.device_name_prefix}」で始まる機器が見つからない")
        client = BleakClient(dev, disconnected_callback=self._on_disconnect, timeout=b.connect_timeout_s)
        try:
            await client.connect()
        except Exception as e:
            raise LinkError(f"BLE: {dev.name}（{dev.address}）に接続できない: {e!r}") from e
        svc = client.services.get_service(b.uart_service_uuid)
        if svc is None or svc.get_characteristic(b.uart_rx_char_uuid) is None:
            await client.disconnect()
            raise LinkError(f"BLE: {dev.name} に UART サービス {b.uart_service_uuid} / {b.uart_rx_char_uuid} がない"
                            "（micro:bit のプログラムに Bluetooth UART を入れたか）")
        self._client, self._lost = client, None
        log.info("BLE: %s（%s）に接続した", dev.name, dev.address)

    def _check_connected(self) -> None:
        if self._client is None or self._lost is not None or not self._client.is_connected:
            raise LinkError(f"BLE: 接続していない（{self._lost or '未接続'}）")

    async def _write(self, data: bytes) -> float:
        """ロックの中で呼ぶ。書き込み 1 回の時間（応答ありなら往復）を返す。"""
        t0 = time.monotonic()
        try:
            await self._client.write_gatt_char(self.ble.uart_rx_char_uuid, data, response=self.ble.write_with_response)
        except Exception as e:
            self._lost = f"書き込みの失敗: {e!r}"
            raise LinkError(f"BLE: 指令 {data!r} を書き込めない: {e!r}") from e
        self.n_sent += 1
        self.last_sent = time.monotonic()
        return self.last_sent - t0

    async def send(self, cmd: Command) -> None:
        self._check_connected()
        data = encode_command(cmd, self.robot, self.ble)
        async with self._lock:
            check_not_expired(cmd)
            await self._write(data)

    async def write_line(self, text: str) -> float:
        """生の 1 行（改行なしの ASCII、例えば "M,30,30,1"）を書く。BLE の確かめ（scripts/check_ble.py）用で、書式は
        確かめない（読めない指令で micro:bit が止まるかも確かめるため）。書き込み 1 回の時間を返す。"""
        if not text or not text.isascii() or not text.isprintable():
            raise ValueError(f"改行なしの表示できる ASCII の 1 行にする: {text!r}")
        data = (text + "\n").encode("ascii")
        if len(data) > self.ble.max_payload_bytes:
            raise ValueError(f"{text!r} が {len(data)} バイトで、1 回の書き込みの上限 {self.ble.max_payload_bytes} を超える")
        self._check_connected()
        async with self._lock:
            return await self._write(data)

    def status(self) -> LinkStatus:
        ok = self._client is not None and self._lost is None and self._client.is_connected
        return LinkStatus(ok, "接続中" if ok else (self._lost or "未接続"), self.n_sent, self.last_sent)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None
