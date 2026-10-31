"""実機の機器一式（USB カメラ、BLE の micro:bit、ロボットの上の ArUco、背景との差の検出と追跡、VLM の同定）とランタイムを
つなぐ（doc/real_robot_runbook.md）。話しかける画面のセッション（field.session.TalkSession）から、SimRig と同じ
呼び方（running / make_runtime / run_instruction / goal_rects / outcome / camera.frames / hom）で使う。

- 作業領域は校正（data/calibration/camera.yaml）で求めた大きさ。名前つきのゴールはその大きさへ写す（task.goal_rects_for_area）。
- 物体の検出は、校正の後に撮った空の床（背景）との差（perception_oblique.BackgroundDetector）。ロボットを除く範囲は
  field.yaml の real（実測の車体と高さ）。追跡・爪の中の補い・見失ったときの仕切り直しは Gazebo のリハーサルと同じ
  field.target_tracking.TargetTracking。
- 指示を受けたときの物体の同定は VLM（3090 の音声サーバーの /identify）。
- 真値は無い。成功は「ランタイムが追跡で全対象をゴールの中と判断した」こと。
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import math
import threading
from pathlib import Path
from typing import Awaitable, Callable

from ..calibration import CameraModel
from ..config import AllCfg
from ..geometry import Pose2D
from ..markers import load_markers_cfg
from ..perception import PerceptionCfg
from ..perception_oblique import BackgroundDetector
from ..robot_model import SIDES, jaw_polygons
from ..seg_detector import SegDetector
from ..task import goal_rects_for_area
from ..task_manager import TidyInstruction
from .calibration_web import area_problems, background_meta_path, load_background
from .interfaces import FieldStop
from .config import FieldCfg, RealCfg
from .hardware import ArucoRobotPose, BleRobotLink, MarkerNotVisible, OpenCVCamera, load_robot_marker_yaw
from .interfaces import FrameUnusable
from .motion_gain import OnlineMotionGain
from .runtime import FieldRuntime, MotionModel, Watched
from .target_tracking import TargetTracking

log = logging.getLogger(__name__)
# 物を探す範囲の、外の余白に足す物体の半分の長さ（ミニカー 80 mm の半分 + 余裕 10 mm、未検証）
ROI_OBJECT_MARGIN_M = 0.05
ROOT = Path(__file__).resolve().parents[3]   # リポジトリの根（config/seg_detector.yaml のモデルのパスの基準）


def real_robot_polygons(cfg: AllCfg, real: RealCfg, pose: Pose2D, jaw_angle: float) -> list[list[tuple[float, float]]]:
    """実機のロボットの部品の床の外形（車体の後ろ、グリッパーの取り付け部、残りが爪）。車体は実測（field.yaml の real）、
    爪は sim.yaml の形。高さは後ろが real.robot_mask_height_m、取り付け部が real.mount_height_m。"""
    x0, x1 = real.body_x_m
    xm = real.mount_x_m
    w = real.body_half_width_m
    box = lambda a, b: [pose.transform_point(p) for p in ((b, w), (a, w), (a, -w), (b, -w))]
    jaws = [[pose.transform_point(p) for p in poly] for side in SIDES for poly in jaw_polygons(cfg.sim, side, jaw_angle)]
    return [box(x0, xm), box(xm, x1), *jaws]


# 塊が大きすぎるときの画像を保存する間隔の下限（手が入ったままのとき、10 fps で溜めない。未検証）
ERROR_SAVE_INTERVAL_S = 5.0


class RecordingDetector(BackgroundDetector):
    """大きすぎる塊を捨てたとき、その画像と「物と見た所」（赤）・ロボットを除いた範囲（青）・塊の枠（黄）を残す（2026-10-22、
    実機で「塊が大きすぎる」で止まり、何が写っていたかログだけでは分からなかった。その後、止めずに捨てるようにした）。"""

    def __init__(self, *a, error_dir: Path, **k):
        super().__init__(*a, **k)
        self.error_dir = error_dir
        self._last_saved = -math.inf

    def __call__(self, frame, robot, jaw_angle):
        """大きすぎる塊は捨てて（drop_large）、ほかの物は返す。捨てた塊があれば、画像を ERROR_SAVE_INTERVAL_S ごとに残す。"""
        dets = super().__call__(frame, robot, jaw_angle)
        if self.last_large:
            import cv2
            import numpy as np
            import time as _time
            if _time.monotonic() - self._last_saved >= ERROR_SAVE_INTERVAL_S:
                self._last_saved = _time.monotonic()
                self.error_dir.mkdir(parents=True, exist_ok=True)
                stamp = _time.strftime("%Y%m%d_%H%M%S")
                bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
                mask = self.mask(frame.image.shape[:2], robot, jaw_angle)
                vis = bgr.copy()
                vis[self.foreground(frame.image, mask) > 0] = (0, 0, 255)
                vis[mask > 0] = (vis[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
                for _, (x0, y0, x1, y1) in self.last_large:
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 3)
                cv2.imwrite(str(self.error_dir / f"{stamp}_frame.png"), bgr)
                cv2.imwrite(str(self.error_dir / f"{stamp}_overlay.jpg"), vis)
                log.warning("背景との差の大きすぎる塊を捨てた（物ではない。カメラが動いたか、人・影が入った。続けば校正し直す）: "
                            "%s（画像 %s）", [(round(a, 4), b) for a, b in self.last_large], self.error_dir / stamp)
        return dets


class RecordingSegDetector(RecordingDetector):
    """RecordingDetector の塊を作るところだけを学習した検出器にしたもの（config/field.yaml の real.detector: learned）。
    大きすぎる塊を捨てたときの画像（背景との差を赤で）はそのまま残す。"""

    def __init__(self, model, *a, **k):
        super().__init__(*a, **k)
        self.model = model
        self.last_instances: list = []

    blobs = SegDetector.blobs


class RecordingPose(ArucoRobotPose):
    """ロボットのマーカーが見えないフレームを、最後に見えた姿勢とともに記録する（ERROR_SAVE_INTERVAL_S ごとに 1 枚、
    最後の姿勢のまわりを切り出す）。2026-10-22、実機で片付けの途中にマーカーが続けて見えなくなり止まったが、どこで・なぜかが
    ログから分からなかった。"""

    def __init__(self, *a, error_dir: Path, **k):
        super().__init__(*a, **k)
        self.error_dir = error_dir
        self._last_saved = -math.inf

    def estimate(self, frame):
        try:
            p = super().estimate(frame)
        except MarkerNotVisible:
            import time as _time
            import cv2
            lp = self.last_pose
            where = "なし" if lp is None else f"({lp.x:+.3f}, {lp.y:+.3f}) m、向き {math.degrees(lp.yaw):+.0f}°"
            if _time.monotonic() - self._last_saved >= ERROR_SAVE_INTERVAL_S:
                self._last_saved = _time.monotonic()
                self.error_dir.mkdir(parents=True, exist_ok=True)
                img = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
                if lp is not None:
                    u, v = (int(c) for c in self.cam.to_px([lp.xy], self.m.field.robot_marker_height_m)[0])
                    h, w = img.shape[:2]
                    img = img[max(0, v - 200):min(h, v + 200), max(0, u - 300):min(w, u + 300)]
                path = self.error_dir / f"{_time.strftime('%Y%m%d_%H%M%S')}_marker_lost.jpg"
                cv2.imwrite(str(path), img)
                log.warning("ロボットのマーカーが見えない。最後に見えた姿勢 %s（画像 %s）", where, path)
            raise
        self.last_pose = p
        return p


class RealRig:
    def __init__(self, cfg: AllCfg, f: FieldCfg, pcfg: PerceptionCfg, calib_path: Path, background_path: Path,
                 robot_marker_path: Path, identify_server, camera: OpenCVCamera, error_dir: Path):
        """identify_server: voice.client.ServerClient（/identify を持つもの）。None なら同定できない（指示を受けると止まる）。
        camera: アプリが開く 1 つのカメラ（校正の画面と共有する。同じカメラを 2 つ開くと先に開いた側が読めなくなる）。
        error_dir: 検出で止まったときの画像の保存先（部屋が写るので data/camera/ の下）。"""
        self.cfg, self.pcfg = cfg, pcfg
        # 実機だけの値で現場の設定を置き換える（色の確かめの上限。field.yaml の real）
        self.f = dataclasses.replace(
            f, perception=dataclasses.replace(f.perception, identity_chroma_tolerance=f.real.identity_chroma_tolerance),
            control=dataclasses.replace(f.control, persist=f.real.persist))
        self.paths = (calib_path, background_path, robot_marker_path)
        self._reload_lock = threading.Lock()
        self.error_dir = error_dir
        self.identify_server = identify_server
        self.camera = camera
        self.link = BleRobotLink(f.ble, cfg.sim.robot)
        # 走りながら割り出す速さの倍率（電池で変わる）。指示をまたいで持ち越す
        r = f.real
        self.motion_gain = OnlineMotionGain(
            MotionModel(r.motion_delay_s, r.motion_lag_s, r.motion_gain_linear, r.motion_gain_angular),
            window_pairs=r.motion_gain_window_pairs, min_pairs=r.motion_gain_min_pairs, command_ttl_s=f.control.command_ttl_s,
            max_linear_mps=cfg.sim.robot.linear_speed_mps, max_angular_radps=cfg.sim.robot.angular_speed_radps)
        self._key = self._files_key()
        self._load()

    def _files_key(self) -> tuple:
        """校正・背景の記録・マーカーの向きのファイルの中身（変わったら読み込み直す）。"""
        calib, _, marker = self.paths
        return tuple(p.read_bytes() if p.exists() else None for p in (calib, background_meta_path(calib), marker))

    def _load(self) -> None:
        from .rehearsal import load_background_cfg
        from .vlm_identify import VlmObjectIdentifier
        cfg, f = self.cfg, self.f
        calib_path, background_path, robot_marker_path = self.paths
        self.hom = CameraModel.load(calib_path)
        self.area_size = tuple(self.hom.work_area_size_m)
        m = load_markers_cfg()
        self.pose = RecordingPose(self.hom, m, load_robot_marker_yaw(robot_marker_path), error_dir=self.error_dir)
        background = load_background(calib_path, background_path)
        self.background_rgb = background
        bcfg = dataclasses.replace(load_background_cfg(), robot_height_m=f.real.robot_mask_height_m,
                                   robot_margin_m=f.real.robot_margin_m, close_px=f.real.close_px,
                                   color_split=f.real.color_split, merged_max_length_m=f.real.merged_max_length_m,
                                   track_chroma_gate=f.real.track_chroma_gate)
        g = cfg.sim.gripper
        # 物を探す範囲: 作業領域 + 学習で失敗にしない外の余白（walls.out_of_area_margin_m）+ 物体の半分の長さ（中心が余白の
        # 端にある物の塊を切らない。ミニカー 80 mm の半分 40 mm + 余裕 10 mm、未検証）。物とみなすのは、塊の中心が作業領域 +
        # 外の余白の中にあるものだけ（外の帯のソファの脚・影は大きくても止めない。2026-10-22）
        margin = cfg.sim.walls.out_of_area_margin_m + ROI_OBJECT_MARGIN_M
        roi = (self.area_size[0] / 2 + margin, self.area_size[1] / 2 + margin)
        args = (self.hom, background, bcfg, lambda r, j: real_robot_polygons(cfg, f.real, r, j), g.jaw_bottom_z_m + g.jaw_height_m,
                roi)
        kw = dict(keep_half_m=(self.area_size[0] / 2 + cfg.sim.walls.out_of_area_margin_m,
                               self.area_size[1] / 2 + cfg.sim.walls.out_of_area_margin_m),
                  drop_large=True, error_dir=self.error_dir,
                  body_heights_m=(f.real.robot_mask_height_m, f.real.mount_height_m))
        if f.real.detector == "background":
            self.detector = RecordingDetector(*args, **kw)
        elif f.real.detector == "learned":
            from .. import seg_detector
            scfg = seg_detector.load_seg_detector_cfg()
            path = Path(scfg.model) if Path(scfg.model).is_absolute() else ROOT / scfg.model
            self.detector = RecordingSegDetector(seg_detector.SegModel(path, scfg), *args, **kw)
        else:
            raise ValueError(f"real.detector は background か learned: {f.real.detector}")
        top = max(o.size_m[2] for o in cfg.sim.objects.values())   # 枠は一番高い物体（ミニカー）の上まで
        self.identifier = (None if self.identify_server is None
                           else VlmObjectIdentifier(self.identify_server, self.hom, top,
                                                    green_lime_split_hue_deg=f.real.green_lime_split_hue_deg))
        log.info("実機: 作業領域 %.3f × %.3f m、物体の検出 %s（ロボットを除く高さ %.3f m）、同定 %s", *self.area_size,
                 "背景との差" if f.real.detector == "background" else "学習した検出器", bcfg.robot_height_m,
                 "VLM" if self.identifier is not None else "なし")

    def reload_if_changed(self) -> bool:
        """校正・背景・マーカーの向きが読み込んだときから変わっていれば、読み込み直す（管理のアプリの中で校正し直したとき）。"""
        with self._reload_lock:   # 映像の配信・管理の検出・セッションが別のスレッドから呼ぶ
            key = self._files_key()
            if key == self._key:
                return False
            log.info("校正・背景・マーカーの向きが変わったので読み込み直す")
            self._load()
            self._key = key
            return True

    def goal_rects(self) -> dict:
        return goal_rects_for_area(self.cfg.sim, self.area_size)

    @staticmethod
    def outcome(report: dict) -> tuple[bool, str]:
        """実機に真値は無い。ランタイムは追跡で全対象がゴールの中と判断したときだけ完了を返す。"""
        return True, "追跡でゴールの中（実機に真値はない）"

    def reset_board(self, board) -> None:
        raise RuntimeError("実機の盤面は人が置き直す（置き直しの指令は無い）")

    def check_camera(self) -> None:
        """今のカメラの映像が校正の背景と合うか（カメラが動いていないか）。合わなければ FieldStop("camera_moved")。起動のときではなく
        動き始めるときに確かめる（起動したまま校正し直して、そのまま実機で動かすため。2026-10-25）。基準は起動のときのカメラ選びと同じ
        real.camera_match_max_diff。"""
        from .hardware import pick_camera_by_background
        try:
            frame = self.camera.frames.get()
        except LookupError as e:
            raise FieldStop("no_camera", "カメラの映像がまだ来ていない") from e
        try:
            pick_camera_by_background({0: frame.value.image}, self.background_rgb,
                                      self.f.real.camera_match_max_diff)
        except RuntimeError as e:
            raise FieldStop("camera_moved", f"カメラの映像が校正の背景と合わない。校正の画面で校正し直す（{e}）") from e

    def make_runtime(self, policy, telemetry=None, zones: tuple = ()) -> FieldRuntime:
        self.reload_if_changed()
        self.check_camera()
        problems = area_problems(self.cfg, load_markers_cfg(), self.hom)
        if problems:   # 前の版で保存した校正（2026-10-26 より前は確かめずに保存した）
            raise FieldStop("area_unfit", "。".join(problems))
        if self.identifier is None:
            raise RuntimeError("VLM の同定が無い（音声サーバーを指定する）")
        m, (nl, na) = self.motion_gain.model(), self.motion_gain.counts()
        log.info("先読みの速さの倍率（走りながら割り出した）: 前進 %.2f（組 %d）、旋回 %.2f（組 %d）。組 %d 未満は設定の値",
                 m.gain_linear, nl, m.gain_angular, na, self.f.real.motion_gain_min_pairs)
        return FieldRuntime(self.cfg, self.f, self.pcfg, self.camera, self.link, self.pose, self.identifier, self.hom,
                            policy, telemetry, zones,
                            tracking_factory=lambda p: TargetTracking(self.cfg, self.f.perception, p, self.detector),
                            area_size=self.area_size,
                            motion_model=self.motion_gain.model(), motion_gain=self.motion_gain)

    async def run_instruction(self, rt: FieldRuntime, get_instruction: Callable[[], Awaitable[TidyInstruction]],
                              watch: dict[str, Watched]) -> dict:
        return await rt.run(get_instruction, watch)

    @contextlib.asynccontextmanager
    async def running(self, extra_watch: dict[str, Watched] | None = None, settle_s: float = 0.0):
        """カメラを動かし、ランタイムに渡す watch を返す。抜けるときにカメラを止め、BLE を切る（切れるとロボットは止まる）。"""
        cam_t = asyncio.create_task(self.camera.run(), name="camera")
        watch = {"camera": Watched(cam_t, True), **(extra_watch or {})}
        try:
            yield watch
        finally:
            cam_t.cancel()
            await asyncio.gather(cam_t, return_exceptions=True)
            await self.link.close()
