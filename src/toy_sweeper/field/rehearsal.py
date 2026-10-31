"""Gazebo を 1 Action ずつ進めるリハーサル。pixi 環境（rclpy）でのみ使える。

実時間ではなく、撮影 → 検出・追跡 → Policy → Gazebo を 0.25 s 進める、を繰り返す。2026-10-18、学習と並べると
Gazebo（CPU で描画、フル HD の斜め上のカメラ）が実時間に追いつかなかった（0.157 s の遅れで止まった）ため。
検出（背景との差）・追跡・同定・Controller は現場のランタイムと同じ部品を使う。実時間の問題（遅れ、処理落ち）は
2D の実時間シミュレーション（field.sim）で確かめる。

話しかける画面（field.session.TalkSession）からは SimRig と同じ呼び方（running / make_runtime / run_instruction /
reset_board / camera.frames）で使える。
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import math
import threading
import time

import numpy as np

from ..config import AllCfg, load_yaml, CONFIG_DIR, strict_dataclass_from_dict
from ..geometry import Pose2D
from ..perception import PerceptionCfg, Tracker, chroma_distance
from ..perception_env import robot_polygons
from ..perception_oblique import BackgroundDetectCfg, BackgroundDetector
from ..snapshot import inside_goal
from ..spaces import Action, GripperState, drive_velocity
from ..task import EpisodeSpec
from ..task_manager import TidyInstruction
from .config import FieldCfg
from .interfaces import FieldStop, Frame, SimTruth, TrackView
from .latest import Latest
from .runtime import Controller, TrackedState, make_assignment
from .target_tracking import TargetTracking

log = logging.getLogger(__name__)
# _audit_identity: 対象の追跡が本人の真の位置からこれ以上離れ、別の物体の方が近ければ取り違えとみなす。観測のずれ
# （Gazebo のリハーサルで p95 5〜8 mm、挟んでいる間 7〜16 mm）より十分大きく、LEGO の長辺 32 mm 程度（記録用、未検証）
IDENTITY_AUDIT_MIN_M = 0.03
# _log_lost: 見失った対象の真の位置からこの距離以内の検出を、その対象の検出とみなす（上と同じ理由、記録用、未検証）
LOST_NEAR_M = 0.03



def lost_reason(truth_xy, dets, *, hidden: bool, ref_rgb, ref_len: float, chroma_tol: float, size_ratio: float,
                near_m: float) -> str:
    """見失った対象が、真の位置でなぜ追跡に付かないか（調べるためだけの記録、2026-10-19。フローリングの床の 60 盤面の失敗の
    半分が見失いだったため）。真の位置から near_m 以内の検出のうち、色度の差が chroma_tol 未満で長さの比が size_ratio 以下の
    ものがあれば "detected"、色の合うものが無ければ "color_off"、色は合うが大きさが違えば "size_off"。近くに検出が無ければ、
    ロボットの陰なら "hidden_by_robot"、そうでなければ "no_detection"。"""
    near = [d for d in dets if math.dist(d.pose.xy, truth_xy) <= near_m]
    if not near:
        return "hidden_by_robot" if hidden else "no_detection"
    color = [d for d in near if chroma_distance(d.mean_rgb, ref_rgb) < chroma_tol]
    if not color:
        return "color_off"
    if any(max(d.size_m[0], ref_len) / max(min(d.size_m[0], ref_len), 1e-6) <= size_ratio for d in color):
        return "detected"
    return "size_off"


def load_background_cfg() -> BackgroundDetectCfg:
    return strict_dataclass_from_dict(BackgroundDetectCfg, load_yaml("perception_oblique.yaml", CONFIG_DIR),
                                      "perception_oblique")


def unidentified_at_start(identifier) -> list[str] | None:
    """開始時に同定しなかった物体の名前。真値を知る同定（field.sim.SimTruthIdentifier）でだけ分かる。VLM の同定は真の物体を
    知らないので None（数えない）。"""
    from .sim import SimTruthIdentifier
    return identifier.missing if isinstance(identifier, SimTruthIdentifier) else None


def judged_targets(identifier, identified: list[str], truth_matching: list[str]) -> list[str]:
    """完了の真値の確かめで見る物体の名前。真値の同定（field.sim.SimTruthIdentifier）は、同定した対象（名前が真の物体の名前）。
    VLM の同定の名前（obj + 追跡の番号）は真の物体と関係ないので、指示に合う真の物体（truth_matching）。"""
    from .sim import SimTruthIdentifier
    return list(identified) if isinstance(identifier, SimTruthIdentifier) else list(truth_matching)


def truth_names(identifier, track_of: dict[str, int], tracks: dict, truth_poses: dict, tolerance_m: float) -> dict[str, str]:
    """同定した名前 → 真の物体の名前（調べるための記録が真値を引くため）。真値の同定はそのまま。VLM の同定の名前は真の物体と
    関係ないので、開始時の追跡の位置にいちばん近い真の物体（tolerance_m より遠ければ対応なし）。"""
    from .sim import SimTruthIdentifier
    if isinstance(identifier, SimTruthIdentifier):
        return {n: n for n in track_of}
    out = {}
    for n, tid in track_of.items():
        xy = tracks[tid].pose.xy
        m = min(truth_poses, key=lambda k: math.dist(truth_poses[k].xy, xy))
        if math.dist(truth_poses[m].xy, xy) <= tolerance_m:
            out[n] = m
    return out


class _Perception:
    def __init__(self):
        self.tracked: Latest[TrackedState] = Latest("tracked")


class RehearsalRuntime:
    """1 つの指示の実行の状態（映像の重ね描き draw_overlay と TalkSession が読む項目は FieldRuntime と同じ名前）。"""

    def __init__(self, policy, zones: tuple):
        self.policy, self.zones = policy, tuple(zones)
        self.phase = "init"
        self.controller: Controller | None = None
        self.assignment = None
        self.last_decision = None
        self.perception = _Perception()
        self.stop_requested: str | None = None

    def request_stop(self, source: str) -> None:
        log.warning("停止の要求: %s", source)
        self.stop_requested = self.stop_requested or source


class GazeboRehearsalRig:
    def __init__(self, cfg: AllCfg, f: FieldCfg, pcfg: PerceptionCfg, board: EpisodeSpec, seed: int, backend, camera,
                 relaunch=None, real_rules: bool = False):
        """backend: GazeboBackend。camera: gazebo.camera.OverheadCamera（frame() で今の sim 時刻の画像）。
        relaunch(board) → (backend, camera): World を新しい盤面で起動し直す（World には盤面の物体のモデルしか無いので、
        盤面を替えるときに要る）。None なら同じ物体の盤面にしか置き直せない。
        real_rules（2026-10-23）: 実機の判定の決まり（config/field.yaml の real、field.real_rig と同じ）で動かす。検出は
        close_px・color_split・探す範囲（作業領域 + 外の余白）・大きすぎる塊を捨てる、取り違えの色の上限は
        real.identity_chroma_tolerance、止めずに続ける（real.persist: 仕切り直しの回数で止めない。歩数の上限では止める。
        リハーサルは終わらせて数えるため）、開始時に同定できない物体があっても始める（SimTruthIdentifier の allow_missing）。
        ロボットを除く範囲（real.robot_mask_height_m・body_*）は実機の車体の値なので使わない（Gazebo の車体は sim.yaml）。"""
        self.relaunch = relaunch
        from .sim import SimArucoStandIn, SimTruthIdentifier
        self.real_rules = real_rules
        if real_rules:
            f = dataclasses.replace(
                f, perception=dataclasses.replace(f.perception, identity_chroma_tolerance=f.real.identity_chroma_tolerance),
                control=dataclasses.replace(f.control, persist=f.real.persist))
        self.cfg, self.f, self.pcfg, self.board = cfg, f, pcfg, board
        self.b, self.cam_src = backend, camera
        self.pose = SimArucoStandIn(f.sim, seed)
        self.identifier = SimTruthIdentifier(f.perception.association_tolerance_m, allow_missing=real_rules)
        self.truth_of: dict[str, str] = {}   # 同定した名前 → 真の物体の名前（truth_names、指示ごとに作る）
        self.hom = None             # calibrate() で求める CameraModel（映像の重ね描きにも使う）
        self.detector = None
        self.tracker_factory = lambda: Tracker(pcfg, self.detector.cfg.track_chroma_gate)   # calibrate() の検出の設定
        self.on_observe = None      # 観測のたびに呼ぶ on_observe(rt, TrackedState)（動画、scripts/eval_rehearsal.py --video）
        self.frames: Latest[Frame] = Latest("camera")
        self.camera = self          # web.create_app は rig.camera.frames を読む
        self._lock = threading.Lock()   # Gazebo を回すのは一度に 1 つのスレッドだけ
        from collections import Counter
        self.stats: Counter = Counter()  # 挟み損ね・付け直し・仕切り直しの回数（scripts/eval_rehearsal.py が数える）
        self._swaps: set = set()          # 取り違えを log に出した (対象, 相手) の組（_audit_identity）
        # 調べるための記録（_audit_identity）: フレームごと・対象ごとの (対象, 真値で正しい追跡か, RGB の差, 色度の差 ×255, 爪)
        self.color_audit: list = []
        self.tt: TargetTracking | None = None   # 1 つの指示の間の追跡（背景との差の検出のとき。_run で作る）
        self._seq = 0
        self.b.reset(board.robot, list(board.objects), GripperState.OPEN)

    # ---------- 校正 ----------
    def calibrate(self) -> None:
        """ロボットと物体を視野の外に出した空の床を撮り、床のマーカーでカメラの姿勢を求め、その画像を背景にする
        （実機の「校正する」と同じ: 何も置いていない床の写真）。その後、盤面を戻す。"""
        import cv2
        from ..calibration import estimate_extrinsics, intrinsics_from_fov
        from ..markers import load_markers_cfg
        with self._lock:
            away = Pose2D(0.0, -3.0, 0.0)    # カメラの後ろ（視野の外）
            self.b.reset(away, [], GripperState.OPEN)
            empty = self.cam_src.frame()
            gz = self.cfg.sim.gazebo
            intr = intrinsics_from_fov(tuple(gz.camera_image_px), gz.camera_hfov_rad)
            cam = estimate_extrinsics(cv2.cvtColor(empty, cv2.COLOR_RGB2BGR), intr, load_markers_cfg(), self.cfg.sim,
                                      self.cfg.sim.walls.thickness_m)
            log.info("カメラの校正: 位置 %s m、再投影誤差 %.2f px", np.round(cam.position(), 3).tolist(), cam.reproj_px)
            self.hom = cam
            g = self.cfg.sim.gripper
            if self.real_rules:   # field.real_rig.RealRig と同じ検出の設定と範囲
                from .real_rig import ROI_OBJECT_MARGIN_M
                r = self.f.real
                bcfg = dataclasses.replace(load_background_cfg(), close_px=r.close_px, color_split=r.color_split,
                                           merged_max_length_m=r.merged_max_length_m, track_chroma_gate=r.track_chroma_gate)
                hx, hy = (v / 2 for v in self.cfg.sim.work_area.size_m)
                out = self.cfg.sim.walls.out_of_area_margin_m
                self.detector = BackgroundDetector(cam, empty, bcfg, lambda r_, j: robot_polygons(self.cfg, r_, j),
                                                   g.jaw_bottom_z_m + g.jaw_height_m,
                                                   roi_half_m=(hx + out + ROI_OBJECT_MARGIN_M, hy + out + ROI_OBJECT_MARGIN_M),
                                                   keep_half_m=(hx + out, hy + out), drop_large=True)
            else:
                self.detector = BackgroundDetector(cam, empty, load_background_cfg(),
                                                   lambda r, j: robot_polygons(self.cfg, r, j), g.jaw_bottom_z_m + g.jaw_height_m)
            self.b.reset(self.board.robot, list(self.board.objects), GripperState.OPEN)
            self._publish(self._frame())

    def use_vlm_identifier(self, server) -> None:
        """指示を受けたときの物体の同定を VLM（音声サーバーの /identify）にする。calibrate() の後に呼ぶ。"""
        from .vlm_identify import VlmObjectIdentifier
        if self.hom is None:
            raise RuntimeError("calibrate() が先")
        top = max(o.size_m[2] for o in self.cfg.sim.objects.values())   # 枠は一番高い物体（ミニカー）の上まで
        self.identifier = VlmObjectIdentifier(server, self.hom, top)

    def use_truth_perception(self) -> None:
        """物体の検出と追跡に真値を使う（比べるため）。calibrate() の後に呼ぶ。"""
        from .sim import TruthDetector, TruthTracker
        if self.hom is None:
            raise RuntimeError("calibrate() が先")
        self.detector = TruthDetector(self.cfg, self.hom)
        self.tracker_factory = TruthTracker

    # ---------- 撮影と観測 ----------
    def _truth(self) -> SimTruth:
        g = self.cfg.sim.gripper
        gr = self.b.gripper_state()
        return SimTruth(robot=self.b.robot_pose(), objects=self.board.objects,
                        poses={o.name: self.b.object_pose(o.name) for o in self.board.objects},
                        jaw_angle=g.jaw_closed_angle_rad if gr == GripperState.CLOSED else g.jaw_open_angle_rad,
                        gripper=gr)

    def _frame(self) -> Frame:
        self._seq += 1
        return Frame(self.cam_src.frame(), time.monotonic(), self._seq, self._truth())

    def _publish(self, frame: Frame) -> None:
        self.frames.set(frame, frame.stamp)

    def _observe(self, rt: RehearsalRuntime, tracker) -> TrackedState:
        frame = self._frame()
        self._publish(frame)
        t = time.perf_counter()
        robot = self.pose.estimate(frame)
        if self.tt is not None:
            tracks = self.tt.process(frame, robot, frame.truth.gripper)
        else:
            tracks = tracker.update(self.detector(frame, robot, frame.truth.jaw_angle))
        if rt.assignment is not None:
            if log.isEnabledFor(logging.DEBUG) and self.tt is not None:
                self._debug_frame(rt, tracks, robot, self.tt.last_dets + self.tt.last_held, self.tt.last_held, frame.truth)
            self._audit_identity(rt, tracks, frame.truth)
        views = {tid: TrackView(tid, tr.pose, tr.size_m, tr.mean_rgb, tr.missed, tr.hits) for tid, tr in tracks.items()}
        ts = TrackedState(frame, robot, views, time.perf_counter() - t)
        rt.perception.tracked.set(ts, frame.stamp)
        if self.on_observe is not None:
            self.on_observe(rt, ts)
        return ts

    def _audit_identity(self, rt: RehearsalRuntime, tracks: dict, truth: SimTruth) -> None:
        """調べるためだけの記録（動きは変えない。2026-10-19、リハーサルの失敗の多くが見失い・取り違えだったため）: 対象の
        追跡（今のフレームで見えているもの）が、真の位置では別の物体の方に近ければ、取り違えとして記録する（対象と相手の
        組ごとに最初の 1 回だけ log に出す）。実カメラでは真値が無いので使えない。"""
        a = rt.assignment
        for n in a.episode.target_names:
            tid = a.track_of[n]
            if tid not in tracks or tracks[tid].missed > 0 or n not in self.truth_of:
                continue
            p = tracks[tid].pose.xy
            d_self = math.dist(p, truth.poses[self.truth_of[n]].xy)
            m, d_m = min(((o.name, math.dist(p, truth.poses[o.name].xy)) for o in truth.objects), key=lambda x: x[1])
            swapped = m != self.truth_of[n] and d_self > IDENTITY_AUDIT_MIN_M
            rgb, ref = tracks[tid].mean_rgb, a.ref_rgb[n]
            gc = self._gripper_center(truth.robot)
            self.color_audit.append((n, not swapped, round(math.dist(rgb, ref), 1),
                                     round(chroma_distance(rgb, ref), 2), truth.gripper.name,
                                     tuple(round(x) for x in rgb), round(1000 * math.dist(p, gc)),
                                     tuple(round(1000 * x) for x in tracks[tid].size_m)))
            if swapped:
                self.stats["truth_swap_frames"] += 1
                if (n, m) not in self._swaps:
                    self._swaps.add((n, m))
                    o = next(x for x in truth.objects if x.name == m)
                    log.info("取り違え（真値で確かめた）: 対象 %s の追跡 %d が %s（%s %s）の位置 %.0f mm にある（%s 本人からは %.0f mm）。"
                             "爪 %s、ロボットから %.0f mm", n, tid, m, o.color, o.type.name, 1000 * d_m, n, 1000 * d_self,
                             truth.gripper.name, 1000 * math.dist(p, truth.robot.xy))

    def _gripper_center(self, robot: Pose2D) -> tuple[float, float]:
        from ..robot_model import gripper_center_x
        return robot.transform_point((gripper_center_x(self.cfg.sim), 0.0))

    def _note_close(self, rt: RehearsalRuntime, ts: TrackedState) -> None:
        """爪を閉じる直前（TargetTracking.note_close）と、調べるための記録。"""
        a = rt.assignment
        gc = self._gripper_center(ts.robot)
        self.tt.note_close(ts.tracks, ts.robot)
        # 調べるための記録（2026-10-19、閉じた爪の中の対象を見失った）: 閉じる直前の、対象の追跡と真の位置の爪の中心からの距離
        truth = self._truth()
        tgc = self._gripper_center(truth.robot)
        log.info("爪を閉じる: 挟んだとみなす対象 %s。対象ごとの爪の中心からの距離 mm（追跡、見えていないフレーム数、真値）%s",
                 self.tt.held_target, {n: (round(1000 * math.dist(ts.tracks[a.track_of[n]].pose.xy, gc))
                                           if a.track_of[n] in ts.tracks else None,
                                           ts.tracks[a.track_of[n]].missed if a.track_of[n] in ts.tracks else None,
                                           round(1000 * math.dist(truth.poses[self.truth_of[n]].xy, tgc))
                                           if n in self.truth_of else None) for n in a.episode.target_names})

    def _debug_frame(self, rt: RehearsalRuntime, tracks: dict, robot: Pose2D, dets: list, held: list, truth: SimTruth) -> None:
        """調べるためだけの記録（log が DEBUG のとき、2026-10-19）: フレームごとの対象の追跡と真値、爪の中心の近くの検出。"""
        gc = self._gripper_center(robot)
        a = rt.assignment
        row = {}
        for n in a.episode.target_names:
            tid = a.track_of[n]
            t = tracks.get(tid)
            row[n] = (tid, None if t is None else round(1000 * math.dist(t.pose.xy, gc)), None if t is None else t.missed,
                      round(1000 * math.dist(truth.poses[self.truth_of[n]].xy, self._gripper_center(truth.robot)))
                      if n in self.truth_of else None)
        near = [(round(1000 * math.dist(d.pose.xy, gc)), round(1000 * d.size_m[0]), tuple(round(v) for v in d.mean_rgb))
                for d in dets if math.dist(d.pose.xy, gc) < 0.06]
        log.debug("フレーム %d: 爪 %s 挟んだ対象 %s 補い %d | 対象（追跡, 追跡の爪の中心からの mm, missed, 真値の mm）%s | "
                  "爪の中心から 60 mm 以内の検出（mm, 長さ, 色）%s", self._seq, truth.gripper.name, self.tt.held_target, len(held),
                  row, near)

    def _log_lost(self, rt: RehearsalRuntime, tracker, c, when: str) -> None:
        """調べるためだけの記録（動きは変えない。2026-10-19）: 追跡に付いていない対象ごとに、真の位置での理由（lost_reason）、
        爪の中心・最後の観測位置からの距離、爪の状態を log に出し、理由を stats（lost_<理由>）に数える。"""
        a = rt.assignment
        truth = self._truth()
        gc = self._gripper_center(truth.robot)
        for n in a.episode.target_names:
            tid = a.track_of[n]
            if (tid in tracker.tracks and tracker.tracks[tid].missed == 0) or n not in self.truth_of:
                continue
            xy = truth.poses[self.truth_of[n]].xy
            tt = self.tt
            hidden = tt.last_mask is not None and self.detector.hidden_by_robot(tt.last_mask, xy)
            why = lost_reason(xy, tt.last_dets, hidden=hidden, ref_rgb=a.ref_rgb[n], ref_len=tt.ref_len[n],
                              chroma_tol=self.f.perception.identity_chroma_tolerance,
                              size_ratio=c.reassociate_size_ratio, near_m=LOST_NEAR_M)
            near = [(round(1000 * math.dist(d.pose.xy, xy)), tuple(round(v) for v in d.mean_rgb),
                     round(chroma_distance(d.mean_rgb, a.ref_rgb[n]), 1), round(1000 * d.size_m[0]))
                    for d in tt.last_dets if math.dist(d.pose.xy, xy) <= 2 * LOST_NEAR_M]
            log.info("見失いの真値（%s）: 対象 %s は %s。真の位置 (%.3f, %.3f)、爪の中心から %.0f mm、最後の観測から %.0f mm、"
                     "爪 %s、近くの検出（距離 mm、色、色度の差、長さ mm）%s", when, n, why, xy[0], xy[1],
                     1000 * math.dist(xy, gc), 1000 * math.dist(xy, tt.last_xy[n]), truth.gripper.name, near)
            if when == "見失った時点":
                self.stats[f"lost_{why}"] += 1

    def _recover(self, rt: RehearsalRuntime, tracker, ts: TrackedState, c) -> TrackedState:
        """見失った対象の仕切り直し: 爪を開いて下がり、撮り直して、同じ色・大きさの物体を広めに探して付け直す
        （付け直しは TargetTracking.reattach_after_recover）。"""
        T = self.cfg.task.control.action_duration_s
        log.info("対象を見失ったので仕切り直す: 爪を開いて %d 歩下がる", c.recover_back_steps)
        self._log_lost(rt, tracker, c, "見失った時点")
        self.tt.begin_recover()
        rt.phase = "recovering"
        self.b.actuate_gripper(False)
        for _ in range(c.recover_back_steps):
            self.b.drive(*drive_velocity(self.cfg.sim.robot, Action.BACKWARD), T)
            ts = self._observe(rt, tracker)
        # 実機の決まり（persist、field.runtime と同じ）: 見つからない対象はあきらめて、ほかの対象を続ける（2026-10-23）
        self.tt.settle_after_recover(self.f.control.persist, on_fail=lambda: self._log_lost(rt, tracker, c, "下がった後"))
        if not rt.assignment.episode.target_names:
            raise FieldStop("target_lost", f"仕切り直しても対象 {rt.assignment.given_up} が見つからない")
        rt.phase = "control"
        rt.policy.reset()
        # 下がった分を「直前 1 歩の動き」に入れない（4 歩分は Observation の値域を超える。2026-10-18、例外で止まった）
        rt.controller.last_robot = ts.robot
        rt.controller.prev_action = Action.BACKWARD
        rt.controller.gripper = GripperState.OPEN       # 仕切り直しで爪を開いた
        return ts

    # ---------- SimRig と同じ呼び方 ----------
    @contextlib.asynccontextmanager
    async def running(self, extra_watch=None, settle_s: float = 0.0):
        if self.detector is None:
            raise RuntimeError("calibrate() が先")
        yield dict(extra_watch or {})

    def reload_if_changed(self) -> bool:
        """校正のファイルは無い（calibrate() で Gazebo のカメラから求める。field.rig.SimRig と同じ形にするため）。"""
        return False

    @property
    def area_size(self) -> tuple[float, float]:
        """作業領域の (幅, 奥行き)（Gazebo の World の大きさ sim.work_area、真値）。"""
        return tuple(self.cfg.sim.work_area.size_m)

    def goal_rects(self) -> dict:
        from ..task import goal_rects
        return goal_rects(self.cfg.sim)

    @staticmethod
    def outcome(report: dict) -> tuple[bool, str]:
        """SimRig.outcome と同じ（真値でゴールの中か）。"""
        return all(report["true_targets_in_goal"].values()), f"真値でゴールの中 {report['true_targets_in_goal']}"

    def make_runtime(self, policy, telemetry=None, zones: tuple = ()) -> RehearsalRuntime:
        return RehearsalRuntime(policy, zones)

    def reset_board(self, board: EpisodeSpec) -> None:
        """盤面を置き直す。物体のモデルが変わるなら Gazebo を起動し直す（カメラは同じ位置なので、校正と背景は使い続ける）。"""
        from ..gazebo.sdf_gen import board_models
        with self._lock:
            if board_models(board.objects) != board_models(self.board.objects):
                if self.relaunch is None:
                    raise RuntimeError("物体の違う盤面には、Gazebo を起動し直さないと置き直せない（relaunch が無い）")
                log.info("盤面の物体が変わるので Gazebo を起動し直す")
                self.b, self.cam_src = self.relaunch(board)
            self.board = board
            self.b.reset(board.robot, list(board.objects), GripperState.OPEN)
            self._publish(self._frame())

    async def run_instruction(self, rt: RehearsalRuntime, get_instruction, watch) -> dict:
        ins = await get_instruction()
        return await asyncio.to_thread(self._run, rt, ins)

    def _outside_m(self, goal, otype, pose) -> float:
        from ..geometry import box_corners, signed_distance_to_rect
        from ..spaces import OBJECT_TYPE_NAMES
        size = self.cfg.sim.objects[OBJECT_TYPE_NAMES[otype]].size_m[:2]
        return max(signed_distance_to_rect(c, goal) for c in box_corners(pose, size))

    def _note_obs_error(self, rt: RehearsalRuntime, ts) -> None:
        """見えている（この歩で検出された、挟んでいない）対象の、観測 − 真値（カメラの検出の誤差を測るため）。"""
        from ..geometry import wrap_angle
        a = rt.assignment
        for n in a.episode.target_names:
            tid = a.track_of[n]
            if tid not in ts.tracks or ts.tracks[tid].missed > 0 or (self.tt is not None and n == self.tt.held_target):
                continue
            if n not in self.truth_of:
                continue
            o, t = ts.tracks[tid].pose, self.b.object_pose(self.truth_of[n])
            # 長方形の向きは 180° の区別がないので、差は [-π/2, π/2) に折り返す
            dyaw = wrap_angle(2 * (o.yaw - t.yaw)) / 2
            self.obs_errors.append((o.x - t.x, o.y - t.y, dyaw))

    def _run(self, rt: RehearsalRuntime, ins: TidyInstruction) -> dict:
        with self._lock:
            T = self.cfg.task.control.action_duration_s
            tracker = self.tracker_factory()
            bg = isinstance(self.detector, BackgroundDetector)
            self.tt = TargetTracking(self.cfg, self.f.perception, self.pcfg, self.detector, tracker, self.stats) if bg else None
            rt.phase = "opening_gripper"
            self.b.actuate_gripper(False)
            rt.phase = "assigning"
            ts = self._observe(rt, tracker)
            ts = self._observe(rt, tracker)      # 追跡が落ち着くように 2 枚目で割り当てる
            # v30 以降の方策（観測に作業領域の大きさ）には、Gazebo の World の大きさ（sim.work_area、真値）を渡す
            area = self.area_size if self.cfg.task.observation.area_size else None
            rt.assignment = make_assignment(self.cfg, self.identifier, ts, ins, rt.zones, self.f.perception.goal_margin_m,
                                            area_size=area)
            self.truth_of = truth_names(self.identifier, rt.assignment.track_of, ts.tracks, self._truth().poses,
                                        self.f.perception.association_tolerance_m)
            if set(self.truth_of) != set(rt.assignment.track_of):
                log.info("真の物体と対応しない同定（調べるための記録から外す）: %s",
                         sorted(set(rt.assignment.track_of) - set(self.truth_of)))
            missing = unidentified_at_start(self.identifier)
            if missing:   # 実機の決まり（allow_missing）: 1 つの塊に見えて同定しなかった物体
                self.stats["start_unidentified"] += len(missing)
                log.info("開始時に同定しなかった物体（ほかの物とくっついて 1 つの塊に見える）: %s", missing)
            # 調べるための記録（2026-10-19）: 同定した物体ごとの、描画された色（追跡の画素の平均）。色の上限の根拠を測る
            specs = {o.name: o for o in rt.assignment.episode.objects}
            log.info("同定した物体の色: %s", {n: (specs[n].color, specs[n].type.name, tuple(round(x) for x in ts.tracks[t].mean_rgb))
                                            for n, t in rt.assignment.track_of.items()})
            a = rt.assignment
            if self.tt is not None:
                self.tt.start(a, ts.tracks)
            ctl = Controller(self.cfg, self.pcfg, rt.policy, rt.assignment, self.f.perception)
            rt.controller = ctl
            rt.policy.reset()
            rt.phase = "control"
            t0 = time.monotonic()
            n_recover, n_unseen = 0, 0
            self.obs_errors: list[tuple[float, float, float]] = []   # 見えている対象の (Δx, Δy, Δyaw)、観測 − 真値
            for k in range(ctl.max_steps() + 1):
                self._note_obs_error(rt, ts)
                # ロボットが作業領域 + 外の余白（学習で失敗にする範囲）を出たら止める（2026-10-23、実機の決まりで見失った対象の
                # 仕切り直し（4 歩下がる）を止めずに繰り返し、作業領域の 1.6 m 後ろまで下がって Observation の値域を外れた）
                rp, (hx, hy) = self.b.robot_pose(), (v / 2 + self.cfg.sim.walls.out_of_area_margin_m for v in self.cfg.sim.work_area.size_m)
                if abs(rp.x) > hx or abs(rp.y) > hy:
                    raise FieldStop("robot_out_of_area", f"ロボットが作業領域の外に出た（{rp.x:.2f}, {rp.y:.2f}）")
                if rt.stop_requested is not None:
                    raise FieldStop("stop_request", f"停止の要求（{rt.stop_requested}）")
                c = self.detector.cfg if self.tt is not None else None
                unseen = [n for n in rt.assignment.episode.target_names
                          if rt.assignment.track_of[n] in ts.tracks and ts.tracks[rt.assignment.track_of[n]].missed > 0]
                n_unseen = n_unseen + 1 if unseen else 0
                if c is not None and n_unseen >= c.recover_unseen_steps:
                    if n_recover >= c.recover_max and not self.f.control.persist:
                        raise FieldStop("target_hidden", f"仕切り直しても対象 {unseen} が見えない")
                    n_recover += 1
                    log.info("対象 %s が %d 歩見えないので仕切り直す", unseen, n_unseen)
                    ts = self._recover(rt, tracker, ts, c)
                    n_unseen = 0
                    continue
                try:
                    d = ctl.decide(ts)
                except FieldStop as e:
                    # 実機の決まり（persist、field.runtime と同じ）では、色が変わった（入れ替わったおそれ）ときも仕切り直す
                    again = ("target_lost", "target_identity") if self.f.control.persist else ("target_lost",)
                    if e.reason not in again or c is None or (n_recover >= c.recover_max and not self.f.control.persist):
                        raise
                    n_recover += 1
                    ts = self._recover(rt, tracker, ts, c)
                    n_unseen = 0
                    continue
                rt.last_decision = d
                if d.done:
                    ep = rt.assignment.episode
                    types = {o.name: o.type for o in self.board.objects}
                    # 真値の判定は元の大きさのゴールで（ep.goal は perception.goal_margin_m だけ縮めてある）
                    from ..task import goal_rects
                    full = goal_rects(self.cfg.sim)[ep.goal_name]
                    # 指示に合う真の物体
                    from ..task_manager import select_targets
                    truth_t = select_targets(list(self.board.objects), {o.name: o.pose for o in self.board.objects}, ins,
                                             self.board.robot, self.cfg)
                    judged = judged_targets(self.identifier, ep.target_names, truth_t)
                    in_goal = {n: inside_goal(self.cfg.sim, full, types[n], self.b.object_pose(n)) for n in judged}
                    # 真値の同定のとき: 指示に合う真の物体のうち、開始時に同定しなかったもの（実機の決まりの allow_missing）。
                    # これも入って成功。VLM の同定では judged が指示に合う真の物体のすべてなので無い
                    missed = {n: inside_goal(self.cfg.sim, full, types[n], self.b.object_pose(n))
                              for n in truth_t if n not in judged}
                    # 4 隅のうち最も外の隅の、元の大きさのゴールの縁までの距離（m、入っていれば 0 以下）。真値と観測
                    true_out = {n: self._outside_m(full, types[n], self.b.object_pose(n)) for n in judged}
                    obs_types = {o.name: o.type for o in ep.objects}   # 同定の名前（VLM では真の物体の名前と別）
                    obs_out = {n: self._outside_m(full, obs_types[n], ts.tracks[a.track_of[n]].pose)
                               for n in ep.target_names if a.track_of[n] in ts.tracks}
                    log.info("完了: %d 歩（%.0f s）、真値でゴールの中 %s、縁からのはみ出し 真値 %s 観測 %s", k,
                             time.monotonic() - t0, in_goal, {n: round(1000 * v, 1) for n, v in true_out.items()},
                             {n: round(1000 * v, 1) for n, v in obs_out.items()})
                    if missed:
                        log.info("開始時に同定しなかった対象（真値）: ゴールの中 %s", missed)
                    return {"result": "success", "steps": k, "targets": list(ep.target_names),
                            "true_targets_in_goal": in_goal, "true_missed_targets_in_goal": missed,
                            "given_up": list(a.given_up), "true_outside_mm": {n: 1000 * v for n, v in true_out.items()},
                            "obs_outside_mm": {n: 1000 * v for n, v in obs_out.items()}}
                if d.action == Action.CLOSE_GRIPPER and self.tt is not None:
                    self._note_close(rt, ts)
                if d.action in (Action.OPEN_GRIPPER, Action.CLOSE_GRIPPER):
                    self.b.actuate_gripper(d.action == Action.CLOSE_GRIPPER)
                else:
                    self.b.drive(*drive_velocity(self.cfg.sim.robot, d.action), T)
                ctl.commit(d.action)
                ts = self._observe(rt, tracker)
            raise FieldStop("timeout", f"{ctl.max_steps()} 歩で終わらない")
