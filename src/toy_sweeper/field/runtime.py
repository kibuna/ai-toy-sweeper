"""現場側の非同期ランタイム（MacBook、ROS なし、asyncio）。

    カメラ ──Latest[Frame]──▶ 検出・追跡（スレッド）──Latest[TrackedState]──▶ 制御（0.25 s 周期）──▶ 指令の門 ──▶ RobotLink
                                                                            ▲                     ▲
                                              安全監視（0.05 s 周期）: フレームの古さ・追跡の古さ・通信・停止の要求・制御の途絶

- センサーのデータは Latest（最新の値だけ）で受け渡す。待ち行列は使わない。
- 重い処理・止まる呼び出し（検出、ArUco、同定、描画、cv2 の読み込み）は asyncio.to_thread でスレッドに出す。
- 止める理由が生じたら、指令の門（CommandGate）を閉じて停止を送り、FieldStop を投げる（黙って続けない）。
"""
from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np

from ..config import AllCfg
from ..geometry import Pose2D, Rect, wrap_angle
from ..perception import Homography, PerceptionCfg, Tracker, chroma_distance, detect_objects
from ..robot_model import gripper_center_x
from ..perception_env import robot_polygons
from ..snapshot import build_snapshot, pending_fixes, target_held
from ..spaces import (Action, GripperState, ObservationBuilder, TaskMode, WorldSnapshot,
                      drive_velocity)
from ..task import EpisodeSpec, goal_rects, goal_rects_for_area
from ..task_manager import matches_instruction, TidyInstruction, select_targets
from .config import FieldCfg, FieldPerceptionCfg
from .interfaces import (CameraSource, Command, CommandExpired, DriveCommand, FieldStop, Frame, GripperCommand, LinkError,
                         FrameUnusable, ObjectIdentifier, PoseLost, RobotLink, RobotPoseEstimator, StopCommand, TrackView)
from .latest import Latest

log = logging.getLogger(__name__)

# 現場のタスクは Stage の盤面生成器から引いたものではないので、EpisodeSpec.stage にはこの印を入れる（観測には使わない）
FIELD_STAGE = -1
# Policy に渡す観測の古さ。学習では常に 0（ToySweeperEnv）なので、実際の古さ（約 0.2 s）は入れず 0 にする。
# 入れると学習で一度も見ていない値になる。古さは安全監視の上限（safety.max_obs_age_s）で抑える
OBS_AGE_FED_TO_POLICY_S = 0.0
# 開始時に開く指令を送るまでの、ロボットのマスク（検出から除く領域）に使う爪の状態。実機では開始前の状態は分からない
INITIAL_GRIPPER_FOR_MASK = GripperState.OPEN
# ロボットの向きの跳びの上限: 前に取れた姿勢からの経過時間に回れる量（旋回の速さ）の YAW_JUMP_RATE_FACTOR 倍 + YAW_JUMP_MARGIN_RAD
# を超えたら、マーカーの向きの読み違いとして扱う（2026-10-26、実機で 1 フレームに約 150° 跳んだ。10 fps で回れるのは 1 フレーム
# 約 16°、2 倍 + 17° で 49°。マーカーの向きのぶれは 1〜2°。どれも未検証）。同じ跳びが JUMP_ACCEPT_FRAMES フレーム続いたら
# 本当に向きが変わった（持ち上げて置き直した）とみなす
# 止めずに待つ（persist）間、観測が古いのがこの秒数続いたら 1 回知らせる（マーカーが読めない所で止まったまま、何も言わずに 31 秒
# 待った。2026-10-26。止まって待つ 1 回の多くは 1 s 未満なので、それより十分長い 5 s。未検証）
STALE_NOTICE_S = 5.0
STALE_NOTICE_TEXT = "カメラで ロボットが みえないよ。みえる ところに うごかしてね"
# マーカーが読めなくなって止まって待つ間は、最後の動きを逆にして 1 歩ずつ戻る（2026-10-26。
# 実機で奥へ進んで読めなくなり、20〜30 s 止まったままだった）。
# UNDO_MAX: 戻る歩数の上限。読めない間も姿勢の見積もり（pose_predict_max_s 0.5 s = 4 周期）で動くので、その 4 歩 + 2（未検証）。
# UNDO_WAIT_STEPS: 1 歩戻るごとに、マーカーが読めるかを待つ周期（カメラ 10 fps の 2 フレーム分、未検証）
UNDO_MAX = 6
UNDO_WAIT_STEPS = 2
# 記録した動きを戻し終えても読めないとき、最後に見えた姿勢から作業領域の中心へ近づく向きに前進・後退する歩数の上限
# （2026-10-30 実機 13:04: 奥の隅で画面の外へ出て、6 歩戻っても見えず止まって待ち続けた。12 歩は 0.145 m/s × 0.125 s で
# 約 22 cm。学習で失敗にしない外の余白 10 cm + マーカーの半分の対角 + 余裕。未検証）
UNDO_TOWARD_CENTER_MAX = 12


def step_toward_center(pose: Pose2D) -> Action:
    """作業領域の中心（原点）へ近づく向きの 1 歩（前進か後退。向きは変えない）。"""
    ahead = math.cos(pose.yaw) * -pose.x + math.sin(pose.yaw) * -pose.y
    return Action.FORWARD if ahead >= 0 else Action.BACKWARD
YAW_JUMP_RATE_FACTOR = 2.0
YAW_JUMP_MARGIN_RAD = 0.3
JUMP_ACCEPT_FRAMES = 3


@dataclass(frozen=True)
class MotionModel:
    """送った走行の指令に対するロボットの動き方（姿勢の先読みに使う）。指令は delay_s 遅れて倍率 gain_* で効き始め、実際の
    速さは時定数 lag_s で追いつく（止めた後も lag_s のあいだ惰性で動く）。scripts/fit_motion_from_telemetry.py で割り出す。"""
    delay_s: float
    lag_s: float
    gain_linear: float
    gain_angular: float


# 指令がその場から学習どおりの速さで効く（シミュレーションの世界・2026-10-30 より前の先読み）
IDEAL_MOTION = MotionModel(0.0, 0.0, 1.0, 1.0)
MOTION_DT_S = 0.002   # 動き方のモデルで先読みするときの積分の刻み（2 ms。遅れ・時定数の 1/40 以下）


# ---------- 時間の記録 ----------
class TimingStats:
    def __init__(self):
        self.samples: dict[str, list[float]] = {}

    def add(self, name: str, v: float) -> None:
        self.samples.setdefault(name, []).append(float(v))

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for k, v in self.samples.items():
            a = np.asarray(v)
            out[k] = {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "p50": float(np.percentile(a, 50)),
                      "p95": float(np.percentile(a, 95)), "max": float(a.max()), "min": float(a.min())}
        return out


class LoggingTelemetry:
    """3090 側への状態の送信の代わり（今はログに JSON で書くだけ。WebSocket は後で）。"""

    def __init__(self, name: str = "toy_sweeper.field.telemetry"):
        self.log = logging.getLogger(name)

    def publish(self, kind: str, data: dict) -> None:
        if self.log.isEnabledFor(logging.DEBUG):
            self.log.debug("%s", json.dumps({"kind": kind, **data}, ensure_ascii=False, default=str))


# ---------- 検出・追跡 ----------
@dataclass(frozen=True)
class TrackedState:
    frame: Frame
    robot: Pose2D                     # ArUco（シミュレーションでは代わり）の推定
    tracks: dict[int, TrackView]
    perception_s: float               # 姿勢の推定 + 検出 + 追跡にかかった時間


class Perception:
    def __init__(self, cfg: AllCfg, pcfg: PerceptionCfg, camera: CameraSource, pose: RobotPoseEstimator,
                 hom: Homography, gripper_at: Callable[[float], GripperState], stats: TimingStats, detector=None,
                 tracker=None, tracking=None, predict=None, predict_max_s: float = 0.0,
                 max_yaw_rate_radps: float | None = None):
        """detector: (frame, ロボットの姿勢, 爪の角度) → 検出のリスト。None ならカメラ画像から検出する（detect_objects）。
        tracker: update(検出) → {ID: Track}。None なら近い順の対応付け（perception.Tracker）。
        シミュレーションの真値を使う検出と追跡（field.sim.TruthDetector / TruthTracker）を差し込める。
        tracking: field.target_tracking.TargetTracking（斜めのカメラの背景との差の検出と、爪の中の補い・付け直し。
        リハーサルと同じもの）。渡したら detector・tracker は使わない。"""
        if tracking is not None and (detector is not None or tracker is not None):
            raise ValueError("tracking を渡すときは detector・tracker を渡さない（tracking が検出と追跡を持つ）")
        self.tracking = tracking
        self.cfg, self.pcfg = cfg, pcfg
        self.detector = detector
        self.camera, self.pose, self.hom = camera, pose, hom
        self.gripper_at = gripper_at
        # 姿勢が一瞬取れないときの見積もり: predict(最後の姿勢, その時刻, 今の時刻) → 姿勢（直前の指令から）。最後に取れてから
        # predict_max_s までだけ使う（0 は使わない）
        self.predict, self.predict_max_s = predict, predict_max_s
        self.last_pose: tuple[Pose2D, float] | None = None
        self.predicted_frames = 0
        # 毎フレームの姿勢を渡す先（撮った時刻・姿勢・読めたか見積もりか）。実機の動き方を走りの記録から割り出す（2026-10-30）
        self.on_pose = None
        # 向きの跳びを読み違いとして捨てる（None は確かめない）。続けて捨てたフレーム数と、捨てた回数
        self.max_yaw_rate_radps = max_yaw_rate_radps
        self._jumps = 0
        self.yaw_jump_frames = 0
        self.stats = stats
        self.tracker = Tracker(pcfg, 0.0) if tracker is None else tracker   # 上からの合成画像の検出（色の門なし、今まで）
        self.tracked: Latest[TrackedState] = Latest("tracked")
        self.skipped_frames = 0
        self.pose_lost_frames = 0   # 知覚に使えず飛ばしたフレーム（姿勢が取れない、塊が大きすぎる）

    def _process(self, frame: Frame, gripper: GripperState) -> tuple[Pose2D, dict[int, TrackView], float]:
        t = time.perf_counter()
        try:
            robot = self.pose.estimate(frame)
            lp = self.last_pose
            if self.max_yaw_rate_radps is not None and lp is not None:
                limit = (YAW_JUMP_RATE_FACTOR * self.max_yaw_rate_radps * max(0.0, frame.stamp - lp[1])
                         + YAW_JUMP_MARGIN_RAD)
                jump = abs(wrap_angle(robot.yaw - lp[0].yaw))
                if jump > limit and self._jumps + 1 < JUMP_ACCEPT_FRAMES:
                    self._jumps += 1
                    self.yaw_jump_frames += 1
                    raise PoseLost(f"ロボットの向きが {math.degrees(jump):.0f}° 跳んだ（上限 {math.degrees(limit):.0f}°）。"
                                   "マーカーの向きの読み違いとして扱う")
            self._jumps = 0
            self.last_pose = (robot, frame.stamp)
            measured = True
        except PoseLost:
            lp = self.last_pose
            if self.predict is None or lp is None or frame.stamp - lp[1] > self.predict_max_s:
                raise
            robot = self.predict(lp[0], lp[1], frame.stamp)
            self.predicted_frames += 1
            measured = False
        if self.on_pose is not None:
            self.on_pose({"stamp": frame.stamp, "seq": frame.seq, "x": robot.x, "y": robot.y, "yaw": robot.yaw,
                          "measured": measured})
        g = self.cfg.sim.gripper
        jaw_angle = g.jaw_closed_angle_rad if gripper == GripperState.CLOSED else g.jaw_open_angle_rad
        if self.tracking is not None:
            tracks = self.tracking.process(frame, robot, gripper)
            views = {tid: TrackView(tid, tr.pose, tr.size_m, tr.mean_rgb, tr.missed, tr.hits) for tid, tr in tracks.items()}
            return robot, views, time.perf_counter() - t
        if self.detector is None:
            dets = detect_objects(frame.image, self.hom, robot_polygons(self.cfg, robot, jaw_angle), self.pcfg)
        else:
            dets = self.detector(frame, robot, jaw_angle)
        tracks = self.tracker.update(dets)
        views = {tid: TrackView(tid, tr.pose, tr.size_m, tr.mean_rgb, tr.missed, tr.hits) for tid, tr in tracks.items()}
        return robot, views, time.perf_counter() - t

    async def run(self) -> None:
        seq = 0
        while True:
            item = await self.camera.frames.wait_newer(seq, None)
            if seq:
                self.skipped_frames += item.seq - seq - 1
            seq = item.seq
            frame = item.value
            try:
                robot, views, dt = await asyncio.to_thread(self._process, frame, self.gripper_at(frame.stamp))
            except FrameUnusable as e:   # このフレームは飛ばす（続けば観測が古くなり、制御が止まって待つ → 止める）
                self.pose_lost_frames += 1
                log.warning("フレーム %d を飛ばす: %s", frame.seq, e)
                continue
            self.stats.add("perception_s", dt)
            self.stats.add("frame_age_at_tracked_s", time.monotonic() - frame.stamp)
            self.tracked.set(TrackedState(frame, robot, views, dt), frame.stamp)


# ---------- 制御（観測 → Action） ----------
@dataclass
class Assignment:
    episode: EpisodeSpec
    track_of: dict[str, int]          # 物体名 → 追跡の ID（開始時の同定で決めた）
    instruction: TidyInstruction
    ref_rgb: dict[str, tuple[float, float, float]]   # 同定した物体の、同定したときの画素の平均色（追跡の入れ替わりの検出）
    given_up: list[str] = field(default_factory=list)   # 見つからずにあきらめた対象（give_up）

    def give_up(self, names: list[str]) -> None:
        """見つからない対象をあきらめる（対象から外し、given_up に残す。2026-10-23）。物体は盤面に
        残るが、追跡の ID は消えたもののままなので観測には入らない（見えても名前のない物として扱う）。"""
        bad = [n for n in names if n not in self.episode.target_names]
        if bad:
            raise ValueError(f"対象でない物体はあきらめられない: {bad}（対象 {self.episode.target_names}）")
        self.episode = dataclasses.replace(
            self.episode, target_names=tuple(n for n in self.episode.target_names if n not in names))
        self.given_up.extend(names)

    def restore(self, names: list[str]) -> None:
        """あきらめた対象を対象に戻す（元の追跡がまた見えたとき。FieldRuntime._reidentify）。"""
        bad = [n for n in names if n not in self.given_up]
        if bad:
            raise ValueError(f"あきらめていない物体は戻せない: {bad}（あきらめた {self.given_up}）")
        self.episode = dataclasses.replace(self.episode, target_names=self.episode.target_names + tuple(names))
        self.given_up = [n for n in self.given_up if n not in names]


@dataclass(frozen=True)
class Decision:
    done: bool
    action: Action | None
    snapshot: WorldSnapshot
    primary: str | None
    color_drift: float                # 今回見えた対象の、同定したときからの平均色の変化の最大（RGB の距離）


def shrink_goal(goal: Rect, margin_m: float) -> Rect:
    """ゴールを縁から margin_m だけ内側へ縮める（config/field.yaml perception.goal_margin_m）。"""
    if not 0.0 <= margin_m < min(goal.size) / 2:
        raise ValueError(f"goal_margin_m {margin_m} はゴールの大きさ {goal.size} の半分より小さい 0 以上")
    return Rect(goal.center, (goal.size[0] - 2 * margin_m, goal.size[1] - 2 * margin_m))


def make_assignment(cfg: AllCfg, identifier: ObjectIdentifier, ts: TrackedState, ins: TidyInstruction,
                    zones: tuple[Rect, ...], goal_margin_m: float,
                    area_size: tuple[float, float] | None = None, min_hits: int = 0) -> Assignment:
    """指示を受けたときの割り当て: 追跡を同定し（シミュレーションは真値、実機は VLM）、指示に合う対象と置き場を決める。
    実時間のランタイムと、Gazebo を 1 歩ずつ進めるリハーサル（field.rehearsal）が使う。
    area_size: 作業領域の (幅, 奥行き)（v30。実機は校正で求めた大きさ）。名前つきのゴールをその大きさへ写し
    （task.goal_rects_for_area）、盤面に入れる（観測の大きさ）。None は sim.work_area（今まで）。"""
    # min_hits: 同定に出すのは、これ以上のフレームで見え続けて今も見えている追跡だけ（実機、perception.identify_min_seen_s）。
    # 0 は全部（シミュレーション・リハーサル、今まで）。外れた追跡は名前のない追跡のまま
    steady = ts.tracks if min_hits <= 0 else {tid: t for tid, t in ts.tracks.items() if t.missed == 0 and t.hits >= min_hits}
    ids = identifier.identify(ts.frame, steady)
    objects = list(ids.values())
    poses = {o.name: o.pose for o in objects}
    full = (goal_rects(cfg.sim) if area_size is None else goal_rects_for_area(cfg.sim, area_size))[ins.destination]
    targets = select_targets(objects, poses, ins, ts.robot, cfg, goal=full)
    if not targets:
        done = [o.name for o in objects if matches_instruction(o, ins)]
        if done:   # 指示に合う物はみなゴールの中（2026-10-26）
            raise FieldStop("already_done", f"指示 {ins} に合う物 {done} は、もうゴールの中")
        raise FieldStop("no_targets", f"指示 {ins} に合う、ゴールの外の物体がない（同定 {len(objects)} 個）")
    # 観測の偏りで縁のすぐ外を中と見誤らないよう、縮めたゴールへ運ぶ（置き場の一覧からは元の大きさで除く）
    goal = shrink_goal(full, goal_margin_m)
    ep = EpisodeSpec(FIELD_STAGE, ts.robot, tuple(objects), tuple(targets), None, ins.destination, goal,
                     tuple(z for z in zones if z != full), area_size=None if area_size is None else tuple(area_size))
    track_of = {o.name: tid for tid, o in ids.items()}
    ref_rgb = {n: ts.tracks[tid].mean_rgb for n, tid in track_of.items()}
    n_unnamed = sum(tid not in ids for tid in ts.tracks)
    log.info("対象 %s → %s（同定 %d 個、名前のない追跡 %d 本、フレーム %d）", list(targets), ins.destination,
             len(objects), n_unnamed, ts.frame.seq)
    return Assignment(ep, track_of, ins, ref_rgb)


# 指令が効くまでの見積もりに使う、最近の送りの回数（8 Hz で 2 s。接続ごとに送りの時間が 57・120 ms と変わったので、
# 起動からの平均でなく最近の平均。未検証）
COMMAND_LEAD_SAMPLES = 16


def command_lead_s(stats: TimingStats) -> float:
    """指令を送り始めてから届くまでの見積もり（最近 COMMAND_LEAD_SAMPLES 回の send_s の平均）。まだ送っていなければ 0。"""
    v = stats.samples.get("send_s", [])[-COMMAND_LEAD_SAMPLES:]
    return float(np.mean(v)) if v else 0.0


def to_now(ts: TrackedState, robot_now: Pose2D, gripper: GripperState, cfg: AllCfg) -> TrackedState:
    """写真の時刻の追跡を、今の時刻のロボットの姿勢 robot_now（出した指令から見積もった。FieldRuntime._predict_pose）に合わせる。
    閉じた爪の間の物（snapshot.target_held と同じ判定）はロボットといっしょに動かし、床の物はそのまま。学習では観測の古さは
    常に 0（OBS_AGE_FED_TO_POLICY_S）なので、方策に渡す前に古さを埋める（2026-10-27 実機: 平均 0.2 s 古く、旋回のたびに 50〜90°
    行き過ぎて左右に振れた）。"""
    r0 = ts.robot
    if gripper != GripperState.CLOSED:
        return dataclasses.replace(ts, robot=robot_now)
    c0, s0 = math.cos(r0.yaw), math.sin(r0.yaw)
    c1, s1 = math.cos(robot_now.yaw), math.sin(robot_now.yaw)
    tracks = {}
    for tid, t in ts.tracks.items():
        p = t.pose
        if target_held(cfg.sim, cfg.task, r0, gripper, p):
            dx, dy = p.x - r0.x, p.y - r0.y
            lx, ly = c0 * dx + s0 * dy, -s0 * dx + c0 * dy
            p = Pose2D(robot_now.x + c1 * lx - s1 * ly, robot_now.y + s1 * lx + c1 * ly,
                       wrap_angle(p.yaw + robot_now.yaw - r0.yaw))
            t = dataclasses.replace(t, pose=p)
        tracks[tid] = t
    return dataclasses.replace(ts, robot=robot_now, tracks=tracks)


class Controller:
    """追跡の結果から観測を作り、Policy で Action を決める（I/O なし）。env の snapshot_from と同じ作り方。"""

    def __init__(self, cfg: AllCfg, pcfg: PerceptionCfg, policy, a: Assignment, fp: FieldPerceptionCfg):
        self.cfg, self.pcfg, self.policy, self.a = cfg, pcfg, policy, a
        self.fp = fp
        self._off = {n: 0 for n in a.episode.target_names}   # 色が上限を超えて続いたフレーム数（対象ごと）
        self._confirm_backs = 0   # 完了の前に色を確かめるため続けて下がった回数
        self.builder = ObservationBuilder(cfg.sim, cfg.task)
        self.gripper = GripperState.OPEN
        self.prev_action = Action.STOP
        self.last_primary = a.episode.target_names[0]
        self.last_robot = None
        self.motion_resets = 0   # 止まっていた後に、直前 1 歩の動きを 0 にした回数（note_stopped）
        self.motion_jumps = 0    # 1 周期では動けない姿勢の差（読み違い・待った後の跳び）で、直前 1 歩の動きを 0 にした回数
        self.n_steps = 0
        self._n_targets = len(a.episode.target_names)
        # 挟み損ねたとみなす、爪の中心から主対象までの距離: 学習の挟み損ね（sim2d.grasp_fail）は物を kick_m の上限まで弾く。
        # その先に物の半分の長さ（いちばん長い物）。これより遠い対象を閉じた爪で見た場面は、学習に無い（2026-10-27 実機で、
        # 閉じた爪の方策が後ろ 12 cm の対象を置いて前進 100% でゴールへ向かった）
        self.missed_grasp_m = cfg.sim.sim2d.grasp_fail.kick_m[1] + max(o.size_m[0] for o in cfg.sim.objects.values()) / 2
        self.missed_grasp_opens = 0   # 挟み損ねて爪を開いた回数   # 始めの対象の数（あきらめても歩数の上限は減らさない）

    def max_steps(self) -> int:
        """env.max_steps と同じ上限（始めの対象の数で増える）。"""
        e = self.cfg.task.episode
        return e.max_steps + e.max_steps_per_extra_target * (self._n_targets - 1)

    def note_stopped(self) -> None:
        """ロボットを止めて待った（観測が古い、同定し直し、仕切り直しで下がった）。次の「直前 1 歩の動き」は 0 にする（待つ前の
        姿勢との差は 1 歩の動きではない。2026-10-26、実機で 17 cm・87° になり Observation が値域の外でアプリごと終わった）。"""
        self.last_robot = None
        self.motion_resets += 1

    def _swapped(self, n: str, rgb, d: float) -> bool:
        """対象 n の追跡の色 rgb（同定したときから色度 d）が、別の物体に入れ替わったように見えるか。上限を超え、かつ、ほかの
        同定した物体の色の方が近いか、上限の 2 倍を超えた。2026-10-30 実機: ゴールに置いた赤い LEGO が同定したときから 16 変わり
        （上限 14。置いた場所の照明・見える面で変わる）、ほかの物とは 49 以上離れていたのに、入れ替わりとみなして仕切り直しを
        くり返した。2 倍（28）は、同じ物で見た最大 16.2 と、爪・車体の色が混ざった塊（赤い LEGO から約 50）・黒（110）の間（未検証）"""
        tol = self.fp.identity_chroma_tolerance
        if d <= tol:
            return False
        return d > 2 * tol or any(chroma_distance(rgb, ref) < d for m, ref in self.a.ref_rgb.items() if m != n)

    def decide(self, ts: TrackedState) -> Decision:
        ep = self.a.episode
        if self.last_primary not in ep.target_names:   # 主対象をあきらめた（Assignment.give_up）
            self.last_primary = ep.target_names[0]
        lost = [n for n in ep.target_names if self.a.track_of[n] not in ts.tracks]
        if lost:
            raise FieldStop("target_lost", f"対象 {lost} の追跡が消えた（{self.pcfg.max_missed} フレーム見えない）")
        # 追跡の入れ替わり（対象の追跡が近くの別の物体に移る）を色で見つける。このフレームで対応が付いた追跡だけを見る
        # （見失っている間は最後の色のまま）。2026-10-17、Stage 6 で赤い対象の追跡が隣の黒い LEGO に移り、
        # 黒い LEGO をゴールへ運んで「完了」になった
        # 2026-10-19: 明るさを除いた色度で比べ、上限を超えたのが identity_persist_frames 回続いたら止める。爪を閉じていて対象が
        # 爪の近くにあるときは見ない（閉じた爪の画素が混ざる。config/field.yaml の根拠）
        drift = 0.0
        fp = self.fp
        gc = ts.robot.transform_point((gripper_center_x(self.cfg.sim), 0.0))
        for n in ep.target_names:
            tr = ts.tracks[self.a.track_of[n]]
            r = math.dist(tr.pose.xy, gc)
            if tr.missed > 0 or r <= fp.identity_skip_open_radius_m or (
                    self.gripper == GripperState.CLOSED and r <= fp.identity_skip_closed_radius_m):
                continue
            d = chroma_distance(tr.mean_rgb, self.a.ref_rgb[n])
            drift = max(drift, d)
            self._off[n] = self._off[n] + 1 if self._swapped(n, tr.mean_rgb, d) else 0
            if self._off[n] >= fp.identity_persist_frames:
                # 止めた（仕切り直す）ら数え直す（2026-10-30 実機: 数が残り、仕切り直しのたびにすぐまた仕切り直した）
                k, self._off[n] = self._off[n], 0
                raise FieldStop("target_identity", f"対象 {n} の追跡の色（色度）が同定したときから {d:.1f} 変わったのが "
                                f"{k} フレーム続いた（上限 {fp.identity_chroma_tolerance}）。別の物体に入れ替わったおそれ")
        poses = {n: ts.tracks[tid].pose for n, tid in self.a.track_of.items() if tid in ts.tracks}
        # 物体名と結び付いていない追跡（同定の後に出た塊）は観測に入れない。2026-10-30、実機の録画（16:45 の赤）で、爪の中の
        # LEGO が赤と茶の 2 つの塊に割れて、ロボットの前 10 cm に毎フレーム 1〜2 個の「LEGO」として方策に渡っていた
        # （障害物の表示が正しかったことがない）。本物の物は同定で名前が付くので、障害物はそれだけにする
        extra = ()
        # 直前 1 周期の移動（v9）。ArUco で測ったロボットの姿勢の差を、前の姿勢のロボット座標で表す。止まっていた後
        # （note_stopped）は 0
        prev = self.last_robot if self.last_robot is not None else ts.robot
        c, sn = math.cos(prev.yaw), math.sin(prev.yaw)
        dx, dy = ts.robot.x - prev.x, ts.robot.y - prev.y
        motion = (c * dx + sn * dy, -sn * dx + c * dy, wrap_angle(ts.robot.yaw - prev.yaw))
        b = self.builder
        if max(abs(motion[0]) / b.step_m, abs(motion[1]) / b.step_m, abs(motion[2]) / b.step_rad) > b.bound:
            # Observation の値域を超える差は 1 歩の動きではない（マーカーの向きの読み違い、マーカーが読めずに待った後の跳び）。
            # 2026-10-30 実機 12:44:37: 同定し直しの後、1 周期で向きが −103° 変わり、値域の外で指示ごとエラーで終わった
            log.warning("直前 1 周期の姿勢の差（前 %.0f mm、左 %.0f mm、%.0f°）は 1 歩では動けない。読み違い・跳びとして 0 にする",
                        1000 * motion[0], 1000 * motion[1], math.degrees(motion[2]))
            self.motion_jumps += 1
            motion = (0.0, 0.0, 0.0)
        self.last_robot = ts.robot
        time_left = max(0.0, (self.max_steps() - self.n_steps) / self.max_steps())
        snap, primary = build_snapshot(ep, self.cfg.sim, self.cfg.task, poses, ts.robot, self.gripper, self.prev_action,
                                       self.last_primary, extra_obstacles=extra, obs_age_s=OBS_AGE_FED_TO_POLICY_S,
                                       robot_motion=motion, time_left=time_left)
        self.last_primary = primary
        unseen = [n for n in ep.target_names if ts.tracks[self.a.track_of[n]].missed > 0]
        exhausted = False
        # v9: もともと外にあった対象外がゴールに入っていたら完了にしない（取り出すまで Policy を続ける）。
        # 2026-10-19: 置き場（前の指示のゴール）から出した対象外も（学習の成功の条件 _n_home_out == 0 とそろえる）
        ret, rem = pending_fixes(ep, self.cfg.sim, poses)
        # success.require_cleanup: false（評価と同じく全対象が入れば成功）では、直すべき対象外を待たない
        if snap.n_remaining == 0 and (ret or rem) and self.cfg.task.success.require_cleanup:
            log.info("全対象がゴールに入ったように見えるが、置き場から出た %s、ゴールの中の対象外 %s がある。完了にしない",
                     ret, rem)
        elif snap.n_remaining == 0 and any(self._off[n] > 0 for n in ep.target_names):
            # 色が上限を超えている対象がある（取り違えかもしれない）。続いた回数が足りなくても、その位置では完了にしない
            log.info("全対象がゴールに入ったように見えるが、%s の色が同定したときと違う。完了にしない",
                     [n for n in ep.target_names if self._off[n] > 0])
        elif snap.n_remaining == 0 and self.gripper == GripperState.OPEN and (unseen or self._unverified(ts, gc)):
            # 爪を開いた後、見えていない対象（爪の中で隠れている）や、爪の中（色を見ない範囲）の対象があれば、その位置では
            # 完了にせず、下がってから確かめる（config/field.yaml completion_confirm_back_steps）。2026-10-19: 見えない
            # 対象は前は Policy に戻していたが、開く → 見えない → 方策がまた閉じる、を繰り返して時間切れになった（盤面 20006）
            which = unseen or self._unverified(ts, gc)
            if self._confirm_backs < fp.completion_confirm_back_steps:
                self._confirm_backs += 1
                log.info("全対象がゴールに入ったように見えるが、%s が%s。下がって確かめる（%d 回目）", which,
                         "見えていない" if unseen else "開いた爪の中で色を確かめられない", self._confirm_backs)
                return Decision(False, Action.BACKWARD, snap, primary, drift)
            if unseen:
                # 下がっても見えない: その追跡は爪の中に無い（挟み損ねたのに補った偽の追跡）。見失ったとして仕切り直しに
                # 任せる（2026-10-19、盤面 20029: Policy に戻すと、閉じる → 偽の追跡を補う → 開く → 見えない、を
                # 繰り返して時間切れ）
                n_backs, self._confirm_backs = self._confirm_backs, 0   # 仕切り直しの後も同じ Controller を使う
                raise FieldStop("target_lost", f"爪を開いて {n_backs} 歩下がっても {unseen} が見えない")
            log.info("下がっても %s が爪の中にある（ロボットに付いてくる追跡）。完了にせず Policy を続ける", which)
            exhausted = True
        elif snap.n_remaining == 0 and unseen:
            # 見失っている間の追跡は最後の位置のまま。その位置では完了にしない（Policy を続ける）
            log.info("全対象がゴールに入ったように見えるが、%s がこのフレームで見えていない。完了にしない", unseen)
        elif snap.n_remaining == 0:
            if self.gripper == GripperState.CLOSED:
                # 挟んでいる物体は爪のマスクで削られ、追跡の位置が 1 cm 前後ずれる（シミュレーションで 7〜16 mm）。
                # そのまま完了にすると、実際はゴールの外のことがある。爪を開いてから、
                # 次の周期の新しいフレームで判定し直す（入っていなければ Policy が続ける）
                log.info("全対象がゴールに入ったように見えるが爪が閉じている。開いてから確かめる")
                return Decision(False, Action.OPEN_GRIPPER, snap, primary, drift)
            return Decision(True, None, snap, primary, drift)
        if not exhausted:
            self._confirm_backs = 0
        if self.gripper == GripperState.CLOSED and snap.task_mode == TaskMode.DELIVER:
            tr = ts.tracks.get(self.a.track_of[primary])
            if tr is not None and tr.missed == 0 and math.dist(tr.pose.xy, gc) > self.missed_grasp_m:
                self.missed_grasp_opens += 1
                log.info("挟み損ねた: 爪が閉じているが、対象 %s は爪の中心から %.0f mm（学習で弾かれる範囲 %.0f mm の外）。爪を開く",
                         primary, 1000 * math.dist(tr.pose.xy, gc), 1000 * self.missed_grasp_m)
                return Decision(False, Action.OPEN_GRIPPER, snap, primary, drift)
        action = Action(int(self.policy.predict(self.builder.build(snap))))
        return Decision(False, action, snap, primary, drift)

    def _unverified(self, ts: TrackedState, gc) -> list[str]:
        """色を見ない範囲（爪の中心から identity_skip_open_radius_m 以内）にある対象。"""
        return [n for n in self.a.episode.target_names
                if math.dist(ts.tracks[self.a.track_of[n]].pose.xy, gc) <= self.fp.identity_skip_open_radius_m]

    def commands(self, action: Action, now: float, ttl_s: float, next_seq: Callable[[], int]) -> list[Command]:
        """Action を指令に直す。グリッパーは、先に走行を止めてから開閉する（env と同じく開閉の間は動かない）。"""
        if action in (Action.OPEN_GRIPPER, Action.CLOSE_GRIPPER):
            return [DriveCommand(next_seq(), now, ttl_s, 0.0, 0.0),
                    GripperCommand(next_seq(), now, ttl_s, action == Action.CLOSE_GRIPPER)]
        v, w = drive_velocity(self.cfg.sim.robot, action)
        return [DriveCommand(next_seq(), now, ttl_s, v, w)]

    def commit(self, action: Action) -> None:
        if action == Action.CLOSE_GRIPPER:
            self.gripper = GripperState.CLOSED
        elif action == Action.OPEN_GRIPPER:
            self.gripper = GripperState.OPEN
        self.prev_action = action
        self.n_steps += 1


# ---------- 指令の門と安全監視 ----------
class CommandGate:
    """指令は必ずここを通す。一度閉じたら（trip）走行・グリッパーの指令は通さず、停止だけを通す。"""

    def __init__(self, link: RobotLink):
        self.link = link
        self._lock = asyncio.Lock()
        self.tripped: FieldStop | None = None
        self._seq = itertools.count(1)
        self.last_sent: float | None = None

    def next_seq(self) -> int:
        return next(self._seq)

    def trip(self, stop: FieldStop) -> FieldStop:
        """門を閉じる（同期。最初の理由を残す）。"""
        if self.tripped is None:
            self.tripped = stop
        return self.tripped

    async def send(self, cmd: Command) -> None:
        async with self._lock:
            if self.tripped is not None and not isinstance(cmd, StopCommand):
                raise self.tripped
            await self.link.send(cmd)
            self.last_sent = time.monotonic()

    async def stop(self) -> None:
        await self.send(StopCommand(self.next_seq(), time.monotonic()))


class Supervisor:
    """止める条件の判定。check は I/O なしで、止める理由（FieldStop）か None を返す。"""

    def __init__(self, f: FieldCfg, frames: Latest[Frame], tracked: Latest[TrackedState], link: RobotLink, start: float):
        self.s = f.safety
        self.ttl = f.control.command_ttl_s
        self.frames, self.tracked, self.link = frames, tracked, link
        self.start = start
        self.stop_requested: str | None = None
        self.persist = f.control.persist   # 止めずに待つ（実機）: 古いフレーム・追跡、通信の切れは制御の側で待つ・つなぎ直す
        self.controlling = False
        self.last_command: float | None = None

    def start_control(self, now: float) -> None:
        """制御を始める。制御の止まりは、ここから数える。最後の指令（爪を開く）からの時間には、指示を受けたときの同定（実機の
        VLM で 1〜4 s）が入るので、そこから数えると最初の指令の前に止まった（2026-10-22、実機で「最後の指令から 3.903 s」）。"""
        self.last_command = now
        self.controlling = True

    def check(self, now: float) -> FieldStop | None:
        s = self.s
        if self.stop_requested is not None:
            return FieldStop("stop_request", f"停止の要求（{self.stop_requested}）")
        st = self.link.status()
        if not st.connected and not self.persist:
            return FieldStop("link_lost", f"ロボットとの通信: {st.detail}")
        for name, latest, what in (() if self.persist else
                                   (("camera", self.frames, "カメラのフレーム"), ("perception", self.tracked, "追跡の結果"))):
            if latest.seq == 0:
                if now - self.start > s.startup_timeout_s:
                    return FieldStop(f"no_{name}", f"開始から {s.startup_timeout_s} s たっても{what}が来ない")
                continue
            age = latest.age(now)
            if age > s.max_frame_age_s:
                return FieldStop(f"stale_{name}", f"{what}が {age:.3f} s 古い（上限 {s.max_frame_age_s} s）")
        if self.controlling and self.last_command is not None and now - self.last_command > self.ttl:
            return FieldStop("control_stalled", f"最後の指令から {now - self.last_command:.3f} s（期限 {self.ttl} s）")
        return None


# ---------- 全体 ----------
@dataclass
class Watched:
    """ランタイムの外で動かしているタスク（カメラ・SimWorld・録画・音声）。例外で終わったら止める。

    forever=True のタスクが終わること自体も異常（カメラが黙って止まった、など）。"""
    task: asyncio.Task
    forever: bool


class FieldRuntime:
    INVERSE = {Action.FORWARD: Action.BACKWARD, Action.BACKWARD: Action.FORWARD,
               Action.TURN_LEFT: Action.TURN_RIGHT, Action.TURN_RIGHT: Action.TURN_LEFT}

    def __init__(self, cfg: AllCfg, f: FieldCfg, pcfg: PerceptionCfg, camera: CameraSource, link: RobotLink,
                 pose: RobotPoseEstimator, identifier: ObjectIdentifier, hom: Homography, policy, telemetry=None,
                 zones: tuple[Rect, ...] = (), detector=None, tracker=None, tracking_factory=None,
                 area_size: tuple[float, float] | None = None, motion_model: MotionModel = IDEAL_MOTION,
                 motion_gain=None):
        """motion_model: 姿勢の先読みに使う動き方（実機は config/field.yaml の real.motion_*）。
        motion_gain: 走りながら速さの倍率を割り出すもの（field.motion_gain.OnlineMotionGain、実機だけ）。あれば、送った走行の
        指令と読めた姿勢を渡し、毎周期 motion_model をその見積もりにする。
        zones: 前の指示のゴールなど、物体を出さない置き場（v18）。今の指示のゴールと同じものは除いて渡す。
        detector・tracker: Perception を見よ（どちらも None ならカメラ画像から検出）。
        tracking_factory: PerceptionCfg（追跡を消すまでのフレーム数をカメラの fps に合わせた後のもの）→
        field.target_tracking.TargetTracking。渡すと、検出と追跡はそれが行い、対象を見失ったときの仕切り直し（爪を開いて
        下がり、付け直す。リハーサルと同じ）も行う。
        area_size: 作業領域の (幅, 奥行き)（make_assignment を見よ。実機は校正の大きさ）。"""
        self.cfg, self.f = cfg, f
        self.area_size = area_size
        self.zones = tuple(zones)
        hold = max(1, round(f.perception.track_hold_s * camera.fps))
        self.identify_min_hits = max(1, round(f.perception.identify_min_seen_s * camera.fps))
        if hold != pcfg.max_missed:
            log.info("追跡を消すまでのフレーム数: perception.yaml の %d（0.25 s 周期で 2 秒）を、%.1f fps で %.1f s の %d にする",
                     pcfg.max_missed, camera.fps, f.perception.track_hold_s, hold)
        self.pcfg = dataclasses.replace(pcfg, max_missed=hold)
        self.camera, self.link = camera, link
        self.identifier = identifier
        self.policy = policy
        self.telemetry = telemetry or LoggingTelemetry()
        self.stats = TimingStats()
        self.stale_holds = 0   # 観測が古くて止まって待った周期の数
        self.reidentified = 0   # 見失って同定し直した回数（_reidentify）
        self.restored = 0       # あきらめた対象の元の追跡がまた見えて、同定し直さずに戻した回数（_reidentify）
        self.on_notice = None   # 止めずに待つ間の知らせ（TalkSession が読み上げにつなぐ）。None は知らせない
        self.undos = 0          # マーカーが読めない間に、最後の動きを逆にして戻った歩数
        self.undo_log: list[Action] = []
        self.center_steps = 0   # 戻し終えても読めないとき、作業領域の中心へ近づく向きに動いた歩数（UNDO_TOWARD_CENTER_MAX）
        self._gripper_log: list[tuple[float, GripperState]] = []
        self.tracking = None if tracking_factory is None else tracking_factory(self.pcfg)
        self.perception = Perception(cfg, self.pcfg, camera, pose, hom, self._gripper_at, self.stats, detector, tracker,
                                     self.tracking, self._predict_pose, f.perception.pose_predict_max_s,
                                     cfg.sim.robot.angular_speed_radps)
        self.perception.on_pose = self._note_pose
        self._drive_log: list[tuple[float, float, float]] = []   # 送った走行の指令（時刻、v、ω）。姿勢の見積もりに使う
        self.motion_model = motion_model
        self.motion_gain = motion_gain
        self.gate = CommandGate(link)
        self.supervisor: Supervisor | None = None
        self.assignment: Assignment | None = None
        self.controller: Controller | None = None
        self.last_decision: Decision | None = None
        self.phase = "init"
        self._main: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._start = time.monotonic()
        self._pending_stop: str | None = None

    # ---------- 外から ----------
    def request_stop(self, source: str) -> None:
        """停止の要求（音声の停止語、キーボード）。イベントループのスレッドから呼ぶ。他のスレッドからは request_stop_threadsafe。"""
        log.warning("停止の要求: %s", source)
        if self.supervisor is None:
            self._pending_stop = source
        else:
            self.supervisor.stop_requested = self.supervisor.stop_requested or source

    def request_stop_threadsafe(self, source: str) -> None:
        if self._loop is None:
            raise RuntimeError("ランタイムがまだ動いていない")
        self._loop.call_soon_threadsafe(self.request_stop, source)

    def _note_pose(self, d: dict) -> None:
        """Perception の姿勢（マーカーで読めた・見積もった）。走りの記録に残し、読めた姿勢は倍率の見積もりに渡す。"""
        self.telemetry.publish("pose", d)
        if self.motion_gain is not None and d["measured"]:
            self.motion_gain.add_pose(d["stamp"], d["x"], d["y"], d["yaw"])

    def _note_drive(self, t: float, linear: float, angular: float) -> None:
        self._drive_log.append((t, linear, angular))
        if self.motion_gain is not None:
            self.motion_gain.add_send(t, linear, angular)

    def _refresh_motion(self) -> None:
        """先読みの動き方を、走りながら割り出した倍率にする（2026-10-30、電池で速さが ±3 割変わる）。"""
        if self.motion_gain is not None:
            self.motion_model = self.motion_gain.model()

    def _predict_pose(self, pose: Pose2D, t0: float, t1: float) -> Pose2D:
        """t0 の姿勢から、送った走行の指令（各指令は次の指令か期限 command_ttl_s まで続く）で t1 の姿勢を見積もる（差動二輪）。
        動き方（self.motion_model）が理想でなければ、遅れ・倍率・時定数を入れて刻んで積む（2026-10-30）。"""
        log_ = [e for e in self._drive_log]   # 制御のスレッドが足すので写す
        ttl = self.f.control.command_ttl_s
        m = self.motion_model
        if m != IDEAL_MOTION:
            def cmd(t):   # 時刻 t に効いている指令（v, ω）
                v = w = 0.0
                for k, (ts, vv, ww) in enumerate(log_):
                    if ts > t:
                        break
                    te = min(log_[k + 1][0] if k + 1 < len(log_) else math.inf, ts + ttl)
                    v, w = (vv, ww) if t < te else (0.0, 0.0)
                return v, w
            # t0 の前から積み始めて、t0 での速さ（惰性）を作る。遅れ + 時定数の 6 倍で 0.25 % まで追いつく
            t = t0 - m.delay_s - 6 * m.lag_s
            a = 1.0 if m.lag_s <= 0 else min(1.0, MOTION_DT_S / m.lag_s)
            v = w = 0.0
            x, y, yaw = pose.x, pose.y, pose.yaw
            while t < t1:
                dt = min(MOTION_DT_S, t1 - t)
                cv, cw = cmd(t - m.delay_s)
                v += (m.gain_linear * cv - v) * a
                w += (m.gain_angular * cw - w) * a
                if t >= t0:
                    x, y = x + v * dt * math.cos(yaw + w * dt / 2), y + v * dt * math.sin(yaw + w * dt / 2)
                    yaw = wrap_angle(yaw + w * dt)
                t += dt
            return Pose2D(x, y, yaw)
        x, y, yaw = pose.x, pose.y, pose.yaw
        for k, (ts, v, w) in enumerate(log_):
            te = min(log_[k + 1][0] if k + 1 < len(log_) else math.inf, ts + ttl)
            a, b = max(ts, t0), min(te, t1)
            if b <= a:
                continue
            dt = b - a
            if abs(w) < 1e-9:
                x, y = x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw)
            else:
                x += v / w * (math.sin(yaw + w * dt) - math.sin(yaw))
                y -= v / w * (math.cos(yaw + w * dt) - math.cos(yaw))
                yaw = wrap_angle(yaw + w * dt)
        return Pose2D(x, y, yaw)

    def _gripper_at(self, stamp: float) -> GripperState:
        state = INITIAL_GRIPPER_FOR_MASK
        for t, g in self._gripper_log:
            if t > stamp:
                break
            state = g
        return state

    # ---------- 本体 ----------
    async def _send_all(self, cmds: list[Command]) -> None:
        for c in cmds:
            # 送ったすべての指令を記録する（仕切り直し・戻る指令も。実機の動き方を走りの記録から割り出す、2026-10-30）
            self.telemetry.publish("send", {"t": time.monotonic(), "seq": c.seq, **(
                {"linear": c.linear, "angular": c.angular} if isinstance(c, DriveCommand) else {"close": c.close})})
            # 通信の失敗は、安全監視より先に制御の周期で気づくこともある。どちらでも同じ理由で止める
            try:
                await self.gate.send(c)
            except CommandExpired as e:
                raise FieldStop("command_expired", str(e)) from e
            except LinkError as e:
                if not self.f.control.persist:
                    raise FieldStop("link_lost", f"指令 seq={c.seq} を送れない: {e}") from e
                # つなぎ直して続ける（ロボットは切れた時点で自分で止まっている）。残りの指令は古くなるので送らず、次の周期から
                log.warning("指令 seq=%d を送れない（%s）。つなぎ直す", c.seq, e)
                try:
                    await self.link.connect()
                except LinkError as e2:
                    raise FieldStop("link_lost", f"つなぎ直せない: {e2}") from e2
                log.info("つなぎ直した")
                return
            if isinstance(c, GripperCommand):
                self._gripper_log.append((time.monotonic(), GripperState.CLOSED if c.close else GripperState.OPEN))
            elif isinstance(c, DriveCommand):
                self._note_drive(time.monotonic(), c.linear, c.angular)
        self.supervisor.last_command = self.gate.last_sent

    async def _assign(self, ins: TidyInstruction, min_hits: int = 0) -> Assignment:
        """min_hits: 同定に出す追跡の、見え続けたフレーム数の下限（同定し直しだけ。指示を受けたときは追跡器が始まったばかりで
        どの追跡も 1〜2 フレームなので 0）。"""
        s = self.f.safety
        cur = self.perception.tracked.seq
        try:
            item = await self.perception.tracked.wait_newer(cur, None if self.f.control.persist else s.startup_timeout_s)
        except TimeoutError:
            raise FieldStop("no_perception", f"指示の後 {s.startup_timeout_s} s たっても追跡の結果が来ない") from None
        a = await asyncio.to_thread(make_assignment, self.cfg, self.identifier, item.value, ins, self.zones,
                                    self.f.perception.goal_margin_m, self.area_size, min_hits)
        if self.tracking is not None:
            self.tracking.start(a, item.value.tracks)
        return a

    async def _reidentify(self, gone: list[str]) -> bool:
        """仕切り直しで付け直せなかったとき: ロボットを止め、今の追跡を同定し直して割り当てを作り直す（ゴールに置いた物は
        select_targets が除く）。同定の間（VLM で 1〜3 s）は制御の止まりを数えない。指示に合う物が無ければ target_lost。"""
        log.info("付け直せない対象 %s: 今の映像を同定し直して、指示に合う物を探す", gone)
        self.phase = "reidentifying"
        c = self.f.control
        await self._send_all([DriveCommand(self.gate.next_seq(), time.monotonic(), c.command_ttl_s, 0.0, 0.0)])
        self.supervisor.controlling = False
        # 止まってから identify_min_seen_s 後に撮った映像まで待つ。下がった後に見えたばかりの物も、同定に出す下限
        # （identify_min_hits）まで見え続けてから同定する（2026-10-30 実機 002_105703: 爪の前の赤いブロックが下がって見えてから
        # 1 s たたないうちに同定し、同定に出されずに target_lost で止まった）
        t_stop = time.monotonic()
        item = self.perception.tracked.get()
        while item.stamp < t_stop + self.f.perception.identify_min_seen_s:
            item = await self.perception.tracked.wait_newer(self.perception.tracked.seq, None)
        # 元の追跡が今また見えていて色も同じなら、その対象に戻す（VLM に聞かない）。2026-10-27 実機: 濃い緑の LEGO の追跡は
        # 同じ色のまま残っていたのに、VLM が other・black と答えて外し、すぐ消える別の塊を対象にして仕切り直しをくり返した
        a = self.assignment
        tol = self.f.perception.identity_chroma_tolerance
        back = [n for n in gone if a.track_of[n] in item.value.tracks and item.value.tracks[a.track_of[n]].missed == 0
                and chroma_distance(item.value.tracks[a.track_of[n]].mean_rgb, a.ref_rgb[n]) <= tol]
        if back:
            a.restore(back)
            self.restored += 1
            log.info("見失った対象 %s の元の追跡が、同じ色でまた見えている。同定し直さずに戻す", back)
            self.supervisor.start_control(time.monotonic())
            self.phase = "recovering"
            return False
        try:
            a = await self._assign(self.assignment.instruction, self.identify_min_hits)
        except FieldStop as e:
            if e.reason == "already_done":   # 見失った物は、ロボットの陰でゴールに入っていた
                log.info("同定し直したら、指示に合う物はみなゴールの中: %s", e.detail)
                self.phase = "control"
                return True
            if e.reason != "no_targets":
                raise
            raise FieldStop("target_lost", f"同定し直しても、指示に合う物が見つからない（見失った対象 {gone}）") from e
        finally:
            self.supervisor.start_control(time.monotonic())
        self.reidentified += 1
        self.assignment = a
        self.controller = Controller(self.cfg, self.pcfg, self.policy, a, self.f.perception)
        self.phase = "recovering"
        return False

    async def _control(self) -> dict:
        c, s = self.f.control, self.f.safety
        T = self.cfg.task.control.action_duration_s
        ctl = self.controller
        sup = self.supervisor
        sup.start_control(time.monotonic())
        t0 = time.monotonic()
        prev = None
        tt = self.tracking
        rc = tt.c if tt is not None else None
        n_unseen, n_recover = 0, 0
        stale_since: float | None = None   # 観測が古くて止まって待ち始めた時刻
        noticed = False                    # この待ちを知らせたか
        moves: list[Action] = []           # 最近の走行の Action（マーカーが読めないときに逆にして戻る。UNDO_MAX まで）
        next_undo_k = 0                    # 次に戻ってよい周期
        center_left = UNDO_TOWARD_CENTER_MAX   # この読めない間に、中心へ近づく向きにまだ動いてよい歩数
        grip_hold_until = 0.0              # 爪の開け閉めを待つ（control.gripper_settle_s）終わりの時刻
        extra_backs = 0                    # 仕切り直しで、対象がロボットの陰から出るまで足した後退の歩数
        recover: list = []     # 仕切り直しの残りの周期（Action、最後の None は付け直し）
        recover_sent_t = 0.0   # 仕切り直しの最後の指令を送った時刻（付け直しは、その動きが終わった後に撮った映像で決める）
        for k in itertools.count():
            target = t0 + k * T
            await asyncio.sleep(max(0.0, target - time.monotonic()))
            now = time.monotonic()
            late = now - target
            self.stats.add("tick_lateness_s", late)
            if prev is not None:
                self.stats.add("tick_period_s", now - prev)
            prev = now
            if late > c.max_tick_lateness_s:
                if not c.persist:
                    raise FieldStop("control_overrun", f"周期 {k} が予定より {late:.3f} s 遅れた（上限 {c.max_tick_lateness_s} s）")
                log.warning("周期 %d が予定より %.3f s 遅れた（上限 %.1f s）。続ける", k, late, c.max_tick_lateness_s)
                if late > T:
                    # 遅れた周期を続けて回して取り戻さない（指令が立て続けに出る）。予定を今から数え直す（2026-10-27 実機）
                    t0 = now - k * T
            if now < grip_hold_until:
                # 爪が閉じ切る・開き切るまで走らずに待つ（config/field.yaml の根拠）。止まれを送り続ける（指令が途切れると安全監視
                # が control_stalled で止める）
                await self._send_all([DriveCommand(self.gate.next_seq(), time.monotonic(), c.command_ttl_s, 0.0, 0.0)])
                continue
            if not c.persist and k >= ctl.max_steps():
                raise FieldStop("timeout", f"{ctl.max_steps()} 周期（{ctl.max_steps() * T:.0f} s）で終わらない")
            item = self.perception.tracked.get()
            age = now - item.stamp
            self.stats.add("obs_age_s", age)
            if age > s.max_obs_age_s:
                # 古い観測では動かない: この周期はロボットを止めて待つ（Policy は呼ばない）。古いのが stale_obs_hold_s 続いたら止める
                # （2026-10-22、実機で 1 回だけ 467 ms になり、片付け全体が止まった）
                stale_since = now if stale_since is None else stale_since
                lp = self.perception.last_pose
                if (c.persist and moves and k >= next_undo_k and lp is not None
                        and now - lp[1] > s.max_obs_age_s):
                    # マーカーが読めない: 最後の動きを逆にして 1 歩戻り、読めるかを UNDO_WAIT_STEPS 周期待つ
                    back = self.INVERSE[moves.pop()]
                    self.undos += 1
                    self.undo_log.append(back)
                    log.warning("周期 %d: ロボットのマーカーが %.2f s 読めない。最後の動きの逆 %s で 1 歩戻る（%d 歩目）", k,
                                now - lp[1], back.name, self.undos)
                    await self._send_all(ctl.commands(back, time.monotonic(), c.command_ttl_s, self.gate.next_seq))
                    ctl.note_stopped()
                    next_undo_k = k + 1 + UNDO_WAIT_STEPS
                    continue
                if (c.persist and not moves and center_left > 0 and k >= next_undo_k and lp is not None
                        and now - lp[1] > s.max_obs_age_s):
                    # 戻し終えても読めない: 最後に見えた姿勢から作業領域の中心へ近づく向きに 1 歩（2026-10-30 実機 13:04、
                    # UNDO_TOWARD_CENTER_MAX の根拠）
                    step = step_toward_center(lp[0])
                    center_left -= 1
                    self.center_steps += 1
                    log.warning("周期 %d: ロボットのマーカーが %.2f s 読めない。戻し終えたので、作業領域の中心へ %s で 1 歩"
                                "（残り %d 歩）", k, now - lp[1], step.name, center_left)
                    await self._send_all(ctl.commands(step, time.monotonic(), c.command_ttl_s, self.gate.next_seq))
                    ctl.note_stopped()
                    next_undo_k = k + 1 + UNDO_WAIT_STEPS
                    continue
                if (c.persist and not noticed and now - stale_since >= STALE_NOTICE_S
                        and self.on_notice is not None):
                    noticed = True
                    log.warning("観測が古いまま %.0f s 待っている。知らせる", now - stale_since)
                    self.on_notice(STALE_NOTICE_TEXT)
                if not c.persist and now - stale_since >= s.stale_obs_hold_s:
                    raise FieldStop("stale_observation", f"制御に使う追跡が {age:.3f} s 古い（上限 {s.max_obs_age_s} s）状態が "
                                    f"{now - stale_since:.2f} s 続いた（上限 {s.stale_obs_hold_s} s）")
                self.stale_holds += 1
                ctl.note_stopped()
                log.warning("周期 %d: 追跡が %.3f s 古い（上限 %.1f s）ので止まって待つ", k, age, s.max_obs_age_s)
                await self._send_all([DriveCommand(self.gate.next_seq(), time.monotonic(), c.command_ttl_s, 0.0, 0.0)])
                continue
            stale_since, noticed = None, False
            center_left = UNDO_TOWARD_CENTER_MAX
            if recover:
                # 仕切り直しの途中（爪を開く → 下がる × recover_back_steps → 付け直し。リハーサルと同じ）
                step = recover.pop(0)
                if step is None and item.stamp < recover_sent_t + T:
                    # 下がり終える前に撮った映像では決めない（観測は 0.2〜0.5 s 古い。2026-10-30 実機 002_105703: 下がり切る前の
                    # 映像で「見つからない」とあきらめた。その直後の映像では爪の 113 mm 先に見えていた）。待つ間は止まれを送る
                    # （送らないと指令が途切れて安全監視が止める。13:40:33 の control_stalled。最後の後退の惰性も TTL まで続かない）
                    recover = [None]
                    await self._send_all([DriveCommand(self.gate.next_seq(), time.monotonic(), c.command_ttl_s, 0.0, 0.0)])
                    continue
                if step is None:
                    # 見えない対象の最後の位置が、まだロボットの陰（検出から除く範囲）なら、さらに 1 歩ずつ下がる。2026-10-22、実機で
                    # 奥のゴールに置いた LEGO がロボットの車体と台紙の陰になり、4 歩（約 7 cm）下がっても見えず、仕切り直しを 21 回
                    # くり返した
                    m = tt.last_mask
                    hidden = [n for n in self.assignment.episode.target_names
                              if m is not None and tt.detector.hidden_by_robot(m, tt.last_xy[n])
                              and (self.assignment.track_of[n] not in item.value.tracks
                                   or item.value.tracks[self.assignment.track_of[n]].missed > 0)]
                    if hidden and extra_backs < self.f.perception.recover_extra_back_steps:
                        extra_backs += 1
                        recover = [None]
                        recover_sent_t = time.monotonic()
                        await self._send_all(ctl.commands(Action.BACKWARD, time.monotonic(), c.command_ttl_s,
                                                          self.gate.next_seq))
                        ctl.commit(Action.BACKWARD)
                        continue
                    extra_backs = 0
                    # 止めずに続ける（persist）なら、見つからない対象はあきらめて、ほかの対象を続ける（2026-10-23）
                    gone = tt.settle_after_recover(c.persist)
                    if gone and c.persist:
                        # あきらめる前に、今の映像を同定し直して（実機は VLM）、指示に合うゴールの外の物を対象にし直す（2026-10-26、
                        # 見失ったらすぐあきらめず、もう一度同定するため）
                        if await self._reidentify(gone):
                            log.info("完了: 指示に合う物はみなゴールの中（同定し直して確かめた、%d 周期）", k)
                            return {"result": "success", "steps": k, "control_s": time.monotonic() - t0,
                                    "given_up": []}
                        ctl = self.controller
                    if not self.assignment.episode.target_names:
                        raise FieldStop("target_lost", f"仕切り直しても対象 {self.assignment.given_up} が見つからない")
                    self.policy.reset()
                    # 下がった分・同定し直しで待った間を「直前 1 歩の動き」に入れない（4 歩分は Observation の値域を超える。
                    # リハーサルの 2026-10-18）
                    ctl.note_stopped()
                    self.phase = "control"
                    n_unseen = 0
                else:
                    recover_sent_t = time.monotonic()
                    await self._send_all(ctl.commands(step, time.monotonic(), c.command_ttl_s, self.gate.next_seq))
                    ctl.commit(step)
                    continue
            ep = self.assignment.episode
            unseen = [n for n in ep.target_names
                      if self.assignment.track_of[n] in item.value.tracks
                      and item.value.tracks[self.assignment.track_of[n]].missed > 0]
            # 完了の前の確かめで下がっている間（Controller._confirm_backs）は数えない。置いた物はロボットの陰で見えないのが
            # ふつうで、数えると確かめの途中で仕切り直し（と VLM、3〜4 s）が割り込んだ（2026-10-27 実機の成功 2 本とも）
            n_unseen = n_unseen + 1 if unseen and ctl._confirm_backs == 0 else 0
            t_dec = time.perf_counter()
            why = None    # 仕切り直す理由
            if tt is not None and n_unseen >= rc.recover_unseen_steps:
                if not c.persist and n_recover >= rc.recover_max:
                    raise FieldStop("target_hidden", f"仕切り直しても対象 {unseen} が見えない")
                why = f"対象 {unseen} が {n_unseen} 周期見えない"
            else:
                try:
                    # 観測の古さを埋めて渡す（学習では古さ 0。to_now）
                    ts_now = item.value.frame.stamp
                    # 終点は、この周期の指令が効き始める時刻。理想の動き方なら今 + 最近の送りの平均（command_lead_s）。動き方の
                    # モデルがあれば今 + delay_s（delay_s は撮った時刻の基準で割り出した、カメラの遅れ + 指令が効くまでの遅れ。
                    # 今送る指令が効き始める時の姿勢は、撮った時刻の基準で今 + delay_s に写る。2026-10-30）
                    self._refresh_motion()
                    m = self.motion_model
                    t_eff = time.monotonic() + (command_lead_s(self.stats) if m == IDEAL_MOTION else m.delay_s)
                    d = ctl.decide(to_now(item.value, self._predict_pose(item.value.robot, ts_now, t_eff),
                                          ctl.gripper, self.cfg))
                except FieldStop as e:
                    # 見失った・色が変わった（入れ替わったおそれ）ときは止めずに仕切り直す（下がって同じ色の物に付け直す）。2026-10-22、
                    # 実機で正しく追跡していた緑の LEGO の色がロボットの影で 11.3 変わり、止まった
                    if (e.reason not in ("target_lost", "target_identity") or tt is None
                            or (not c.persist and n_recover >= rc.recover_max)):
                        raise
                    why = str(e)
            if why is not None:
                n_recover += 1
                log.info("仕切り直す（%d 回目、%s）: 爪を開いて %d 周期下がる", n_recover, why, rc.recover_back_steps)
                tt.begin_recover()
                self.phase = "recovering"
                recover = [Action.OPEN_GRIPPER] + [Action.BACKWARD] * rc.recover_back_steps + [None]
                step = recover.pop(0)
                recover_sent_t = time.monotonic()
                await self._send_all(ctl.commands(step, time.monotonic(), c.command_ttl_s, self.gate.next_seq))
                ctl.commit(step)
                continue
            self.stats.add("decide_s", time.perf_counter() - t_dec)
            self.stats.add("target_color_drift", d.color_drift)
            self.last_decision = d
            if d.done:
                log.info("完了: 全対象がゴールに入った（追跡の位置で判定、%d 周期、%.1f s）", k, now - t0)
                return {"result": "success", "steps": k, "control_s": now - t0, "given_up": list(self.assignment.given_up)}
            if d.action == Action.CLOSE_GRIPPER and tt is not None:
                tt.note_close(item.value.tracks, item.value.robot)
            await self._send_all(ctl.commands(d.action, time.monotonic(), c.command_ttl_s, self.gate.next_seq))
            ctl.commit(d.action)
            if d.action in (Action.OPEN_GRIPPER, Action.CLOSE_GRIPPER):
                grip_hold_until = time.monotonic() + c.gripper_settle_s
                ctl.note_stopped()
            if d.action in self.INVERSE:
                moves = (moves + [d.action])[-UNDO_MAX:]
            self.stats.add("send_s", time.monotonic() - now)
            self.telemetry.publish("control", {
                "k": k, "action": d.action.name, "primary": d.primary, "n_remaining": d.snapshot.n_remaining,
                "robot": dataclasses.astuple(item.value.robot), "obs_age_s": round(age, 4), "late_s": round(late, 4),
                "frame": item.value.frame.seq})
            if d.action in (Action.OPEN_GRIPPER, Action.CLOSE_GRIPPER) or ctl.gripper == GripperState.CLOSED:
                # 爪の開け閉めを調べる記録（2026-10-27、挟めたように見えても開け閉めを繰り返した）
                tr = item.value.tracks.get(self.assignment.track_of[d.primary]) if d.primary in self.assignment.track_of else None
                gc = item.value.robot.transform_point((gripper_center_x(self.cfg.sim), 0.0))
                log.info("周期 %d: %s（爪 %s、持っている %s、主対象 %s は爪の中心から %s mm、見えない %s フレーム）", k,
                         d.action.name, ctl.gripper.name, d.snapshot.target_held, d.primary,
                         None if tr is None else round(1000 * math.dist(tr.pose.xy, gc)), None if tr is None else tr.missed)
            elif k % 16 == 0:
                log.info("周期 %d: %s、主対象 %s、残り %d、観測の古さ %.0f ms、遅れ %.1f ms", k, d.action.name, d.primary,
                         d.snapshot.n_remaining, 1000 * age, 1000 * late)

    async def _run_main(self, get_instruction: Callable[[], Awaitable[TidyInstruction]]) -> dict:
        s = self.f.safety
        self.phase = "startup"
        try:
            await self.perception.tracked.wait_newer(0, None if self.f.control.persist else s.startup_timeout_s)
        except TimeoutError:
            raise FieldStop("no_perception", f"開始から {s.startup_timeout_s} s たっても追跡の結果が来ない") from None
        self.phase = "waiting_instruction"
        log.info("指示を待つ")
        ins = await get_instruction()
        log.info("指示: %s", ins)
        # 開始時の爪の状態は実機では分からないので開く（シミュレーションは開いた状態で始まる）。開き終わるまで待つ
        self.phase = "opening_gripper"
        await self._send_all([DriveCommand(self.gate.next_seq(), time.monotonic(), self.f.control.command_ttl_s, 0.0, 0.0),
                              GripperCommand(self.gate.next_seq(), time.monotonic(), self.f.control.command_ttl_s, False)])
        # 待つ時間は、方策が爪を開け閉めした後と同じ control.gripper_settle_s（実機の爪が開き切るまで。2026-10-30）。待つ間も
        # 1 行動ごとに止まれを送る（指令が TTL 途切れると、ロボットの見張りが止める・安全監視が control_stalled にする）
        until = time.monotonic() + self.f.control.gripper_settle_s
        while (left := until - time.monotonic()) > 0:
            await asyncio.sleep(min(left, self.cfg.task.control.action_duration_s))
            if until - time.monotonic() > 0:
                await self._send_all([DriveCommand(self.gate.next_seq(), time.monotonic(), self.f.control.command_ttl_s,
                                                   0.0, 0.0)])
        self.phase = "assigning"
        self.assignment = await self._assign(ins)
        self.controller = Controller(self.cfg, self.pcfg, self.policy, self.assignment, self.f.perception)
        self.policy.reset()
        self.phase = "control"
        out = await self._control()
        out["targets"] = list(self.assignment.episode.target_names)
        return out

    async def _safety_loop(self) -> None:
        while True:
            await asyncio.sleep(self.f.safety.check_period_s)
            stop = self.supervisor.check(time.monotonic())
            if stop is not None:
                first = self.gate.trip(stop)
                log.error("安全監視で止める: %s", first)
                try:
                    await self.gate.stop()
                except LinkError as e:
                    first.add_note(f"停止を送れなかった（ロボット側のウォッチドッグで止まるのを待つ）: {e!r}")
                    log.error("停止を送れない: %r。ロボット側のウォッチドッグ（%.2f s）で止まるはず", e, self.f.control.command_ttl_s)
                self._main.cancel()
                return

    async def _wait(self, owned: dict[str, asyncio.Task], watch: dict[str, Watched]) -> dict:
        watch = dict(watch)
        owned = dict(owned)
        while True:
            tasks = {self._main, *owned.values(), *(w.task for w in watch.values())}
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if self._main in done:
                if self._main.cancelled():
                    if self.gate.tripped is not None:
                        raise self.gate.tripped
                    raise asyncio.CancelledError()
                return self._main.result()   # FieldStop などの例外はそのまま上へ
            for name, t in list(owned.items()):
                if t in done:
                    if name == "safety" and not t.cancelled() and t.exception() is None:
                        del owned[name]      # 止めた。main のキャンセルを待つ
                        continue
                    exc = None if t.cancelled() else t.exception()
                    raise FieldStop("task_failed", f"{name} が止まった: {exc!r}") from exc
            for name, w in list(watch.items()):
                if w.task in done:
                    exc = None if w.task.cancelled() else w.task.exception()
                    if exc is None and not w.forever and not w.task.cancelled():
                        del watch[name]      # 終わってよいタスク（音声の WAV を流し終えた等）
                        continue
                    raise FieldStop("task_failed", f"{name} が止まった: {exc!r}") from exc

    async def run(self, get_instruction: Callable[[], Awaitable[TidyInstruction]],
                  watch: dict[str, Watched] | None = None) -> dict:
        """指示を受けて片付けを終えるまで動かす。成功なら記録（dict）を返し、それ以外は FieldStop などの例外。

        どの終わり方でも、最後にロボットへ停止を送る。"""
        self._loop = asyncio.get_running_loop()
        self._start = time.monotonic()
        self.supervisor = Supervisor(self.f, self.camera.frames, self.perception.tracked, self.link, self._start)
        if self._pending_stop is not None:
            self.supervisor.stop_requested = self._pending_stop
        await self.link.connect()
        owned = {"perception": asyncio.create_task(self.perception.run(), name="perception"),
                 "safety": asyncio.create_task(self._safety_loop(), name="safety")}
        self._main = asyncio.create_task(self._run_main(get_instruction), name="main")
        primary: BaseException | None = None
        try:
            report = await self._wait(owned, watch or {})
            report["elapsed_s"] = time.monotonic() - self._start
            return report
        except BaseException as e:
            primary = e
            if isinstance(e, FieldStop):
                self.gate.trip(e)
                log.error("止めた: %s", e)
            raise
        finally:
            self.gate.trip(self.gate.tripped or FieldStop("finished", "終了"))
            try:
                await self.gate.stop()
            except LinkError as e:
                log.error("最後の停止を送れない: %r（ロボット側のウォッチドッグで止まるはず）", e)
                if primary is None:
                    raise
                primary.add_note(f"最後の停止を送れなかった: {e!r}")
            finally:
                for t in [self._main, *owned.values()]:
                    t.cancel()
                results = await asyncio.gather(self._main, *owned.values(), return_exceptions=True)
                for name, r in zip(["main", *owned], results):
                    if isinstance(r, BaseException) and not isinstance(r, (asyncio.CancelledError, FieldStop)) \
                            and r is not primary and r is not getattr(primary, "__cause__", None):
                        log.error("%s が例外で終わっていた: %r", name, r)
                        if primary is not None:
                            primary.add_note(f"{name} も例外で終わっていた: {r!r}")
                self._log_summary()
                if isinstance(primary, FieldStop):
                    primary.report = self.report()

    def report(self) -> dict:
        return {"phase": self.phase, "steps": None if self.controller is None else self.controller.n_steps,
                "targets": None if self.assignment is None else list(self.assignment.episode.target_names),
                "skipped_frames": self.perception.skipped_frames, "timing": self.stats.summary()}

    def _log_summary(self) -> None:
        sm = self.stats.summary()
        for k in ("tick_period_s", "tick_lateness_s", "obs_age_s", "perception_s", "decide_s", "send_s"):
            if k in sm:
                v = sm[k]
                log.info("時間 %-18s n=%4d 平均 %7.2f ms  p95 %7.2f ms  最大 %7.2f ms  最小 %7.2f ms", k, v["n"],
                         1000 * v["mean"], 1000 * v["p95"], 1000 * v["max"], 1000 * v["min"])
        log.info("検出が追いつかず飛ばしたフレーム: %d", self.perception.skipped_frames)


async def given_instruction(ins: TidyInstruction) -> TidyInstruction:
    """指示を直接与えるとき（ファイルから）の get_instruction。"""
    return ins
