"""シミュレーションの機器一式（SimWorld + SimCamera + SimRobotLink + ArUco の代わり + 真値の同定）とランタイムをつなぐ。

scripts/field_app.py・scripts/talk_sim.py と test/test_field_sim.py が使う。終わった後も少しの間 SimWorld を動かし続け、
停止がロボットに届いて止まったことを真値で確かめる（drift_after_stop_m）。
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import sys
import time
from typing import Awaitable, Callable

from ..config import AllCfg
from ..perception import PerceptionCfg
from ..perception_env import scene_homography
from ..snapshot import inside_goal
from ..task import EpisodeSpec, goal_rects
from ..task_manager import TidyInstruction
from .config import FieldCfg
from .interfaces import FieldStop
from .runtime import FieldRuntime, Watched
from .sim import SimArucoStandIn, SimCamera, SimRobotLink, SimTruthIdentifier, SimWorld

log = logging.getLogger(__name__)


def board_identification_error(cfg: AllCfg, pcfg: PerceptionCfg, board: EpisodeSpec, tolerance_m: float) -> str | None:
    """開始時の画像で、全物体を別々に検出・同定できるか（できなければ理由）。

    接して置かれた物体（Stage 6 の cluster）は 1 つの塊に見え、SimTruthIdentifier が同定できない。
    find_board が盤面を探すときに使う（同定できない盤面は理由を記録して飛ばす）。
    """
    from ..perception import Tracker, detect_objects
    from ..perception_env import robot_polygons
    from ..render import draw_scene
    from .interfaces import Frame, SimTruth, TrackView
    from ..spaces import GripperState
    jaw = cfg.sim.gripper.jaw_open_angle_rad
    poses = {o.name: o.pose for o in board.objects}
    img, _ = draw_scene(cfg, board.robot, board.objects, poses, jaw)
    tracks = Tracker(pcfg, 0.0).update(   # 1 フレームだけなので色の門は効かない
        detect_objects(img, scene_homography(cfg), robot_polygons(cfg, board.robot, jaw), pcfg))
    truth = SimTruth(board.robot, board.objects, poses, jaw, GripperState.OPEN)
    views = {tid: TrackView(tid, t.pose, t.size_m, t.mean_rgb, t.missed, t.hits) for tid, t in tracks.items()}
    try:
        SimTruthIdentifier(tolerance_m).identify(Frame(img, 0.0, 1, truth), views)
    except RuntimeError as e:
        return str(e)
    return None


def find_board(cfg: AllCfg, pcfg: PerceptionCfg, tolerance_m: float, stage: int, seed_start: int, search_limit: int,
               accept: Callable[[EpisodeSpec], bool] = lambda ep: True) -> tuple[int, EpisodeSpec]:
    """seed_start から順に、accept を満たし、開始時の画像で全物体を別々に同定できる盤面を探す（飛ばした理由は記録する）。"""
    from ..task import TaskSampler
    import numpy as np
    if stage == 9:
        # 現場のランタイムは置き場を開始の位置から決める（field/runtime.py）。Stage 9（v24）の置き場の指定は渡せない
        raise ValueError("Stage 9（置き場の指定のある盤面、v24）は現場のランタイムの盤面に使えない")
    sampler = TaskSampler(cfg.sim, cfg.task)
    for seed in range(seed_start, seed_start + search_limit):
        board = sampler.sample(stage, np.random.default_rng(seed))
        if not accept(board):
            continue
        err = board_identification_error(cfg, pcfg, board, tolerance_m)
        if err is not None:
            log.info("Seed %d は飛ばす: %s", seed, err)
            continue
        return seed, board
    raise RuntimeError(f"Seed {seed_start}〜{seed_start + search_limit - 1} に条件を満たし、全物体を同定できる盤面がない")


class SimRig:
    def __init__(self, cfg: AllCfg, f: FieldCfg, pcfg: PerceptionCfg, board: EpisodeSpec, seed: int):
        self.cfg, self.f, self.pcfg, self.board = cfg, f, pcfg, board
        self.world = SimWorld(cfg, f.sim, board, seed)
        self.camera = SimCamera(cfg, f.sim, self.world)
        self.link = SimRobotLink(self.world)
        self.pose = SimArucoStandIn(f.sim, seed)
        self.identifier = SimTruthIdentifier(f.perception.association_tolerance_m)
        self.hom = scene_homography(cfg)
        self.drift_after_stop_m: float | None = None
        # 物体の検出と追跡。None はカメラ画像から・近い順の対応付け（現場と同じ）。use_truth_perception で真値から
        self.detector = None
        self.tracker_factory = None
        self.tracking_factory = None   # PerceptionCfg → TargetTracking（FieldRuntime の tracking_factory）

    def reset_board(self, board: EpisodeSpec) -> None:
        """動いている世界の盤面を置き直す（指示を実行していないときだけ呼ぶ）。"""
        self.board = board
        self.world.reset_board(board)

    def use_truth_perception(self) -> None:
        """物体の検出と追跡に真値を使う（talk_sim.py --perception truth）。"""
        from .sim import TruthDetector, TruthTracker
        self.detector = TruthDetector(self.cfg, self.hom)
        self.tracker_factory = TruthTracker

    def reload_if_changed(self) -> bool:
        """シミュレーションに校正のファイルは無い（RealRig と同じ形にするため）。"""
        return False

    def goal_rects(self) -> dict:
        """名前つきのゴール（作業領域は sim.work_area の固定）。"""
        return goal_rects(self.cfg.sim)

    @staticmethod
    def outcome(report: dict) -> tuple[bool, str]:
        """ランタイムが完了とした指示の、真値での結果（全対象がゴールの中か）。"""
        return all(report["true_targets_in_goal"].values()), f"真値でゴールの中 {report['true_targets_in_goal']}"

    def pause(self) -> None:
        """物理と描画を止める（管理のアプリで実機を動かしている間。CPU を取らず、実時間からの遅れで止まらない）。"""
        self.world.pause()
        self.camera.frozen = True

    def resume(self) -> None:
        self.world.resume()
        self.camera.frozen = False

    @property
    def area_size(self) -> tuple[float, float]:
        """作業領域の (幅, 奥行き)（sim.work_area の固定）。"""
        return tuple(self.cfg.sim.work_area.size_m)

    def make_runtime(self, policy, telemetry=None, zones: tuple = ()) -> FieldRuntime:
        # 観測に作業領域の大きさを入れる方策（v30〜）には、盤面に大きさを渡す（無いと snapshot が例外）
        area = self.area_size if self.cfg.task.observation.area_size else None
        return FieldRuntime(self.cfg, self.f, self.pcfg, self.camera, self.link, self.pose, self.identifier, self.hom,
                            policy, telemetry, zones, self.detector,
                            None if self.tracker_factory is None else self.tracker_factory(), self.tracking_factory,
                            area_size=area)

    def truth_report(self, rt: FieldRuntime) -> dict:
        """真値での結果（ランタイムは追跡の位置で判定するので、それが正しかったかをここで確かめる）。"""
        poses = self.world.object_poses()
        out = {"world_watchdog_stops": self.world.watchdog_stops, "world_max_lag_s": self.world.max_lag_s,
               "camera_missed_slots": self.camera.missed_slots,
               "camera_render_mean_s": sum(self.camera.render_s) / max(len(self.camera.render_s), 1)}
        if rt.assignment is not None:
            ep = rt.assignment.episode
            types = {o.name: o.type for o in self.board.objects}
            out["true_targets_in_goal"] = {n: inside_goal(self.cfg.sim, ep.goal, types[n], poses[n]) for n in ep.target_names}
            if rt.perception.tracked.seq:
                tracks = rt.perception.tracked.get().value.tracks
                out["tracked_err_m"] = {n: (math.dist(tracks[tid].pose.xy, poses[n].xy) if tid in tracks else None)
                                        for n, tid in rt.assignment.track_of.items() if n in ep.target_names}
            out["gripper_at_end"] = None if rt.controller is None else rt.controller.gripper.name
            out["true_non_target_max_move_m"] = max(
                (math.dist(poses[o.name].xy, o.pose.xy) for o in self.board.objects if o.name not in ep.target_names),
                default=0.0)
        return out

    async def run(self, rt: FieldRuntime, get_instruction: Callable[[], Awaitable[TidyInstruction]],
                  extra_watch: dict[str, Watched] | None = None, settle_s: float = 0.3) -> dict:
        """世界とカメラを動かし、1 つの指示を実行して止める。"""
        async with self.running(extra_watch, settle_s) as watch:
            return await self.run_instruction(rt, get_instruction, watch)

    async def run_instruction(self, rt: FieldRuntime, get_instruction: Callable[[], Awaitable[TidyInstruction]],
                              watch: dict[str, Watched]) -> dict:
        """running() の中で 1 つの指示を実行する（続けて何度でも呼べる）。"""
        try:
            report = await rt.run(get_instruction, watch)
            report.update(self.truth_report(rt))
            return report
        except FieldStop as e:
            e.report = {**(e.report or {}), **self.truth_report(rt)}
            raise

    @contextlib.asynccontextmanager
    async def running(self, extra_watch: dict[str, Watched] | None = None, settle_s: float = 0.3):
        """世界とカメラのタスクを動かし、ランタイムに渡す watch を返す。抜けるときに止まっているかを確かめて片付ける。"""
        world_t = asyncio.create_task(self.world.run(), name="world")
        while self.world.time is None:   # 世界のスレッドが動き出すまで（カメラが最初の状態を読めるように）
            if world_t.done():
                await world_t
            await asyncio.sleep(0.005)
        cam_t = asyncio.create_task(self.camera.run(), name="camera")
        watch = {"world": Watched(world_t, True), "camera": Watched(cam_t, True), **(extra_watch or {})}
        try:
            yield watch
        finally:
            if not world_t.done():
                # 停止が届く（指令の遅れ）まで待ってから、止まっているかを真値で確かめる
                await asyncio.sleep(self.f.sim.command_latency_s + 3 * self.world.dt)
                p0 = self.world.robot_pose()
                t0 = time.monotonic()
                await asyncio.sleep(settle_s)
                p1 = self.world.robot_pose()
                self.drift_after_stop_m = math.dist(p0.xy, p1.xy)
                log.info("停止の後 %.2f s のロボットの移動: %.2f mm", time.monotonic() - t0, 1000 * self.drift_after_stop_m)
            for t in (cam_t, world_t):
                t.cancel()
            in_flight = sys.exc_info()[1]
            for name, r in zip(("camera", "world"), await asyncio.gather(cam_t, world_t, return_exceptions=True)):
                if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                    log.error("%s が例外で終わっていた: %r", name, r)
                    if in_flight is None:
                        raise r
