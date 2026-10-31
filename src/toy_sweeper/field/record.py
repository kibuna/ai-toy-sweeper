"""カメラのフレームを mp4 に記録する（追跡の結果を重ねて描く）。制御には使わない。

最新のフレームだけを受け取るので、書き込みが追いつかなければフレームが抜ける（抜けた数は数えて最後に記録する）。
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np

from ..perception import Homography
from .latest import Latest

log = logging.getLogger(__name__)


def draw_overlay(img: np.ndarray, rt, hom: Homography, goals: dict | None = None) -> np.ndarray:
    """カメラのフレームに追跡の結果（対象は赤、同定した物体は白、名前のない追跡は紫）、ロボットの向き、状態を重ねる。
    goals（名前 → Rect）を渡すと、ゴールの枠も描く（今の指示のゴールは太く）。rt が None なら追跡と状態は描かない。"""
    out = img.copy()

    def px(xy) -> tuple[int, int]:
        u, v = hom.to_px(np.array([xy]))[0]
        return int(round(u)), int(round(v))
    active = None if rt is None or rt.assignment is None else rt.assignment.episode.goal_name
    for name, g in (goals or {}).items():
        # 4 隅を移して多角形で描く（斜めのカメラでは床の長方形は台形に写る。2 隅の画素の長方形ではずれた、2026-10-25）
        hx, hy = g.size[0] / 2, g.size[1] / 2
        corners = np.int32([px((g.center[0] + sx * hx, g.center[1] + sy * hy)) for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))])
        on = name == active
        cv2.polylines(out, [corners], True, (40, 160, 70) if on else (90, 90, 90), 3 if on else 1, cv2.LINE_AA)
        x0, y0 = corners[0]
        cv2.putText(out, name, (int(x0) + 4, int(y0) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    if rt is None:
        return out
    if rt.perception.tracked.seq:
        ts = rt.perception.tracked.get().value
        names = {} if rt.assignment is None else {tid: n for n, tid in rt.assignment.track_of.items()}
        targets = set() if rt.assignment is None else set(rt.assignment.episode.target_names)
        for tid, tr in ts.tracks.items():
            n = names.get(tid)
            color = (255, 0, 0) if n in targets else ((255, 255, 255) if n else (255, 0, 255))
            cv2.circle(out, px(tr.pose.xy), 6, color, 2, cv2.LINE_AA)
        r = ts.robot
        tip = (r.x + 0.06 * math.cos(r.yaw), r.y + 0.06 * math.sin(r.yaw))
        cv2.arrowedLine(out, px(r.xy), px(tip), (0, 255, 255), 2, cv2.LINE_AA)
    d = rt.last_decision
    text = f"{rt.phase}"
    if d is not None and rt.controller is not None:
        text += f"  step {rt.controller.n_steps}  {d.action.name if d.action else 'DONE'}  left {d.snapshot.n_remaining}"
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


class FrameRecorder:
    def __init__(self, path: Path, fps: float, hom: Homography, runtime):
        self.path, self.fps, self.hom, self.rt = path, fps, hom, runtime
        self._writer: cv2.VideoWriter | None = None
        self.n_written = 0
        self.n_skipped = 0

    def _overlay(self, img: np.ndarray) -> np.ndarray:
        return draw_overlay(img, self.rt, self.hom)

    def _write(self, rgb: np.ndarray) -> None:
        if self._writer is None:
            h, w = rgb.shape[:2]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"VideoWriter を開けない: {self.path}")
        self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.n_written += 1

    async def run(self, frames: Latest) -> None:
        seq = 0
        while True:
            item = await frames.wait_newer(seq, None)
            if seq:
                self.n_skipped += item.seq - seq - 1
            seq = item.seq
            # キャンセルされても書きかけのフレームは書き終えてから止まる（close と同時に書かないように）
            fut = asyncio.ensure_future(asyncio.to_thread(self._write, self._overlay(item.value.image)))
            try:
                await asyncio.shield(fut)
            except asyncio.CancelledError:
                await fut
                raise

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            log.info("動画: %s（%d フレーム、抜け %d）", self.path, self.n_written, self.n_skipped)


# 止めた後、フレームが来なくても待つ上限（hold_after_s に足す）。カメラが止まっていても止められるように
STOP_WAIT_EXTRA_S = 2.0


class TaskRecorder:
    """指示ごとの動画（config/field.yaml の record、2026-10-25）: start（指示を受けて動き始める）から stop（終わった・止まった）の
    hold_after_s 後までのカメラのフレームに、追跡・ゴール・状態を重ね（draw_overlay）、下に指示・経過・結果を書いて 1 本の mp4 に。
    同じフレームを重ね描きなしでも <名前>_raw.mp4 に残す（2026-10-28。
    重ね描きの印が、動画に検出を当て直すときに物と見分けられなかった）。
    フレームは来たものを書くだけ（書き込みが追いつかなければ抜ける。抜けた数は最後に記録する）。run() を別の task で回す。"""

    def __init__(self, frames: Latest, out_dir: Path, fps: float, hold_after_s: float):
        self.frames, self.out_dir, self.fps, self.hold_after_s = frames, out_dir, fps, hold_after_s
        self.session = None          # draw_overlay の rt・校正・ゴールを読む TalkSession（attach）
        self.n_videos = 0
        self._writer: cv2.VideoWriter | None = None
        self._raw_writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._active = False
        self._title = ""
        self._result: str | None = None
        self._t0 = 0.0
        self._stop_at: float | None = None
        self._done = asyncio.Event()
        self._lock = asyncio.Lock()
        self._n_written = self._n_skipped = 0

    def attach(self, session) -> None:
        self.session = session

    @property
    def recording(self) -> bool:
        return self._active

    async def start(self, title: str) -> Path:
        if self._active:
            raise RuntimeError(f"前の動画を記録中に start（{self._path}）")
        if self.session is None:
            raise RuntimeError("attach の前に start")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.n_videos += 1
        self._path = self.out_dir / f"{self.n_videos:03d}_{time.strftime('%H%M%S')}.mp4"
        self._title, self._result, self._stop_at = title, None, None
        self._t0 = time.monotonic()
        self._n_written = self._n_skipped = 0
        self._done = asyncio.Event()
        self._active = True
        log.info("動画を記録し始める: %s", self._path)
        return self._path

    async def stop(self, result: str) -> None:
        """結果を書き、hold_after_s の間はそのまま書いてから閉じる（閉じるまで待つ）。"""
        if not self._active:
            raise RuntimeError("記録していないのに stop")
        self._result = result
        self._stop_at = time.monotonic() + self.hold_after_s
        try:
            await asyncio.wait_for(self._done.wait(), self.hold_after_s + STOP_WAIT_EXTRA_S)
        except TimeoutError:
            log.warning("止めた後のフレームが来ない。そのまま閉じる")
            async with self._lock:
                self._close()

    def close(self, result: str) -> None:
        """記録中なら、すぐ閉じる（アプリの終了）。"""
        if self._active:
            self._result = result
            self._close()

    def _close(self) -> None:
        if not self._active:
            return
        for w in (self._writer, self._raw_writer):
            if w is not None:
                w.release()
        self._writer = self._raw_writer = None
        self._active = False
        log.info("動画: %s（%d フレーム、抜け %d、%.0f 秒、%s）", self._path, self._n_written, self._n_skipped,
                 time.monotonic() - self._t0, self._result)
        self._done.set()

    def _compose(self, img: np.ndarray) -> np.ndarray:
        s = self.session
        out = draw_overlay(img, s.rt, s.rig.hom, s.rig.goal_rects())
        text = f"{self._title}   {time.monotonic() - self._t0:.1f} s" + (f"   -> {self._result}" if self._result else "")
        y = out.shape[0] - 12
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _open_writer(self, path: Path, shape) -> cv2.VideoWriter:
        h, w = shape[:2]
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not wr.isOpened():
            raise RuntimeError(f"VideoWriter を開けない: {path}")
        return wr

    def _write(self, rgb: np.ndarray, raw: np.ndarray) -> None:
        if self._writer is None:
            self._writer = self._open_writer(self._path, rgb.shape)
            self._raw_writer = self._open_writer(self._path.with_name(self._path.stem + "_raw.mp4"), raw.shape)
        self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self._raw_writer.write(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
        self._n_written += 1

    async def run(self) -> None:
        seq = 0
        while True:
            item = await self.frames.wait_newer(seq, None)
            skipped = item.seq - seq - 1 if seq else 0
            seq = item.seq
            if not self._active:
                continue
            async with self._lock:
                if not self._active:
                    continue
                if self._n_written:
                    self._n_skipped += skipped
                img = self._compose(item.value.image)
                await asyncio.to_thread(self._write, img, item.value.image)
                if self._stop_at is not None and time.monotonic() >= self._stop_at:
                    self._close()


def attach_task_recorder(session, frames: Latest, fcfg, fps: float, out_dir: Path) -> TaskRecorder | None:
    """config/field.yaml の record.tasks なら、指示ごとの動画（TaskRecorder）を session につなぐ（run() は呼ぶ側が別の task で
    回す）。管理のアプリ（scripts/field_app.py）と 3090 の talk_sim.py（Gazebo の世界）で共通。"""
    if not fcfg.record.tasks:
        return None
    rec = TaskRecorder(frames, out_dir, fps, fcfg.record.hold_after_s)
    session.attach_recorder(rec)
    log.info("指示ごとの動画（重ね描きあり・なし）: %s", out_dir)
    return rec
