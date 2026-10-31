"""音声クライアント（toy_sweeper.voice.client.VoiceClient）とランタイムをつなぐ。

VoiceClient.run はフレームを順に読む止まる処理なので、スレッド（asyncio.to_thread）で動かす。
コールバックはそのスレッドから呼ばれるので、loop.call_soon_threadsafe でイベントループへ渡す。
- 指示（on_instruction）→ Latest[TidyInstruction]。ランタイムは最初の指示を待って使う。
- 停止語（on_stop）→ FieldRuntime.request_stop("voice")（VLM もサーバーも通さない）。
動いている間に来た 2 つ目の指示は使わず、ロボットを止める（言い直しで別のことをさせたいのに、前の指示で動き続けないように）。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Iterable, Iterator

import numpy as np

from ..task_manager import TidyInstruction
from .latest import Latest

log = logging.getLogger(__name__)


class VoiceBridge:
    def __init__(self, loop: asyncio.AbstractEventLoop, runtime):
        self.loop = loop
        self.runtime = runtime
        self.instructions: Latest[TidyInstruction] = Latest("instruction")

    # ---------- 音声のスレッドから ----------
    def on_instruction(self, ins: TidyInstruction) -> None:
        self.loop.call_soon_threadsafe(self._instruction, ins, time.monotonic())

    def on_stop(self) -> None:
        self.runtime.request_stop_threadsafe("voice")

    # ---------- イベントループで ----------
    def _instruction(self, ins: TidyInstruction, stamp: float) -> None:
        if self.instructions.seq > 0:
            log.error("動いている間に新しい指示が来た: %s。前の指示のまま動き続けないよう止める", ins)
            self.runtime.request_stop("new_instruction")
        self.instructions.set(ins, stamp)

    async def get_instruction(self) -> TidyInstruction:
        """FieldRuntime.run に渡す。最初の指示が来るまで待つ（人が言うまで来ないので期限なし）。"""
        return (await self.instructions.wait_newer(0, None)).value


def paced(frames: Iterable[np.ndarray], frame_s: float) -> Iterator[np.ndarray]:
    """WAV のフレームを実時間の速さで渡す（マイクの代わりに WAV を流すとき、停止語が実際の時刻に届くように）。"""
    t0 = time.monotonic()
    for i, f in enumerate(frames):
        wait = t0 + (i + 1) * frame_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        yield f
