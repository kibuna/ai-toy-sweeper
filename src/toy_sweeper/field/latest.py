"""最新の値だけを持つ入れ物（センサーのデータに待ち行列を使わない）。

古い値を溜めて順に処理すると、遅れが積み上がって古いデータで制御してしまう。受け手はいつも最新の値を見て、
取りこぼした分は数えて記録する（seq の飛び）。asyncio のイベントループの中だけで使う（スレッドからは
loop.call_soon_threadsafe(latest.set, ...) で渡す）。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Stamped(Generic[T]):
    value: T
    seq: int        # 1 から始まる通し番号
    stamp: float    # その値の時刻（time.monotonic の秒）。フレームなら撮影時刻


class Latest(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self._item: Stamped[T] | None = None
        self._event = asyncio.Event()

    @property
    def seq(self) -> int:
        """今の通し番号。まだ値がなければ 0。"""
        return 0 if self._item is None else self._item.seq

    def set(self, value: T, stamp: float) -> Stamped[T]:
        """値を置き換える。時刻が前の値より古ければ ValueError（時刻の取り違えを黙って通さない）。"""
        if self._item is not None and stamp < self._item.stamp:
            raise ValueError(f"{self.name}: 時刻が戻った（{stamp:.6f} < {self._item.stamp:.6f}）")
        self._item = Stamped(value, self.seq + 1, stamp)
        ev, self._event = self._event, asyncio.Event()
        ev.set()
        return self._item

    def get(self) -> Stamped[T]:
        if self._item is None:
            raise LookupError(f"{self.name}: まだ値がない")
        return self._item

    def age(self, now: float | None = None) -> float:
        """最新の値の時刻からの経過（秒）。値がなければ LookupError。"""
        return (time.monotonic() if now is None else now) - self.get().stamp

    async def wait_newer(self, seq: int, timeout_s: float | None) -> Stamped[T]:
        """通し番号 seq より新しい値が来るまで待って返す。timeout_s を過ぎたら TimeoutError（None は無期限）。"""
        async def wait():
            while self.seq <= seq:
                await self._event.wait()
            return self.get()
        return await asyncio.wait_for(wait(), timeout_s)
