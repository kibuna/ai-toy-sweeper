"""話しかけるとシミュレーションが動くセッション。

SimRig の世界とカメラを動かしたまま、指示ごとに FieldRuntime を作り直して実行する。終わった指示のゴールは、
次の指示の置き場（EpisodeSpec.zones、v18）として渡す。入力（文・音声）は外（ブラウザの画面）から
say / hear で受け取る。指示の構造化と書き起こしは音声サーバー（scripts/voice_server.py）に任せる。
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from ..config import AllCfg, strict_dataclass_from_dict
from ..task import EpisodeSpec
from ..task_manager import TidyInstruction
from . import kid_words as kw
from .interfaces import FieldStop
from .rig import SimRig
from .runtime import FieldRuntime, Watched, given_instruction

log = logging.getLogger(__name__)

# 画面に出す履歴の件数。1 つの指示で 3〜4 件（入力、確認、結果）なので、10 指示ぶん
EVENT_HISTORY = 40


@dataclass(frozen=True)
class Event:
    t: float          # セッション開始からの秒
    kind: str         # heard（子どもの言葉）/ robot（ロボットの返事。読み上げる）/ info（読み上げない）
    text: str         # 画面に出す文
    speech: str | None = None   # 読み上げる文（robot だけ。kid_words.for_speech）


class TalkSession:
    def __init__(self, cfg: AllCfg, rig: SimRig, policy, structure: Callable[[str], Awaitable[dict]],
                 transcribe: Callable[[bytes], Awaitable[dict]] | None, stop_words: tuple[str, ...],
                 new_board: Callable[[], EpisodeSpec] | None = None, record_dir: Path | None = None):
        """structure: 文 → 音声サーバーの /instruction の応答。transcribe: WAV → /transcribe の応答（無ければ音声は断る）。
        new_board: リセットで置く新しい盤面を返す（無ければリセットは断る）。
        record_dir: 録音（WAV）と書き起こしを残す場所（子どもの声の評価データを作るため。この PC の外へは出さない）。"""
        if not stop_words:
            raise ValueError("stop_words が空")
        self.cfg, self.rig, self.policy = cfg, rig, policy
        self.structure, self.transcribe, self.new_board = structure, transcribe, new_board
        self.record_dir = record_dir
        # やりとり（聞き取った指示・返事・指示の結果）を起きたその場で 1 行ずつ残す JSONL（2026-10-28）。
        # None は残さない
        self.conversation_path: Path | None = None
        self._n_audio = 0
        self.stop_words = tuple(stop_words)
        self.queue: asyncio.Queue[TidyInstruction] = asyncio.Queue()
        self.busy = False
        self.rt: FieldRuntime | None = None        # 実行中（なければ最後）のランタイム。映像の重ね描きに使う
        self.done_goals: list = []                 # 終わった指示のゴール（次の指示の置き場）
        self.events: deque[Event] = deque(maxlen=EVENT_HISTORY)
        self.results: list[dict] = []
        self._t0 = time.monotonic()
        self._asking = False                       # 音声サーバーに問い合わせ中（その間の次の指示は断る）
        self.recorder = None                       # 指示ごとの動画（attach_recorder、config/field.yaml の record）

    def attach_recorder(self, recorder) -> None:
        """指示を受けて動き始めてから終わるまでを、指示ごとに動画にする（field.record.TaskRecorder。run() は外で回す）。"""
        self.recorder = recorder
        recorder.attach(self)

    def _event(self, kind: str, text: str, via: str | None = None) -> None:
        self.events.append(Event(round(time.monotonic() - self._t0, 1), kind, text,
                                 kw.for_speech(text) if kind == "robot" else None))
        log.info("[%s] %s", kind, text)
        self._converse({"kind": kind, "text": text, **({"via": via} if via else {})})

    def _converse(self, row: dict) -> None:
        if self.conversation_path is None:
            return
        self.conversation_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.conversation_path, "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}, ensure_ascii=False, default=str) + "\n")

    def _result(self, ins: TidyInstruction, result: str, report) -> None:
        self.results.append({"instruction": ins, "result": result, "report": report})
        self._converse({"kind": "result", "instruction": dataclasses.asdict(ins), "result": result})

    # ---------- 入力 ----------
    def stop(self, source: str) -> bool:
        """実行中なら止める（止めたら True）。"""
        if self.rt is not None and self.busy:
            self.rt.request_stop(source)
            log.info("止める（%s）", source)
            self._event("robot", kw.STOPPING)
            return True
        self._event("robot", kw.NOT_MOVING)
        return False

    async def say(self, text: str, via: str = "text") -> dict:
        """文の指示。停止語を含めば VLM を通さずすぐ止める。実行中・問い合わせ中の新しい指示は断る。
        via: voice（録音を書き起こした）か text（画面に打った）。やりとりの記録に残す。"""
        if via not in ("voice", "text"):
            raise ValueError(f"via は voice か text: {via}")
        text = text.strip()
        if not text:
            return {"ok": False, "reason": "空の指示"}
        self._event("heard", text, via)
        if any(w in text for w in self.stop_words):
            return {"ok": True, "stopped": self.stop("停止語")}
        if self.busy or self._asking or not self.queue.empty():
            self._event("robot", kw.BUSY)
            return {"ok": False, "reason": "実行中"}
        self._asking = True
        try:
            res = await self.structure(text)
        finally:
            self._asking = False
        if not res["ok"]:
            # 画面には短い言葉だけ出し、理由（InstructionError など）はログと応答に残す
            log.info("断った理由: %s", res["reason"])
            self._event("robot", kw.NOT_UNDERSTOOD)
            return {"ok": False, "reason": res["reason"]}
        ins = strict_dataclass_from_dict(TidyInstruction, res["instruction"], "instruction")
        log.info("指示: %s", ins)
        sentence = kw.confirm(ins)
        self._event("robot", sentence)
        self.queue.put_nowait(ins)
        return {"ok": True, "confirmation": sentence, "instruction": res["instruction"]}

    async def reset(self) -> dict:
        """盤面を新しく置き直す。動いている間と、指示を読み取っている間は断る。置き場（前の指示のゴール）も忘れる。"""
        if self.new_board is None:
            log.info("この盤面は置き直せない（Gazebo のモード）")
            self._event("robot", kw.CANNOT_RESET)
            return {"ok": False, "reason": "置き直せない"}
        if self.busy or self._asking or not self.queue.empty():
            self._event("robot", kw.BUSY)
            return {"ok": False, "reason": "実行中"}
        board = await asyncio.to_thread(self.new_board)
        self._asking = True     # 置き直している間（Gazebo は起動し直しで十数秒）は指示を受けない
        try:
            await asyncio.to_thread(self.rig.reset_board, board)
        finally:
            self._asking = False
        self.done_goals.clear()
        self.rt = None
        log.info("盤面を置き直した: 物体 %d 個", len(board.objects))
        self._event("robot", kw.RESET)
        return {"ok": True}

    async def hear(self, wav: bytes) -> dict:
        """録音（WAV、16 kHz）の指示。書き起こしてから say と同じ扱い。"""
        if self.transcribe is None:
            raise RuntimeError("音声の書き起こしが設定されていない")
        r = await self.transcribe(wav)
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            self._n_audio += 1
            name = f"{self._n_audio:03d}.wav"
            (self.record_dir / name).write_bytes(wav)
            with open(self.record_dir / "transcripts.jsonl", "a") as f:
                f.write(json.dumps({"file": name, "ok": r["ok"], "text": r.get("text"), "reason": r.get("reason")},
                                   ensure_ascii=False) + "\n")
        if not r["ok"]:
            log.info("書き起こせなかった理由: %s", r["reason"])
            self._event("robot", kw.NOT_HEARD)
            return {"ok": False, "reason": r["reason"]}
        return {"text": r["text"], **await self.say(r["text"], "voice")}

    # ---------- 実行 ----------
    async def serve(self, watch: dict[str, Watched]) -> None:
        """指示を待って 1 つずつ実行し続ける（SimRig.running() の中で呼ぶ）。"""
        log.info("指示を待つ")
        while True:
            ins = await self.queue.get()
            self.busy = True
            goal = self.rig.goal_rects()[ins.destination]
            t0 = time.monotonic()
            result = "error"
            try:
                # カメラが動いた（camera_moved）などは、止まった理由を言って次の指示を待つ（校正し直せばそのまま続けられる）
                rt = self.rig.make_runtime(self.policy, zones=tuple(self.done_goals))
                rt.on_notice = lambda text: self._event("robot", text)
                self.rt = rt
                log.info("開始: %s（置き場 %d 個）", ins, len(self.done_goals))
                if self.recorder is not None:
                    await self.recorder.start(f"{' '.join(ins.target_colors)} {' '.join(ins.target_classes)}"
                                              f"{''.join(' -' + c for c in ins.excluded_classes)} -> {ins.destination}")
                rep = await self.rig.run_instruction(rt, lambda: given_instruction(ins), watch)
                ok, how = self.rig.outcome(rep)
                result = "success" if ok else "wrong"
                self._result(ins, result, rep)
                log.info("終わった: %s（%.0f 秒、%d 歩、%s）", "成功" if ok else "追跡では完了だが真値では外",
                         time.monotonic() - t0, rep["steps"], how)
                self._event("robot", kw.done(ins, len(rep["given_up"])) if ok else kw.DONE_BUT_LEFT)
                if goal not in self.done_goals:
                    self.done_goals.append(goal)
            except FieldStop as e:
                result = e.reason
                self._result(ins, e.reason, e.report)
                log.info("止まった（%s）: %s", e.reason, e)
                self._event("robot", kw.STOPPED if e.reason == "stop_request"
                            else kw.already_done(ins) if e.reason == "already_done" else kw.failed(e.reason, ins))
            except Exception as e:
                # 思わぬ例外: この指示だけを終えて（ロボットはランタイムの後始末で止まる）理由を言い、次の指示を待つ。例外は
                # 握りつぶさず、ERROR で記録を残す（2026-10-26、実機のマーカーの読み違いでアプリごと終わった）
                result = "error"
                self._result(ins, "error", {"error": repr(e)})
                log.exception("思わぬ例外でこの指示を終えた: %s", e)
                self._event("robot", kw.failed("internal_error", ins))
            finally:
                if self.recorder is not None and self.recorder.recording:
                    await self.recorder.stop(result)
                self.busy = False

    def refresh_calibration(self) -> None:
        """動いていなければ、校正・背景が変わっていたら読み込み直す（画面の重ね描きの前に呼ぶ。校正し直した後、次の指示を待たずに
        枠を新しい校正で描く）。動いている間は読み込み直さない（ランタイムが古い校正の検出・姿勢を持っている）。"""
        if not self.busy:
            self.rig.reload_if_changed()

    def status(self) -> dict:
        rt = self.rt
        return {"busy": self.busy, "asking": self._asking,
                "headline": kw.THINKING if self._asking else (kw.WORKING if self.busy else kw.WAITING),
                "phase": None if rt is None else rt.phase,
                "step": None if rt is None or rt.controller is None else rt.controller.n_steps,
                "events": [e.__dict__ for e in self.events]}
