"""現場側（MacBook、CPU）のクライアント。発話の区間ごとに、停止語 → ウェイクワード → 指示の順に扱う。

    待ち受け（idle）──ウェイクワード──▶ 指示待ち（waiting_command）──次の発話──▶ サーバーへ（書き起こし → 指示）
          ▲                                   │ command_timeout_s を過ぎたら待ち受けに戻る
          └───────────── 確認の読み上げ・on_instruction ◀───┘
    停止語はどの状態でも、サーバーに送らずすぐ on_stop を呼ぶ。

サーバーとの通信は標準ライブラリ（urllib）だけで行う。サーバーの 4xx / 5xx・応答の形の違い・通信の失敗は
VoiceServerError にする（黙って次へ進まない）。サーバーが ok: false を返したのは想定内の結果で、言い直しを頼む。
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from ..config import ConfigError, strict_dataclass_from_dict
from ..task_manager import TidyInstruction
from .audio import Segment, Segmenter, wav_bytes
from .config import VoiceCfg
from .speech import NOT_HEARD, REPHRASE, STOPPING, TOO_LONG, Speaker, confirmation_sentence
from .wake import WakeDetector, normalize

log = logging.getLogger(__name__)


class VoiceServerError(RuntimeError):
    pass


def _check(d, keys: dict, where: str) -> dict:
    if not isinstance(d, dict):
        raise VoiceServerError(f"{where}: 応答が JSON オブジェクトでない: {d!r}")
    for k, tp in keys.items():
        if k not in d or not isinstance(d[k], tp) or (tp is not bool and isinstance(d[k], bool)):
            raise VoiceServerError(f"{where}: 応答の {k} がない、または型が違う: {d!r}")
    return d


class ServerClient:
    def __init__(self, base_url: str, timeout_s: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, data: bytes, content_type: str) -> dict:
        req = urllib.request.Request(self.base_url + path, data=data, method="POST",
                                     headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            raise VoiceServerError(f"{path}: HTTP {e.code}: {e.read()[:500]!r}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise VoiceServerError(f"{path}: サーバーにつながらない（{self.base_url}）: {e}") from e
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise VoiceServerError(f"{path}: 応答が JSON でない: {body[:500]!r}") from e

    def transcribe(self, wav: bytes) -> dict:
        d = _check(self._post("/transcribe", wav, "audio/wav"), {"ok": bool, "seconds": (int, float)}, "/transcribe")
        return _check(d, {"text": str} if d["ok"] else {"reason": str}, "/transcribe")

    def identify(self, image_png_b64: str, boxes: list[list[int]]) -> dict:
        """物体の同定（音声サーバーの /identify）。{"objects": [{"class", "color"} | {"error"}], "seconds"}。"""
        d = _check(self._post("/identify", json.dumps({"image_png_b64": image_png_b64, "boxes": boxes}).encode(),
                              "application/json"), {"objects": list, "seconds": (int, float)}, "/identify")
        if len(d["objects"]) != len(boxes):
            raise VoiceServerError(f"/identify: 枠 {len(boxes)} 個に対して答えが {len(d['objects'])} 個")
        return d

    def instruction(self, text: str) -> dict:
        d = _check(self._post("/instruction", json.dumps({"text": text}).encode(), "application/json"),
                   {"ok": bool}, "/instruction")
        if d["ok"]:
            return _check(d, {"instruction": dict, "raw": str, "seconds": (int, float)}, "/instruction")
        _check(d, {"reason": str}, "/instruction")
        if "raw" not in d or not (d["raw"] is None or isinstance(d["raw"], str)):
            raise VoiceServerError(f"/instruction: 応答の raw がない、または型が違う: {d!r}")
        return d


@dataclass
class Turn:
    """1 つの発話区間の扱いの記録（遅れの計測と、動作確認の表示に使う）。"""
    start_s: float
    duration_s: float
    detection: str
    transcript_local: str
    wake_s: float
    outcome: str = "ignored"          # ignored / wake / stop / instruction / rejected / not_heard / too_long / timeout
    text: str = ""                    # サーバーの書き起こし
    transcribe_s: float | None = None  # サーバーでの処理時間（GPU）
    transcribe_rtt_s: float | None = None  # 送ってから返るまで（通信を含む）
    instruction_s: float | None = None
    instruction_rtt_s: float | None = None
    instruction: dict | None = None
    said: list[str] = field(default_factory=list)


class VoiceClient:
    def __init__(self, cfg: VoiceCfg, wake: WakeDetector, server, speaker: Speaker,
                 on_instruction: Callable[[TidyInstruction], None], on_stop: Callable[[], None]):
        self.cfg = cfg
        self.wake = wake
        self.server = server
        self.speaker = speaker
        self.on_instruction = on_instruction
        self.on_stop = on_stop
        self.state = "idle"
        self.wake_end_s = 0.0
        self.turns: list[Turn] = []

    def run(self, frames: Iterable[np.ndarray], segmenter: Segmenter) -> None:
        for f in frames:
            seg = segmenter.feed(f)
            if seg is not None:
                self.handle(seg)
        segmenter.finish()

    def _say(self, turn: Turn, text: str) -> None:
        turn.said.append(text)
        self.speaker.say(text)

    def handle(self, seg: Segment) -> Turn:
        det = self.wake.detect(seg.audio)
        turn = Turn(seg.start_s, seg.duration_s, det.kind, det.transcript, det.seconds)
        self.turns.append(turn)
        log.info("発話 %.2f–%.2f s: 検出 %s「%s」（%.2f s）", seg.start_s, seg.end_s, det.kind, det.transcript,
                 det.seconds)
        if det.kind == "stop":
            self.on_stop()
            self.state = "idle"
            turn.outcome = "stop"
            self._say(turn, STOPPING)
            return turn
        if self.state == "waiting_command" and seg.start_s - self.wake_end_s > self.cfg.wake.command_timeout_s:
            log.info("ウェイクワードから %.1f s 以内に指示がなかったので待ち受けに戻る", self.cfg.wake.command_timeout_s)
            self.state = "idle"
            if det.kind != "wake":
                turn.outcome = "timeout"
                return turn
        if det.kind == "wake" and not det.trailing_command:
            self.state = "waiting_command"
            self.wake_end_s = seg.end_s
            turn.outcome = "wake"
            return turn
        if det.kind == "wake" or self.state == "waiting_command":
            self.state = "idle"
            self._command(seg, turn)
        return turn

    def _command(self, seg: Segment, turn: Turn) -> None:
        if seg.too_long:
            turn.outcome = "too_long"
            self._say(turn, TOO_LONG)
            return
        t0 = time.perf_counter()
        tr = self.server.transcribe(wav_bytes(seg.audio, self.cfg.audio.sample_rate_hz))
        turn.transcribe_rtt_s, turn.transcribe_s = time.perf_counter() - t0, tr["seconds"]
        if not tr["ok"] or not normalize(tr["text"]):
            log.info("書き起こせなかった: %s", tr.get("reason", "空の書き起こし"))
            turn.outcome = "not_heard"
            self._say(turn, NOT_HEARD)
            return
        turn.text = tr["text"]
        log.info("書き起こし「%s」（サーバー %.2f s、往復 %.2f s）", turn.text, turn.transcribe_s, turn.transcribe_rtt_s)
        t0 = time.perf_counter()
        r = self.server.instruction(turn.text)
        turn.instruction_rtt_s = time.perf_counter() - t0
        if not r["ok"]:
            log.info("指示を実行しない: %s（VLM の出力: %r）", r["reason"], r["raw"])
            turn.outcome = "rejected"
            self._say(turn, REPHRASE)
            return
        turn.instruction_s = r["seconds"]
        try:
            ins = strict_dataclass_from_dict(TidyInstruction, r["instruction"], "instruction")
        except ConfigError as e:
            raise VoiceServerError(f"/instruction の instruction の形が違う: {r['instruction']!r}") from e
        turn.instruction = r["instruction"]
        turn.outcome = "instruction"
        log.info("指示: %s（VLM %.2f s、往復 %.2f s）", r["instruction"], turn.instruction_s, turn.instruction_rtt_s)
        self._say(turn, confirmation_sentence(ins))
        self.on_instruction(ins)
