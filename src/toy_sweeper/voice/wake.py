"""ウェイクワードと停止語の検出。

試作は「VAD の区間ごとに小さい Whisper（CPU）で書き起こし、語が含まれるかを見る」（TranscriptWakeDetector）。
WakeDetector の形を満たせば、openWakeWord などの専用のキーワード検出器に置き換えられる
（openWakeWord の既成のモデルは英語のみなので、日本語には自前で学習したモデルが要る）。
"""
from __future__ import annotations

import dataclasses
import time
import unicodedata
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

import numpy as np

from .config import WakeCfg

Kind = Literal["wake", "stop", "none"]


@dataclass(frozen=True)
class Detection:
    kind: Kind
    transcript: str           # 書き起こし（専用の検出器なら空でよい）
    trailing_command: bool    # ウェイクワードの後ろに指示も続いている（同じ区間を指示として送る）
    seconds: float            # 検出にかかった時間


class WakeDetector(Protocol):
    def detect(self, audio: np.ndarray) -> Detection: ...


def normalize(text: str) -> str:
    """NFKC、ひらがな→カタカナ、小文字化、記号・空白を除く。書き起こしと語の照合に使う。"""
    out = []
    for ch in unicodedata.normalize("NFKC", text).lower():
        if "ぁ" <= ch <= "ゖ":
            ch = chr(ord(ch) + 0x60)
        if unicodedata.category(ch)[0] in "PZSC":
            continue
        out.append(ch)
    return "".join(out)


def approx_find(text: str, pat: str) -> tuple[int, int]:
    """text のどこかの部分文字列と pat の編集距離の最小値と、その部分文字列の終わりの位置（最小が複数なら最も前）。
    Sellers の方法（編集距離の表の 1 行目を 0 にして、どこからでも始められるようにする）。"""
    prev = list(range(len(pat) + 1))          # text の 0 文字目まで（空）に対する各 pat[:i] の距離
    best, best_end = prev[-1], 0
    for j, ch in enumerate(text, 1):
        cur = [0]
        for i, cp in enumerate(pat, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (cp != ch)))
        if cur[-1] < best:
            best, best_end = cur[-1], j
        prev = cur
    return best, best_end


class TranscriptWakeDetector:
    """書き起こしに停止語（完全一致）・ウェイクワード（編集距離 wake_max_edits まで）が含まれるかを見る。"""

    def __init__(self, transcribe: Callable[[np.ndarray], str], cfg: WakeCfg):
        self.transcribe = transcribe
        self.wake = [normalize(w) for w in cfg.wake_words]
        self.stop = [normalize(w) for w in cfg.stop_words]
        self.min_command_chars = cfg.min_command_chars
        self.max_edits = cfg.wake_max_edits

    def detect(self, audio: np.ndarray) -> Detection:
        t0 = time.perf_counter()
        text = self.transcribe(audio)
        return dataclasses.replace(self.match(text), seconds=time.perf_counter() - t0)

    def match(self, text: str) -> Detection:
        t = normalize(text)
        if any(s in t for s in self.stop):
            return Detection("stop", text, False, 0.0)
        ends = [end for w in self.wake for d, end in [approx_find(t, w)] if d <= self.max_edits]
        if not ends:
            return Detection("none", text, False, 0.0)
        return Detection("wake", text, len(t) - min(ends) >= self.min_command_chars, 0.0)


class FasterWhisperTranscriber:
    """faster-whisper（CTranslate2、CPU）による書き起こし。torch を使わない。"""

    def __init__(self, cfg: WakeCfg, language: str):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(cfg.model, device="cpu", compute_type=cfg.compute_type, cpu_threads=cfg.cpu_threads)
        self.cfg = cfg
        self.language = language

    def __call__(self, audio: np.ndarray) -> str:
        if audio.dtype != np.int16:
            raise ValueError(f"音声は int16: {audio.dtype}")
        # 区間は VAD で切り出し済みなので vad_filter は使わない。区間ごとに独立なので前の文脈は使わない。
        # 温度は 0 に固定し、faster-whisper の既定の「温度を上げてサンプリングし直す」フォールバックを切る
        # （2026-10-16、既定のままだと同じ区間が「ロボット」にも「レバッド」にもなり、結果が乱数で変わった）。
        # フォールバックの判定にだけ使う閾値（compression_ratio / log_prob）も切る。no_speech_threshold も切り、
        # 無音らしいという理由で書き起こしを黙って捨てない（VAD で音声と判定した区間なので）
        segments, _ = self.model.transcribe(audio.astype(np.float32) / 32768.0, language=self.language,
                                            beam_size=self.cfg.beam_size, vad_filter=False,
                                            condition_on_previous_text=False, without_timestamps=True,
                                            hotwords=self.cfg.hotwords or None, temperature=0.0,
                                            compression_ratio_threshold=None, log_prob_threshold=None,
                                            no_speech_threshold=None)
        return "".join(s.text for s in segments).strip()
