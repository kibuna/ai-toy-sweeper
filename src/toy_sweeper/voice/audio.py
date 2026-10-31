"""音声の入出力と、VAD による発話の区切り。

音声は 16 kHz・モノラル・16 bit 整数（np.int16）で扱う。形式が違えば VoiceInputError（変換して読まない）。
マイク（sounddevice）は mic_frames の中でだけ import する（音声機器のない環境でもテストが動くように）。
"""
from __future__ import annotations

import io
import logging
import queue
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Protocol

import numpy as np

from .config import AudioCfg, VADCfg

log = logging.getLogger(__name__)


class VoiceInputError(ValueError):
    pass


# ---------- WAV ----------

def read_wav(src: str | Path | BinaryIO, sample_rate_hz: int) -> np.ndarray:
    try:
        with wave.open(str(src) if isinstance(src, Path) else src, "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype()) != (1, 2, sample_rate_hz, "NONE"):
                raise VoiceInputError(f"WAV は {sample_rate_hz} Hz・モノラル・16 bit PCM でなければならない: "
                                      f"{w.getframerate()} Hz, {w.getnchannels()} ch, {8 * w.getsampwidth()} bit")
            return np.frombuffer(w.readframes(w.getnframes()), np.int16).copy()
    except (wave.Error, EOFError) as e:
        raise VoiceInputError(f"WAV として読めない: {e}") from e


def wav_bytes(audio: np.ndarray, sample_rate_hz: int) -> bytes:
    if audio.dtype != np.int16 or audio.ndim != 1:
        raise VoiceInputError(f"音声は 1 次元の int16: {audio.dtype}, {audio.shape}")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate_hz)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def frames_from_wav(path: Path, cfg: AudioCfg) -> Iterator[np.ndarray]:
    """WAV をフレームに分ける。最後の端数のフレーム（frame_ms 未満）は 0 で埋めて 1 フレームにする。
    発話の途中でファイルが終わっていれば Segmenter.finish が VoiceInputError にするので、埋めた 0 が発話を隠すことはない。"""
    x = read_wav(path, cfg.sample_rate_hz)
    n = cfg.frame_samples
    pad = (-len(x)) % n
    x = np.concatenate([x, np.zeros(pad, np.int16)])
    for i in range(0, len(x), n):
        yield x[i:i + n]


def mic_frames(cfg: AudioCfg) -> Iterator[np.ndarray]:
    """マイクからフレームを順に返す（止めるまで続く）。取りこぼし（入力のあふれ）は例外にする。"""
    import sounddevice as sd
    q: queue.Queue = queue.Queue()
    errors: list[str] = []

    def callback(indata, frames, time_info, status):
        if status:
            errors.append(str(status))
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=cfg.sample_rate_hz, channels=1, dtype="int16", blocksize=cfg.frame_samples,
                        callback=callback):
        log.info("マイクの入力を開始（%d Hz、%d ms ごと）", cfg.sample_rate_hz, cfg.frame_ms)
        while True:
            frame = q.get()
            if errors:
                raise VoiceInputError(f"マイクの入力で問題が起きた（取りこぼしの可能性）: {errors}")
            yield frame


# ---------- VAD ----------

class FrameClassifier(Protocol):
    def is_speech(self, frame: np.ndarray) -> bool: ...


class SileroVadClassifier:
    """Silero VAD（v6、faster-whisper に同梱の onnx）を 1 フレーム（512 サンプル）ずつ動かす。

    faster-whisper の SileroVADModel は音声全体をまとめて処理する形なので、同じ入出力（直前 64 サンプルの文脈、
    状態 h・c）を自前で 1 フレームずつ受け渡す。webrtcvad は白色雑音を音声と判定したので使わない。
    """
    CONTEXT = 64

    def __init__(self, cfg: AudioCfg, threshold: float):
        import onnxruntime
        from faster_whisper.utils import get_assets_path
        path = Path(get_assets_path()) / "silero_vad_v6.onnx"
        if not path.is_file():
            raise FileNotFoundError(f"Silero VAD のモデルがない（faster-whisper の版の違い？）: {path}")
        if cfg.frame_samples != 512:
            raise VoiceInputError(f"Silero VAD は 16 kHz で 512 サンプルのフレーム: {cfg.frame_samples}")
        opts = onnxruntime.SessionOptions()
        # 1 フレームが小さく、スレッドを増やしても速くならない（faster-whisper も 1 スレッドで動かしている）
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self.sess = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=opts)
        self.threshold = threshold
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.ctx = np.zeros(self.CONTEXT, np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = frame.astype(np.float32) / 32768.0
        out, self.h, self.c = self.sess.run(None, {"input": np.concatenate([self.ctx, x])[None], "h": self.h,
                                                   "c": self.c})
        self.ctx = x[-self.CONTEXT:]
        return float(np.ravel(out)[0])

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.prob(frame) >= self.threshold


@dataclass(frozen=True)
class Segment:
    audio: np.ndarray        # int16
    start_s: float           # 入力の先頭からの時刻（区間の最初のサンプル）
    too_long: bool           # max_utterance_s に達して閉じた（発話はまだ続いていたかもしれない）

    @property
    def duration_s(self) -> float:
        from .config import SAMPLE_RATE_HZ
        return len(self.audio) / SAMPLE_RATE_HZ

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


class Segmenter:
    """フレームを 1 つずつ受け取り、発話の区間が閉じたら Segment を返す。

    - 「音声」フレームが start_speech_ms 続いたら発話の始まり。その前の pre_roll_ms も区間に含める
    - 「音声でない」フレームが end_silence_ms 続いたら発話の終わり（その無音も区間に含める）
    - max_utterance_s に達したら too_long の区間として閉じる。続きの音声は新しい発話として扱う
    """

    def __init__(self, audio: AudioCfg, vad: VADCfg, classifier: FrameClassifier):
        self.n = audio.frame_samples
        self.frame_s = audio.frame_ms / 1000
        self.start_frames = vad.start_speech_ms // audio.frame_ms
        self.end_frames = vad.end_silence_ms // audio.frame_ms
        self.max_frames = int(vad.max_utterance_s * 1000) // audio.frame_ms
        self.classifier = classifier
        self.ring: deque = deque(maxlen=vad.pre_roll_ms // audio.frame_ms + self.start_frames)
        self.t = 0                  # 受け取ったフレーム数
        self.speech_run = 0
        self.silence_run = 0
        self.current: list[np.ndarray] | None = None
        self.current_start = 0

    def feed(self, frame: np.ndarray) -> Segment | None:
        if frame.dtype != np.int16 or frame.shape != (self.n,):
            raise VoiceInputError(f"フレームは int16 で {self.n} サンプル: {frame.dtype}, {frame.shape}")
        speech = self.classifier.is_speech(frame)
        self.t += 1
        if self.current is None:
            self.ring.append(frame)
            self.speech_run = self.speech_run + 1 if speech else 0
            if self.speech_run >= self.start_frames:
                self.current = list(self.ring)
                self.current_start = self.t - len(self.ring)
                self.silence_run = 0
                self.ring.clear()
            return None
        self.current.append(frame)
        self.silence_run = 0 if speech else self.silence_run + 1
        if self.silence_run >= self.end_frames:
            return self._close(too_long=False)
        if len(self.current) >= self.max_frames:
            log.warning("発話が上限 %.1f s に達したので区間を閉じる（too_long）", self.max_frames * self.frame_s)
            return self._close(too_long=True)
        return None

    def _close(self, too_long: bool) -> Segment:
        seg = Segment(np.concatenate(self.current), self.current_start * self.frame_s, too_long)
        self.current = None
        self.speech_run = 0
        return seg

    def finish(self) -> None:
        """入力の終わり。発話の途中で終わっていれば VoiceInputError（途中までの音声を黙って使わない）。"""
        if self.current is not None:
            raise VoiceInputError(f"入力が発話の途中で終わった（{self.current_start * self.frame_s:.2f} s から）。"
                                  "WAV の末尾には end_silence_ms 以上の無音が要る")
