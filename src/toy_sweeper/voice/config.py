"""config/voice.yaml の読み込み。値の検査もここで行う（不正なら ConfigError、既定値で代用しない）。"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import CONFIG_DIR, ConfigError, load_yaml, strict_dataclass_from_dict

SAMPLE_RATE_HZ = 16000            # Whisper の入力の仕様。設定値もこれでなければならない
SILERO_FRAME_MS = 32              # Silero VAD（v6）は 16 kHz で 512 サンプルずつ受け付ける


@dataclass(frozen=True)
class AudioCfg:
    sample_rate_hz: int
    frame_ms: int

    @property
    def frame_samples(self) -> int:
        return self.sample_rate_hz * self.frame_ms // 1000


@dataclass(frozen=True)
class VADCfg:
    threshold: float
    start_speech_ms: int
    end_silence_ms: int
    pre_roll_ms: int
    max_utterance_s: float


@dataclass(frozen=True)
class WakeCfg:
    model: str
    compute_type: str
    cpu_threads: int
    beam_size: int
    hotwords: str
    wake_words: tuple[str, ...]
    wake_max_edits: int
    stop_words: tuple[str, ...]
    min_command_chars: int
    command_timeout_s: float


@dataclass(frozen=True)
class ServerCfg:
    asr_model_id: str
    asr_dtype: str
    asr_num_beams: int
    asr_max_new_tokens: int
    max_audio_s: float
    asr_prompt: str


@dataclass(frozen=True)
class ClientCfg:
    request_timeout_s: float
    say_voice: str
    say_rate_wpm: int


@dataclass(frozen=True)
class VoiceCfg:
    language: str
    audio: AudioCfg
    vad: VADCfg
    wake: WakeCfg
    server: ServerCfg
    client: ClientCfg


def validate_voice_cfg(c: VoiceCfg) -> None:
    from .wake import normalize
    if c.audio.sample_rate_hz != SAMPLE_RATE_HZ:
        raise ConfigError(f"audio.sample_rate_hz は {SAMPLE_RATE_HZ}（Whisper の入力）でなければならない: {c.audio.sample_rate_hz}")
    if c.audio.frame_ms != SILERO_FRAME_MS:
        raise ConfigError(f"audio.frame_ms は Silero VAD の {SILERO_FRAME_MS} ms でなければならない: {c.audio.frame_ms}")
    if not 0 < c.vad.threshold < 1:
        raise ConfigError(f"vad.threshold は 0 と 1 の間: {c.vad.threshold}")
    for name in ("start_speech_ms", "end_silence_ms", "pre_roll_ms"):
        v = getattr(c.vad, name)
        if v <= 0 or v % c.audio.frame_ms:
            raise ConfigError(f"vad.{name} は frame_ms（{c.audio.frame_ms}）の正の倍数: {v}")
    if c.vad.max_utterance_s * 1000 <= c.vad.pre_roll_ms + c.vad.start_speech_ms:
        raise ConfigError("vad.max_utterance_s が短すぎる（発話の始まりの判定より短い）")
    if c.vad.max_utterance_s > c.server.max_audio_s:
        raise ConfigError(f"vad.max_utterance_s（{c.vad.max_utterance_s}）がサーバーの上限 max_audio_s（{c.server.max_audio_s}）を超える")
    if not c.wake.wake_words or not c.wake.stop_words:
        raise ConfigError("wake.wake_words と wake.stop_words は空にできない")
    wake = {normalize(w) for w in c.wake.wake_words}
    stop = {normalize(w) for w in c.wake.stop_words}
    if "" in wake | stop:
        raise ConfigError("正規化すると空になる語がある")
    if c.wake.wake_max_edits < 0 or any(2 * c.wake.wake_max_edits >= len(w) for w in wake):
        raise ConfigError(f"wake.wake_max_edits（{c.wake.wake_max_edits}）は 0 以上で、各ウェイクワードの文字数の半分未満")
    for w in wake:
        for s in stop:
            if w in s or s in w:
                raise ConfigError(f"ウェイクワード {w} と停止語 {s} が重なる")
    for name in ("cpu_threads", "beam_size", "min_command_chars"):
        if getattr(c.wake, name) <= 0:
            raise ConfigError(f"wake.{name} は正の値: {getattr(c.wake, name)}")
    if c.wake.command_timeout_s <= 0 or c.client.request_timeout_s <= 0:
        raise ConfigError("待ち時間は正の値")
    if c.server.asr_num_beams <= 0 or c.server.asr_max_new_tokens <= 0 or c.server.max_audio_s <= 0:
        raise ConfigError("server の値は正の値")


def load_voice_cfg(config_dir=CONFIG_DIR) -> VoiceCfg:
    c = strict_dataclass_from_dict(VoiceCfg, load_yaml("voice.yaml", config_dir), "voice")
    validate_voice_cfg(c)
    return c
