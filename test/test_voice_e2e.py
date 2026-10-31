"""音声入力の結合テスト（本物のモデルとサーバー）。既定では skip する。

    # 1. GPU の PC でサーバーを起動
    ~/.pixi/bin/pixi run -e vlm python scripts/voice_server.py --host 127.0.0.1 --port 8765
    # 2. 別の端末で
    TOY_SWEEPER_VOICE_E2E=1 TOY_SWEEPER_VOICE_SERVER=http://127.0.0.1:8765 uv run pytest test/test_voice_e2e.py -v

TOY_SWEEPER_VOICE_CPU_THREADS を設定すると、config の wake.cpu_threads の代わりに使う（開発機で学習と並べるとき）。

フィクスチャ（test/data/voice/、合成音声）を、Silero VAD → faster-whisper（CPU）→ サーバー（Whisper + VLM）に通す。
"""
import dataclasses
import os
from pathlib import Path

import pytest

from toy_sweeper.task_manager import TidyInstruction
from toy_sweeper.voice.audio import Segmenter, SileroVadClassifier, frames_from_wav
from toy_sweeper.voice.client import ServerClient, VoiceClient
from toy_sweeper.voice.config import load_voice_cfg
from toy_sweeper.voice.speech import NOT_HEARD, REPHRASE, STOPPING, RecordingSpeaker
from toy_sweeper.voice.wake import FasterWhisperTranscriber, TranscriptWakeDetector

FIXTURES = Path(__file__).resolve().parent / "data" / "voice"

pytestmark = pytest.mark.skipif(os.environ.get("TOY_SWEEPER_VOICE_E2E") != "1",
                                reason="本物のモデルとサーバーが要る。TOY_SWEEPER_VOICE_E2E=1 と "
                                       "TOY_SWEEPER_VOICE_SERVER=<URL> を設定して実行する")


@pytest.fixture(scope="module")
def setup():
    url = os.environ.get("TOY_SWEEPER_VOICE_SERVER")
    if not url:
        raise RuntimeError("TOY_SWEEPER_VOICE_E2E=1 のときは TOY_SWEEPER_VOICE_SERVER（サーバーの URL）が要る")
    c = load_voice_cfg()
    if threads := os.environ.get("TOY_SWEEPER_VOICE_CPU_THREADS"):
        c = dataclasses.replace(c, wake=dataclasses.replace(c.wake, cpu_threads=int(threads)))
    return c, TranscriptWakeDetector(FasterWhisperTranscriber(c.wake, c.language), c.wake), ServerClient(
        url, c.client.request_timeout_s)


def _run(setup, *names):
    c, wake, server = setup
    speaker = RecordingSpeaker()
    got, stops = [], []
    client = VoiceClient(c, wake, server, speaker, got.append, lambda: stops.append(True))
    frames = (f for n in names for f in frames_from_wav(FIXTURES / f"{n}.wav", c.audio))
    client.run(frames, Segmenter(c.audio, c.vad, SileroVadClassifier(c.audio, c.vad.threshold)))
    return client, speaker, got, stops


RED_LEGO = TidyInstruction(("lego",), ("red",), (), "top_right")


@pytest.mark.parametrize("name", ["wake_then_red_lego", "wake_with_command"])
def test_red_lego(setup, name):
    client, speaker, got, stops = _run(setup, name)
    assert got == [RED_LEGO]
    assert speaker.said == ["赤いレゴを右上に運びます"]
    assert stops == []


def test_stop_without_server(setup):
    client, speaker, got, stops = _run(setup, "stop")
    assert stops == [True] and speaker.said == [STOPPING]
    assert all(t.transcribe_rtt_s is None for t in client.turns)


def test_unsupported_instruction_asks_rephrase(setup):
    client, speaker, got, stops = _run(setup, "wake_then_floor")
    assert got == [] and speaker.said == [REPHRASE]


def test_chatter_and_noise_ignored(setup):
    client, speaker, got, stops = _run(setup, "chatter", "noise")
    assert got == [] and speaker.said == [] and stops == []
    assert NOT_HEARD not in speaker.said


def test_sequence(setup):
    """続けて流す: 雑談 → 指示 → 停止。"""
    client, speaker, got, stops = _run(setup, "chatter", "wake_then_red_lego", "stop")
    assert got == [RED_LEGO] and stops == [True]
    assert speaker.said == ["赤いレゴを右上に運びます", STOPPING]
