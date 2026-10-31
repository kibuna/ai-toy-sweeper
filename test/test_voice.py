"""音声入力（toy_sweeper.voice）の単体テスト。マイク・GPU・モデルは使わない。

フィクスチャの WAV（test/data/voice/）は scripts/make_voice_fixtures.py で合成したもの（pyopenjtalk、オフライン）。
"""
import dataclasses
import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from toy_sweeper.config import ConfigError, load_all
from toy_sweeper.task_manager import TidyInstruction, load_vocabulary
from toy_sweeper.voice.audio import (Segment, Segmenter, VoiceInputError, SileroVadClassifier, frames_from_wav,
                                     read_wav, wav_bytes)
from toy_sweeper.voice.client import ServerClient, VoiceClient, VoiceServerError
from toy_sweeper.voice.config import load_voice_cfg, validate_voice_cfg
from toy_sweeper.voice.speech import (NOT_HEARD, REPHRASE, STOPPING, TOO_LONG, RecordingSpeaker,
                                      confirmation_sentence)
from toy_sweeper.voice.wake import Detection, TranscriptWakeDetector, normalize

FIXTURES = Path(__file__).resolve().parent / "data" / "voice"
SR = 16000


@pytest.fixture(scope="module")
def vcfg():
    return load_voice_cfg()


# ---------- 設定 ----------

def test_config_loads(vcfg):
    assert vcfg.audio.frame_samples == 512
    assert vcfg.language == "ja"


@pytest.mark.parametrize("section,key,value", [
    ("audio", "frame_ms", 30),            # Silero VAD が受け付けない
    ("audio", "sample_rate_hz", 44100),   # Whisper の入力でない
    ("vad", "end_silence_ms", 100),       # frame_ms の倍数でない
    ("vad", "threshold", 1.0),
    ("wake", "stop_words", ()),
    ("wake", "stop_words", ("ロボット止まって",)),  # ウェイクワードと重なる
    ("vad", "max_utterance_s", 60.0),     # サーバーの上限を超える
    ("wake", "wake_max_edits", 2),        # 「ロボット」（4 文字）の半分
])
def test_config_rejects(vcfg, section, key, value):
    bad = dataclasses.replace(vcfg, **{section: dataclasses.replace(getattr(vcfg, section), **{key: value})})
    with pytest.raises(ConfigError):
        validate_voice_cfg(bad)


# ---------- WAV ----------

def test_wav_roundtrip():
    x = (np.arange(1600) % 200 - 100).astype(np.int16)
    y = read_wav(io.BytesIO(wav_bytes(x, SR)), SR)
    assert np.array_equal(x, y)


def _wav(x, sr=SR, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.astype(np.int16).tobytes())
    buf.seek(0)
    return buf


@pytest.mark.parametrize("sr,ch", [(44100, 1), (SR, 2)])
def test_read_wav_rejects_format(sr, ch):
    with pytest.raises(VoiceInputError):
        read_wav(_wav(np.zeros(3200), sr, ch), SR)


# ---------- VAD による区切り ----------

class FakeVad:
    """フレームの最初の値が 0 でなければ音声とみなす。"""
    def is_speech(self, frame):
        return bool(frame[0] != 0)


def _frames(pattern, n=512):
    """pattern: '1' が音声フレーム、'0' が無音フレーム。"""
    return [np.full(n, 1000 if c == "1" else 0, np.int16) for c in pattern]


def _segment_all(vcfg, frames, vad=None):
    seg = Segmenter(vcfg.audio, vcfg.vad, vad or FakeVad())
    out = [s for f in frames if (s := seg.feed(f)) is not None]
    seg.finish()
    return out


def test_segmenter_basic(vcfg):
    # 無音 20、音声 30、無音 30（1 フレーム 32 ms、end_silence 704 ms = 22 フレーム）
    out = _segment_all(vcfg, _frames("0" * 20 + "1" * 30 + "0" * 30))
    assert len(out) == 1
    s = out[0]
    assert not s.too_long
    # 発話は 20 フレーム目から。pre_roll 320 ms = 10 フレーム前から含む
    assert s.start_s == pytest.approx(10 * 0.032)
    # 音声 30 + 前 10 + 終わりの無音 22 フレーム
    assert len(s.audio) == (10 + 30 + 22) * 512


def test_segmenter_ignores_blip(vcfg):
    # 2 フレームの物音（start_speech 96 ms = 3 フレーム未満）は発話にしない
    assert _segment_all(vcfg, _frames("0" * 10 + "11" + "0" * 40)) == []


def test_segmenter_keeps_short_pause(vcfg):
    # 512 ms（16 フレーム）の間は一つの発話のまま
    out = _segment_all(vcfg, _frames("0" * 5 + "1" * 20 + "0" * 16 + "1" * 20 + "0" * 30))
    assert len(out) == 1


def test_segmenter_splits_long_pause(vcfg):
    out = _segment_all(vcfg, _frames("0" * 5 + "1" * 20 + "0" * 40 + "1" * 20 + "0" * 30))
    assert len(out) == 2


def test_segmenter_too_long(vcfg):
    # 10 s = 312 フレームを超える発話は too_long で閉じる（黙って続きを捨てたり切り詰めて送ったりしない）
    out = _segment_all(vcfg, _frames("1" * 400 + "0" * 30))
    assert out[0].too_long
    assert len(out[0].audio) / SR <= vcfg.vad.max_utterance_s + 1e-9
    assert all(not s.too_long for s in out[1:])


def test_segmenter_end_in_speech_raises(vcfg):
    seg = Segmenter(vcfg.audio, vcfg.vad, FakeVad())
    for f in _frames("1" * 20):
        seg.feed(f)
    with pytest.raises(VoiceInputError):
        seg.finish()


def test_segmenter_rejects_bad_frame(vcfg):
    seg = Segmenter(vcfg.audio, vcfg.vad, FakeVad())
    with pytest.raises(VoiceInputError):
        seg.feed(np.zeros(100, np.int16))


def test_silero_silence_and_noise(vcfg):
    vad = SileroVadClassifier(vcfg.audio, vcfg.vad.threshold)
    zeros = [np.zeros(512, np.int16)] * 100
    assert _segment_all(vcfg, zeros, vad) == []
    assert _segment_all(vcfg, list(frames_from_wav(FIXTURES / "noise.wav", vcfg.audio)), vad) == []


@pytest.mark.parametrize("db", [-50, -30, -20])
def test_silero_white_noise(vcfg, db):
    """白色雑音（換気扇などの定常雑音の代わり）を発話にしない。webrtcvad はこれを音声と判定した。"""
    rng = np.random.default_rng(0)
    x = np.clip(rng.normal(0, 32768 * 10 ** (db / 20), 5 * SR), -32768, 32767).astype(np.int16)
    frames = [x[i:i + 512] for i in range(0, len(x) - 511, 512)]
    assert _segment_all(vcfg, frames, SileroVadClassifier(vcfg.audio, vcfg.vad.threshold)) == []


def test_silero_fixture_two_utterances(vcfg):
    """「ロボット」（1.2 s の間）「赤いレゴを右上に片付けて」は 2 つの発話に分かれる。"""
    vad = SileroVadClassifier(vcfg.audio, vcfg.vad.threshold)
    out = _segment_all(vcfg, list(frames_from_wav(FIXTURES / "wake_then_red_lego.wav", vcfg.audio)), vad)
    assert len(out) == 2
    assert out[0].duration_s < 2.0 < out[1].duration_s


def test_silero_fixture_one_breath(vcfg):
    """「ロボット、赤いレゴを右上に片付けて」（読点の間 約 0.5 s）は 1 つの発話。"""
    vad = SileroVadClassifier(vcfg.audio, vcfg.vad.threshold)
    out = _segment_all(vcfg, list(frames_from_wav(FIXTURES / "wake_with_command.wav", vcfg.audio)), vad)
    assert len(out) == 1


# ---------- ウェイクワード・停止語 ----------

def test_normalize():
    assert normalize("ろぼっと、") == "ロボット"
    assert normalize(" Robot! ") == "robot"
    assert normalize("ＲＯＢＯＴ") == "robot"
    assert normalize("赤いレゴを。") == "赤イレゴヲ"


def _detector(vcfg, text):
    return TranscriptWakeDetector(lambda audio: text, vcfg.wake)


@pytest.mark.parametrize("text,kind,trailing", [
    ("ロボット", "wake", False),
    ("ろぼっと。", "wake", False),
    ("ねえ、ロボット", "wake", False),
    ("Robot.", "wake", False),
    ("ロボット、赤いレゴを右上に片付けて", "wake", True),
    ("ロボットです", "wake", False),          # 付け足しは指示とみなさない（min_command_chars）
    ("止まって", "stop", False),
    ("ストップ!", "stop", False),
    ("ロボット止まって", "stop", False),       # 停止語が優先
    ("今日はいい天気ですね", "none", False),
    ("", "none", False),
])
def test_wake_detector(vcfg, text, kind, trailing):
    d = _detector(vcfg, text).detect(np.zeros(1600, np.int16))
    assert (d.kind, d.trailing_command) == (kind, trailing)
    assert d.transcript == text
    assert d.seconds >= 0


@pytest.mark.parametrize("k,text,kind,trailing", [
    (0, "レボット", "none", False),
    (1, "レボット", "wake", False),
    (1, "ねえラボット", "wake", False),
    (1, "ロボト、赤いレゴを右上に片付けて", "wake", True),   # 1 文字欠け。後ろの指示の長さは一致した位置から数える
    (1, "ボート", "none", False),                            # 2 文字違う
    (2, "ボート", "wake", False),
    (1, "止まって", "stop", False),
])
def test_wake_fuzzy(vcfg, k, text, kind, trailing):
    det = TranscriptWakeDetector(lambda audio: text, dataclasses.replace(vcfg.wake, wake_max_edits=k))
    d = det.match(text)
    assert (d.kind, d.trailing_command) == (kind, trailing)


def test_approx_find():
    from toy_sweeper.voice.wake import approx_find
    assert approx_find("アイウエオ", "ウエ") == (0, 4)
    assert approx_find("アイウエオ", "ウX") == (1, 3)
    assert approx_find("", "ウエ") == (2, 0)


# ---------- 確認の文 ----------

@pytest.mark.parametrize("ins,expect", [
    (TidyInstruction(("lego",), ("red",), (), "top_right"), "赤いレゴを右上に運びます"),
    (TidyInstruction(("toy_car",), (), ("lego",), "bottom_left"), "レゴはそのままにして、ミニカーを左下に運びます"),
    (TidyInstruction(("lego",), ("green", "yellow"), (), "bottom_right"), "黄色と緑のレゴを右下に運びます"),
    (TidyInstruction(("toy_car", "lego"), (), (), "top_left"), "レゴとミニカーを左上に運びます"),
    (TidyInstruction(("lego",), ("green",), (), "top_left"), "緑のレゴを左上に運びます"),
])
def test_confirmation_sentence(ins, expect):
    assert confirmation_sentence(ins) == expect


def test_confirmation_unknown_raises():
    with pytest.raises(ValueError):
        confirmation_sentence(TidyInstruction(("lego",), (), (), "kitchen"))


# ---------- クライアントの流れ（偽のサーバー・偽の検出器） ----------

GOOD_INS = {"target_classes": ["lego"], "target_colors": ["red"], "excluded_classes": [], "destination": "top_right"}


class FakeServer:
    def __init__(self, text="赤いレゴを右上に片付けて", instruction=None, transcribe_ok=True):
        self.text = text
        self.instruction_reply = instruction or {"ok": True, "instruction": GOOD_INS, "raw": "{}", "seconds": 3.0}
        self.transcribe_ok = transcribe_ok
        self.wavs, self.texts = [], []

    def transcribe(self, wav: bytes) -> dict:
        self.wavs.append(wav)
        if not self.transcribe_ok:
            return {"ok": False, "reason": "生成が上限で打ち切られた", "seconds": 1.0}
        return {"ok": True, "text": self.text, "seconds": 0.5}

    def instruction(self, text: str) -> dict:
        self.texts.append(text)
        return self.instruction_reply


class ScriptedWake:
    """区間ごとに、決めておいた検出結果を順に返す。"""
    def __init__(self, detections):
        self.detections = list(detections)

    def detect(self, audio):
        return self.detections.pop(0)


def _det(kind, trailing=False, text=""):
    return Detection(kind, text, trailing, 0.1)


def _seg(start_s, dur_s=1.0, too_long=False):
    return Segment(np.zeros(int(dur_s * SR), np.int16), start_s, too_long)


class Harness:
    def __init__(self, vcfg, detections, server=None):
        self.server = server or FakeServer()
        self.speaker = RecordingSpeaker()
        self.instructions, self.stops = [], []
        self.client = VoiceClient(vcfg, ScriptedWake(detections), self.server, self.speaker,
                                  on_instruction=self.instructions.append, on_stop=lambda: self.stops.append(True))


def test_flow_wake_then_command(vcfg):
    h = Harness(vcfg, [_det("wake"), _det("none")])
    h.client.handle(_seg(0.0))
    assert h.server.wavs == [] and h.client.state == "waiting_command"
    h.client.handle(_seg(2.0, 2.5))
    assert h.server.texts == ["赤いレゴを右上に片付けて"]
    assert h.instructions == [TidyInstruction(("lego",), ("red",), (), "top_right")]
    assert h.speaker.said == ["赤いレゴを右上に運びます"]
    assert h.client.state == "idle"
    # 送った WAV はその区間の音声そのもの
    assert len(read_wav(io.BytesIO(h.server.wavs[0]), SR)) == int(2.5 * SR)
    t = h.client.turns[-1]
    assert t.outcome == "instruction" and t.transcribe_s == 0.5 and t.instruction_s == 3.0


def test_flow_wake_and_command_in_one_segment(vcfg):
    h = Harness(vcfg, [_det("wake", trailing=True)])
    h.client.handle(_seg(0.0, 3.0))
    assert len(h.instructions) == 1 and len(h.server.wavs) == 1


def test_flow_no_wake_ignored(vcfg):
    h = Harness(vcfg, [_det("none"), _det("none")])
    h.client.handle(_seg(0.0))
    h.client.handle(_seg(2.0))
    assert h.server.wavs == [] and h.speaker.said == [] and h.instructions == []


def test_flow_stop_without_server(vcfg):
    h = Harness(vcfg, [_det("stop")])
    h.client.handle(_seg(0.0))
    assert h.stops == [True]
    assert h.server.wavs == [] and h.server.texts == []
    assert h.speaker.said == [STOPPING]


def test_flow_stop_while_waiting_command(vcfg):
    h = Harness(vcfg, [_det("wake"), _det("stop")])
    h.client.handle(_seg(0.0))
    h.client.handle(_seg(2.0))
    assert h.stops == [True] and h.server.wavs == [] and h.client.state == "idle"


def test_flow_rejected_instruction_asks_rephrase(vcfg):
    srv = FakeServer(text="床を拭いて", instruction={"ok": False, "reason": "InstructionError: ...", "raw": "{}"})
    h = Harness(vcfg, [_det("wake"), _det("none")], srv)
    h.client.handle(_seg(0.0))
    h.client.handle(_seg(2.0))
    assert h.instructions == [] and h.speaker.said == [REPHRASE]
    assert h.client.turns[-1].outcome == "rejected"


def test_flow_empty_transcript(vcfg):
    h = Harness(vcfg, [_det("wake"), _det("none")], FakeServer(text=" 。"))
    h.client.handle(_seg(0.0))
    h.client.handle(_seg(2.0))
    assert h.server.texts == [] and h.speaker.said == [NOT_HEARD]


def test_flow_transcribe_not_ok(vcfg):
    h = Harness(vcfg, [_det("wake"), _det("none")], FakeServer(transcribe_ok=False))
    h.client.handle(_seg(0.0))
    h.client.handle(_seg(2.0))
    assert h.server.texts == [] and h.speaker.said == [NOT_HEARD]


def test_flow_too_long_command(vcfg):
    h = Harness(vcfg, [_det("wake"), _det("none")])
    h.client.handle(_seg(0.0))
    h.client.handle(_seg(2.0, 10.0, too_long=True))
    assert h.server.wavs == [] and h.speaker.said == [TOO_LONG] and h.client.state == "idle"


def test_flow_command_timeout(vcfg):
    h = Harness(vcfg, [_det("wake"), _det("none")])
    h.client.handle(_seg(0.0))
    h.client.handle(_seg(1.0 + vcfg.wake.command_timeout_s + 0.5))
    assert h.server.wavs == [] and h.client.state == "idle"


def test_flow_bad_instruction_from_server_raises(vcfg):
    """サーバーが ok: true なのに形の違う instruction を返したら例外（黙って使わない）。"""
    srv = FakeServer(instruction={"ok": True, "instruction": {"target_classes": ["lego"]}, "raw": "", "seconds": 1.0})
    h = Harness(vcfg, [_det("wake"), _det("none")], srv)
    h.client.handle(_seg(0.0))
    with pytest.raises(VoiceServerError):
        h.client.handle(_seg(2.0))


def test_run_feeds_frames(vcfg):
    """フレームの列から区切り → 検出 → サーバーまで通す。"""
    h = Harness(vcfg, [_det("wake"), _det("none")])
    frames = _frames("0" * 10 + "1" * 20 + "0" * 40 + "1" * 60 + "0" * 30)
    h.client.run(frames, Segmenter(vcfg.audio, vcfg.vad, FakeVad()))
    assert len(h.instructions) == 1


# ---------- HTTP クライアント（手元に立てた偽のサーバー） ----------

class _Handler(BaseHTTPRequestHandler):
    replies: dict = {}

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        code, body = self.replies[self.path]
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def http_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv
    srv.shutdown()


def _client(srv):
    return ServerClient(f"http://127.0.0.1:{srv.server_address[1]}", timeout_s=5.0)


def test_server_client_ok(http_server):
    _Handler.replies = {"/transcribe": (200, {"ok": True, "text": "赤", "seconds": 0.1}),
                        "/instruction": (200, {"ok": False, "reason": "x", "raw": None})}
    c = _client(http_server)
    assert c.transcribe(b"RIFF")["text"] == "赤"
    assert c.instruction("赤")["ok"] is False


def test_server_client_5xx_raises(http_server):
    _Handler.replies = {"/instruction": (500, "Internal Server Error")}
    with pytest.raises(VoiceServerError, match="500"):
        _client(http_server).instruction("赤")


@pytest.mark.parametrize("body", [{"text": "赤"}, {"ok": True, "text": 1, "seconds": 0.1},
                                  {"ok": True, "text": "赤", "seconds": True}, "not json"])
def test_server_client_bad_shape_raises(http_server, body):
    _Handler.replies = {"/transcribe": (200, body)}
    with pytest.raises(VoiceServerError):
        _client(http_server).transcribe(b"RIFF")


def test_server_client_unreachable():
    with pytest.raises(VoiceServerError):
        ServerClient("http://127.0.0.1:9", timeout_s=2.0).instruction("赤")


# ---------- サーバー（FastAPI、偽の ASR と偽の VLM） ----------

@dataclasses.dataclass
class _Reply:
    text: str
    n_tokens: int = 10
    seconds: float = 0.2


GOOD_JSON = json.dumps({"task": "collect", "target_classes": ["lego"], "target_colors": ["red"],
                        "excluded_classes": [], "destination": "top_right", "constraints": [],
                        "needs_confirmation": False})
# 「床を拭いて」に VLM がもっともらしい JSON を返した場合（check_grounding で止まるべき）
FLOOR_JSON = json.dumps({"task": "collect", "target_classes": ["lego", "toy_car"], "target_colors": [],
                         "excluded_classes": [], "destination": "top_right", "constraints": [],
                         "needs_confirmation": False})


@pytest.fixture
def app_client(vcfg):
    from fastapi.testclient import TestClient

    from toy_sweeper.voice.server import create_app

    class FakeASR:
        def __call__(self, audio):
            if len(audio) == 0:
                raise AssertionError
            return "赤いレゴを右上に片付けて"

    class FakeVLM:
        replies = {}

        def ask(self, text, max_new_tokens, image=None):
            for k, v in self.replies.items():
                if k in text:
                    if isinstance(v, Exception):
                        raise v
                    return _Reply(v)
            raise AssertionError(text)

    vlm = FakeVLM()
    app = create_app(FakeASR(), vlm, vcfg, load_all(), load_vocabulary(), max_new_tokens=512)
    return TestClient(app, raise_server_exceptions=False), vlm


def test_app_transcribe(app_client):
    c, _ = app_client
    r = c.post("/transcribe", content=wav_bytes(np.zeros(SR, np.int16), SR), headers={"Content-Type": "audio/wav"})
    assert r.status_code == 200
    assert r.json()["ok"] is True and r.json()["text"] == "赤いレゴを右上に片付けて" and r.json()["seconds"] >= 0


@pytest.mark.parametrize("body", [b"not a wav", _wav(np.zeros(SR), 44100).getvalue(),
                                  _wav(np.zeros(SR), SR, 2).getvalue(), wav_bytes(np.zeros(0, np.int16), SR),
                                  wav_bytes(np.zeros(31 * SR, np.int16), SR)])
def test_app_transcribe_rejects(app_client, body):
    c, _ = app_client
    assert c.post("/transcribe", content=body).status_code == 400


def test_app_instruction_ok(app_client):
    c, vlm = app_client
    vlm.replies = {"赤いレゴ": "```json\n" + GOOD_JSON + "\n```"}
    r = c.post("/instruction", json={"text": "赤いレゴを右上に片付けて"}).json()
    assert r["ok"] is True
    assert r["instruction"] == GOOD_INS
    assert r["seconds"] >= 0


@pytest.mark.parametrize("text,reply,reason", [
    ("床を拭いて", FLOOR_JSON, "InstructionError"),          # 根拠のない出力は止める
    ("赤いレゴを右上に", "わかりました", "VLMOutputError"),     # JSON でない
])
def test_app_instruction_rejected(app_client, text, reply, reason):
    c, vlm = app_client
    vlm.replies = {text: reply}
    r = c.post("/instruction", json={"text": text})
    assert r.status_code == 200
    assert r.json()["ok"] is False and r.json()["reason"].startswith(reason) and r.json()["raw"] == reply


def test_app_instruction_truncated_is_rejected(app_client):
    from toy_sweeper.vlm import VLMOutputError
    c, vlm = app_client
    vlm.replies = {"赤": VLMOutputError("生成が上限 512 トークンで打ち切られた")}
    r = c.post("/instruction", json={"text": "赤いレゴを右上に"}).json()
    assert r["ok"] is False and r["raw"] is None


def test_app_instruction_server_error_is_5xx(app_client):
    c, vlm = app_client
    vlm.replies = {"赤": RuntimeError("CUDA out of memory")}
    assert c.post("/instruction", json={"text": "赤いレゴを右上に"}).status_code == 500


@pytest.mark.parametrize("body", [{"text": "  "}, {"txt": "赤"}, {"text": "赤", "extra": 1}])
def test_app_instruction_bad_request(app_client, body):
    c, _ = app_client
    assert c.post("/instruction", json=body).status_code in (400, 422)


def test_identify_objects_crops_and_parses():
    """/identify: 枠ごとに切り出して VLM に種類と色を聞く。決まった値以外は例外（黙って使わない）。"""
    import base64
    import cv2
    from fastapi.testclient import TestClient
    from toy_sweeper.vlm_config import load_vlm_cfg
    from toy_sweeper.voice.server import create_app

    class ColorVLM:   # 切り出しの平均色で答える偽物
        def ask(self, text, max_new_tokens, image=None):
            r, g, b = np.asarray(image).reshape(-1, 3).mean(0)
            return _Reply('{"class": "lego", "color": "%s"}' % ("red" if r > b else "blue"))

    img = np.full((200, 300, 3), 200, np.uint8)
    img[40:70, 30:80] = (220, 40, 40)
    img[120:150, 200:260] = (40, 60, 220)
    png = base64.b64encode(cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()).decode()
    app = create_app(lambda a: "", ColorVLM(), load_voice_cfg(), load_all(), load_vocabulary(), max_new_tokens=512,
                     vlm_cfg=dataclasses.replace(load_vlm_cfg(), crop_margin_px=2, identify_mode="crops"))
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/identify", json={"image_png_b64": png, "boxes": [[30, 40, 80, 70], [200, 120, 260, 150]]}).json()
    assert [(o["class"], o["color"]) for o in r["objects"]] == [("lego", "red"), ("lego", "blue")]
    assert c.post("/identify", json={"image_png_b64": png, "boxes": [[0, 0, 400, 50]]}).status_code == 400   # 画像の外

    class BadVLM:
        def ask(self, text, max_new_tokens, image=None):
            return _Reply('{"class": "dinosaur", "color": "red"}')
    app = create_app(lambda a: "", BadVLM(), load_voice_cfg(), load_all(), load_vocabulary(), max_new_tokens=512,
                     vlm_cfg=dataclasses.replace(load_vlm_cfg(), identify_mode="crops"))
    r = TestClient(app, raise_server_exceptions=False).post("/identify", json={"image_png_b64": png, "boxes": [[30, 40, 80, 70]]})
    assert r.json()["objects"][0]["error"]


def test_identify_grid_mode():
    """grid: 切り出しを番号つきのマスにまとめて 1 回で聞く。番号がそろわなければ全部 error。"""
    import base64
    import cv2
    from fastapi.testclient import TestClient
    from toy_sweeper.vlm_config import load_vlm_cfg
    from toy_sweeper.voice.server import create_app
    calls = []

    class GridVLM:
        def __init__(self, reply):
            self.reply = reply

        def ask(self, text, max_new_tokens, image=None):
            calls.append(image.size)
            return _Reply(self.reply)
    img = np.full((200, 300, 3), 200, np.uint8)
    png = base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()
    boxes = [[30, 40, 80, 70], [200, 120, 260, 150], [10, 10, 40, 40]]
    good = '[{"id": 2, "class": "toy_car", "color": "pink"}, {"id": 1, "class": "lego", "color": "lime"}, {"id": 3, "class": "other", "color": "black"}]'
    cfg = dataclasses.replace(load_vlm_cfg(), identify_mode="grid")
    c = TestClient(create_app(lambda a: "", GridVLM(good), load_voice_cfg(), load_all(), load_vocabulary(), 512, vlm_cfg=cfg))
    r = c.post("/identify", json={"image_png_b64": png, "boxes": boxes}).json()
    assert [(o["class"], o["color"]) for o in r["objects"]] == [("lego", "lime"), ("toy_car", "pink"), ("other", None)]   # other の色は使わない（_check_object）
    assert len(calls) == 1 and calls[0] == (2 * cfg.crop_size_px, 2 * cfg.crop_size_px)   # 3 個は 2 × 2 マスで 1 回
    bad = '[{"id": 1, "class": "lego", "color": "red"}]'
    c = TestClient(create_app(lambda a: "", GridVLM(bad), load_voice_cfg(), load_all(), load_vocabulary(), 512, vlm_cfg=cfg))
    assert all("error" in o for o in c.post("/identify", json={"image_png_b64": png, "boxes": boxes}).json()["objects"])


def test_check_object_other_ignores_color():
    """おもちゃでない物（other）の色は使わないので、決まった色の言葉でなくても受け付ける（色は None）。2026-10-22、実機の画像で
    ソファの脚の影に VLM が {"class": "other", "color": "brown"} と答え、同定ごと止まった。おもちゃの色は今までどおり確かめる。"""
    from toy_sweeper.voice.server import _check_object
    assert _check_object({"class": "other", "color": "brown"}, "") == {"class": "other", "color": None}
    assert "error" in _check_object({"class": "lego", "color": "brown"}, "")
    assert "error" in _check_object({"class": "chair", "color": "red"}, "")
    assert _check_object({"class": "lego", "color": "red"}, "") == {"class": "lego", "color": "red"}
