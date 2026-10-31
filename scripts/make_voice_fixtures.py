"""音声入力の試験用の WAV を、オフラインの日本語音声合成（pyopenjtalk）で作る。ネットワークの音声合成は使わない。

    uv run python scripts/make_voice_fixtures.py --fixtures test/data/voice
    uv run python scripts/make_voice_fixtures.py --asr-set data/generated/voice_asr

--fixtures: 単体テストと結合テストに使う短い WAV（リポジトリに入れる）
  wake_then_red_lego.wav  「ロボット」1.2 s の間「赤いレゴを右上に片付けて」（2 つの発話）
  wake_with_command.wav   「ロボット、赤いレゴを右上に片付けて」（一息、1 つの発話）
  stop.wav                「止まって」
  wake_then_floor.wav     「ロボット」1.2 s の間「床を拭いて」（対応外の指示）
  chatter.wav             「今日はいい天気ですね」（ウェイクワードなし）
  noise.wav               白色雑音のみ（-40 dBFS、3 s）
--asr-set: data/vlm_eval/instructions.jsonl の日本語の指示を、話す速さと声の高さを変えて 3 通りずつ合成する
  （サーバーの Whisper の比較用、scripts/eval_voice_asr.py）。manifest.jsonl に WAV と元の文と正解を書く。

音声はすべて 16 kHz・モノラル・16 bit。pyopenjtalk の出力（48 kHz）を 1/3 に間引き、ピークを -3 dBFS に揃える。
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyopenjtalk
from scipy.signal import resample_poly

from toy_sweeper.voice.audio import wav_bytes
from toy_sweeper.voice.config import SAMPLE_RATE_HZ

ROOT = Path(__file__).resolve().parents[1]
TTS_SR = 48000
PEAK = 32767 * 10 ** (-3 / 20)
LEAD_S, GAP_S, TAIL_S = 0.5, 1.2, 1.5     # 先頭の無音、ウェイクワードと指示の間、末尾の無音（end_silence より長い）
# ASR 比較用の 3 通り（速さ、半音）。合成音声の単調さを少しでも崩すため。値に根拠はない（比較用の揺らぎ）
VARIANTS = [(1.0, 0.0), (0.85, 3.0), (1.15, -3.0)]


def synth(text: str, speed: float = 1.0, half_tone: float = 0.0) -> np.ndarray:
    x, sr = pyopenjtalk.tts(text, speed=speed, half_tone=half_tone)
    if sr != TTS_SR:
        raise RuntimeError(f"pyopenjtalk の出力が {TTS_SR} Hz でない: {sr}")
    y = resample_poly(x, SAMPLE_RATE_HZ, TTS_SR)
    return (y / np.abs(y).max() * PEAK).astype(np.int16)


def silence(s: float) -> np.ndarray:
    return np.zeros(int(s * SAMPLE_RATE_HZ), np.int16)


def write(path: Path, *parts: np.ndarray) -> None:
    path.write_bytes(wav_bytes(np.concatenate(parts), SAMPLE_RATE_HZ))
    print(f"{path}  {sum(len(p) for p in parts) / SAMPLE_RATE_HZ:.2f} s")


def make_fixtures(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    wake = synth("ロボット")
    write(out / "wake_then_red_lego.wav", silence(LEAD_S), wake, silence(GAP_S), synth("赤いレゴを右上に片付けて"),
          silence(TAIL_S))
    write(out / "wake_with_command.wav", silence(LEAD_S), synth("ロボット、赤いレゴを右上に片付けて"), silence(TAIL_S))
    write(out / "stop.wav", silence(LEAD_S), synth("止まって"), silence(TAIL_S))
    write(out / "wake_then_floor.wav", silence(LEAD_S), wake, silence(GAP_S), synth("床を拭いて"), silence(TAIL_S))
    write(out / "chatter.wav", silence(LEAD_S), synth("今日はいい天気ですね"), silence(TAIL_S))
    rng = np.random.default_rng(0)
    rms = 32768 * 10 ** (-40 / 20)
    write(out / "noise.wav", np.clip(rng.normal(0, rms, 3 * SAMPLE_RATE_HZ), -32768, 32767).astype(np.int16))


def make_asr_set(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, line in enumerate((ROOT / "data" / "vlm_eval" / "instructions.jsonl").read_text().splitlines()):
        item = json.loads(line)
        if not re.search(r"[぀-ヿ一-鿿]", item["text"]):
            continue                      # 英語の指示は音声では対象外（language: ja）
        for j, (speed, half_tone) in enumerate(VARIANTS):
            name = f"ins{i:02d}_v{j}.wav"
            write(out / name, silence(LEAD_S), synth(item["text"], speed, half_tone), silence(TAIL_S))
            rows.append({"wav": name, "text": item["text"], "expect": item["expect"], "speed": speed,
                         "half_tone": half_tone})
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"{len(rows)} 件 → {out / 'manifest.jsonl'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path)
    ap.add_argument("--asr-set", type=Path)
    a = ap.parse_args()
    if a.fixtures is None and a.asr_set is None:
        ap.error("--fixtures か --asr-set のどちらかが要る")
    if a.fixtures:
        make_fixtures(a.fixtures if a.fixtures.is_absolute() else ROOT / a.fixtures)
    if a.asr_set:
        make_asr_set(a.asr_set if a.asr_set.is_absolute() else ROOT / a.asr_set)


if __name__ == "__main__":
    main()
