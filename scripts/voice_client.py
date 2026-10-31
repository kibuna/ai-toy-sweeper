"""音声入力のクライアント（MacBook、CPU）。uv の環境で実行する。

    # マイクから（macOS の say で確認を話す）
    uv run python scripts/voice_client.py --server http://<3090 の PC>:8765 --mic --speaker say
    # WAV を順に流す（マイクの代わり。ファイルは続けて 1 本の入力として扱う）
    uv run python scripts/voice_client.py --server http://127.0.0.1:8765 --speaker print \
        --wav test/data/voice/wake_then_red_lego.wav test/data/voice/stop.wav

発話を VAD で区切り、各区間を小さい Whisper（CPU）で書き起こして停止語・ウェイクワードを探す。
ウェイクワードの次の発話（または同じ区間の続き）をサーバーに送り、検証済みの指示を受け取ったら確認を話して
on_instruction に渡す（今は表示するだけ。後でタスク管理・制御ループにつなぐ）。停止語は on_stop をすぐ呼ぶ。
ログは dev_log/voice/client_<日時>.log にも書く。--wav のときは最後に区間ごとの結果と時間の表を出す。
"""
import argparse
import dataclasses
import json
import logging
import time
from itertools import chain
from pathlib import Path

from toy_sweeper.voice.audio import Segmenter, SileroVadClassifier, frames_from_wav, mic_frames
from toy_sweeper.voice.client import ServerClient, VoiceClient
from toy_sweeper.voice.config import load_voice_cfg
from toy_sweeper.voice.speech import PrintSpeaker, SaySpeaker
from toy_sweeper.voice.wake import FasterWhisperTranscriber, TranscriptWakeDetector

ROOT = Path(__file__).resolve().parents[1]


def fmt(x):
    return "-" if x is None else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="サーバーの URL（例: http://192.168.0.10:8765）")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mic", action="store_true", help="マイクから入力する")
    src.add_argument("--wav", type=Path, nargs="+", help="マイクの代わりに流す WAV（16 kHz・モノラル・16 bit）")
    ap.add_argument("--speaker", choices=["say", "print"], required=True,
                    help="確認の話し方（say: macOS の say コマンド、print: 標準出力に書く）")
    ap.add_argument("--cpu-threads", type=int,
                    help="config の wake.cpu_threads の代わりに使うスレッド数（開発機で他の処理と並べるとき）")
    a = ap.parse_args()
    log_dir = ROOT / "dev_log" / "voice"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_dir / f"client_{time.strftime('%Y%m%d_%H%M%S')}.log")])
    log = logging.getLogger("voice_client")
    c = load_voice_cfg()
    if a.cpu_threads is not None:
        if a.cpu_threads <= 0:
            ap.error("--cpu-threads は正の値")
        log.info("wake.cpu_threads を %d から %d に上書き（--cpu-threads）", c.wake.cpu_threads, a.cpu_threads)
        c = dataclasses.replace(c, wake=dataclasses.replace(c.wake, cpu_threads=a.cpu_threads))
    speaker = SaySpeaker(c.client.say_voice, c.client.say_rate_wpm) if a.speaker == "say" else PrintSpeaker()
    t0 = time.perf_counter()
    wake = TranscriptWakeDetector(FasterWhisperTranscriber(c.wake, c.language), c.wake)
    log.info("ウェイクワード用の Whisper（%s、CPU）を読み込んだ（%.1f s）", c.wake.model, time.perf_counter() - t0)

    def on_instruction(ins):
        print("[指示] " + json.dumps(dataclasses.asdict(ins), ensure_ascii=False), flush=True)

    def on_stop():
        print("[停止]", flush=True)

    client = VoiceClient(c, wake, ServerClient(a.server, c.client.request_timeout_s), speaker, on_instruction, on_stop)
    segmenter = Segmenter(c.audio, c.vad, SileroVadClassifier(c.audio, c.vad.threshold))
    if a.mic:
        log.info("待ち受けを開始（ウェイクワード %s、停止語 %s）。Ctrl+C で終了", c.wake.wake_words, c.wake.stop_words)
        client.run(mic_frames(c.audio), segmenter)
        return
    wavs = [p if p.is_absolute() else Path.cwd() / p for p in a.wav]
    client.run(chain.from_iterable(frames_from_wav(p, c.audio) for p in wavs), segmenter)
    print("\n| 区間（s） | 長さ | 検出 | 手元の書き起こし | 手元 Whisper | 結果 | サーバーの書き起こし | 書き起こし（GPU / 往復） "
          "| VLM（GPU / 往復） | 話した文 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for t in client.turns:
        print(f"| {t.start_s:.2f} | {t.duration_s:.2f} | {t.detection} | {t.transcript_local} | {t.wake_s:.2f} | {t.outcome} "
              f"| {t.text} | {fmt(t.transcribe_s)} / {fmt(t.transcribe_rtt_s)} | {fmt(t.instruction_s)} / "
              f"{fmt(t.instruction_rtt_s)} | {' '.join(t.said)} |")


if __name__ == "__main__":
    main()
