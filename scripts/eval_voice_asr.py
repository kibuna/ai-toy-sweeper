"""サーバー側の Whisper の比較（RTX 3090、pixi の vlm 環境）。

    uv run python scripts/make_voice_fixtures.py --asr-set data/generated/voice_asr
    ~/.pixi/bin/pixi run -e vlm python scripts/eval_voice_asr.py --asr-set data/generated/voice_asr \
        --models openai/whisper-large-v3-turbo kotoba-tech/kotoba-whisper-v2.0 --out data/evaluations/voice_asr

合成音声の指示（data/vlm_eval/instructions.jsonl の日本語 19 件 × 話し方 3 通り）を書き起こし、
1. 文字誤り率（CER、正規化した文字列の編集距離 / 元の文の文字数）と、正規化後に元の文と一致した割合
2. 書き起こしを VLM と検証（/instruction と同じ処理）に通したときの正しさ（scripts/eval_vlm.py と同じ判定）
3. 書き起こしの時間
を比べる。基準として、元の文を VLM に通した場合の正しさも出す。WAV は前後の無音ごと渡す（クライアントの区間とほぼ同じ）。
"""
import argparse
import dataclasses
import json
import statistics
import subprocess
import time
from pathlib import Path

from toy_sweeper.config import load_all
from toy_sweeper.task_manager import load_vocabulary
from toy_sweeper.vlm import QwenVL
from toy_sweeper.vlm_config import load_vlm_cfg
from toy_sweeper.voice.audio import read_wav
from toy_sweeper.voice.config import load_voice_cfg
from toy_sweeper.voice.server import ASROutputError, WhisperASR, structure_instruction
from toy_sweeper.voice.wake import normalize

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("target_classes", "target_colors", "excluded_classes", "destination")


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def correct(expect: dict, r: dict) -> bool:
    if expect["needs_confirmation"]:
        return not r["ok"]                   # 止まれたか
    return r["ok"] and all(sorted(expect[k]) == sorted(r["instruction"][k]) if isinstance(expect[k], list)
                           else expect[k] == r["instruction"][k] for k in FIELDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-set", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    vcfg, cfg, vlm_cfg, vocab = load_voice_cfg(), load_all(), load_vlm_cfg(), load_vocabulary()
    src = a.asr_set if a.asr_set.is_absolute() else ROOT / a.asr_set
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in (src / "manifest.jsonl").read_text().splitlines()]
    vlm = QwenVL(vlm_cfg)
    ins_cache: dict[str, dict] = {}

    def structured(text):
        if text not in ins_cache:
            ins_cache[text] = structure_instruction(text, vlm, cfg, vocab, vlm_cfg.max_new_tokens_instruction)
        return ins_cache[text]

    base_ok = sum(correct(it["expect"], structured(it["text"])) for it in items)
    lines = [f"commit {subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()}"
             f" / VLM {vlm_cfg.model_id} / 合成音声 {len(items)} 件（{src.name}）", "",
             f"基準（元の文を VLM に通す）: 正しい {base_ok} / {len(items)}", "",
             "| モデル | CER | 正規化後に一致 | VLM まで通して正しい | 書き起こし（中央値 / 最大） |", "|---|---|---|---|---|"]
    for m in a.models:
        asr = WhisperASR(dataclasses.replace(vcfg.server, asr_model_id=m), vcfg.language, vcfg.audio.sample_rate_hz)
        asr(read_wav(src / items[0]["wav"], vcfg.audio.sample_rate_hz))       # 初回（読み込み・CUDA の準備）を計測から外す
        rows = []
        for it in items:
            audio = read_wav(src / it["wav"], vcfg.audio.sample_rate_hz)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                text = asr(audio)
            except ASROutputError as e:
                text = f"<ASROutputError: {e}>"
            dt = time.perf_counter() - t0
            ref, hyp = normalize(it["text"]), normalize(text)
            r = structured(text)
            rows.append({**it, "model": m, "transcript": text, "seconds": dt, "edits": edit_distance(ref, hyp),
                         "ref_chars": len(ref), "exact": ref == hyp, "result": r, "correct": correct(it["expect"], r)})
            print(f"{'OK' if rows[-1]['correct'] else 'NG'} {dt:.2f} s {it['text']} → {text}", flush=True)
        name = m.replace("/", "__")
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        cer = sum(r["edits"] for r in rows) / sum(r["ref_chars"] for r in rows)
        secs = [r["seconds"] for r in rows]
        lines.append(f"| {m} | {cer:.3f} | {sum(r['exact'] for r in rows)} / {len(rows)} | "
                     f"{sum(r['correct'] for r in rows)} / {len(rows)} | {statistics.median(secs):.2f} s / {max(secs):.2f} s |")
        del asr
        torch.cuda.empty_cache()
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
