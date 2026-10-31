"""VLM の評価。pixi の vlm 環境で実行する。

    ~/.pixi/bin/pixi run -e vlm python scripts/eval_vlm.py --boards data/generated/vlm_boards --out data/evaluations/vlm_qwen3vl2b

1. 指示の構造化: data/vlm_eval/instructions.jsonl の各指示を JSON にし、task_manager で検証（形式と、
   指示文との突き合わせ check_grounding）して正解と比べる。
   正解が needs_confirmation: true のものは、VLM が true を返すか、検証で弾かれれば「止まれた」とする。
2. 物体の分類: scripts/render_vlm_dataset.py の盤面画像から物体を切り出し、種類と色を正解と比べる（合成画像）。
"""
import argparse
import json
import statistics
import subprocess
from pathlib import Path

from PIL import Image

from toy_sweeper.config import load_all
from toy_sweeper.task_manager import InstructionError, check_grounding, load_vocabulary, parse_instruction
from toy_sweeper.vlm import QwenVL, VLMOutputError, extract_json, instruction_prompt, object_prompt
from toy_sweeper.vlm_config import load_vlm_cfg

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("target_classes", "target_colors", "excluded_classes", "destination")


def eval_instructions(vlm, cfg, vcfg, path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        item = json.loads(line)
        exp = item["expect"]
        r = vlm.ask(instruction_prompt(item["text"], cfg), vcfg.max_new_tokens_instruction)
        row = {"text": item["text"], "expect": exp, "raw": r.text, "seconds": r.seconds, "tokens": r.n_tokens}
        try:
            d = extract_json(r.text)
            row["json"] = d
            ins = parse_instruction(d, cfg)
            check_grounding(item["text"], ins, load_vocabulary())
            row["accepted"] = True
            got = {"target_classes": sorted(ins.target_classes), "target_colors": sorted(ins.target_colors),
                   "excluded_classes": sorted(ins.excluded_classes), "destination": ins.destination}
            row["got"] = got
        except VLMOutputError as e:
            row["accepted"], row["error"] = False, f"VLMOutputError: {e}"
        except InstructionError as e:
            row["accepted"], row["error"] = False, f"InstructionError: {e}"
        if exp["needs_confirmation"]:
            row["correct"] = not row["accepted"]          # 止まれたか
        else:
            row["correct"] = row["accepted"] and all(
                sorted(exp[k]) == row["got"][k] if isinstance(exp[k], list) else exp[k] == row["got"][k] for k in FIELDS)
        rows.append(row)
        print(("OK " if row["correct"] else "NG ") + item["text"] + f"  ({r.seconds:.1f} s)", flush=True)
    return rows


def eval_objects(vlm, vcfg, boards: Path) -> list[dict]:
    rows = []
    labels = json.loads((boards / "labels.json").read_text())
    prompt = object_prompt()
    for b in labels:
        img = Image.open(boards / b["image"]).convert("RGB")
        for o in b["objects"]:
            x0, y0, x1, y1 = o["bbox"]
            m = vcfg.crop_margin_px
            crop = img.crop((max(0, x0 - m), max(0, y0 - m), min(img.width, x1 + m), min(img.height, y1 + m)))
            crop = crop.resize((vcfg.crop_size_px, vcfg.crop_size_px), Image.LANCZOS)
            r = vlm.ask(prompt, vcfg.max_new_tokens_object, crop)
            row = {"image": b["image"], "name": o["name"], "class": o["class"], "color": o["color"], "raw": r.text,
                   "seconds": r.seconds}
            try:
                d = extract_json(r.text)
                row["got_class"], row["got_color"] = d.get("class"), d.get("color")
            except VLMOutputError as e:
                row["error"] = str(e)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", help="config/vlm.yaml の model_id の代わりに使うモデル（比較用）")
    a = ap.parse_args()
    import dataclasses
    cfg, vcfg = load_all(), load_vlm_cfg()
    if a.model:
        vcfg = dataclasses.replace(vcfg, model_id=a.model)
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    vlm = QwenVL(vcfg)
    ins = eval_instructions(vlm, cfg, vcfg, ROOT / "data" / "vlm_eval" / "instructions.jsonl")
    objs = eval_objects(vlm, vcfg, a.boards if a.boards.is_absolute() else ROOT / a.boards)
    (out / "instructions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ins) + "\n")
    (out / "objects.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in objs) + "\n")

    normal = [r for r in ins if not r["expect"]["needs_confirmation"]]
    stop = [r for r in ins if r["expect"]["needs_confirmation"]]
    n_obj = len(objs)
    cls = sum(r.get("got_class") == r["class"] for r in objs)
    col = sum(r.get("got_color") == r["color"] for r in objs)
    both = sum(r.get("got_class") == r["class"] and r.get("got_color") == r["color"] for r in objs)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    lines = [f"model: {vcfg.model_id} / commit {commit}", "",
             "| 項目 | 結果 |", "|---|---|",
             f"| 指示（実行すべき {len(normal)} 件）: 検証を通った | {sum(r['accepted'] for r in normal)} / {len(normal)} |",
             f"| 指示（実行すべき {len(normal)} 件）: 中身まで正しい | {sum(r['correct'] for r in normal)} / {len(normal)} |",
             f"| 指示（止まるべき {len(stop)} 件）: 止まれた | {sum(r['correct'] for r in stop)} / {len(stop)} |",
             f"| 指示 1 件の推論時間（中央値） | {statistics.median(r['seconds'] for r in ins):.1f} s |",
             f"| 物体（合成画像 {n_obj} 個）: 種類 | {cls} / {n_obj} |",
             f"| 物体: 色 | {col} / {n_obj} |",
             f"| 物体: 種類と色の両方 | {both} / {n_obj} |",
             f"| 物体 1 個の推論時間（中央値） | {statistics.median(r['seconds'] for r in objs):.2f} s |"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
