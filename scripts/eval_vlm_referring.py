"""「右の赤いレゴ」のような 1 個を指す指示を、2 つの方式で比べる。pixi の vlm 環境で実行する。

    ~/.pixi/bin/pixi run -e vlm python scripts/eval_vlm_referring.py --boards data/generated/vlm_boards --n 60 \
        --out data/evaluations/vlm_referring

- text: 盤面の物体の一覧（id、色、種類、床面の座標）を文字で渡す（LLM として使う）。
- image: 各物体に番号を描き込んだ画像を渡す（VLM、Set-of-Mark）。
- query: 指示を「種類・色・選び方」に構造化させ（画像も一覧も渡さない）、1 個を選ぶのは
  task_manager.resolve_selector の幾何計算に任せる。決められないときは聞き返す（不正解として数え、別に件数も出す）。
問題は scripts/render_vlm_dataset.py が作った referring.jsonl（答えが一意のものだけ）。
"""
import argparse
import json
import statistics
from pathlib import Path

from PIL import Image

from toy_sweeper.config import CONFIG_DIR
from toy_sweeper.task import COLORS
from toy_sweeper.task_manager import InstructionError, resolve_selector
from toy_sweeper.vlm import QwenVL, VLMOutputError, extract_json
from toy_sweeper.vlm_config import load_vlm_cfg

ROOT = Path(__file__).resolve().parents[1]
JA_COLOR = {"red": "赤", "blue": "青", "yellow": "黄", "green": "緑", "white": "白", "black": "黒", "lime": "黄緑",
            "light_blue": "水色", "pink": "ピンク", "orange": "オレンジ"}


def object_list(objs) -> str:
    return "\n".join(f'  id={o["id"]}: {JA_COLOR[o["color"]]}の{o["class"]} (x={o["x"]:+.2f}, y={o["y"]:+.2f})' for o in objs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True, help="問題数（側と隣を半々）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modes", nargs="+", default=["text", "image", "query"], choices=["text", "image", "query"])
    a = ap.parse_args()
    boards = a.boards if a.boards.is_absolute() else ROOT / a.boards
    labels = {b["image"]: b for b in json.loads((boards / "labels.json").read_text())}
    qs = [json.loads(l) for l in (boards / "referring.jsonl").read_text().splitlines()]
    pick = [q for q in qs if q["kind"] == "side"][: a.n // 2] + [q for q in qs if q["kind"] == "near"][: a.n - a.n // 2]
    if len(pick) != a.n:
        raise ValueError(f"問題が足りない: {len(pick)} < {a.n}")
    vcfg = load_vlm_cfg()
    vlm = QwenVL(vcfg)
    t_text = (CONFIG_DIR / "prompts" / "referring_text.txt").read_text()
    t_img = (CONFIG_DIR / "prompts" / "referring_image.txt").read_text()
    t_query = (CONFIG_DIR / "prompts" / "referring_query.txt").read_text().replace(
        "{colors}", ", ".join(f'"{c}"' for c in COLORS))
    rows = []
    for q in pick:
        b = labels[q["image"]]
        for mode in a.modes:
            if mode == "query":
                r = vlm.ask(t_query.replace("{instruction}", q["text"]), vcfg.max_new_tokens_object)
                stopped = False
                try:
                    got = [resolve_selector(extract_json(r.text), b["objects"])]
                except (VLMOutputError, InstructionError, KeyError, TypeError):
                    got, stopped = None, True
                rows.append({**q, "mode": mode, "got": got, "correct": got == q["answer"], "stopped": stopped,
                             "seconds": r.seconds, "raw": r.text})
                continue
            if mode == "text":
                prompt = t_text.replace("{objects}", object_list(b["objects"])).replace("{instruction}", q["text"])
                r = vlm.ask(prompt, vcfg.max_new_tokens_object)
            else:
                img = Image.open(boards / b["marked_image"]).convert("RGB")
                r = vlm.ask(t_img.replace("{instruction}", q["text"]), vcfg.max_new_tokens_object, img)
            try:
                got = extract_json(r.text).get("target_ids")
            except VLMOutputError:
                got = None
            rows.append({**q, "mode": mode, "got": got, "correct": got == q["answer"], "seconds": r.seconds, "raw": r.text})
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    lines = [f"model: {vcfg.model_id}", "", "| 方式 | 位置で指す（右・左・奥・手前） | 隣で指す | 1 問の時間（中央値） |", "|---|---:|---:|---:|"]
    for mode, label in (("text", "盤面の一覧を文字で（LLM）"), ("image", "番号入りの画像で（VLM）"),
                        ("query", "条件を構造化して幾何計算で選ぶ")):
        rs = [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        side = [r for r in rs if r["kind"] == "side"]
        near = [r for r in rs if r["kind"] == "near"]
        lines.append(f"| {label} | {sum(r['correct'] for r in side)} / {len(side)} | {sum(r['correct'] for r in near)} / {len(near)} "
                     f"| {statistics.median(r['seconds'] for r in rs):.1f} s |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
