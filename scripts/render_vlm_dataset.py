"""VLM の物体分類の評価用に、Stage 6 の盤面画像（文字・目印なし）と正解を作る。uv 環境で実行する。

    uv run python scripts/render_vlm_dataset.py --seeds 30000 30019 --out data/generated/vlm_boards

出力:
- <out>/board_<seed>.png（盤面）、<out>/board_<seed>_marked.png（各物体に ID 番号を描き込んだもの、Set-of-Mark 用）
- <out>/labels.json（物体ごとの ID・種類・色・床面の位置・画素の外接矩形）
- <out>/referring.jsonl（1 個を指す指示と正解の ID。答えが一意に決まるものだけ）
合成画像なので実カメラの精度の代わりにはならない（実カメラの評価は実画像 20〜50 枚で行う）。
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from toy_sweeper.config import load_all
from toy_sweeper.render import draw_scene
from toy_sweeper.spaces import OBJECT_TYPE_NAMES
from toy_sweeper.task import TaskSampler

ROOT = Path(__file__).resolve().parents[1]


JA_COLOR = {"red": "赤い", "blue": "青い", "yellow": "黄色い", "green": "緑の", "white": "白い", "black": "黒い",
            "lime": "黄緑の", "light_blue": "水色の", "pink": "ピンクの", "orange": "オレンジの"}
JA_CLASS = {"lego": "レゴ", "toy_car": "ミニカー"}
# 位置の言葉と、その向きの座標（右 = +x、奥 = +y。カメラ画像の右・上）
SIDES = {"右": (0, 1), "左": (0, -1), "奥": (1, 1), "手前": (1, -1)}
SIDE_MARGIN_M = 0.10      # いちばん端のものが 2 番目より 10 cm 以上端にあるときだけ使う（曖昧さを避ける）
NEAR_MARGIN_M = 0.05      # いちばん近いものが 2 番目より 5 cm 以上近いときだけ使う


def mark_ids(img, objs):
    """各物体の外接矩形の右上に ID 番号を描く（Set-of-Mark）。"""
    out = img.copy()
    for o in objs:
        x0, y0, x1, y1 = o["bbox"]
        org = (min(x1 + 2, out.shape[1] - 22), max(y0 - 2, 14))
        cv2.rectangle(out, (org[0] - 2, org[1] - 13), (org[0] + 18, org[1] + 3), (255, 255, 255), -1)
        cv2.putText(out, str(o["id"]), org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def referring_questions(image, objs):
    """答えが一意に決まる「1 個を指す指示」を作る。"""
    qs = []
    groups = {}
    for o in objs:
        groups.setdefault((o["color"], o["class"]), []).append(o)
    for (color, cls), g in groups.items():
        if len(g) < 2:
            continue
        for word, (axis, sign) in SIDES.items():
            key = (lambda o: sign * (o["x"] if axis == 0 else o["y"]))
            s = sorted(g, key=key, reverse=True)
            if key(s[0]) - key(s[1]) >= SIDE_MARGIN_M:
                qs.append({"image": image, "kind": "side",
                           "text": f"{word}にある{JA_COLOR[color]}{JA_CLASS[cls]}を片付けて", "answer": [s[0]["id"]]})
    for (color, cls), g in groups.items():
        if len(g) != 1:
            continue
        ref = g[0]
        for tcls in ("lego", "toy_car"):
            cands = sorted((o for o in objs if o["id"] != ref["id"] and o["class"] == tcls),
                           key=lambda o: (o["x"] - ref["x"]) ** 2 + (o["y"] - ref["y"]) ** 2)
            if len(cands) >= 2:
                d0 = ((cands[0]["x"] - ref["x"]) ** 2 + (cands[0]["y"] - ref["y"]) ** 2) ** 0.5
                d1 = ((cands[1]["x"] - ref["x"]) ** 2 + (cands[1]["y"] - ref["y"]) ** 2) ** 0.5
                if d1 - d0 >= NEAR_MARGIN_M:
                    qs.append({"image": image, "kind": "near",
                               "text": f"{JA_COLOR[color]}{JA_CLASS[cls]}のいちばん近くにある{JA_CLASS[tcls]}を片付けて",
                               "answer": [cands[0]["id"]]})
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, required=True, help="最初と最後の Seed（両端含む）")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = load_all()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    sampler = TaskSampler(cfg.sim, cfg.task)
    labels, questions = [], []
    for seed in range(a.seeds[0], a.seeds[1] + 1):
        ep = sampler.sample(6, np.random.default_rng(seed))
        img, boxes = draw_scene(cfg, ep.robot, ep.objects, {o.name: o.pose for o in ep.objects})
        name = f"board_{seed}.png"
        cv2.imwrite(str(out / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        objs = [{"id": int(o.name[3:]), "name": o.name, "class": OBJECT_TYPE_NAMES[o.type], "color": o.color,
                 "x": round(o.pose.x, 3), "y": round(o.pose.y, 3), "bbox": boxes[o.name]} for o in ep.objects]
        cv2.imwrite(str(out / f"board_{seed}_marked.png"), cv2.cvtColor(mark_ids(img, objs), cv2.COLOR_RGB2BGR))
        labels.append({"image": name, "marked_image": f"board_{seed}_marked.png", "objects": objs})
        questions += referring_questions(name, objs)
    (out / "labels.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False))
    (out / "referring.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n")
    print(f"{len(labels)} boards, {sum(len(b['objects']) for b in labels)} objects, {len(questions)} questions -> {out}")


if __name__ == "__main__":
    main()
