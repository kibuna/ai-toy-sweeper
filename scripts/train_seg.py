"""学習する物体の検出器（YOLO のインスタンスセグメンテーション）を、Gazebo で作ったデータ（scripts/gen_seg_data.py）で
学習し、ONNX に書き出す。uv の依存グループ detector で動かす:

    uv run --group detector python scripts/train_seg.py --data data/generated/seg_gazebo_v1 --name seg_v1 --epochs 30 --batch 4

出力は checkpoints/<name>/（ultralytics の weights/best.pt と、推論用の best.onnx）。
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = ROOT / "checkpoints/pretrained/yolo26n-seg.pt"
# 学習の画像の大きさ（長辺）。LEGO（32 mm）が 1920 幅で約 30 px、1280 で約 20 px（未検証）
IMGSZ = 1280
# 推論の入力（高さ, 幅）: 1920×1080 を 2/3 にした 1280×720 を、32 の倍数の 736 に詰める
EXPORT_HW = (736, 1280)
log = logging.getLogger("train_seg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True, help="VRAM を見て決める（3090 は共用）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="学習の画像の使う割合（時間を測るため）")
    a = ap.parse_args()
    data = a.data if a.data.is_absolute() else ROOT / a.data
    out = ROOT / "checkpoints" / a.name
    if out.exists():
        raise FileExistsError(f"{out} がある（別の名前にする）")
    for split in ("train", "val"):
        n = len(list((data / "images" / split).glob("*.jpg")))
        if n == 0:
            raise FileNotFoundError(f"{data}/images/{split} に画像が無い")
    yml = data / "data.yaml"
    yml.write_text(f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n  0: lego\n  1: toy_car\n")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from ultralytics import YOLO
    t0 = time.time()
    model = YOLO(str(PRETRAINED))
    # 拡張は ultralytics の既定（上下反転なし flipud=0、左右反転 0.5、mosaic 1.0、HSV のゆらぎ）。損失の重みも既定。未検証
    model.train(data=str(yml), imgsz=IMGSZ, epochs=a.epochs, batch=a.batch, workers=a.workers, fraction=a.fraction,
                project=str(ROOT / "checkpoints"), name=a.name, exist_ok=False, seed=0, plots=True, flipud=0.0)
    log.info("学習: %.0f s", time.time() - t0)
    best = out / "weights" / "best.pt"
    onnx = YOLO(str(best)).export(format="onnx", imgsz=EXPORT_HW, opset=17, dynamic=False, simplify=False)
    shutil.move(onnx, out / "best.onnx")
    log.info("書き出し: %s", out / "best.onnx")


if __name__ == "__main__":
    main()
