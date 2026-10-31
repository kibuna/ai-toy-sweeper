"""物体の検出を、Gazebo で作った検証の画像（scripts/gen_seg_data.py）の真値と突き合わせて比べる。

検出器は 2 つ: 背景との差（BackgroundDetector、実機の決まり: close_px・color_split・探す範囲）と、学習した検出器
（SegDetector、--model の ONNX）。どちらも同じ探す範囲・ロボットの範囲で、床の中心を真値と 2 cm 以内で 1 対 1 に対応付ける。

- 見つけた割合: 見えるはずの物体（中心が画像の中、ロボットの陰でない、作業領域 + 外の余白の中）のうち、対応が付いたもの
- くっつきを分けた割合: 接して置いた 2 個（meta の touching）が両方見えるとき、両方に対応が付いたもの
- 余りの検出: 対応の付かない検出（2 個がくっついた塊の中心など）
- 床の中心の誤差（mm）

    uv run python scripts/eval_seg_detector.py --data data/generated/seg_gazebo_v1 --model checkpoints/seg_v1/best.onnx \\
        --out data/evaluations/seg_v1_offline
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from toy_sweeper.calibration import camera_from_gazebo_pose
from toy_sweeper.config import load_all
from toy_sweeper.field.config import load_field_cfg
from toy_sweeper.field.real_rig import ROI_OBJECT_MARGIN_M
from toy_sweeper.field.rehearsal import load_background_cfg
from toy_sweeper.geometry import Pose2D
from toy_sweeper.perception_env import robot_polygons
from toy_sweeper.perception_oblique import BackgroundDetector, BlobTooLarge
from toy_sweeper.seg_data import BASE_OVERRIDES
from toy_sweeper.seg_detector import SegDetector, SegModel, load_seg_detector_cfg

ROOT = Path(__file__).resolve().parents[1]
MATCH_M = 0.02      # 検出と真値を対応付ける距離の上限（field の association_tolerance_m と同じ考え。未検証）
log = logging.getLogger("eval_seg_detector")


def match(dets_xy: np.ndarray, truth_xy: np.ndarray, max_m: float) -> dict[int, int]:
    """近い順に 1 対 1 で対応付ける（真値の番号 → 検出の番号）。"""
    if len(dets_xy) == 0 or len(truth_xy) == 0:
        return {}
    d = np.linalg.norm(truth_xy[:, None] - dets_xy[None], axis=2)
    out, used = {}, set()
    for i, j in zip(*np.unravel_index(np.argsort(d, axis=None), d.shape)):
        if d[i, j] > max_m:
            break
        if i in out or j in used:
            continue
        out[int(i)], _ = int(j), used.add(int(j))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="画像の数の上限（0 は全部）")
    a = ap.parse_args()
    data = a.data if a.data.is_absolute() else ROOT / a.data
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    metas = [json.loads(l) for l in (data / "meta" / f"{a.split}.jsonl").read_text().splitlines()]
    if a.limit:
        metas = metas[:a.limit]
    cfg = load_all(overrides=BASE_OVERRIDES)
    f = load_field_cfg(cfg)
    r = f.real
    bcfg = dataclasses.replace(load_background_cfg(), close_px=r.close_px, color_split=r.color_split,
                               merged_max_length_m=r.merged_max_length_m, track_chroma_gate=r.track_chroma_gate)
    hx, hy = (v / 2 for v in cfg.sim.work_area.size_m)
    outm = cfg.sim.walls.out_of_area_margin_m
    keep = (hx + outm, hy + outm)
    roi = (hx + outm + ROI_OBJECT_MARGIN_M, hy + outm + ROI_OBJECT_MARGIN_M)
    g = cfg.sim.gripper
    polys = lambda r_, j: robot_polygons(cfg, r_, j)
    seg_model = SegModel(a.model if a.model.is_absolute() else ROOT / a.model, load_seg_detector_cfg())
    heights = {k.upper(): o.size_m[2] for k, o in cfg.sim.objects.items()}
    stats = {n: defaultdict(list) for n in ("background", "learned")}
    world_dets: dict = {}
    for m in metas:
        w = m["world"]
        if w not in world_dets:
            cam = camera_from_gazebo_pose(m["camera_pose"], m["camera_hfov_rad"], (1920, 1080))
            bg = cv2.imread(str(data / "backgrounds" / f"{a.split}_w{w:04d}.jpg"))[..., ::-1].copy()
            args = (cam, bg, bcfg, polys, g.jaw_bottom_z_m + g.jaw_height_m)
            kw = dict(roi_half_m=roi, keep_half_m=keep, drop_large=True)
            world_dets = {w: {"background": BackgroundDetector(*args, **kw), "learned": SegDetector(seg_model, *args, **kw)}}
        img = cv2.imread(str(data / "images" / a.split / f"{m['image']}.jpg"))[..., ::-1].copy()
        robot = Pose2D(*m["robot"])
        jaw = g.jaw_closed_angle_rad if m["gripper"] == "CLOSED" else g.jaw_open_angle_rad
        objs = m["objects"]
        txy = np.array([o["pose"][:2] for o in objs])
        for name, det in world_dets[w].items():
            rmask = det.mask(img.shape[:2], robot, jaw)
            visible = []
            for o in objs:
                x, y = o["pose"][:2]
                u, v = det.cam.to_px(np.array([[x, y]]), heights[o["type"]] / 2)[0]
                visible.append(abs(x) <= keep[0] and abs(y) <= keep[1] and 0 <= u < 1920 and 0 <= v < 1080
                               and not det.hidden_by_robot(rmask, (x, y)))
            t0 = time.perf_counter()
            try:
                dets = det.detect(img, rmask)
            except BlobTooLarge:
                dets = []
                stats[name]["too_large"].append(1)
            stats[name]["ms"].append(1000 * (time.perf_counter() - t0))
            mt = match(np.array([d.pose.xy for d in dets]).reshape(-1, 2), txy, MATCH_M)
            s = stats[name]
            for i, o in enumerate(objs):
                if not visible[i]:
                    continue
                key = "held" if o["name"] == m["held"] else o["type"]
                s[f"found_{key}"].append(i in mt)
                if i in mt:
                    s[f"err_mm_{o['type']}"].append(1000 * math.dist(dets[mt[i]].pose.xy, o["pose"][:2]))
            idx = {o["name"]: i for i, o in enumerate(objs)}
            for p, q in m["touching"]:
                if visible[idx[p]] and visible[idx[q]]:
                    s["touching_split"].append(idx[p] in mt and idx[q] in mt)
            s["extra"].append(len(dets) - len(mt))
    summary = {"images": len(metas)}
    for name, s in stats.items():
        row = {}
        for k, v in sorted(s.items()):
            v = np.asarray(v, float)
            if k.startswith("found_") or k == "touching_split":
                row[k] = {"rate": round(float(v.mean()), 4), "n": len(v)}
            elif k.startswith("err_mm_"):
                row[k] = {"mean": round(float(v.mean()), 2), "p95": round(float(np.percentile(v, 95)), 2), "n": len(v)}
            elif k == "extra":
                row["extra_per_image"] = round(float(v.mean()), 3)
            elif k == "ms":
                row["ms_mean"] = round(float(v.mean()), 1)
            else:
                row[k] = int(v.sum())
        summary[name] = row
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    log.info("%s", json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
