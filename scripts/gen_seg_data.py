"""学習する検出器のデータを Gazebo で作る。pixi 環境で動かす。

Gazebo を 1 回起動するごとに条件（カメラ・照明・床、seg_data.sample_conditions）と物体の組（Stage 6〜8 の盤面の物体）を
引き、物体を置き直して（seg_data.sample_layout）RGB の画像と物体ごとのマスクを撮る。YOLO の形式で書く:

    <out>/images/<split>/w<世界>_<枚>.jpg
    <out>/labels/<split>/w<世界>_<枚>.txt     クラス（0 LEGO、1 ミニカー）と輪郭（画像の幅・高さで割った座標）
    <out>/meta/<split>.jsonl                  1 枚ごとの条件・ロボット・物体の真値（床の位置の誤差を測るため）
    <out>/conditions/<split>_w<世界>.yaml     その世界の条件（config/experiments の形）
    <out>/backgrounds/<split>_w<世界>.jpg     その世界の何も置いていない床（背景との差の検出と比べるため）

--backgrounds-only は、前に作った世界の条件と物体の組で、背景だけを撮り直す。

    ~/.pixi/bin/pixi run python scripts/gen_seg_data.py --out data/generated/seg_gazebo --split train --worlds 0 100 --frames 100
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from sensor_msgs.msg import Image

from toy_sweeper.config import load_all
from toy_sweeper.geometry import Pose2D
from toy_sweeper.gazebo.backend import GazeboBackend, _ns
from toy_sweeper.gazebo.camera import OverheadCamera
from toy_sweeper.gazebo.launcher import running_simulation
from toy_sweeper.gazebo.sdf_gen import SEG_TOPIC, board_models
from toy_sweeper.spaces import GripperState
from toy_sweeper.seg_data import BASE_OVERRIDES, instances_from_labels, sample_conditions, sample_layout
from toy_sweeper.task import TaskSampler

ROOT = Path(__file__).resolve().parents[1]
# 分け方: 盤面の Seed の始めと、条件の乱数の Seed。リハーサルの盤面（20000〜）と重ねない
SPLITS = {"train": (100000, 1), "val": (200000, 2)}
STAGES = (6, 7, 8)          # 物体の組を取る Stage（物体の多い盤面。Stage 8 は 27 個まで）
MIN_PIXELS = 20             # これより少ない画素の物体（ほぼ隠れた物）はラベルにしない（LEGO の全体は 300〜1000 画素。未検証）
JPEG_QUALITY = 95
log = logging.getLogger("gen_seg_data")


class SegCamera:
    """物体ごとのマスクのカメラ（labels_map）を受け取る。"""

    def __init__(self, b: GazeboBackend):
        self.latest = None
        b.node.create_subscription(Image, SEG_TOPIC + "/labels_map", self._cb, 2)

    def _cb(self, m: Image) -> None:
        if m.encoding != "rgb8":
            raise ValueError(f"想定外の labels_map の形式: {m.encoding}")
        self.latest = (_ns(m.header.stamp), np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3).copy())


def capture_background(b: GazeboBackend, cam: OverheadCamera, path: Path) -> None:
    """ロボットと物体を視野の外に出した床（field.rehearsal.GazeboRehearsalRig.calibrate と同じ）。"""
    b.reset(Pose2D(0.0, -3.0, 0.0), [], GripperState.OPEN)
    cv2.imwrite(str(path), cam.frame()[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def yolo_line(cls: int, poly: np.ndarray, w: int, h: int) -> str:
    xy = (poly.astype(float) + 0.5) / (w, h)
    return f"{cls} " + " ".join(f"{v:.5f}" for v in xy.ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--worlds", type=int, nargs=2, required=True, metavar=("FROM", "TO"), help="世界の番号の範囲 [FROM, TO)")
    ap.add_argument("--frames", type=int, required=True, help="1 つの世界で撮る枚数")
    ap.add_argument("--backgrounds-only", action="store_true", help="前に作った世界の背景だけを撮る")
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    for d in ("images", "labels"):
        (out / d / a.split).mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    (out / "conditions").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "backgrounds").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(out / "logs" / f"{a.split}_{a.worlds[0]}_{a.worlds[1]}.log"),
                                  logging.StreamHandler()])
    board_seed0, cond_seed = SPLITS[a.split]
    for k in range(*a.worlds):
        bg_path = out / "backgrounds" / f"{a.split}_w{k:04d}.jpg"
        cpath = out / "conditions" / f"{a.split}_w{k:04d}.yaml"
        if a.backgrounds_only:
            if bg_path.exists():
                continue
            if not cpath.exists():
                raise FileNotFoundError(f"世界 {k} の条件が無い: {cpath}")
        elif (out / "labels" / a.split / f"w{k:04d}_{a.frames - 1:03d}.txt").exists():
            log.info("世界 %d は済み", k)
            continue
        t0 = time.time()
        rng = np.random.default_rng([cond_seed, k])
        cond = sample_conditions(rng)
        if a.backgrounds_only:
            if yaml.safe_load(cpath.read_text()) != cond:
                raise RuntimeError(f"世界 {k} の条件が前と違う（seg_data.sample_conditions が変わった）")
        else:
            cpath.write_text(yaml.safe_dump(cond, sort_keys=False))
        overrides = BASE_OVERRIDES + (cpath,)
        cfg = load_all(overrides=overrides)
        stage = STAGES[k % len(STAGES)]
        board = TaskSampler(cfg.sim, cfg.task).sample(stage, np.random.default_rng(board_seed0 + k))
        n_inst = 0
        with running_simulation(out / "logs" / f"gazebo_{a.split}_w{k:04d}.log", camera=True, overrides=overrides,
                                board=list(board.objects), markers=True):
            b = GazeboBackend(cfg, models=board_models(board.objects))
            try:
                cam, seg = OverheadCamera(b), SegCamera(b)
                capture_background(b, cam, bg_path)
                if a.backgrounds_only:
                    log.info("世界 %d の背景", k)
                    continue
                with open(out / "meta" / f"{a.split}.jsonl", "a") as meta:
                    for j in range(a.frames):
                        lay = sample_layout(cfg, list(board.objects), rng)
                        b.reset(lay.robot, lay.objects, lay.gripper)
                        img = cam.frame()
                        t = cam.latest[0]
                        b._wait(lambda: seg.latest is not None and seg.latest[0] >= t, "物体ごとのマスク")
                        if seg.latest[0] != t:
                            raise RuntimeError(f"RGB とマスクの時刻が違う: {t} と {seg.latest[0]}")
                        h, w = img.shape[:2]
                        inst = instances_from_labels(seg.latest[1], MIN_PIXELS)
                        n_inst += len(inst)
                        name = f"w{k:04d}_{j:03d}"
                        cv2.imwrite(str(out / "images" / a.split / f"{name}.jpg"), img[..., ::-1],
                                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        (out / "labels" / a.split / f"{name}.txt").write_text(
                            "".join(yolo_line(c, p, w, h) + "\n" for c, p, _ in inst))
                        meta.write(json.dumps({
                            "image": name, "world": k, "stage": stage, "board_seed": board_seed0 + k,
                            "camera_pose": cond["sim"]["gazebo"]["camera_pose"],
                            "camera_hfov_rad": cond["sim"]["gazebo"]["camera_hfov_rad"],
                            "robot": [lay.robot.x, lay.robot.y, lay.robot.yaw], "gripper": lay.gripper.name,
                            "held": lay.held, "touching": lay.touching,
                            "objects": [{"name": o.name, "type": o.type.name, "color": o.color,
                                         "pose": [*b.object_pose(o.name).xy, b.object_pose(o.name).yaw]} for o in lay.objects],
                            "n_instances": len(inst)}) + "\n")
            finally:
                b.close()
        log.info("世界 %d（Stage %d、物体 %d 個）: %d 枚、物体のラベル %d、%.0f s", k, stage, len(board.objects), a.frames,
                 n_inst, time.time() - t0)


if __name__ == "__main__":
    main()
