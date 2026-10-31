"""Gazebo のリハーサル（field.rehearsal）の成功率をまとめて測る。pixi の環境で実行する。

    ~/.pixi/bin/pixi run python scripts/eval_rehearsal.py --model checkpoints/<run>/final.zip --stage 6 --seed 20000 \\
        --boards 10 --perception camera --identify truth --out data/evaluations/rehearsal_<名前>

盤面ごとに Gazebo を起動し（盤面の物体のモデルだけ）、校正し、指示を 1 つ実行する。指示は盤面の --target-type（既定 lego。
ミニカーの Stage 4 は toy_car）の色から 1 つ、
行き先は名前付きのゴールから選ぶ（Seed から決まる）。白の物体は出さない（背景との差で見えない）。
成功は「ランタイムが完了と判断し、かつ真値で全対象がゴールの中」。完了と判断したが真値で外なら「wrong_done」。
出力: --out/results.jsonl（盤面ごと）、summary.json、run.log。--video なら --out/videos/board_<Seed>.mp4（カメラの画像に追跡の
結果・ゴール・状態を重ねたもの、field.record.draw_overlay。1 観測 = 1 フレーム、等速）。
"""
import argparse
import asyncio
import dataclasses
import json
import logging
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from toy_sweeper.config import load_all
from toy_sweeper.field.config import load_field_cfg
from toy_sweeper.field.interfaces import FieldStop
from toy_sweeper.field.record import draw_overlay
from toy_sweeper.field.rehearsal import GazeboRehearsalRig
from toy_sweeper.field.rig import find_board
from toy_sweeper.field.runtime import given_instruction
from toy_sweeper.gazebo.backend import GazeboBackend
from toy_sweeper.gazebo.camera import OverheadCamera
from toy_sweeper.gazebo.launcher import running_simulation
from toy_sweeper.gazebo.sdf_gen import board_models
from toy_sweeper.perception import load_perception_cfg
from toy_sweeper.policies import PPOPolicy
from toy_sweeper.spaces import ObjectType
from toy_sweeper.task import goal_rects
from toy_sweeper.task_manager import TidyInstruction, select_targets

ROOT = Path(__file__).resolve().parents[1]
OVERRIDES = (ROOT / "config/experiments/gazebo_oblique.yaml", ROOT / "config/experiments/no_chassis_penalty.yaml")
log = logging.getLogger("eval_rehearsal")
VIDEO_SIZE = (1280, 720)    # 動画の大きさ（カメラの 1920×1080 の 2/3。見て分かる大きさで、ファイルを小さく）
RESULT_HOLD_S = 2.0         # 動画の最後に結果を見せる秒数
TYPES = {"lego": ObjectType.LEGO, "toy_car": ObjectType.TOY_CAR}


def instruction_for(board, cfg, rng, target_type: str) -> TidyInstruction | None:
    """盤面の target_type（lego / toy_car）の色から 1 つ、行き先は名前付きのゴールから。対象がゴールの外に 1 個以上あるもの。"""
    colors = sorted({o.color for o in board.objects if o.type == TYPES[target_type]})
    if not colors:
        return None
    dests = sorted(goal_rects(cfg.sim))
    poses = {o.name: o.pose for o in board.objects}
    for _ in range(50):
        ins = TidyInstruction((target_type,), (colors[int(rng.integers(len(colors)))],), (), dests[int(rng.integers(len(dests)))])
        if select_targets(list(board.objects), poses, ins, board.robot, cfg):
            return ins
    return None


class BoardVideo:
    """1 盤面の動画（--video）。カメラの画像に追跡の結果・ゴール・状態（draw_overlay）と盤面・指示の行を重ねて、
    VIDEO_SIZE に縮めて書く。終わったら結果の行を付けた最後のフレームを RESULT_HOLD_S 秒続け、ブラウザで再生できる
    H.264 に変換する（OpenCV の mp4v はブラウザで再生できない）。"""

    def __init__(self, path: Path, fps: float, rig, header: str):
        self.path, self.fps, self.rig, self.header = path, fps, rig, header
        self.raw = path.with_suffix(".raw.mp4")
        self.writer = None
        self.last = None

    def _write(self, img: np.ndarray) -> None:
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(self.raw), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, VIDEO_SIZE)
            if not self.writer.isOpened():
                raise RuntimeError(f"VideoWriter を開けない: {self.raw}")
        self.writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def __call__(self, rt, ts) -> None:
        img = cv2.resize(draw_overlay(ts.frame.image, rt, self.rig.hom, self.rig.goal_rects()), VIDEO_SIZE,
                         interpolation=cv2.INTER_AREA)
        cv2.putText(img, self.header, (8, VIDEO_SIZE[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        self.last = img
        self._write(img)

    def close(self, result: str) -> None:
        if self.last is None:
            raise RuntimeError(f"動画のフレームが無い: {self.path}")
        img = self.last.copy()
        cv2.putText(img, result, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (20, 140, 40) if result.startswith("success") else (200, 30, 30), 3, cv2.LINE_AA)
        for _ in range(round(RESULT_HOLD_S * self.fps)):
            self._write(img)
        self.writer.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", str(self.path)], check=True)
        self.raw.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--seed", type=int, help="盤面の Seed（探し始め）。--boards と組で使う")
    ap.add_argument("--boards", type=int)
    ap.add_argument("--board-seeds", type=int, nargs="+",
                    help="この Seed の盤面だけ（前の評価の失敗をやり直すため）。どれかが盤面にならなければ止める")
    ap.add_argument("--perception", choices=["camera", "truth"], required=True)
    ap.add_argument("--identify", choices=["truth", "vlm"], required=True)
    ap.add_argument("--voice-server", default="http://127.0.0.1:8765", help="--identify vlm のとき")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--override", type=Path, nargs="*", default=[])
    ap.add_argument("--video", action="store_true", help="盤面ごとの動画を --out/videos に保存する")
    ap.add_argument("--debug", action="store_true", help="field.rehearsal のフレームごとの記録（調べるため）も run.log に出す")
    ap.add_argument("--target-type", choices=sorted(TYPES), default="lego",
                    help="指示の対象の種類。Stage 4 は対象がミニカー 1 個なので toy_car")
    ap.add_argument("--rules", choices=["rehearsal", "real"], default="rehearsal",
                    help="判定の決まり。real は実機と同じ（GazeboRehearsalRig の real_rules: 検出の設定、取り違えの上限、"
                         "止めずに続ける、開始時に同定できない物体があっても始める）")
    ap.add_argument("--detector", choices=["background", "learned"], default="background",
                    help="カメラの物体の検出。learned は学習した検出器（seg_detector.SegDetector、config/seg_detector.yaml の"
                         "モデル）で、塊を作るところだけを置き換える")
    ap.add_argument("--seg-model", type=Path, help="--detector learned のモデル（ONNX）。省略は config/seg_detector.yaml")
    a = ap.parse_args()
    if a.detector == "learned" and a.perception != "camera":
        ap.error("--detector learned は --perception camera のときだけ")
    if a.seg_model is not None and a.detector != "learned":
        ap.error("--seg-model は --detector learned のときだけ")
    by_range = a.seed is not None and a.boards is not None
    if by_range == (a.board_seeds is not None) or (a.board_seeds is not None and (a.seed, a.boards) != (None, None)):
        ap.error("--seed と --boards の組か、--board-seeds のどちらか一方を指定する")
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    if a.debug:
        logging.getLogger("toy_sweeper.field.rehearsal").setLevel(logging.DEBUG)
    cfg = load_all(overrides=OVERRIDES + tuple(a.override))
    cfg = dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, sampling=dataclasses.replace(cfg.task.sampling,
                                                                                                    goal_mode="zones")))
    # Gazebo の World（床・マーカー）は sim.work_area の大きさだけ。盤面ごとの作業領域（v30）は扱わない
    rng_area = cfg.task.sampling.work_area_size_range_m
    w, d = cfg.sim.work_area.size_m
    assert not rng_area or [list(x) for x in rng_area] == [[w, w], [d, d]], \
        f"作業領域の範囲 {rng_area} が Gazebo の World の {w} × {d} m と違う（config/experiments/gazebo_area08.yaml を重ねる）"
    f = load_field_cfg(cfg)
    pcfg = load_perception_cfg()
    rows = []
    obs_errors = []
    seg_model = None
    if a.detector == "learned":
        from toy_sweeper.seg_detector import SegModel, load_seg_detector_cfg
        scfg = load_seg_detector_cfg()
        mpath = a.seg_model or Path(scfg.model)
        seg_model = SegModel(mpath if mpath.is_absolute() else ROOT / mpath, scfg)
        log.info("学習した検出器: %s", mpath)
    seed = a.seed
    n_boards = a.boards if a.board_seeds is None else len(a.board_seeds)
    for k in range(n_boards):
        if a.board_seeds is not None:
            want = a.board_seeds[k]
            s, board = find_board(cfg, pcfg, f.perception.association_tolerance_m, a.stage, want, 1,
                                  lambda b: all(o.color != "white" for o in b.objects))
            assert s == want, (s, want)
        else:
            s, board = find_board(cfg, pcfg, f.perception.association_tolerance_m, a.stage, seed, 1000,
                                  lambda b: all(o.color != "white" for o in b.objects))
            seed = s + 1
        ins = instruction_for(board, cfg, np.random.default_rng(s), a.target_type)
        if ins is None:
            raise RuntimeError(f"盤面 {s} に指示を作れない")
        t0 = time.time()
        # 確率的な Policy の行動の乱数を盤面ごとに固定する（2026-10-19。同じ盤面の失敗が行動の乱数で大きく揺れ、知覚の
        # 版どうしを比べられなかった。同じ乱数の並びで比べるため）
        import torch
        torch.manual_seed(s)
        row = {"board_seed": s, "n_objects": len(board.objects), "instruction": dataclasses.asdict(ins)}
        with running_simulation(out / f"gazebo_{s}.log", camera=True, overrides=OVERRIDES + tuple(a.override),
                                board=list(board.objects), markers=True):
            b = GazeboBackend(cfg, models=board_models(board.objects))
            try:
                rig = GazeboRehearsalRig(cfg, f, pcfg, board, s, b, OverheadCamera(b), real_rules=a.rules == "real")
                video = None
                if a.video:
                    words = f"{' '.join(ins.target_colors)} {' '.join(ins.target_classes)} -> {ins.destination}"
                    video = BoardVideo(out / "videos" / f"board_{s}.mp4", 1.0 / cfg.task.control.action_duration_s, rig,
                                       f"stage {a.stage}  board {s}  {a.perception} perception  instruction: {words}")
                    rig.on_observe = video
                rig.calibrate()
                if seg_model is not None:
                    from toy_sweeper.seg_detector import SegDetector
                    rig.detector = SegDetector.like(rig.detector, seg_model)
                if a.perception == "truth":
                    rig.use_truth_perception()
                if a.identify == "vlm":
                    from toy_sweeper.voice.client import ServerClient
                    rig.use_vlm_identifier(ServerClient(a.voice_server, 120.0))
                rt = rig.make_runtime(PPOPolicy(a.model if a.model.is_absolute() else ROOT / a.model, deterministic=False, cfg=cfg))
                try:
                    rep = asyncio.run(rig.run_instruction(rt, lambda: given_instruction(ins), {}))
                    # 開始時に同定しなかった対象（--rules real）も、真値でゴールの中なら成功
                    ok = all(rep["true_targets_in_goal"].values()) and all(rep["true_missed_targets_in_goal"].values())
                    row |= {"result": "success" if ok else "wrong_done", "steps": rep["steps"],
                            "n_targets": len(rep["targets"]), "missed_targets": rep["true_missed_targets_in_goal"],
                            "given_up": rep["given_up"],
                            "true_outside_mm": rep.get("true_outside_mm"), "obs_outside_mm": rep.get("obs_outside_mm")}
                except FieldStop as e:
                    row |= {"result": e.reason, "detail": str(e)[:200]}
                row |= {"stats": dict(rig.stats), "wall_s": round(time.time() - t0, 1)}
                if video is not None:
                    video.close(f"{row['result']}  ({row.get('steps', '-')} steps)")
                with open(out / "color_audit.jsonl", "a") as fh:   # 調べるための記録（field.rehearsal._audit_identity）
                    for c in rig.color_audit:
                        fh.write(json.dumps({"board_seed": s, "target": c[0], "correct": c[1], "rgb": c[2], "chroma": c[3],
                                             "gripper": c[4], "rgb_now": c[5], "gripper_mm": c[6], "size_mm": c[7]}) + "\n")
                e = np.array(getattr(rig, "obs_errors", []) or np.zeros((0, 3)))
                obs_errors.append(e)
                if len(e):
                    row["obs_error"] = {"n": len(e), "mean_mm": [round(1000 * v, 2) for v in e[:, :2].mean(0)],
                                        "std_mm": [round(1000 * v, 2) for v in e[:, :2].std(0)],
                                        "yaw_std_rad": round(float(e[:, 2].std()), 4)}
            finally:
                b.close()
        rows.append(row)
        log.info("盤面 %d（%d/%d）: %s", s, k + 1, n_boards, row)
        with open(out / "results.jsonl", "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    res = [r["result"] for r in rows]
    summary = {"model": str(a.model), "stage": a.stage, "target_type": a.target_type, "perception": a.perception, "detector": a.detector,
               "seg_model": str(a.seg_model) if a.seg_model else None, "identify": a.identify, "rules": a.rules,
               "boards": len(rows), "success": res.count("success"), "results": {k: res.count(k) for k in sorted(set(res))},
               "stats": {k: sum(r["stats"].get(k, 0) for r in rows)
                         for k in sorted({"missed_grasp", "reassociate", "recover", "truth_swap_frames"}
                                         .union(*(r["stats"] for r in rows)))}}
    e = np.concatenate([x for x in obs_errors if len(x)]) if any(len(x) for x in obs_errors) else np.zeros((0, 3))
    if len(e):
        summary["obs_error"] = {"n": len(e), "mean_mm": [round(1000 * v, 2) for v in e[:, :2].mean(0)],
                                "std_mm": [round(1000 * v, 2) for v in e[:, :2].std(0)],
                                "abs_p95_mm": [round(1000 * v, 2) for v in np.percentile(np.abs(e[:, :2]), 95, axis=0)],
                                "yaw_std_rad": round(float(e[:, 2].std()), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("まとめ: %s", summary)


if __name__ == "__main__":
    main()
