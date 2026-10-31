"""本番の画面のカメラの校正（/calibrate、doc/camera_setup.md）。後で作る管理の画面に組み込む部品。

写真はアプリが動かしているカメラ（OpenCVCamera.frames）から撮る（ブラウザのカメラではなく、動かすときと同じ画像で
校正する）。4 隅のマーカーは自由な大きさの長方形の角に置いてよく、作業領域の幅・奥行きも床の写真から求める
（calibration.estimate_extrinsics_free_rect）。ゴールの枠は、大きさがシミュレータ（sim.yaml の work_area）と同じときだけ
重ねる（違う大きさでのゴールの置き方は未定）。カメラの場所を変えたら床の写真で外部パラメータを求め直し、カメラ・解像度・ズーム・ピントを変えたら
ボードの写真で内部パラメータから求め直す。保存すると、前の校正は history/ に残す。

GET  /calibrate                 画面
GET  /calibrate/stream          映像（保存済みの校正の作業領域の枠を重ねる。カメラがずれていないかを見る）
GET  /calibrate/status          状態（保存済みの校正、ボードの写真の数、候補）
POST /calibrate/board           ボードの写真を撮る
POST /calibrate/board/clear     撮ったボードの写真を捨てる
POST /calibrate/intrinsics      ボードの写真から内部パラメータを求める
POST /calibrate/floor           床の写真を撮って外部パラメータを求める（候補）
GET  /calibrate/check.jpg       候補の確かめの画像（床の写真に枠・10 cm の格子・ゴールを重ねる）
POST /calibrate/save            候補を保存する。候補を求めた床の写真（物とロボットをどけて撮る）を背景にもする
POST /calibrate/background      背景（ロボットと物をどけた空の床）だけを撮り直す（照明が変わったとき）。背景との差の検出
                                （perception_oblique）に使う。撮ったときの校正（camera.yaml）と結び付ける
校正できないとき（マーカーが見えない、内部パラメータがない など）は 400 で理由を返す。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from ..calibration import (MIN_BOARD_CORNERS, CalibrationError, CameraModel, Intrinsics, calibrate_intrinsics,
                           charuco_board, draw_check, estimate_extrinsics_free_rect, sim_with_area)
from ..config import AllCfg
from ..markers import MarkersCfg
from ..task import goal_rects
from .latest import Latest

log = logging.getLogger(__name__)

# 写真に使うフレームの古さの上限。カメラは 10 fps（config/field.yaml）なので 1 s は 10 枚分。これより古ければカメラが
# 止まっている（未検証の目安）
FRAME_MAX_AGE_S = 1.0
# 作業領域の大きさがシミュレータと同じとみなす差（ゴールの枠を重ねるかだけに使う）。マーカーの置き方の手の誤差の目安（未検証）
SAME_SIZE_TOL_M = 0.01
# 映像の JPEG の品質（field/web.py と同じ。画面で見分けられれば足りる、未検証）
JPEG_QUALITY = 80


class BackgroundError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def background_meta_path(calib_path: Path) -> Path:
    return calib_path.with_name("background.yaml")


# 画像の縁から、マーカーが切れずに読めるための余白（画素。未検証）
VIEW_EDGE_PX = 10


def area_problems(cfg, m, cam: CameraModel) -> list[str]:
    """校正の作業領域で実機を動かせない理由（無ければ空。保存を断る）。大きさが方策の学習の範囲
    （task.sampling.work_area_size_range_m）の外なら、方策は見たことのない大きさで動く（2026-10-26、奥行き 0.469 m の校正）。"""
    out = []
    w, d = cam.work_area_size_m
    rng = cfg.task.sampling.work_area_size_range_m
    if rng:
        (w0, w1), (d0, d1) = rng
        if not (w0 <= w <= w1 and d0 <= d <= d1):
            out.append(f"作業領域 {100 * w:.0f} × {100 * d:.0f} cm が、方策の学習の大きさ（幅 {100 * w0:.0f}〜{100 * w1:.0f} cm、"
                       f"奥行き {100 * d0:.0f}〜{100 * d1:.0f} cm）の外。床のマーカーを置き直す")
    return out


def area_warnings(cfg, m, cam: CameraModel) -> list[str]:
    """校正の作業領域で気をつけること（知らせるだけで保存はする）。ロボット（マーカーの中心）が作業領域 + 外の余白
    （walls.out_of_area_margin_m）まで出たとき、マーカー全体（半分の対角 + VIEW_EDGE_PX）が画像から出る所がある: そこでは
    マーカーが切れて読めない（2026-10-26、手前の縁から 10 cm 出たロボットのマーカーが画像の下で切れた）。前は保存を断っていたが、
    ロボットがほとんど行かない隅のわずかなはみ出しでも断ったので、知らせるだけにした（2026-10-30、断るのが多すぎた）。"""
    w, d = cam.work_area_size_m
    f = m.field
    reach = cfg.sim.walls.out_of_area_margin_m + f.robot_marker_size_m / math.sqrt(2)
    hx, hy = w / 2 + reach, d / 2 + reach
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    pts = np.vstack([np.linspace(a, b, 20) for a, b in zip(corners, corners[1:] + corners[:1])])
    px = cam.to_px(pts, f.robot_marker_height_m)
    iw, ih = cam.intr.size
    bad = (px[:, 0] < VIEW_EDGE_PX) | (px[:, 0] > iw - VIEW_EDGE_PX) | (px[:, 1] < VIEW_EDGE_PX) | (px[:, 1] > ih - VIEW_EDGE_PX)
    if bad.any():
        return [f"ロボットが作業領域の外の余白（{100 * cfg.sim.walls.out_of_area_margin_m:.0f} cm）まで出たとき、マーカーが画像の"
                "外に出る所がある（そこではロボットを見失いやすい）"]
    return []


def load_background(calib_path: Path, background_path: Path) -> np.ndarray:
    """背景（RGB）。今の校正（calib_path）で撮ったものでなければ BackgroundError（校正し直した後の古い背景を使わない）。"""
    meta = background_meta_path(calib_path)
    if not meta.exists() or not background_path.exists():
        raise BackgroundError(f"背景がない（{background_path}）。校正の画面で、ロボットと物をどけて「背景を撮る」")
    d = yaml.safe_load(meta.read_text())
    if d["camera_yaml_sha256"] != _sha256(calib_path):
        raise BackgroundError(f"背景（{d['taken']}）は今の校正より前の校正で撮った。校正の画面で撮り直す")
    if d["image"] != background_path.name:
        raise BackgroundError(f"背景の記録 {meta} の画像 {d['image']} が {background_path.name} と違う")
    bgr = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise BackgroundError(f"背景を読めない: {background_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class CalibrationStudio:
    def __init__(self, cfg: AllCfg, m: MarkersCfg, frames: Latest, calib_path: Path, photo_dir: Path,
                 wall_thickness_m: float, background_path: Path):
        """background_path: 背景の画像（部屋が写るので git に入れない data/camera/ に置く）。記録は calib_path の隣の background.yaml。"""
        self.cfg, self.m, self.frames = cfg, m, frames
        self.calib_path, self.photo_dir, self.background_path = calib_path, photo_dir, background_path
        self.wall_thickness_m = wall_thickness_m
        self._board_det = cv2.aruco.CharucoDetector(charuco_board(m))
        self.n_board_corners = (m.intrinsics.squares[0] - 1) * (m.intrinsics.squares[1] - 1)
        self.board_images: list[np.ndarray] = []
        self.new_intr: Intrinsics | None = None
        self.candidate: CameraModel | None = None
        self._candidate_img: np.ndarray | None = None   # 候補を求めた床の写真（BGR。保存のとき背景にする）
        self._check: bytes | None = None
        self._n_photos = {"board": 0, "floor": 0}

    def saved(self) -> CameraModel | None:
        return CameraModel.load(self.calib_path) if self.calib_path.exists() else None

    def snapshot(self, max_age_s: float | None = FRAME_MAX_AGE_S) -> np.ndarray:
        """今のフレーム（BGR）。無い・古いときは CalibrationError。max_age_s=None は古さを見ない（試験用）。"""
        try:
            item = self.frames.get()
        except LookupError:
            raise CalibrationError("カメラのフレームがない（カメラが動いていない）") from None
        if max_age_s is not None and (age := time.monotonic() - item.stamp) > max_age_s:
            raise CalibrationError(f"カメラのフレームが {age:.1f} s 古い（上限 {max_age_s} s。カメラが止まっている）")
        return cv2.cvtColor(item.value.image, cv2.COLOR_RGB2BGR)

    def _keep(self, kind: str, img: np.ndarray) -> Path:
        self._n_photos[kind] += 1
        self.photo_dir.mkdir(parents=True, exist_ok=True)
        p = self.photo_dir / f"{kind}_{self._n_photos[kind]:03d}.png"
        if not cv2.imwrite(str(p), img):
            raise OSError(f"写真を保存できない: {p}")
        return p

    def add_board(self, img: np.ndarray) -> dict:
        p = self._keep("board", img)
        ch, _, _, _ = self._board_det.detectBoard(img)
        k = 0 if ch is None else len(ch)
        used = k >= MIN_BOARD_CORNERS
        if used:
            self.board_images.append(img)
        log.info("ボードの写真 %s: 角 %d/%d（%s）", p.name, k, self.n_board_corners, "使う" if used else "使わない")
        return {"file": p.name, "corners": k, "of": self.n_board_corners, "used": used, "count": len(self.board_images)}

    def clear_board(self) -> None:
        self.board_images.clear()
        self.new_intr = None

    def solve_intrinsics(self) -> dict:
        intr, errs = calibrate_intrinsics(self.board_images, self.m, MIN_BOARD_CORNERS)
        self.new_intr = intr
        self.candidate = None
        log.info("内部パラメータ: ボード %d 枚、再投影誤差 %.3f px", len(self.board_images), intr.rms_px)
        return {"rms_px": intr.rms_px, "per_photo_px": [None if math.isnan(e) else e for e in errs],
                "focal_px": [intr.K[0, 0], intr.K[1, 1]], "center_px": [intr.K[0, 2], intr.K[1, 2]],
                "dist": intr.dist.tolist(), "size": list(intr.size)}

    def intrinsics(self) -> Intrinsics:
        """このページで求めた内部パラメータ。無ければ保存済みの校正のもの。どちらも無ければ CalibrationError。"""
        if self.new_intr is not None:
            return self.new_intr
        saved = self.saved()
        if saved is None:
            raise CalibrationError("内部パラメータがない。先にボードの写真を撮って求める")
        return saved.intr

    def solve_floor(self, img: np.ndarray) -> dict:
        p = self._keep("floor", img)
        cam = estimate_extrinsics_free_rect(img, self.intrinsics(), self.m, self.cfg.sim, self.wall_thickness_m)
        check = self._draw(img, cam)
        ok, buf = cv2.imencode(".jpg", check, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("確かめの画像を JPEG にできない")
        self.candidate, self._check, self._candidate_img = cam, buf.tobytes(), img
        cv2.imwrite(str(p.with_name(p.stem + "_check.png")), check)
        log.info("外部パラメータの候補（%s）: 再投影誤差 %.2f px、作業領域 %s m（長方形からのずれ %.3f m）、カメラの位置 %s",
                 p.name, cam.reproj_px, np.round(cam.work_area_size_m, 3).tolist(), cam.rect_deviation_m,
                 np.round(cam.position(), 3).tolist())
        return {"file": p.name, "floor_reproj_px": cam.reproj_px, "camera_position_m": cam.position().tolist(),
                "intrinsics_rms_px": cam.intr.rms_px, **self._area(cam)}

    def _area(self, cam: CameraModel) -> dict:
        return {"work_area_size_m": list(cam.work_area_size_m), "rect_deviation_m": cam.rect_deviation_m,
                "problems": area_problems(self.cfg, self.m, cam),
                "warnings": area_warnings(self.cfg, self.m, cam),
                "same_as_sim": self._same_as_sim(cam),
                "sim_work_area_size_m": list(self.cfg.sim.work_area.size_m)}

    def _same_as_sim(self, cam: CameraModel) -> bool:
        return bool(np.allclose(cam.work_area_size_m, self.cfg.sim.work_area.size_m, atol=SAME_SIZE_TOL_M))

    def _draw(self, bgr: np.ndarray, cam: CameraModel) -> np.ndarray:
        goals = goal_rects(self.cfg.sim) if self._same_as_sim(cam) else {}
        return draw_check(bgr, cam, sim_with_area(self.cfg.sim, cam.work_area_size_m), goals)

    def check_jpeg(self) -> bytes:
        if self._check is None:
            raise CalibrationError("確かめの画像がまだない（床の写真を撮る）")
        return self._check

    def save(self) -> dict:
        if self.candidate is None:
            raise CalibrationError("保存する校正がまだない（床の写真を撮る）")
        problems = area_problems(self.cfg, self.m, self.candidate)
        if problems:
            raise CalibrationError("保存しない: " + "。".join(problems))
        previous = None
        if self.calib_path.exists():
            hist = self.calib_path.parent / "history"
            hist.mkdir(parents=True, exist_ok=True)
            previous = hist / f"camera_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
            if previous.exists():
                raise CalibrationError(f"履歴 {previous} がもうある（1 秒おいて保存し直す）")
            self.calib_path.rename(previous)
        self.candidate.save(self.calib_path)
        log.info("校正を保存した: %s（前の校正: %s）", self.calib_path, previous)
        # 床の写真は物とロボットをどけて撮るので、そのまま背景にする（校正と背景を 1 回で。2026-10-22）
        bg = self.take_background(self._candidate_img)
        return {"saved": str(self.calib_path), "previous": None if previous is None else str(previous),
                "background": bg["saved"]}

    def take_background(self, bgr: np.ndarray) -> dict:
        s = self.saved()
        if s is None:
            raise CalibrationError("背景は校正の後に撮る（今の校正と結び付けるため）")
        if (bgr.shape[1], bgr.shape[0]) != s.intr.size:
            raise CalibrationError(f"背景 {bgr.shape[1]}x{bgr.shape[0]} と校正の画像の大きさ {s.intr.size} が違う")
        self.background_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.background_path), bgr):
            raise OSError(f"背景を保存できない: {self.background_path}")
        taken = time.strftime("%Y-%m-%d %H:%M:%S")
        background_meta_path(self.calib_path).write_text(yaml.safe_dump(
            {"image": self.background_path.name, "camera_yaml_sha256": _sha256(self.calib_path), "taken": taken,
             "size": [bgr.shape[1], bgr.shape[0]]}, sort_keys=False))
        log.info("背景を撮った: %s（%s）", self.background_path, taken)
        return {"saved": str(self.background_path), "taken": taken, "size": [bgr.shape[1], bgr.shape[0]]}

    def background_status(self) -> dict:
        try:
            load_background(self.calib_path, self.background_path)
        except BackgroundError as e:
            return {"ok": False, "reason": str(e)}
        return {"ok": True, **yaml.safe_load(background_meta_path(self.calib_path).read_text())}

    def status(self) -> dict:
        s = self.saved()
        return {"saved": None if s is None else {"path": str(self.calib_path), "floor_reproj_px": s.reproj_px,
                                                 "intrinsics_rms_px": s.intr.rms_px,
                                                 "camera_position_m": s.position().tolist(), **self._area(s)},
                "board_photos": len(self.board_images), "new_intrinsics": self.new_intr is not None,
                "candidate": None if self.candidate is None else {"floor_reproj_px": self.candidate.reproj_px,
                                                                  "camera_position_m": self.candidate.position().tolist(),
                                                                  **self._area(self.candidate)},
                "photo_dir": str(self.photo_dir), "background": self.background_status()}

    def overlay(self, rgb: np.ndarray) -> np.ndarray:
        """映像に保存済みの校正の枠を重ねる（無ければそのまま）。"""
        s = self.saved()
        if s is None or (rgb.shape[1], rgb.shape[0]) != s.intr.size:
            return rgb
        return cv2.cvtColor(self._draw(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), s), cv2.COLOR_BGR2RGB)


def calibration_router(studio: CalibrationStudio) -> APIRouter:
    r = APIRouter()

    def run(f, *args):
        try:
            return f(*args)
        except CalibrationError as e:
            raise HTTPException(400, str(e)) from e

    @r.get("/calibrate", response_class=HTMLResponse)
    def page():
        return PAGE

    @r.get("/calibrate/status")
    def status():
        return studio.status()

    @r.post("/calibrate/board")
    def board():
        return run(lambda: studio.add_board(studio.snapshot()))

    @r.post("/calibrate/board/clear")
    def board_clear():
        studio.clear_board()
        return studio.status()

    @r.post("/calibrate/intrinsics")
    def intrinsics():
        return run(studio.solve_intrinsics)

    @r.post("/calibrate/floor")
    def floor():
        return run(lambda: studio.solve_floor(studio.snapshot()))

    @r.get("/calibrate/check.jpg")
    def check():
        return Response(run(studio.check_jpeg), media_type="image/jpeg")

    @r.post("/calibrate/save")
    def save():
        return run(studio.save)

    @r.post("/calibrate/background")
    def background():
        return run(lambda: studio.take_background(studio.snapshot()))

    async def mjpeg():
        seq = 0
        while True:
            item = await studio.frames.wait_newer(seq, None)
            seq = item.seq
            img = await asyncio.to_thread(studio.overlay, item.value.image)
            ok, buf = await asyncio.to_thread(cv2.imencode, ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise RuntimeError("JPEG にできない")
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

    @r.get("/calibrate/stream")
    def stream():
        return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

    return r


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>カメラの校正</title>
<style>
:root { --bg:#f6f4ef; --fg:#222; --muted:#666; --card:#fff; --line:#ddd; --ok:#2a9d4b; --ng:#d33; --btn:#2f6fde; }
@media (prefers-color-scheme: dark) { :root { --bg:#1b1b1d; --fg:#eee; --muted:#aaa; --card:#26262a; --line:#3a3a3f; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font-family:system-ui, "Hiragino Sans", sans-serif; }
main { max-width:1200px; margin:auto; padding:16px; }
img { width:100%; background:#000; border-radius:8px; display:block; }
section { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; margin:12px 0; }
h2 { margin:4px 0 8px; } h3 { margin:4px 0 8px; font-size:17px; }
.row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin:8px 0; }
button { font-size:16px; padding:10px 16px; border:0; border-radius:8px; background:var(--btn); color:#fff; cursor:pointer; }
button.sub { background:#777; }
p.note { color:var(--muted); margin:4px 0; font-size:14px; }
ol { padding-left:20px; margin:6px 0; } li.ok { color:var(--ok); } li.ng { color:var(--ng); }
pre { white-space:pre-wrap; margin:6px 0; font-size:14px; }
</style></head><body><main>
<h2>カメラの校正</h2>
<p class="note"><a href="/" style="color:inherit">話しかける画面（実機を動かす）</a> ・ <a href="/admin" style="color:inherit">管理の画面</a></p>
<pre id="status"></pre>
<img id="view" src="/calibrate/stream" alt="カメラの映像">
<p class="note">保存済みの校正があれば、作業領域の枠（赤）・10 cm の格子（白）・ゴール（緑）を映像に重ねる。床のテープとずれていたらカメラが動いている。</p>
<section><h3>場所を変えたとき: 床の写真で校正し直す</h3>
<p class="note">4 隅のマーカーは好きな大きさの長方形の角に置く（奥の 2 枚は内側の角、手前の 2 枚は角から手前へはみ出させる）。
作業領域から物とロボットをどけ、マーカー（ID 0〜3）だけにして撮る。幅と奥行きも求める。保存すると、この写真を背景（物を見つけるときの空の床）にもする。</p>
<div class="row"><button id="floor">床の写真を撮って求める</button><button id="save">保存する</button></div>
<img id="check" alt="確かめの画像" style="display:none">
</section>
<section><h3>照明が変わったとき: 背景だけ撮り直す</h3>
<p class="note">物の見つけ方は「空の床との差」。背景は校正の保存のときにも撮る。照明が大きく変わった・マーカーを外したときは、ロボットと物をどけて撮り直す。</p>
<div class="row"><button id="bg">背景を撮り直す</button></div>
</section>
<section><h3>カメラ・解像度・ズーム・ピントを変えたとき: ボードの写真でレンズから測り直す</h3>
<p class="note">ボードを手に持って 15〜20 枚。画面の四隅・端・真ん中、いろいろな傾きと距離で。ぶれないように止めて撮る。</p>
<div class="row"><button id="board">ボードの写真を撮る</button><button id="intr">内部パラメータを求める</button>
<button id="clear" class="sub">ボードの写真を捨てる</button></div>
</section>
<ol id="log"></ol>
<script>
function add(ok, text) { const li = document.createElement("li"); li.className = ok ? "ok" : "ng"; li.textContent = text;
  document.getElementById("log").prepend(li); }
const f2 = (x) => Number(x).toFixed(2);
const area = (c) => `${c.work_area_size_m.map(x => (x * 100).toFixed(1)).join(" × ")} cm` +
  `（角の長方形からのずれ ${(c.rect_deviation_m * 100).toFixed(1)} cm）` +
  (c.same_as_sim ? "" : `（シミュレータの ${c.sim_work_area_size_m.map(x => x * 100).join(" × ")} cm と違う。ゴールの枠は未定なので描かない）`) +
  ((c.problems || []).length ? `\n⚠ 保存できない: ${c.problems.join("。")}` : "") +
  ((c.warnings || []).length ? `\n⚠ 注意（保存はできる）: ${c.warnings.join("。")}` : "");
async function call(path) {
  const r = await fetch(path, {method: "POST"}); const j = await r.json();
  if (!r.ok) { add(false, j.detail || r.statusText); return null; } return j;
}
async function refresh() {
  const s = await (await fetch("/calibrate/status")).json();
  const sv = s.saved ? `保存済み: 作業領域 ${area(s.saved)}、床の誤差 ${f2(s.saved.floor_reproj_px)} px、` +
    `レンズの誤差 ${f2(s.saved.intrinsics_rms_px)} px、カメラの位置 ${s.saved.camera_position_m.map(f2).join(", ")} m` : "保存済みの校正なし";
  document.getElementById("status").textContent = `${sv}\\nボードの写真 ${s.board_photos} 枚` +
    (s.new_intrinsics ? "（このページで求めた内部パラメータを使う）" : "") + (s.candidate ? "\\n未保存の候補あり" : "") +
    `\\n写真の保存先: ${s.photo_dir}` +
    (s.background.ok ? `\\n背景: ${s.background.taken}` : `\\n背景: なし（${s.background.reason}）`);
}
document.getElementById("floor").onclick = async () => {
  const j = await call("/calibrate/floor"); if (!j) return;
  add(j.floor_reproj_px <= 2, `${j.file}: 作業領域 ${area(j)}、床のマーカーの誤差 ${f2(j.floor_reproj_px)} px` +
    `（目安 1〜2 px 以下）、カメラの位置 ${j.camera_position_m.map(f2).join(", ")} m。枠がマーカーと合っていれば保存する`);
  const c = document.getElementById("check"); c.src = "/calibrate/check.jpg?t=" + Date.now(); c.style.display = "block";
  refresh();
};
document.getElementById("save").onclick = async () => {
  const j = await call("/calibrate/save"); if (!j) return;
  add(true, `保存した: ${j.saved}、背景も ${j.background}` + (j.previous ? `（前の校正: ${j.previous}）` : "")); refresh();
};
document.getElementById("board").onclick = async () => {
  const j = await call("/calibrate/board"); if (!j) return;
  add(j.used, `${j.file}: ボードの角 ${j.corners}/${j.of}` + (j.used ? `（${j.count} 枚目）` : "（少ないので使わない。近づけるか明るく）"));
  refresh();
};
document.getElementById("intr").onclick = async () => {
  const j = await call("/calibrate/intrinsics"); if (!j) return;
  add(j.rms_px <= 0.5, `内部パラメータ: 誤差 ${f2(j.rms_px)} px（目安 0.5 px 以下）、焦点距離 ${j.focal_px.map(f2).join(" / ")} px。` +
    "続けて床の写真を撮る"); refresh();
};
document.getElementById("bg").onclick = async () => {
  const j = await call("/calibrate/background"); if (!j) return; add(true, `背景を撮った: ${j.taken}`); refresh();
};
document.getElementById("clear").onclick = async () => { await call("/calibrate/board/clear"); add(true, "ボードの写真を捨てた"); refresh(); };
refresh();
</script></main></body></html>
"""
