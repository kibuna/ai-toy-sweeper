"""実機の物体の検出（背景との差、field.real_rig）を、その場の映像で確かめる画面（doc/real_robot_runbook.md）。
方策も音声サーバーも要らない。カメラはこのスクリプトだけが開く（管理のアプリ・校正の画面は止めておく）。

    uv run python scripts/check_detection.py --port 8083    # http://localhost:8083/

映像に重ねるもの: 物と見た画素（赤）、ロボットを除く範囲（青）、見つけた物（緑の枠と、床の位置・大きさ）、ロボットの姿勢（黄）。
ロボットと物を作業領域に置き、動かしたり並べ替えたりして、物だけが見つかるかを見る。表示は 1 秒ごと（重い処理を毎フレームしない）。
「この画像を保存」で、今の画像と重ねた画像を data/camera/detection_<日時>/ に残す（しきい値を記録から決め直すため）。
"""
import argparse
import asyncio
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

from toy_sweeper.config import load_all
from toy_sweeper.field.config import load_field_cfg
from toy_sweeper.field.hardware import OpenCVCamera
from toy_sweeper.field.detect_view import detection_overlay
from toy_sweeper.field.real_rig import RealRig
from toy_sweeper.perception import load_perception_cfg

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("check_detection")
# 映像の JPEG の品質と幅（画面で見分けられれば足りる。未検証）
JPEG_QUALITY = 80
VIEW_WIDTH_PX = 1280


async def main_async(a) -> None:
    cfg = load_all() if a.model is None else load_all(config_dir=a.model.parent / "config_snapshot")
    f = load_field_cfg(cfg)
    d = ROOT / "data"
    camera = OpenCVCamera(f.camera)
    rig = RealRig(cfg, f, load_perception_cfg(), d / "calibration" / "camera.yaml", d / "camera" / "background.png",
                  d / "calibration" / "robot_marker.yaml", None, camera,
                  d / "camera" / f"errors_check_{time.strftime('%Y%m%d_%H%M%S')}")
    jaw = cfg.sim.gripper.jaw_open_angle_rad
    save_dir = d / "camera" / f"detection_{time.strftime('%Y%m%d_%H%M%S')}"
    last = {}
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/view.jpg")
    async def view():
        item = camera.frames.get()
        img, info = await asyncio.to_thread(detection_overlay, rig, item.value.image, jaw)
        last.update(rgb=item.value.image, overlay=img, info=info)
        h = int(img.shape[0] * VIEW_WIDTH_PX / img.shape[1])
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (VIEW_WIDTH_PX, h)), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return Response(buf.tobytes(), media_type="image/jpeg")

    @app.get("/info")
    def info():
        return last.get("info", {})

    @app.post("/save")
    def save():
        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        cv2.imwrite(str(save_dir / f"{stamp}_frame.png"), cv2.cvtColor(last["rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(save_dir / f"{stamp}_overlay.jpg"), last["overlay"])
        log.info("保存: %s %s", save_dir / stamp, last["info"])
        return {"saved": str(save_dir / stamp)}

    web = uvicorn.Server(uvicorn.Config(app, host=a.host, port=a.port, log_level="warning", timeout_graceful_shutdown=2.0))
    cam_t = asyncio.create_task(camera.run(), name="camera")
    web_t = asyncio.create_task(web.serve(), name="web")
    log.info("ブラウザで http://localhost:%d/ を開く（終了は Ctrl+C）", a.port)
    done, _ = await asyncio.wait({cam_t, web_t}, return_when=asyncio.FIRST_COMPLETED)
    web.should_exit = True
    cam_t.cancel()
    await asyncio.gather(cam_t, web_t, return_exceptions=True)
    for t in done:
        if not t.cancelled() and t.exception() is not None:
            raise t.exception()


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>検出の確かめ</title>
<style>
:root { --bg:#f6f4ef; --fg:#222; } @media (prefers-color-scheme: dark) { :root { --bg:#1b1b1d; --fg:#eee; } }
body { margin:0; background:var(--bg); color:var(--fg); font-family:system-ui, "Hiragino Sans", sans-serif; }
main { max-width:1300px; margin:auto; padding:16px; } img { width:100%; border-radius:8px; background:#000; }
button { font-size:16px; padding:10px 16px; border:0; border-radius:8px; background:#2f6fde; color:#fff; cursor:pointer; }
pre { white-space:pre-wrap; font-size:13px; }
</style></head><body><main>
<h2>検出の確かめ</h2>
<p>赤: 物と見た画素、青: ロボットを除く範囲、緑の丸: 見つけた物（長さ × 幅 mm）、黄: ロボットの車軸の中点、緑の線: 作業領域、水色の線: 物を探す範囲。</p>
<button id="save">この画像を保存</button>
<img id="v" alt="映像"><pre id="info"></pre>
<script>
const v = document.getElementById("v");
async function tick() {
  const url = "/view.jpg?t=" + Date.now();
  await new Promise(r => { v.onload = r; v.onerror = r; v.src = url; });
  document.getElementById("info").textContent = JSON.stringify(await (await fetch("/info")).json(), null, 1);
  setTimeout(tick, 1000);
}
document.getElementById("save").onclick = async () => alert((await (await fetch("/save", {method: "POST"})).json()).saved);
tick();
</script></main></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="方策（隣の config_snapshot/ の設定で爪の形などを読む）。無ければ config/")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    out = ROOT / "dev_log" / "check_detection" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    try:
        asyncio.run(main_async(a))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
