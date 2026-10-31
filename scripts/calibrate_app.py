"""カメラの校正の画面（toy_sweeper.field.calibration_web）を、実カメラ（config/field.yaml の camera）で起動する。
校正だけを行うときの入口（scripts/field_app.py の /calibrate と同じ画面）。ブラウザで http://localhost:<port>/calibrate を開く。

    uv run python scripts/calibrate_app.py --port 8082
    # 前に撮ったボードの写真から内部パラメータを求めておく（保存済みの校正が無く、ボードを撮り直したくないとき）
    uv run python scripts/calibrate_app.py --port 8082 --board-photos data/camera/<日時>

- 写真: data/camera/<日時>/{board,floor}_NNN.png、床の写真の確かめの画像 floor_NNN_check.png（git に入れない）。
- 校正: data/calibration/camera.yaml。保存すると前のものは data/calibration/history/camera_<日時>.yaml に移す。
- ログ: dev_log/calibrate/<日時>/run.log。
- 手前のマーカーの位置は config/markers.yaml の wall_thickness_m（壁の厚み）で決まる。壁を置いたら先に書く。
"""
import argparse
import asyncio
import logging
import time
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from toy_sweeper.config import load_all
from toy_sweeper.field.calibration_web import CalibrationStudio, calibration_router
from toy_sweeper.field.config import load_field_cfg
from toy_sweeper.field.hardware import OpenCVCamera
from toy_sweeper.markers import load_markers_cfg

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("calibrate_app")
# 止めるときに開いている接続を待つ秒数。写真の保存・計算は 1 回 1 秒ほどなので、その途中で切らない程度（未検証）
SHUTDOWN_WAIT_S = 2.0


async def main_async(a) -> None:
    cfg = load_all()
    fcfg, m = load_field_cfg(cfg), load_markers_cfg()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    camera = OpenCVCamera(fcfg.camera)
    studio = CalibrationStudio(cfg, m, camera.frames, ROOT / "data" / "calibration" / "camera.yaml",
                               ROOT / "data" / "camera" / stamp, m.field.wall_thickness_m,
                               ROOT / "data" / "camera" / "background.png")
    if a.board_photos is not None:
        paths = sorted(a.board_photos.glob("board_*.png"))
        if not paths:
            raise FileNotFoundError(f"{a.board_photos} にボードの写真（board_*.png）がない")
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"画像を読めない: {p}")
            studio.add_board(img)
        r = studio.solve_intrinsics()
        log.info("ボードの写真 %d 枚（%s）から内部パラメータ: 誤差 %.3f px、焦点距離 %s px", len(studio.board_images),
                 a.board_photos, r["rms_px"], [round(f, 1) for f in r["focal_px"]])
    app = FastAPI(title="toy-sweeper calibrate")
    app.include_router(calibration_router(studio))

    @app.get("/")
    def index():
        return RedirectResponse("/calibrate")

    log.info("カメラ %s（%dx%d、%.0f fps）、写真: %s、壁の厚み %.3f m", fcfg.camera.device_name, fcfg.camera.width_px,
             fcfg.camera.height_px, fcfg.camera.fps, studio.photo_dir, m.field.wall_thickness_m)
    log.info("ブラウザで http://localhost:%d/calibrate を開く", a.port)
    # 映像（MJPEG）の接続は終わらないので、止めるときに待つ時間を区切る（区切らないと kill で止まらなかった、2026-10-22）
    web = uvicorn.Server(uvicorn.Config(app, host=a.host, port=a.port, log_level="warning",
                                        timeout_graceful_shutdown=SHUTDOWN_WAIT_S))
    cam_t = asyncio.create_task(camera.run(), name="camera")
    web_t = asyncio.create_task(web.serve(), name="web")
    done, _ = await asyncio.wait({cam_t, web_t}, return_when=asyncio.FIRST_COMPLETED)
    web.should_exit = True
    cam_t.cancel()
    for t in (cam_t, web_t):
        try:
            await t
        except asyncio.CancelledError:
            pass
    for t in done:
        t.result()   # カメラの例外（抜けた・設定を受け付けない）はここで上げる


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--board-photos", type=Path, help="前に撮ったボードの写真（board_*.png）のディレクトリ")
    a = ap.parse_args()
    out = ROOT / "dev_log" / "calibrate" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    fh = logging.FileHandler(out / "run.log")
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=[sh, fh])
    try:
        asyncio.run(main_async(a))
    except KeyboardInterrupt:
        log.info("Ctrl+C で終了")


if __name__ == "__main__":
    main()
