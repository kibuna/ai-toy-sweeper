"""実機の管理の画面（/admin、scripts/field_app.py）。大人が見る: セッションの状態、BLE、時間の記録、最後の VLM の
問い合わせ（入力の枠と応答）、ログの最後の行。止めるのは話しかける画面（/）と同じ /stop。

GET /admin              画面
GET /admin/status       状態（JSON）
GET /admin/log?n=200    ログ（run.log）の最後の n 行
GET /admin/detect.jpg   今のフレームに検出を重ねた画像（field.detect_view。1 秒ごとに読む）
GET /admin/detect_info  その検出の情報（ロボットの姿勢、見つけた物、塊が大きすぎるときの理由）
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import asyncio

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from ..spaces import GripperState
from .detect_view import detection_overlay

# 検出を重ねた画像の幅と JPEG の品質（画面で見分けられれば足りる。未検証）
DETECT_VIEW_WIDTH_PX = 960
JPEG_QUALITY = 80
# ログの行数の上限（画面が重くならない程度。未検証）
MAX_LOG_LINES = 1000


def admin_router(worlds, log_path: Path) -> APIRouter:
    """worlds: field.web.Worlds（今の世界の session・rig を出す）。検出の重ね描き（/admin/detect.jpg）は、背景との差の検出を
    持つ世界（実機、World.detection）だけ。ほかの世界では 404。"""
    r = APIRouter()

    @r.get("/admin", response_class=HTMLResponse)
    def page():
        return PAGE

    @r.get("/admin/status")
    async def status():
        if worlds.current.remote is not None:   # 中継する世界（Gazebo）: 遠くの画面の状態だけ
            return {"world": worlds.current.label, "session": await worlds.current.remote.status()}
        session = worlds.current.session
        rig = session.rig
        rt = session.rt
        link = rig.link.status()
        out = {"world": worlds.current.label, "session": session.status(),
               "link": {"connected": link.connected, "detail": link.detail, "n_sent": link.n_sent},
               "area_size_m": list(rig.area_size),
               "vlm_last": None if rig.identifier is None else rig.identifier.last,
               "results": [{"instruction": str(x["instruction"]), "result": x["result"],
                            "steps": (x["report"] or {}).get("steps")} for x in session.results[-10:]]}
        if rt is not None:
            out["timing_ms"] = {k: {"mean": round(1000 * v["mean"], 1), "p95": round(1000 * v["p95"], 1),
                                    "max": round(1000 * v["max"], 1), "n": v["n"]}
                                for k, v in rt.stats.summary().items()}
            out["skipped_frames"] = rt.perception.skipped_frames
        return out

    @r.get("/admin/log", response_class=PlainTextResponse)
    def log_tail(n: int = 200):
        if not 0 < n <= MAX_LOG_LINES:
            raise HTTPException(400, f"n は 1〜{MAX_LOG_LINES}: {n}")
        with log_path.open(encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=n))

    last_detect: dict = {}

    @r.get("/admin/detect.jpg")
    async def detect_jpg():
        w = worlds.current
        if not w.detection:
            raise HTTPException(404, "今の世界には検出の重ね描きが無い")
        session, rig = w.session, w.session.rig
        await asyncio.to_thread(session.refresh_calibration)
        try:
            item = rig.camera.frames.get()
        except LookupError:   # 起動の直後（カメラの映像がまだ来ていない）
            raise HTTPException(503, "カメラの映像を待っている") from None
        rt = session.rt
        closed = rt is not None and rt.controller is not None and rt.controller.gripper == GripperState.CLOSED
        g = session.cfg.sim.gripper
        img, info = await asyncio.to_thread(detection_overlay, rig, item.value.image,
                                            g.jaw_closed_angle_rad if closed else g.jaw_open_angle_rad)
        last_detect.clear()
        last_detect.update(info)
        h = int(img.shape[0] * DETECT_VIEW_WIDTH_PX / img.shape[1])
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (DETECT_VIEW_WIDTH_PX, h)), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG にできない")
        return Response(buf.tobytes(), media_type="image/jpeg")

    @r.get("/admin/detect_info")
    def detect_info():
        if not worlds.current.detection:
            raise HTTPException(404, "今の世界には検出の重ね描きが無い")
        return last_detect

    return r


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>管理の画面</title>
<style>
:root { --bg:#f6f4ef; --fg:#222; --muted:#666; --card:#fff; --line:#ddd; --stop:#e5484d; }
@media (prefers-color-scheme: dark) { :root { --bg:#1b1b1d; --fg:#eee; --muted:#aaa; --card:#26262a; --line:#3a3a3f; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font-family:system-ui, "Hiragino Sans", sans-serif; }
main { max-width:1200px; margin:auto; padding:16px; display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:12px; }
@media (max-width: 900px) { main { grid-template-columns:1fr; } }
section { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:0; }
h2 { grid-column:1/-1; margin:4px 0; } h3 { margin:2px 0 8px; font-size:16px; }
nav { grid-column:1/-1; display:flex; gap:12px; align-items:center; }
nav a { color:inherit; }
button { font-size:16px; padding:10px 18px; border:0; border-radius:8px; background:var(--stop); color:#fff; cursor:pointer; }
pre { white-space:pre-wrap; word-break:break-all; margin:0; font-size:13px; max-height:60vh; overflow:auto; }
img { width:100%; border-radius:8px; background:#000; }
table { border-collapse:collapse; font-size:14px; } td, th { padding:2px 8px; border-bottom:1px solid var(--line); text-align:left; }
</style></head><body><main>
<h2>管理の画面</h2>
<nav><a href="/">話しかける画面</a><a href="/calibrate">カメラの校正</a><button id="stop">止める</button></nav>
<section><h3>検出（赤: 物と見た所、青: ロボットを除く範囲、緑の丸: 見つけた物、黄: 車軸、緑の線: 作業領域、水色の線: 物を探す範囲）</h3><img id="det" alt="検出"><pre id="detinfo"></pre></section>
<section><h3>状態</h3><pre id="status"></pre></section>
<section><h3>最後の VLM の問い合わせ</h3><pre id="vlm"></pre></section>
<section><h3>時間（ms）</h3><table id="timing"></table></section>
<section style="grid-column:1/-1"><h3>ログ（最後の 200 行）</h3><pre id="log"></pre></section>
<script>
async function refresh() {
  const s = await (await fetch("/admin/status")).json();
  const ss = s.session;
  document.getElementById("status").textContent =
    `[${s.world}] ${ss.headline}（段階 ${ss.phase ?? "-"}、歩 ${ss.step ?? "-"}）\\n` +
    `BLE: ${s.link.connected ? "接続中" : "未接続"}（${s.link.detail}、送った指令 ${s.link.n_sent}）\\n` +
    `作業領域: ${s.area_size_m.map(x => (x * 100).toFixed(1)).join(" × ")} cm` +
    (s.skipped_frames !== undefined ? `\\n検出が追いつかず飛ばしたフレーム: ${s.skipped_frames}` : "") +
    "\\n\\n結果:\\n" + (s.results.map(r => `${r.result}（${r.steps ?? "-"} 歩）${r.instruction}`).join("\\n") || "なし");
  document.getElementById("vlm").textContent = s.vlm_last ? JSON.stringify(s.vlm_last, null, 1) : "まだ問い合わせていない";
  const t = s.timing_ms || {};
  document.getElementById("timing").innerHTML = "<tr><th></th><th>平均</th><th>p95</th><th>最大</th><th>回数</th></tr>" +
    Object.entries(t).map(([k, v]) => `<tr><td>${k}</td><td>${v.mean}</td><td>${v.p95}</td><td>${v.max}</td><td>${v.n}</td></tr>`).join("");
  const log = document.getElementById("log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent = await (await fetch("/admin/log?n=200")).text();
  if (atBottom) log.scrollTop = log.scrollHeight;
}
document.getElementById("stop").onclick = () => fetch("/stop", {method: "POST"});
refresh(); setInterval(refresh, 1000);
const det = document.getElementById("det");
async function detTick() {
  await new Promise(r => { det.onload = r; det.onerror = r; det.src = "/admin/detect.jpg?t=" + Date.now(); });
  const d = await (await fetch("/admin/detect_info")).json();
  document.getElementById("detinfo").textContent =
    (d.detect_error ? `止まる理由: ${d.detect_error}\n` : "") + (d.robot_error ? `ロボット: ${d.robot_error}\n` : "") +
    `見つけた物 ${(d.detections || []).length} 個: ` + (d.detections || []).map(x => `(${x.xy_m.join(", ")}) ${x.size_mm.join("×")}mm`).join("  ");
  setTimeout(detLoop, 1000);
}
async function detLoop() {   // 今の世界に検出が無ければ（シミュ）欄を隠し、あれば 1 秒ごとに読む
  const r = await fetch("/admin/detect_info");
  const sec = det.closest("section");
  if (r.ok) { sec.style.display = ""; await detTick(); } else { sec.style.display = "none"; setTimeout(detLoop, 2000); }
}
detLoop();
</script></main></body></html>
"""
