"""話しかけるシミュレーションのブラウザ画面。

GET /         画面（映像、指示の入力欄、録音ボタン、停止ボタン、履歴）
GET /stream   カメラのフレーム（追跡を重ねる）の MJPEG
GET /status   セッションの状態と履歴（JSON）
POST /say     {"text": 指示の文}
POST /audio   録音（WAV、16 kHz、モノラル、16 bit）
POST /stop    止める
POST /reset   盤面を新しく置き直す
"""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .record import draw_overlay
from .session import TalkSession

log = logging.getLogger(__name__)
# 映像を待つ時間の上限（世界を切り替えたことに気づくため）
STREAM_WAIT_S = 1.0
# 映像の JPEG の品質。画面で物体と追跡の印が見分けられれば足りる（未検証）
JPEG_QUALITY = 80


class SayRequest(BaseModel):
    text: str


class SpeakerRequest(BaseModel):
    page: str


class WorldRequest(BaseModel):
    name: str


@dataclass
class World:
    """画面から動かす 1 つの世界（実機、2D の実時間シミュレーション、別の PC の画面の中継）。"""
    label: str             # 画面に出す名前（「実機」「シミュ」）
    session: TalkSession | None   # 中継する世界（remote）では None
    frames: object         # その世界のカメラの Latest[Frame]
    detection: bool        # 検出の重ね描き（背景との差の検出を持つ実機のリグだけ。field.admin_web）
    pause: object = None   # 選ばれていない間に止める・再開する（シミュレーション: SimRig.pause / resume）。None は止めない
    resume: object = None
    remote: object = None  # field.remote_world.RemoteTalk（Gazebo など別の PC の話しかける画面を中継する）


class Worlds:
    """画面の世界の切り替え（2026-10-22、シミュレーションも同じ画面から切り替えて動かす。実機と
    シミュレーションを別のアプリにしたら、両方の画面が読み上げた）。動いている間は切り替えない。"""

    def __init__(self, worlds: dict[str, World], active: str):
        if active not in worlds:
            raise ValueError(f"世界 {active} が無い: {sorted(worlds)}")
        self.worlds, self.active = worlds, active
        self._switching = threading.Lock()   # 切り替えは別スレッドで走る。同時の 2 回で 2 度起動しない
        for n, w in worlds.items():   # 選ばれていない世界は止めておく
            if n != active and w.pause is not None:
                w.pause()

    @staticmethod
    def single(session: TalkSession, frames) -> "Worlds":
        return Worlds({"main": World("", session, frames, False)}, "main")

    @property
    def current(self) -> World:
        return self.worlds[self.active]

    def switch(self, name: str) -> dict:
        if name not in self.worlds:
            raise HTTPException(400, f"世界 {name} が無い: {sorted(self.worlds)}")
        with self._switching:
            return self._switch(name)

    def _switch(self, name: str) -> dict:
        if name == self.active:
            return {"ok": True, "active": name}
        cur = self.current.session
        if cur is not None and (cur.busy or cur._asking or not cur.queue.empty()):
            return {"ok": False, "reason": "動いている間は切り替えない（とめてから切り替える）"}
        if self.current.pause is not None:
            self.current.pause()
        self.active = name
        if self.current.resume is not None:
            self.current.resume()
        log.info("世界を %s（%s）に切り替えた", name, self.worlds[name].label)
        return {"ok": True, "active": name}


def create_app(worlds: Worlds, sample_rate_hz: int, max_audio_s: float) -> FastAPI:
    """worlds: 画面から動かす世界（Worlds.single で 1 つだけ）。指示・停止・映像は、今の世界（worlds.current）へ。
    映像に重ねる校正（hom）とゴールは、毎回リグ（session.rig）から読む（実機は校正し直すと変わる）。"""
    app = FastAPI(title="toy-sweeper talk")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE.replace("{SAMPLE_RATE}", str(sample_rate_hz)).replace("{MAX_AUDIO_S}", str(max_audio_s))

    speaker = {"page": None}   # 読み上げる画面（最後に開いたもの。前のタブが二重に読まないように）

    @app.post("/speaker")
    def claim_speaker(req: SpeakerRequest):
        speaker["page"] = req.page
        return {"ok": True}

    @app.get("/status")
    async def status(page: str | None = None):
        w = worlds.current
        st = await w.remote.status() if w.remote is not None else w.session.status()
        if page is None:
            # 前の版の画面（page を送らない）は、ロボットの返事（kind robot）を全部読み上げる。アプリを起動し直しても開いたままの
            # 古いタブが二重・三重に読んだ（2026-10-22）ので、その画面には読み上げない種類（said）で渡す
            st = {**st, "events": [{**e, "kind": "said"} if e["kind"] == "robot" else e for e in st["events"]]}
        return {**st, "speaker": page is not None and page == speaker["page"], "world": worlds.active,
                "detection": worlds.current.detection,
                "worlds": [{"name": n, "label": w.label} for n, w in worlds.worlds.items()]}

    @app.post("/world")
    async def world(req: WorldRequest):
        r = worlds.current.remote
        if r is not None and await r.busy():
            return {"ok": False, "reason": "動いている間は切り替えない（とめてから切り替える）"}
        return await asyncio.to_thread(worlds.switch, req.name)   # 中継の起動・停止は ssh で数秒かかる

    @app.post("/say")
    async def say(req: SayRequest):
        w = worlds.current
        if w.remote is not None:
            return await w.remote.post("/say", json={"text": req.text})
        return await w.session.say(req.text)

    @app.post("/audio")
    async def audio(request: Request):
        body = await request.body()
        if not body:
            raise HTTPException(400, "音声が空")
        w = worlds.current
        if w.remote is not None:
            return await w.remote.post("/audio", content=body,
                                       headers={"content-type": request.headers.get("content-type", "audio/wav")})
        return await w.session.hear(body)

    @app.post("/stop")
    async def stop():
        w = worlds.current
        if w.remote is not None:
            return await w.remote.post("/stop")
        return {"stopped": w.session.stop("停止ボタン")}

    @app.post("/reset")
    async def reset():
        w = worlds.current
        if w.remote is not None:
            return await w.remote.post("/reset")
        return await w.session.reset()

    async def mjpeg():
        seqs: dict[str, int] = {}
        while True:
            name = worlds.active
            w = worlds.current
            if w.remote is not None:   # 遠くの映像をそのまま流す（切り替えたら次の世界へ）
                async for chunk in w.remote.stream():
                    yield chunk
                    if worlds.active != name:
                        break
                else:
                    await asyncio.sleep(STREAM_WAIT_S)   # まだ起動していない
                continue
            try:   # 切り替えに気づけるよう、待つのは 1 s まで
                item = await w.frames.wait_newer(seqs.get(name, 0), STREAM_WAIT_S)
            except TimeoutError:
                continue
            seqs[name] = item.seq
            s = w.session
            await asyncio.to_thread(s.refresh_calibration)
            img = draw_overlay(item.value.image, s.rt, s.rig.hom, s.rig.goal_rects())
            ok, buf = await asyncio.to_thread(cv2.imencode, ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise RuntimeError("JPEG にできない")
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

    @app.get("/stream")
    def stream():
        return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>おかたづけロボット</title>
<style>
:root { --bg:#fff8ec; --fg:#2b2b2b; --muted:#7a7a7a; --card:#ffffff; --kid:#dff1ff; --robot:#fff1c9; --mic:#ff7a59;
        --stop:#e5484d; --line:#f0e2c8; }
@media (prefers-color-scheme: dark) { :root { --bg:#1d1b18; --fg:#f2efe9; --muted:#b0aaa0; --card:#2a2723; --kid:#23384a;
        --robot:#4a3d1c; --line:#3d3830; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font-family: "Hiragino Maru Gothic ProN", "BIZ UDGothic", system-ui, sans-serif; }
main { display:grid; grid-template-columns: minmax(0, 640px) minmax(300px, 1fr); gap:16px; padding:16px; max-width:1200px; margin:auto; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } }
img#view { width:100%; border-radius:16px; background:#000; display:block; }
#head { font-size:28px; font-weight:700; margin:4px 0 12px; min-height:1.3em; }
.buttons { display:flex; gap:16px; align-items:center; }
button { border:0; cursor:pointer; color:#fff; font-family:inherit; font-weight:700; }
#mic { width:120px; height:120px; border-radius:50%; background:var(--mic); font-size:52px; box-shadow:0 6px 0 #c95a3e; }
#mic.on { background:#2fb36d; box-shadow:0 6px 0 #1f7f4c; animation: pulse 1s infinite; }
@keyframes pulse { 50% { transform: scale(1.06); } }
#reset { width:80px; height:80px; border-radius:50%; background:#7b8cde; font-size:36px; box-shadow:0 6px 0 #5564a8; }
#stop { height:80px; padding:0 28px; border-radius:40px; background:var(--stop); font-size:28px; box-shadow:0 6px 0 #a83236; }
#hint { color:var(--muted); font-size:18px; margin-top:12px; }
#chat { list-style:none; padding:0; margin:16px 0 0; max-height:55vh; overflow:auto; display:flex; flex-direction:column; gap:10px; }
#chat li { padding:10px 14px; border-radius:18px; font-size:22px; line-height:1.4; max-width:90%; }
#chat li.heard { background:var(--kid); align-self:flex-end; }
#chat li.robot { background:var(--robot); align-self:flex-start; }
#chat li.robot::before { content:"🤖 "; }
details { margin-top:16px; color:var(--muted); }
form { display:flex; gap:8px; margin-top:8px; }
input[type=text] { flex:1; font-size:18px; padding:10px; border-radius:10px; border:1px solid var(--line); background:var(--card); color:var(--fg); }
form button { background:#5b7bd5; padding:0 16px; border-radius:10px; font-size:18px; }
</style></head>
<body><main>
<section><img id="view" src="/stream" alt="ロボットの えいぞう"></section>
<section>
  <div id="head"></div>
  <div class="buttons"><button id="mic" title="はなす">🎤</button><button id="stop">とまって</button>
    <button id="reset" title="あたらしく ならべる">🔄</button></div>
  <div id="hint">🎤を おして、「あかい レゴを みぎうえに かたづけて」って いってね</div>
  <ol id="chat"></ol>
  <details><summary>もじで おくる（おとなよう）</summary>
    <div id="worlds"></div>
    <form id="f"><input id="t" type="text" placeholder="あかい レゴを みぎうえに かたづけて" autocomplete="off"><button>おくる</button></form>
    <label id="detwrap" style="display:none"><input type="checkbox" id="det"> 検出を重ねる（赤: 物と見た所、青: ロボットを除く範囲、
      緑の丸: 見つけた物、緑の線: 作業領域、水色の線: 探す範囲。1 秒ごと）</label>
    <div id="detinfo"></div>
    <p><a href="/admin" style="color:inherit">管理の画面</a> ・ <a href="/calibrate" style="color:inherit">カメラの校正</a></p>
  </details>
</section>
</main>
<script>
const SR = {SAMPLE_RATE}, MAX_S = {MAX_AUDIO_S};
// 検出の重ね描き（今の世界が実機のとき、/status の detection）。オンのあいだは映像の代わりに 1 秒ごとの検出の画像
let detAvailable = false;
(() => {
  const view = document.getElementById("view"), box = document.getElementById("det");
  window.setDetection = (ok) => {   // refresh から呼ぶ。検出の無い世界（シミュ）に切り替えたらオフに戻す
    detAvailable = ok;
    document.getElementById("detwrap").style.display = ok ? "" : "none";
    if (!ok && box.checked) { box.checked = false; box.onchange(); }
  };
  let timer = null;
  async function tick() {
    await new Promise(res => { view.onload = res; view.onerror = res; view.src = "/admin/detect.jpg?t=" + Date.now(); });
    const d = await (await fetch("/admin/detect_info")).json();
    document.getElementById("detinfo").textContent = (d.detect_error ? d.detect_error + " " : "") +
      (d.robot_error ? "ロボット: " + d.robot_error + " " : "") + `見つけた物 ${(d.detections || []).length} 個`;
    if (box.checked) timer = setTimeout(tick, 1000);
  }
  box.onchange = () => {
    clearTimeout(timer);
    if (box.checked) { tick(); } else { view.onload = null; view.src = "/stream"; document.getElementById("detinfo").textContent = ""; }
  };
})();
// 読み上げた返事（時刻と文）。数で覚えると、履歴が上限（session.EVENT_HISTORY）で古いものから消えたとき数がずれて、
// 同じ返事を何度も読み上げた（2026-10-22、実機）
const spoken = new Set(); let first = true;
// 読み上げるのは最後に開いた画面だけ（前に開いたタブがアプリにつながり直して、二回ずつ読んだ。2026-10-22）
const page = Math.random().toString(36).slice(2);
fetch("/speaker", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({page})});
function speak(text) {   // ロボットの返事を読み上げる（ブラウザの音声合成、日本語）
  if (!("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(text.replace(/[「」…！]/g, " ")); u.lang = "ja-JP"; u.rate = 0.95; u.pitch = 1.3;
  speechSynthesis.speak(u);
}
async function post(url, body, type) {
  const r = await fetch(url, {method:"POST", headers: type ? {"Content-Type": type} : {}, body});
  if (!r.ok) throw new Error(url + ": " + r.status + " " + await r.text());
  return r.json();
}
document.getElementById("f").onsubmit = async (e) => {
  e.preventDefault(); const t = document.getElementById("t"); const text = t.value; t.value = "";
  try { await post("/say", JSON.stringify({text}), "application/json"); } catch (err) { alert(err); }
  refresh();
};
document.getElementById("stop").onclick = async () => { await post("/stop"); refresh(); };
document.getElementById("reset").onclick = async () => { await post("/reset"); refresh(); };
let rec = null;
document.getElementById("mic").onclick = async () => {
  const b = document.getElementById("mic"), hint = document.getElementById("hint");
  if (rec) { const r = rec; rec = null; b.classList.remove("on"); b.textContent = "🎤";
             hint.textContent = "🎤を おして、「あかい レゴを みぎうえに かたづけて」って いってね"; await r.finish(); return; }
  speechSynthesis.cancel();
  const stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1}});
  const ctx = new AudioContext(); const src = ctx.createMediaStreamSource(stream);
  const node = ctx.createScriptProcessor(4096, 1, 1); const chunks = [];
  node.onaudioprocess = (ev) => { chunks.push(new Float32Array(ev.inputBuffer.getChannelData(0))); };
  src.connect(node); node.connect(ctx.destination);
  const timer = setTimeout(() => { if (rec) document.getElementById("mic").click(); }, MAX_S * 1000 - 500);
  rec = { finish: async () => {
    clearTimeout(timer); node.disconnect(); src.disconnect(); stream.getTracks().forEach(t => t.stop());
    const n = chunks.reduce((a, c) => a + c.length, 0); const all = new Float32Array(n); let o = 0;
    for (const c of chunks) { all.set(c, o); o += c.length; }
    const ratio = ctx.sampleRate / SR; const m = Math.floor(n / ratio); const pcm = new Int16Array(m);
    for (let i = 0; i < m; i++) {   // 平均で間引く（簡単な低域通過）
      const a = Math.floor(i * ratio), z = Math.min(n, Math.floor((i + 1) * ratio)); let s = 0;
      for (let j = a; j < z; j++) s += all[j];
      pcm[i] = Math.max(-1, Math.min(1, s / Math.max(1, z - a))) * 32767;
    }
    await ctx.close();
    const buf = new ArrayBuffer(44 + pcm.length * 2); const v = new DataView(buf);
    const w = (p, s) => { for (let i = 0; i < s.length; i++) v.setUint8(p + i, s.charCodeAt(i)); };
    w(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, SR, true);
    v.setUint32(28, SR * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data");
    v.setUint32(40, pcm.length * 2, true); new Int16Array(buf, 44).set(pcm);
    try { await post("/audio", buf, "audio/wav"); } catch (err) { alert(err); }
    refresh();
  }};
  b.classList.add("on"); b.textContent = "■"; hint.textContent = "いいおわったら もういちど おしてね";
};
async function refresh() {
  try {
    const s = await (await fetch("/status?page=" + page)).json();
    setDetection(s.detection);
    const wl = document.getElementById("worlds");
    if (s.worlds.length > 1) {
      wl.innerHTML = "どれを うごかす: " + s.worlds.map(w =>
        `<button type="button" data-w="${w.name}" style="background:${w.name === s.world ? "#2fb36d" : "#999"};padding:6px 14px;` +
        `border-radius:8px;margin:4px">${w.label}</button>`).join("");
      for (const b of wl.querySelectorAll("button")) b.onclick = async () => {
        const r = await (await fetch("/world", {method: "POST", headers: {"Content-Type": "application/json"},
                                                 body: JSON.stringify({name: b.dataset.w})})).json();
        if (!r.ok) alert(r.reason);
        refresh();
      };
    }
    document.getElementById("head").textContent = s.headline;
    const ol = document.getElementById("chat"); ol.innerHTML = "";
    for (const e of s.events) {
      if (e.kind === "info") continue;
      const li = document.createElement("li"); li.className = e.kind; li.textContent = e.text; ol.append(li);
    }
    ol.scrollTop = ol.scrollHeight;
    const robot = s.events.filter(e => e.kind === "robot");
    for (const e of robot) {
      const key = `${e.t}|${e.text}`;
      if (spoken.has(key)) continue;
      spoken.add(key);
      if (!first && s.speaker) speak(e.speech ?? e.text);   // 開いたときに昔の返事を読み上げない
    }
    first = false;
  } catch (err) { document.getElementById("head").textContent = "ロボットと つながらないよ"; }
}
setInterval(refresh, 600); refresh();
</script></body></html>
"""
