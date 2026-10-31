"""別の PC の話しかける画面（field.web.create_app）を中継する世界（2026-10-26、Gazebo のシミュレーションも画面から使うため）。

Gazebo は 3090 の PC（Linux、ROS、pixi の環境）でしか動かない。3090 で scripts/talk_sim.py --world gazebo を起動し、ssh の
トンネル越しに、指示・音声・停止・置き直し・状態・映像をそのまま中継する。起動するのは画面で選ばれている間だけ（3090 は学習でも
使う）。
"""
from __future__ import annotations

import logging
import shlex
import subprocess

import httpx

log = logging.getLogger(__name__)

# 遠くの画面への問い合わせの期限（指示の構造化・音声の書き起こしは遠くの音声サーバーで数秒かかる。手元の音声サーバーの
# client.request_timeout_s と同じ程度。未検証）
REMOTE_TIMEOUT_S = 60.0
# 状態の問い合わせの期限。画面は 0.6 s ごとに状態を読む（field.web の refresh）。同じ LAN の 3090 は数十 ms で応えるので、
# 数秒応えなければ起動していない・固まったとみなして準備中と返す（指示と同じ 60 s では、Gazebo が起動に失敗している間、
# 画面を開くのに 1 分かかった。2026-10-27）。値そのものは未検証
STATUS_TIMEOUT_S = 2.0
# 準備中（遠くの画面がまだ応えない）の見出し
STARTING_HEADLINE = "Gazebo を じゅんびちゅう…"


class RemoteTalk:
    """遠くの話しかける画面への中継。launcher（start / stop）で遠くの画面を起動・停止する。"""

    def __init__(self, base_url: str, launcher, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url, self.launcher = base_url, launcher
        self.client = httpx.AsyncClient(base_url=base_url, timeout=REMOTE_TIMEOUT_S, transport=transport)
        self.running = False

    # ---------- 世界の切り替え（field.web.Worlds）から ----------
    def resume(self) -> None:
        self.launcher.start()
        self.running = True

    def pause(self) -> None:
        if self.running:
            self.launcher.stop()
            self.running = False

    async def close(self) -> None:
        self.pause()
        await self.client.aclose()

    # ---------- 中継 ----------
    async def status(self) -> dict:
        try:
            r = await self.client.get("/status", params={"page": "relay"}, timeout=STATUS_TIMEOUT_S)
        except httpx.TransportError:
            return {"busy": False, "asking": False, "headline": STARTING_HEADLINE, "phase": "starting", "step": None,
                    "events": []}
        r.raise_for_status()
        st = r.json()
        return {k: st[k] for k in ("busy", "asking", "headline", "phase", "step", "events")}

    async def busy(self) -> bool:
        st = await self.status()
        return st["busy"] or st["asking"]

    async def post(self, path: str, **kw) -> dict:
        try:
            r = await self.client.post(path, **kw)
        except httpx.TransportError as e:
            return {"ok": False, "reason": f"Gazebo の画面がまだ起動していない（{e}）"}
        r.raise_for_status()
        return r.json()

    async def stream(self):
        """遠くの映像（MJPEG）をそのまま流す。"""
        try:
            async with self.client.stream("GET", "/stream", timeout=None) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
        except httpx.TransportError:
            return


class SshLauncher:
    """ssh で遠くの PC にコマンドを起動し（PID を覚える）、手元からのポートのトンネルを張る。止めるときは PID で止める。
    コマンドは setsid で新しいプロセスグループの頭にし（PID = グループの番号）、止めるときは
    グループ全体（pixi・python・Gazebo）に SIGINT を送る（2026-10-27、bash の PID に送っても talk_sim と Gazebo が残った）。"""

    def __init__(self, host: str, remote_dir: str, remote_cmd: str, local_port: int, remote_port: int, remote_log: str):
        self.host, self.remote_dir, self.remote_cmd = host, remote_dir, remote_cmd
        self.local_port, self.remote_port, self.remote_log = local_port, remote_port, remote_log
        self.pid: int | None = None
        self.tunnel: subprocess.Popen | None = None

    def start(self) -> None:
        if self.pid is not None:
            raise RuntimeError(f"遠くのコマンドがもう動いている（PID {self.pid}）")
        # 背景に回すのは setsid の 1 コマンドだけ（$! がその PID になり、setsid は fork せずにグループの頭になる）
        cmd = (f"cd {shlex.quote(self.remote_dir)} || exit 1; setsid nohup {self.remote_cmd} > {shlex.quote(self.remote_log)} "
               "2>&1 < /dev/null & echo $!")
        out = subprocess.run(["ssh", "-o", "ConnectTimeout=10", self.host, cmd], capture_output=True, text=True,
                             timeout=30, check=True).stdout.strip()
        self.pid = int(out.splitlines()[-1])
        self.tunnel = subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                                        "-L", f"{self.local_port}:localhost:{self.remote_port}", self.host])
        log.info("%s で起動した（PID %d、記録 %s）。トンネル %d → %d（PID %d）", self.host, self.pid, self.remote_log,
                 self.local_port, self.remote_port, self.tunnel.pid)

    def stop(self) -> None:
        if self.pid is not None:
            # talk_sim.py は SIGINT で Gazebo を片付けて終わる
            r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", self.host, f"kill -INT -- -{self.pid}"],
                               capture_output=True, text=True, timeout=30)
            log.info("%s の PID %d を止めた（%s）", self.host, self.pid, (r.stderr or "ok").strip())
            self.pid = None
        if self.tunnel is not None:
            self.tunnel.terminate()
            self.tunnel.wait(timeout=10)
            self.tunnel = None
