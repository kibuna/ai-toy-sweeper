"""別の PC（3090）の話しかける画面を中継する世界（field.remote_world、2026-10-26、Gazebo のシミュレーションも使えるようにするため）。
遠くの画面の代わりに、2D の実時間シミュレーションの話しかける画面（同じ create_app）を ASGI でつなぐ。"""
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from test_session import STOP, _board, _ins, cfg, fcfg  # noqa: F401（同じ fixture を使う）
from toy_sweeper.field.remote_world import RemoteTalk
from toy_sweeper.field.rig import SimRig
from toy_sweeper.field.session import TalkSession
from toy_sweeper.field.web import World, Worlds, create_app
from toy_sweeper.perception import load_perception_cfg
from toy_sweeper.policies import MixedHeuristicPolicy


class FakeLauncher:
    def __init__(self):
        self.started, self.stopped = 0, 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


def _local_world(cfg, fcfg):
    async def structure(text):
        return {"ok": True, "instruction": _ins("red", "top_right")}
    rig = SimRig(cfg, fcfg, load_perception_cfg(), _board(cfg), 0)
    s = TalkSession(cfg, rig, MixedHeuristicPolicy(cfg), structure, None, STOP)
    return World("シミュ", s, rig.camera.frames, False)


def test_remote_world_relays_and_starts_only_when_selected(cfg, fcfg):
    remote_app = create_app(Worlds.single(_local_world(cfg, fcfg).session, None), 16000, 10.0)
    launcher = FakeLauncher()
    remote = RemoteTalk("http://gazebo", launcher, transport=httpx.ASGITransport(app=remote_app))
    ws = Worlds({"sim": _local_world(cfg, fcfg), "gazebo": World("Gazebo", None, None, False, remote.pause, remote.resume,
                                                                     remote=remote)}, "sim")
    assert launcher.started == 0                                    # 選ばれるまで起動しない
    c = TestClient(create_app(ws, 16000, 10.0))
    assert c.post("/world", json={"name": "gazebo"}).json()["ok"] is True
    assert launcher.started == 1
    st = c.get("/status?page=a").json()
    assert st["world"] == "gazebo" and st["busy"] is False and [w["label"] for w in st["worlds"]] == ["シミュ", "Gazebo"]
    r = c.post("/say", json={"text": "あかを みぎうえ"}).json()
    assert r["ok"] is True and "はこぶね" in r["confirmation"]      # 遠くのセッションが受けた
    assert c.post("/stop").json() == {"stopped": False}
    assert c.post("/world", json={"name": "sim"}).json()["ok"] is True
    assert launcher.stopped == 1                                    # ほかへ切り替えたら止める
    asyncio.run(remote.close())


def test_remote_world_reports_starting_when_unreachable(cfg, fcfg):
    """遠くの画面がまだ起動していない（Gazebo の起動に数十秒）間は、準備中と返す（例外にしない）。指示は断る。"""
    def refuse(request):
        raise httpx.ConnectError("つながらない（試験）")
    remote = RemoteTalk("http://gazebo", FakeLauncher(), transport=httpx.MockTransport(refuse))
    ws = Worlds({"gazebo": World("Gazebo", None, None, False, remote.pause, remote.resume, remote=remote)}, "gazebo")
    c = TestClient(create_app(ws, 16000, 10.0))
    st = c.get("/status?page=a").json()
    assert st["busy"] is False and "じゅんび" in st["headline"]
    r = c.post("/say", json={"text": "あかを みぎうえ"}).json()
    assert r["ok"] is False and "起動" in r["reason"]
    asyncio.run(remote.close())


def test_switching_to_current_world_twice_does_not_restart(cfg, fcfg):
    """今の世界をもう一度選んでも何もしない。同時に 2 回選ばれても（画面とほかから。切り替えは別スレッドで走る）起動は 1 回
    （2026-10-27、2 回目の起動が「もう動いている」で例外になった）。"""
    import threading
    import time

    class SlowLauncher(FakeLauncher):
        def start(self):
            time.sleep(0.2)        # ssh での起動の間に、もう 1 回の切り替えが来る
            super().start()
    launcher = SlowLauncher()
    remote = RemoteTalk("http://gazebo", launcher, transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    ws = Worlds({"sim": _local_world(cfg, fcfg), "gazebo": World("Gazebo", None, None, False, remote.pause, remote.resume,
                                                                     remote=remote)}, "sim")
    ts = [threading.Thread(target=ws.switch, args=("gazebo",)) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert ws.switch("gazebo")["ok"] is True
    assert (launcher.started, launcher.stopped) == (1, 0)
    asyncio.run(remote.close())


def test_ssh_launcher_stops_whole_process_group(monkeypatch):
    """遠くのコマンドは setsid で新しいプロセスグループの頭として起動し（PID = グループの番号）、止めるときはグループ全体
    （pixi・python・Gazebo）に SIGINT を送る（2026-10-27、bash -c の PID に送っても下の talk_sim と Gazebo が残った）。"""
    import subprocess
    from types import SimpleNamespace
    from toy_sweeper.field.remote_world import SshLauncher
    ran = []

    def run(args, **kw):
        ran.append(args[-1])
        return SimpleNamespace(stdout="12345\n", stderr="")

    class Tunnel:
        pid = 1

        def __init__(self, args):
            pass

        def terminate(self):
            pass

        def wait(self, timeout):
            pass
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", Tunnel)
    s = SshLauncher("host", "/dir", "pixi run python scripts/talk_sim.py", 8091, 8091, "/tmp/x.log")
    s.start()
    assert "setsid nohup pixi run" in ran[0] and s.pid == 12345
    s.stop()
    assert ran[1] == "kill -INT -- -12345" and s.pid is None


def test_talk_sim_restores_ctrl_c():
    """ssh の背景（nohup … &）から起動すると SIGINT が無視の設定で受け継がれ、Python は Ctrl+C（KeyboardInterrupt）を受けない。
    talk_sim.py は起動で SIGINT を Ctrl+C に戻す（止める合図で Gazebo を片付けて終わる。Gazebo は別のセッションで起動するので、
    グループごとの合図は届かない。2026-10-27、止めた後も 3090 に Gazebo が残った）。"""
    import importlib.util
    import signal
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("talk_sim", Path(__file__).resolve().parents[1] / "scripts/talk_sim.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    orig = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        app.restore_ctrl_c()
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, orig)


def test_remote_status_waits_only_briefly():
    """状態の問い合わせ（画面が 1 秒足らずごとに読む）は短い期限で、遠くが応えなければ準備中と返す（2026-10-27、指示と同じ
    60 s の期限で、Gazebo が起動に失敗している間は画面を開くのに 1 分かかった）。"""
    from toy_sweeper.field.remote_world import STATUS_TIMEOUT_S
    seen = []

    def slow(request):
        seen.append(request.extensions["timeout"]["read"])
        raise httpx.ReadTimeout("応えない（試験）")
    remote = RemoteTalk("http://gazebo", FakeLauncher(), transport=httpx.MockTransport(slow))
    st = asyncio.run(remote.status())
    assert st["phase"] == "starting" and seen == [STATUS_TIMEOUT_S] and STATUS_TIMEOUT_S <= 3.0
    asyncio.run(remote.close())
