"""実機の管理のアプリ（doc/real_robot_runbook.md の 3）。話しかける画面・カメラの校正・管理の画面を 1 つのアプリで出す。
カメラ（C920）と BLE（micro:bit）を開くのはこのアプリだけ（同じカメラを 2 つのプロセスで開くと、先に開いた側が読めなくなる）。

    # 3090 の PC で音声サーバー（Whisper + Qwen3-VL）を起動しておき、MacBook からトンネルでつなぐ
    ssh -N -L 8765:localhost:8765 task@192.168.11.56
    uv run python scripts/field_app.py --model checkpoints/<run>/final.zip --voice-server http://127.0.0.1:8765 --port 8080
    # 実機と 2D の実時間シミュレーションを 1 つのアプリで動かし、話しかける画面のおとなようの欄で切り替える。--world は最初の世界。
    # シミュレーションの盤面は 🔄 で置き直す

- http://localhost:<port>/ 話しかける画面（子ども）、/admin 管理の画面（大人）、/calibrate カメラの校正と背景。
- 設定は方策の学習のときのもの（checkpoints/<run>/config_snapshot/、行動の間隔・速さ・観測）をそのまま読む（--override を
  手で付け忘れない）。現場の設定（config/field.yaml）はそれと合うかを確かめる。
- 校正（data/calibration/camera.yaml）・背景（data/camera/background.png）・ロボットのマーカーの向き
  （data/calibration/robot_marker.yaml）が揃っていなければ、2D シミュレーションと校正の画面（カメラが開けたとき）で起動する（揃えてから起動し直す）。
- 出力: dev_log/field_app/<日時>/run.log、telemetry.jsonl（送った指令・毎フレームの姿勢など、1 行 1 件）、results.json、
  conversation_<世界>.jsonl（聞き取った指示・返事・結果をその場で）、
  audio/（録音と書き起こし。git に入れない）。
"""
import argparse
import asyncio
import dataclasses
import json
import logging
import os
import time
from pathlib import Path

from fastapi.responses import HTMLResponse

from toy_sweeper.config import load_all
from toy_sweeper.field.admin_web import admin_router
from toy_sweeper.field.calibration_web import BackgroundError, CalibrationStudio, calibration_router
from toy_sweeper.field.config import load_field_cfg
from toy_sweeper.field.hardware import OpenCVCamera, RobotMarkerError, verify_camera
from toy_sweeper.field.real_rig import RealRig
from toy_sweeper.field.session import TalkSession
from toy_sweeper.field.web import World, Worlds, create_app
from toy_sweeper.markers import load_markers_cfg
from toy_sweeper.perception import load_perception_cfg
from toy_sweeper.voice.client import ServerClient
from toy_sweeper.voice.config import load_voice_cfg

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "data" / "calibration" / "camera.yaml"
ROBOT_MARKER = ROOT / "data" / "calibration" / "robot_marker.yaml"
BACKGROUND = ROOT / "data" / "camera" / "background.png"
# 止めるときに開いている接続（映像の MJPEG は終わらない）を待つ秒数（scripts/calibrate_app.py と同じ、未検証）
SHUTDOWN_WAIT_S = 2.0
# sim の盤面を探す Seed の数の上限（talk_sim.py と同じ。同定できる盤面がこの中に無ければ例外）
SIM_SEARCH_LIMIT = 1000
# カメラを探す番号の上限（iPhone・内蔵カメラ・C920 の 3 台 + 余裕）
# Gazebo の世界（--gazebo-ssh）: 3090 のリポジトリ・pixi・話しかける画面のポート（8080 実機、8765 音声サーバー、8081〜8083 は
# 試しの画面で使ったので 8091）
GAZEBO_REMOTE_DIR = "/home/task/dev/toy-sweeper"
GAZEBO_PIXI = "/home/task/.pixi/bin/pixi"
GAZEBO_PORT = 8091
log = logging.getLogger("field_app")


def setup_logging(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(out / "run.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    for noisy in ("uvicorn.access", "multipart", "PIL", "bleak"):
        logging.getLogger(noisy).setLevel(logging.INFO if noisy != "uvicorn.access" else logging.WARNING)
    # 毎周期の記録（送った指令・毎フレームの姿勢など）は run.log に混ぜず、1 行 1 件の telemetry.jsonl に（2026-10-30、実機の
    # 動き方を走りの記録から割り出す）
    tel = logging.getLogger("toy_sweeper.field.telemetry")
    tel.setLevel(logging.DEBUG)
    tel.propagate = False
    th = logging.FileHandler(out / "telemetry.jsonl")
    th.setFormatter(logging.Formatter("%(message)s"))
    tel.addHandler(th)
    return out / "run.log"


async def serve(app, host: str, port: int, tasks: list[asyncio.Task]) -> None:
    """Web とほかのタスクを動かし、どれかが終わったら（例外も）全部止める。"""
    import uvicorn
    web = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning",
                                        timeout_graceful_shutdown=SHUTDOWN_WAIT_S))
    web_t = asyncio.create_task(web.serve(), name="web")
    try:
        done, _ = await asyncio.wait([web_t, *tasks], return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t.exception() is not None:
                raise t.exception()
        log.info("終了: %s", [t.get_name() for t in done])
    finally:
        web.should_exit = True
        for t in tasks:
            t.cancel()
        await asyncio.gather(web_t, *tasks, return_exceptions=True)


def make_policy(model: Path, cfg, fcfg):
    """行動の選び方（確率的か決定的か）は config/field.yaml の control.policy_stochastic。"""
    from toy_sweeper import policies
    import torch
    torch.set_num_threads(fcfg.control.torch_threads)
    return policies.PPOPolicy(model, deterministic=not fcfg.control.policy_stochastic, cfg=cfg)


def make_sim_world(a, cfg, fcfg, pcfg, vcfg, server, model: Path, out: Path):
    """2D の実時間シミュレーション（field.rig.SimRig）の世界。設定は方策の学習のときのもの。作業領域は sim.work_area の固定
    （盤面ごとの大きさは使わない）、ゴールは名前つきの位置（goal_mode: zones）。盤面は 🔄 で置き直す。"""
    from toy_sweeper.field.rig import SimRig, find_board
    cfg = dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, sampling=dataclasses.replace(
        cfg.task.sampling, goal_mode="zones", work_area_size_range_m=())))
    tol = fcfg.perception.association_tolerance_m
    seed, board = find_board(cfg, pcfg, tol, a.stage, a.seed, SIM_SEARCH_LIMIT)
    log.info("シミュレーション: Stage %d、Seed %d、物体 %d 個", a.stage, seed, len(board.objects))
    next_seed = [seed + 1]

    def new_board():
        s_, b = find_board(cfg, pcfg, tol, a.stage, next_seed[0], SIM_SEARCH_LIMIT)
        next_seed[0] = s_ + 1
        log.info("置き直し: Stage %d、Seed %d、物体 %d 個", a.stage, s_, len(b.objects))
        return b

    # 実機と同じく止めずに続ける（control.persist）。物体の検出と追跡はシミュレーションの真値（合成画像の検出では、接した物体が
    # 1 つの塊に見えて追跡が消え、止まった。ここで試したいのは話しかけてから片付けるまでの流れ、2026-10-22）
    fcfg = dataclasses.replace(fcfg, control=dataclasses.replace(fcfg.control, persist=True))
    rig = SimRig(cfg, fcfg, pcfg, board, seed)
    rig.use_truth_perception()
    session = TalkSession(cfg, rig, make_policy(model, cfg, fcfg), _structure(server), _transcribe(server),
                          vcfg.wake.stop_words, new_board, record_dir=out / "audio_sim")
    session.conversation_path = out / "conversation_sim.jsonl"   # 聞き取った指示・返事・結果（その場で 1 行ずつ）
    attach_recorder(session, rig.camera.frames, fcfg, fcfg.sim.camera_fps, out, "videos_sim")
    return World("シミュ", session, rig.camera.frames, False, rig.pause, rig.resume), rig


def gazebo_command(model: Path, stochastic: bool, stage: int, seed: int) -> str:
    """3090 で Gazebo の話しかける画面を起動するコマンド。方策を学習したときの設定の差し替え（run_info.json の overrides）を
    渡す（渡さないと観測の作り方が学習と違い、talk_sim.py が起動で止まる）。3090 にも同じ場所に同じ重みがある前提
    （checkpoints は 3090 から写したもの）。"""
    rel = model.relative_to(ROOT)
    overrides = json.loads((model.parent / "run_info.json").read_text())["overrides"]
    return (f"{GAZEBO_PIXI} run python scripts/talk_sim.py --world gazebo --perception camera --identify vlm "
            f"--policy ppo --model {rel}{' --stochastic' if stochastic else ''} --override {' '.join(overrides)} "
            f"--stage {stage} --seed {seed} --voice-server http://127.0.0.1:8765 --port {GAZEBO_PORT}")


def attach_recorder(session, frames, fcfg, fps: float, out: Path, name: str) -> None:
    """指示ごとの動画を data/camera/<起動の日時>/<name>/ に残す（field.record.attach_task_recorder。run() は main で回す）。"""
    from toy_sweeper.field.record import attach_task_recorder
    attach_task_recorder(session, frames, fcfg, fps, ROOT / "data" / "camera" / out.name / name)


def _structure(server):
    async def structure(text):
        return await asyncio.to_thread(server.instruction, text)
    return structure


def _transcribe(server):
    async def transcribe(wav):
        return await asyncio.to_thread(server.transcribe, wav)
    return transcribe


def keep_awake() -> None:
    """このアプリが動いている間、MacBook をスリープさせない（macOS の caffeinate。アプリが終われば解除される）。2026-10-22、
    触らずに 30 分たってスリープに入り、カメラが止まってアプリが終わった。ふたを閉じたときのスリープは防げない。"""
    import shutil
    import subprocess
    if shutil.which("caffeinate") is None:
        raise RuntimeError("caffeinate が無い（macOS のコマンド）。スリープでカメラが止まる")
    p = subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])
    log.info("スリープさせない（caffeinate、PID %d。このアプリが終われば解除）", p.pid)


async def main_async(a, out: Path, log_path: Path) -> None:
    keep_awake()
    model = a.model if a.model.is_absolute() else ROOT / a.model
    snap = model.parent / "config_snapshot"
    cfg = load_all(config_dir=snap)
    fcfg, pcfg, vcfg, m = load_field_cfg(cfg), load_perception_cfg(), load_voice_cfg(), load_markers_cfg()
    if a.detector is not None:   # 共有の設定（config/field.yaml の real.detector）を変えずに試す
        fcfg = dataclasses.replace(fcfg, real=dataclasses.replace(fcfg.real, detector=a.detector))
    log.info("方策 %s、設定 %s（行動の間隔 %.3f s、前進 %.3f m/s、旋回 %.2f rad/s）", model, snap,
             cfg.task.control.action_duration_s, cfg.sim.robot.linear_speed_mps, cfg.sim.robot.angular_speed_radps)
    log.info("行動の選び方: %s（config/field.yaml の control.policy_stochastic）",
             "確率的" if fcfg.control.policy_stochastic else "決定的")
    server = ServerClient(a.voice_server, vcfg.client.request_timeout_s)
    sim_world, sim_rig = make_sim_world(a, cfg, fcfg, pcfg, vcfg, server, model, out)
    worlds = {"sim": sim_world}
    camera = studio = real_rig = None
    try:
        # カメラは名前（config/field.yaml の camera.device_name）のものだけを開く（ほかのカメラを点けない）。背景と合うかは実機が
        # 動き始めるときに確かめる（RealRig.check_camera）
        verify_camera(fcfg.camera)
        log.info("カメラ: %s", fcfg.camera.device_name)
        camera = OpenCVCamera(fcfg.camera)
        studio = CalibrationStudio(cfg, m, camera.frames, CALIB, ROOT / "data" / "camera" / out.name,
                                   m.field.wall_thickness_m, BACKGROUND)
        real_rig = RealRig(cfg, fcfg, pcfg, CALIB, BACKGROUND, ROBOT_MARKER, server, camera,
                           ROOT / "data" / "camera" / f"errors_{out.name}")
        real_session = TalkSession(cfg, real_rig, make_policy(model, cfg, fcfg), _structure(server), _transcribe(server),
                                   vcfg.wake.stop_words, None, record_dir=out / "audio")
        real_session.conversation_path = out / "conversation_real.jsonl"
        attach_recorder(real_session, camera.frames, fcfg, fcfg.camera.fps, out, "videos_real")
        worlds["real"] = World("実機", real_session, camera.frames, True)
    except (RuntimeError, FileNotFoundError, BackgroundError, RobotMarkerError) as e:
        # 実機の準備が揃っていない（校正・背景・マーカーの向きが無い、背景に似たカメラが無い）: シミュレーションと校正の画面で起動する
        log.error("実機は使えない: %s。シミュレーション%sで起動する（揃えてから起動し直す）", e,
                  "と校正の画面" if studio is not None else "")
        real_rig = None
    remote = None
    if a.gazebo_ssh is not None:
        # Gazebo は 3090 で動かし、話しかける画面を中継する（選ばれている間だけ起動。field.remote_world）
        from toy_sweeper.field.remote_world import RemoteTalk, SshLauncher
        cmd = gazebo_command(model, fcfg.control.policy_stochastic, a.stage, a.seed)
        remote = RemoteTalk(f"http://127.0.0.1:{GAZEBO_PORT}",
                            SshLauncher(a.gazebo_ssh, GAZEBO_REMOTE_DIR, cmd, GAZEBO_PORT, GAZEBO_PORT,
                                        f"/tmp/toy_sweeper_gazebo_{out.name}.log"))
        worlds["gazebo"] = World("Gazebo", None, None, False, remote.pause, remote.resume, remote=remote)
    active = a.world if a.world in worlds else "sim"
    ws = Worlds(worlds, active)
    app = create_app(ws, vcfg.audio.sample_rate_hz, vcfg.server.max_audio_s)
    app.include_router(admin_router(ws, log_path))
    if studio is not None:
        app.include_router(calibration_router(studio))
    import contextlib
    async with contextlib.AsyncExitStack() as stack:
        tasks = []
        w_sim = await stack.enter_async_context(sim_rig.running())
        tasks += [asyncio.create_task(sim_world.session.serve(w_sim), name="session_sim"), *(w.task for w in w_sim.values())]
        if real_rig is not None:
            w_real = await stack.enter_async_context(real_rig.running())
            tasks += [asyncio.create_task(worlds["real"].session.serve(w_real), name="session_real"),
                      *(w.task for w in w_real.values())]
        elif camera is not None:
            tasks.append(asyncio.create_task(camera.run(), name="camera"))   # 校正の画面のため
        tasks += [asyncio.create_task(w.session.recorder.run(), name=f"recorder_{n}")
                  for n, w in worlds.items() if w.session is not None and w.session.recorder is not None]
        log.info("ブラウザで http://%s:%d/ （話しかける。おとなようの欄で 実機・シミュ・Gazebo を切り替える）、/admin（管理）%sを開く。"
                 "今は %s。終了は Ctrl+C", a.host, a.port, "、/calibrate（校正）" if studio is not None else "",
                 worlds[active].label)
        try:
            await serve(app, a.host, a.port, tasks)
        finally:
            if remote is not None:   # 3090 の Gazebo を止める
                await remote.close()
            local = {n: w for n, w in worlds.items() if w.session is not None}
            for w in local.values():   # 記録中に終了したら、書いた所までで閉じる（閉じないと mp4 が読めない）
                if w.session.recorder is not None:
                    w.session.recorder.close("app_exit")
            (out / "results.json").write_text(json.dumps(
                {"model": str(model), "worlds": {n: {"label": w.label,
                                                      "results": [{"instruction": dataclasses.asdict(r["instruction"]),
                                                                   "result": r["result"], "report": r["report"]}
                                                                  for r in w.session.results],
                                                      "events": [e.__dict__ for e in w.session.events]}
                                                  for n, w in local.items()}},
                indent=2, ensure_ascii=False, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="PPO の重み（checkpoints/<run>/final.zip。隣の config_snapshot/ を読む）")
    ap.add_argument("--voice-server", required=True, help="音声サーバーの URL（書き起こし・指示の構造化・VLM の同定）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--world", choices=["real", "sim"], default="real",
                    help="最初に動かす世界（画面のおとなようの欄で切り替えられる）。real: 実機（カメラ・BLE）。sim: 2D の実時間"
                         "シミュレーション。実機の準備が揃っていなければ sim")
    ap.add_argument("--detector", choices=["background", "learned"],
                    help="実機の物体の検出（省くと config/field.yaml の real.detector）。learned: 学習した検出器（checkpoints/seg_v1/best.onnx）")
    ap.add_argument("--gazebo-ssh", default="task@192.168.11.56",
                    help="Gazebo の世界を動かす PC（ssh の宛先）。画面で Gazebo を選んだ間だけ、そこで talk_sim.py --world gazebo を"
                         "起動して中継する。none で Gazebo の世界を出さない")
    ap.add_argument("--stage", type=int, default=7, help="sim の盤面の Stage（7: 対象 1〜n 個と対象外。🔄 で置き直す）")
    ap.add_argument("--seed", type=int, default=20000, help="sim の盤面の Seed（探し始め。評価の盤面と同じ 20000 から）")
    a = ap.parse_args()
    if a.gazebo_ssh == "none":
        a.gazebo_ssh = None
    out = ROOT / "dev_log" / "field_app" / time.strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(out)
    try:
        asyncio.run(main_async(a, out, log_path))
    except KeyboardInterrupt:
        log.info("Ctrl+C で終了")
    except Exception:
        log.exception("例外で止めた")
        raise
    log.info("記録: %s", out)


if __name__ == "__main__":
    main()
