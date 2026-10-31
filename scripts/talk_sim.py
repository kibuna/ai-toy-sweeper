"""話しかけるとシミュレーションが動く。ブラウザで http://<host>:<port>/ を開く。

    # 3090 の PC で音声サーバー（Whisper + Qwen3-VL）を起動しておく
    ~/.pixi/bin/pixi run -e vlm python scripts/voice_server.py --host 127.0.0.1 --port 8765
    # セッション
    uv run python scripts/talk_sim.py --voice-server http://127.0.0.1:8765 --policy ppo \
        --model checkpoints/<run>/final.zip --stage 6 --seed 20000 --port 8080 --perception truth
    # Gazebo（斜め上のカメラ、背景との差で検出、VLM で同定。pixi の環境）
    ~/.pixi/bin/pixi run python scripts/talk_sim.py --world gazebo --perception camera --identify vlm \
        --voice-server http://127.0.0.1:8765 --policy ppo --model checkpoints/<run>/final.zip --stage 6 --seed 20000 --port 8080

- 盤面は --stage の生成器（goal_mode: zones）で --seed から探す。全物体を開始時の画像で別々に同定できるもの。
- 同じ盤面で続けて指示を出せる。終わった指示のゴールは次の指示の置き場になる（そこから物体を出さない）。
- 出力: dev_log/talk/<日時>/run.log（DEBUG）、results.json（指示ごとの結果）、audio/（録音の WAV と
  transcripts.jsonl。子どもの声の書き起こしの評価用。この PC の外へは出さない）。
"""
import argparse
import asyncio
import dataclasses
import json
import logging
import signal
import time
from pathlib import Path

from toy_sweeper.config import load_all
from toy_sweeper.field.config import load_field_cfg
from toy_sweeper.field.rig import SimRig, find_board
from toy_sweeper.field.session import TalkSession
from toy_sweeper.field.web import Worlds, create_app
from toy_sweeper.perception import load_perception_cfg
from toy_sweeper.voice.client import ServerClient
from toy_sweeper.voice.config import load_voice_cfg

ROOT = Path(__file__).resolve().parents[1]
# 盤面を探す Seed の数の上限
SEARCH_LIMIT = 1000
# Gazebo のモードで重ねる設定: 斜め上のフル HD カメラ、車体の接触の減点なし（Gazebo のバックエンドは数えない）
GAZEBO_OVERRIDES = (ROOT / "config/experiments/gazebo_oblique.yaml", ROOT / "config/experiments/no_chassis_penalty.yaml")
log = logging.getLogger("talk_sim")


def setup_logging(out: Path) -> None:
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
    for noisy in ("toy_sweeper.field.telemetry", "uvicorn.access", "multipart", "PIL"):
        logging.getLogger(noisy).setLevel(logging.INFO if noisy != "uvicorn.access" else logging.WARNING)


def make_policy(a, cfg, fcfg):
    from toy_sweeper.policies import MixedHeuristicPolicy, PPOPolicy
    if a.policy == "ppo":
        import torch
        torch.set_num_threads(fcfg.control.torch_threads)
        return PPOPolicy(a.model if a.model.is_absolute() else ROOT / a.model, deterministic=not a.stochastic, cfg=cfg)
    return MixedHeuristicPolicy(cfg)


def board_ok(a):
    # Gazebo のカメラ画像から見るときは、明るい床の上で見えない白を盤面に出さない（config/perception_oblique.yaml）
    return (lambda b: all(o.color != "white" for o in b.objects)) if a.world == "gazebo" else (lambda b: True)


def pick_board(a, cfg, fcfg, pcfg):
    seed, board = find_board(cfg, pcfg, fcfg.perception.association_tolerance_m, a.stage, a.seed, SEARCH_LIMIT, board_ok(a))
    log.info("盤面: Stage %d、Seed %d、物体 %d 個（%s）", a.stage, seed, len(board.objects),
             ", ".join(f"{o.color} {o.type.name.lower()}" for o in board.objects))
    return seed, board


async def main_async(a, cfg, fcfg, pcfg, vcfg, out: Path, seed, board, rig) -> None:
    server = ServerClient(a.voice_server, vcfg.client.request_timeout_s)

    async def structure(text):
        return await asyncio.to_thread(server.instruction, text)

    async def transcribe(wav):
        return await asyncio.to_thread(server.transcribe, wav)

    next_seed = [seed + 1]

    def new_board():
        """リセットの盤面: 前の盤面の次の Seed から探す（同じ条件: 全物体を同定できる。Gazebo では白を出さない）。"""
        s, b = find_board(cfg, pcfg, fcfg.perception.association_tolerance_m, a.stage, next_seed[0], SEARCH_LIMIT,
                          board_ok(a))
        next_seed[0] = s + 1
        log.info("リセット: Stage %d、Seed %d、物体 %d 個", a.stage, s, len(b.objects))
        return b

    session = TalkSession(cfg, rig, make_policy(a, cfg, fcfg), structure, transcribe, vcfg.wake.stop_words,
                          new_board, record_dir=out / "audio")
    session.conversation_path = out / "conversation.jsonl"   # 聞き取った指示・返事・結果（その場で 1 行ずつ）
    # 指示ごとの動画（重ね描きあり・なし）。2026-10-28、Gazebo のシミュレーションも録画する。
    # fps は 2D のカメラと同じ sim.camera_fps（Gazebo のフレームは方策の 1 歩ごとで実時間ではないので、動画は早回し）
    from toy_sweeper.field.record import attach_task_recorder
    rec = attach_task_recorder(session, rig.camera.frames, fcfg, fcfg.sim.camera_fps,
                               ROOT / "data" / "camera" / f"talk_{out.name}" / f"videos_{a.world}")
    import uvicorn
    app = create_app(Worlds.single(session, rig.camera.frames), vcfg.audio.sample_rate_hz, vcfg.server.max_audio_s)
    web = uvicorn.Server(uvicorn.Config(app, host=a.host, port=a.port, log_level="warning"))
    async with rig.running() as watch:
        serve = asyncio.create_task(session.serve(watch), name="session")
        web_t = asyncio.create_task(web.serve(), name="web")
        rec_t = None if rec is None else asyncio.create_task(rec.run(), name="recorder")
        log.info("ブラウザで http://%s:%d/ を開く（終了は Ctrl+C）", a.host, a.port)
        try:
            done, _ = await asyncio.wait([serve, web_t, *(w.task for w in watch.values())],
                                         return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception() is not None:
                    raise t.exception()
            log.info("終了: %s", [t.get_name() for t in done])
        finally:
            web.should_exit = True
            serve.cancel()
            await asyncio.gather(serve, web_t, return_exceptions=True)
            if rec_t is not None:
                rec_t.cancel()
                await asyncio.gather(rec_t, return_exceptions=True)
                rec.close("app_exit")
            (out / "results.json").write_text(json.dumps(
                {"board_seed": seed, "stage": a.stage,
                 "results": [{"instruction": dataclasses.asdict(r["instruction"]), "result": r["result"],
                              "report": r["report"]} for r in session.results],
                 "events": [e.__dict__ for e in session.events]}, indent=2, ensure_ascii=False, default=str))


def run_gazebo(a, cfg, fcfg, pcfg, vcfg, out: Path) -> None:
    """Gazebo のリハーサル（field.rehearsal）を、盤面の物体と床のマーカーを置いた World で動かす。"""
    from toy_sweeper.field.rehearsal import GazeboRehearsalRig
    from toy_sweeper.gazebo.backend import GazeboBackend
    from toy_sweeper.gazebo.camera import OverheadCamera
    from toy_sweeper.gazebo.launcher import running_simulation
    from toy_sweeper.gazebo.sdf_gen import board_models
    import contextlib
    seed, board = pick_board(a, cfg, fcfg, pcfg)
    stack = contextlib.ExitStack()
    current = {}

    def launch(bd):
        """盤面 bd の物体だけの World で Gazebo を起動する（前の Gazebo は止める）。"""
        if current:
            current["backend"].close()
            stack.close()
        n = current.get("n", 0) + 1
        stack.enter_context(running_simulation(out / f"gazebo_{n}.log", camera=True,
                                               overrides=tuple(GAZEBO_OVERRIDES) + tuple(a.override),
                                               board=list(bd.objects), markers=True))
        b = GazeboBackend(cfg, models=board_models(bd.objects))
        current.update(backend=b, n=n)
        return b, OverheadCamera(b)

    b, camera = launch(board)
    with stack:
        try:
            rig = GazeboRehearsalRig(cfg, fcfg, pcfg, board, seed, b, camera, relaunch=launch)
            rig.calibrate()
            if a.perception == "truth":
                log.warning("物体の検出と追跡に Gazebo の真値を使う（--perception truth）")
                rig.use_truth_perception()
            if a.identify == "vlm":
                rig.use_vlm_identifier(ServerClient(a.voice_server, vcfg.client.request_timeout_s))
            asyncio.run(main_async(a, cfg, fcfg, pcfg, vcfg, out, seed, board, rig))
        finally:
            current["backend"].close()


def restore_ctrl_c() -> None:
    """SIGINT を Ctrl+C（KeyboardInterrupt）に戻す。ssh の背景（nohup … &）から起動すると無視の設定で受け継がれ、止める合図
    （管理のアプリの field.remote_world.SshLauncher.stop）で Gazebo を片付けずに残った（2026-10-27）。"""
    signal.signal(signal.SIGINT, signal.default_int_handler)


def main():
    restore_ctrl_c()
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-server", required=True, help="音声サーバーの URL（scripts/voice_server.py）")
    ap.add_argument("--policy", required=True, choices=["ppo", "mixed"])
    ap.add_argument("--model", type=Path, help="--policy ppo の重み（.zip）")
    ap.add_argument("--stochastic", action="store_true", help="PPO の行動を確率的に選ぶ（評価の数字は確率的）")
    ap.add_argument("--stage", type=int, required=True, help="盤面の生成器の Stage（6: 散らかった盤面）")
    ap.add_argument("--seed", type=int, required=True, help="盤面の Seed（探し始め）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--override", type=Path, nargs="*", default=[], help="config の差し替え（config/experiments/）")
    ap.add_argument("--world", choices=["2d", "gazebo"], default="2d",
                    help="2d: 2D の実時間シミュレーション。gazebo: Gazebo のリハーサル（斜め上のカメラ、1 Action ずつ、"
                         "pixi の環境で起動する）")
    ap.add_argument("--identify", choices=["vlm", "truth"], default="truth",
                    help="gazebo のとき、指示を受けたときの物体の同定。vlm: 音声サーバーの /identify")
    ap.add_argument("--perception", choices=["camera", "truth"], required=True,
                    help="物体の検出。camera: カメラ画像から（現場と同じ）。truth: シミュレーションの真値から"
                         "（接した物体が 1 つの塊に見えて追跡が入れ替わる問題を切り離す）")
    ap.add_argument("--no-aruco-noise", action="store_true",
                    help="ロボットの姿勢（ArUco の代わり）の誤差を 0 にする。誤差ありでは、マスクからはみ出した車体の切れ端に"
                         "追跡が移って止まることが多い")
    a = ap.parse_args()
    if a.policy == "ppo" and a.model is None:
        ap.error("--policy ppo には --model が必要")
    out = ROOT / "dev_log" / "talk" / time.strftime("%Y%m%d_%H%M%S")
    setup_logging(out)
    cfg = load_all(overrides=(GAZEBO_OVERRIDES if a.world == "gazebo" else ()) + tuple(a.override))
    # 指示の行き先は名前付きのゴール（四隅）なので、盤面も goal_mode: zones で引く
    cfg = dataclasses.replace(cfg, task=dataclasses.replace(
        cfg.task, sampling=dataclasses.replace(cfg.task.sampling, goal_mode="zones")))
    fcfg = load_field_cfg(cfg)
    if a.no_aruco_noise:
        log.warning("ArUco の誤差を 0 にする（--no-aruco-noise）。field.yaml は %.4f m / %.3f rad",
                    fcfg.sim.robot_pose_noise_m, fcfg.sim.robot_yaw_noise_rad)
        fcfg = dataclasses.replace(fcfg, sim=dataclasses.replace(fcfg.sim, robot_pose_noise_m=0.0, robot_yaw_noise_rad=0.0))
    pcfg, vcfg = load_perception_cfg(), load_voice_cfg()
    try:
        if a.world == "2d":
            seed, board = pick_board(a, cfg, fcfg, pcfg)
            rig = SimRig(cfg, fcfg, pcfg, board, seed)
            if a.perception == "truth":
                log.warning("物体の検出と追跡にシミュレーションの真値を使う（--perception truth）。カメラ画像からの検出は確かめない")
                rig.use_truth_perception()
            asyncio.run(main_async(a, cfg, fcfg, pcfg, vcfg, out, seed, board, rig))
        else:
            run_gazebo(a, cfg, fcfg, pcfg, vcfg, out)
    except KeyboardInterrupt:
        log.info("Ctrl+C で終了")
    log.info("記録: %s", out)


if __name__ == "__main__":
    main()
