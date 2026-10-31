"""話しかけるセッション（toy_sweeper.field.session）の試験。音声サーバーは偽物。"""
import asyncio
import dataclasses
import math

import pytest

from toy_sweeper.config import load_all
from toy_sweeper.field.config import load_field_cfg
from toy_sweeper.field.rig import SimRig
from toy_sweeper.field.session import TalkSession
from toy_sweeper.geometry import Pose2D
from toy_sweeper.perception import load_perception_cfg
from toy_sweeper.policies import MixedHeuristicPolicy
from toy_sweeper.sim2d import ObjectSpec
from toy_sweeper.spaces import ObjectType
from toy_sweeper.task import EpisodeSpec, goal_rects

STOP = ("止まって", "ストップ")


@pytest.fixture(scope="module")
def cfg():
    return load_all()


@pytest.fixture(scope="module")
def fcfg(cfg):
    f = load_field_cfg(cfg)   # ArUco の誤差は 0（test_field_sim.py と同じ理由）
    return dataclasses.replace(f, sim=dataclasses.replace(f.sim, robot_pose_noise_m=0.0, robot_yaw_noise_rad=0.0))


def _ins(color, dest):
    return {"target_classes": ["lego"], "target_colors": [color], "excluded_classes": [], "destination": dest}


def _board(cfg):
    g = goal_rects(cfg.sim)
    red = ObjectSpec("obj0", ObjectType.LEGO, Pose2D(0.12, 0.12, math.pi / 4), "red")
    blue = ObjectSpec("obj1", ObjectType.LEGO, Pose2D(-0.12, 0.12, -math.pi / 4), "blue")
    return EpisodeSpec(6, Pose2D(0.0, -0.05, math.pi / 4), (red, blue), ("obj0",), None, "top_right", g["top_right"])


def _session(cfg, fcfg, replies):
    calls = []

    async def structure(text):
        calls.append(text)
        return replies[text]
    rig = SimRig(cfg, fcfg, load_perception_cfg(), _board(cfg), 0)
    return TalkSession(cfg, rig, MixedHeuristicPolicy(cfg), structure, None, STOP), calls


def test_say_queues_instruction_and_handles_stop_and_rejections(cfg, fcfg):
    s, calls = _session(cfg, fcfg, {"赤を右上": {"ok": True, "instruction": _ins("red", "top_right")},
                                    "ぬいぐるみ": {"ok": False, "reason": "InstructionError: 未知"}})

    async def go():
        r = await s.say("赤を右上")
        assert r["ok"] and r["confirmation"] == "あかい レゴを みぎうえに はこぶね！" and s.queue.qsize() == 1
        assert not (await s.say("赤を右上"))["ok"]           # 前の指示がまだ待ち行列にある
        s.queue.get_nowait()
        assert not (await s.say("ぬいぐるみ"))["ok"]
        r = await s.say("ちょっと止まって")                   # 停止語は VLM を通さない
        assert r == {"ok": True, "stopped": False}
        assert not (await s.say("  "))["ok"]
    asyncio.run(go())
    assert calls == ["赤を右上", "ぬいぐるみ"]
    assert [e.kind for e in s.events][:2] == ["heard", "robot"]
    assert all(not any("\u4e00" <= ch <= "\u9fff" for ch in e.text) for e in s.events if e.kind == "robot")  # 漢字なし


def test_second_instruction_gets_first_goal_as_zone(cfg, fcfg):
    """同じ盤面で続けて 2 つの指示を実行し、2 つめの指示では 1 つめのゴールが置き場になる。

    セッションの配線の試験なので、物体の検出は真値から（TruthDetector）。カメラ画像からの検出では、学習と並行して
    動かしたとき 4 回に 1 回、2 つめの指示で対象を見失った（2026-10-18）。"""
    s, _ = _session(cfg, fcfg, {"赤を右上": {"ok": True, "instruction": _ins("red", "top_right")},
                                "青を左上": {"ok": True, "instruction": _ins("blue", "top_left")}})
    s.rig.use_truth_perception()
    g = goal_rects(cfg.sim)

    async def go():
        async with s.rig.running() as watch:
            server = asyncio.create_task(s.serve(watch))
            assert (await s.say("赤を右上"))["ok"]
            while not s.results:
                await asyncio.sleep(0.1)
            assert (await s.say("青を左上"))["ok"]
            while len(s.results) < 2:
                await asyncio.sleep(0.1)
            second = s.rt
            server.cancel()
            await asyncio.gather(server, return_exceptions=True)
        return second
    second = asyncio.run(asyncio.wait_for(go(), 120))
    assert [r["result"] for r in s.results] == ["success", "success"], s.results
    assert second.assignment.episode.zones == (g["top_right"],)
    assert s.done_goals == [g["top_right"], g["top_left"]]


def test_truth_detector_matches_true_poses(cfg, fcfg):
    """真値からの検出は、物体ごとに 1 つ、真の位置と大きさ（長辺、短辺）を持つ。"""
    import numpy as np
    from toy_sweeper.field.interfaces import Frame, SimTruth
    from toy_sweeper.field.sim import TruthDetector
    from toy_sweeper.spaces import GripperState
    rig = SimRig(cfg, fcfg, load_perception_cfg(), _board(cfg), 0)
    b = rig.board
    truth = SimTruth(b.robot, b.objects, {o.name: o.pose for o in b.objects}, 0.0, GripperState.OPEN)
    dets = TruthDetector(cfg, rig.hom)(Frame(np.zeros((10, 10, 3), np.uint8), 0.0, 1, truth), b.robot, 0.0)
    assert [d.pose for d in dets] == [o.pose for o in b.objects]
    assert all(d.size_m == pytest.approx((0.0318, 0.0158), abs=1e-3) for d in dets)
    with pytest.raises(ValueError):
        TruthDetector(cfg, rig.hom)(Frame(np.zeros((10, 10, 3), np.uint8), 0.0, 1, None), b.robot, 0.0)


def test_kid_words():
    from toy_sweeper.field import kid_words as kw
    from toy_sweeper.task_manager import TidyInstruction
    assert kw.confirm(TidyInstruction(("lego", "toy_car"), ("blue",), (), "top_left")) == "あおいのを ひだりうえに はこぶね！"
    assert kw.confirm(TidyInstruction(("lego", "toy_car"), ("green",), (), "top_left")) == "みどりのを ひだりうえに はこぶね！"
    assert kw.confirm(TidyInstruction(("lego", "toy_car"), (), (), "bottom_right")) == "おもちゃを みぎしたに はこぶね！"
    assert kw.done(TidyInstruction(("toy_car",), (), ("lego",), "bottom_left")) == "できたよ！ ミニカーを ぜんぶ ひだりしたに はこんだよ"
    assert kw.confirm(TidyInstruction(("toy_car",), (), ("lego",), "bottom_left")) == "レゴは そのままで、ミニカーを ひだりしたに はこぶね！"


def test_truth_tracker_keeps_ids_by_order():
    from toy_sweeper.field.sim import TruthTracker
    from toy_sweeper.perception import Detection
    a = Detection(Pose2D(0.0, 0.0, 0.0), (0.03, 0.015), (1.0, 0.0, 0.0), (0, 0, 1, 1))
    b = Detection(Pose2D(0.01, 0.0, 0.0), (0.03, 0.015), (0.0, 0.0, 1.0), (0, 0, 1, 1))
    t = TruthTracker()
    assert t.update([a, b])[0].mean_rgb == (1.0, 0.0, 0.0)
    # 位置が入れ替わっても、並びの番号で ID は変わらない（近い順の対応付けなら入れ替わる）
    swapped = t.update([dataclasses.replace(a, pose=b.pose), dataclasses.replace(b, pose=a.pose)])
    assert swapped[0].mean_rgb == (1.0, 0.0, 0.0) and swapped[0].pose == b.pose
    with pytest.raises(ValueError):
        t.update([a])


def test_reset_places_new_board_and_forgets_zones(cfg, fcfg):
    from toy_sweeper.task import TaskSampler
    import numpy as np
    s, _ = _session(cfg, fcfg, {})
    new = TaskSampler(cfg.sim, cfg.task).sample(6, np.random.default_rng(3))
    s.new_board = lambda: new
    s.done_goals.append(goal_rects(cfg.sim)["top_right"])

    async def go():
        async with s.rig.running():
            assert (await s.reset())["ok"]
            await asyncio.sleep(0.3)
            truth, _ = s.rig.world.truth()
            s.busy = True
            refused = await s.reset()
            s.busy = False
            return truth, refused
    truth, refused = asyncio.run(go())
    assert truth.objects == new.objects and s.rig.board is new and s.done_goals == []
    assert not refused["ok"]


def test_hear_records_audio_and_transcript(cfg, fcfg, tmp_path):
    import json
    s, _ = _session(cfg, fcfg, {"黒いレゴ右下": {"ok": True, "instruction": _ins("black", "bottom_right")}})

    async def transcribe(wav):
        return {"ok": True, "text": "黒いレゴ右下", "seconds": 0.1}
    s.transcribe, s.record_dir = transcribe, tmp_path / "audio"
    r = asyncio.run(s.hear(b"RIFF....fake"))
    assert r["ok"] and r["text"] == "黒いレゴ右下"
    assert (tmp_path / "audio" / "001.wav").read_bytes() == b"RIFF....fake"
    row = json.loads((tmp_path / "audio" / "transcripts.jsonl").read_text())
    assert row == {"file": "001.wav", "ok": True, "text": "黒いレゴ右下", "reason": None}


def test_conversation_is_saved_as_it_happens(cfg, fcfg, tmp_path):
    """聞き取った指示（声か、画面に打った文か）とアプリの返事を、起きたその場で 1 行ずつ残す（2026-10-28。results.json はアプリを終えるときだけで、落ちると残らない）。"""
    import json
    from toy_sweeper.field import kid_words as kw
    s, _ = _session(cfg, fcfg, {"黒いレゴ右下": {"ok": True, "instruction": _ins("black", "bottom_right")},
                                "ぬいぐるみ": {"ok": False, "reason": "InstructionError: 未知"}})

    async def transcribe(wav):
        return {"ok": True, "text": "黒いレゴ右下", "seconds": 0.1}
    s.transcribe, s.conversation_path = transcribe, tmp_path / "conversation.jsonl"

    async def go():
        await s.hear(b"RIFF....fake")
        await s.say("ぬいぐるみ")
    asyncio.run(go())
    rows = [json.loads(x) for x in (tmp_path / "conversation.jsonl").read_text().splitlines()]
    assert [(r["kind"], r.get("via"), r["text"]) for r in rows] == [
        ("heard", "voice", "黒いレゴ右下"), ("robot", None, "くろい レゴを みぎしたに はこぶね！"),
        ("heard", "text", "ぬいぐるみ"), ("robot", None, kw.BUSY)]       # 1 つ目が待ち行列にあるので断る
    assert all("time" in r for r in rows)


def test_speech_text_uses_kanji_for_hakobu():
    """読み上げは「はこんだよ」を「運んだよ」にし、空白を抜く（「わこんだよ」と読まれないように）。"""
    from toy_sweeper.field import kid_words as kw
    from toy_sweeper.task_manager import TidyInstruction
    ins = TidyInstruction(("lego",), ("red",), (), "top_right")
    assert kw.for_speech(kw.done(ins)) == "できたよ！あかいレゴをぜんぶみぎうえに運んだよ"
    assert kw.for_speech(kw.confirm(ins)) == "あかいレゴをみぎうえに運ぶね！"


def test_shrink_goal():
    """完了の判定のゴールを縁から縮める（config/field.yaml perception.goal_margin_m）。"""
    import pytest
    from toy_sweeper.field.runtime import shrink_goal
    from toy_sweeper.geometry import Rect
    g = shrink_goal(Rect((0.3, -0.3), (0.2, 0.2)), 0.008)
    assert g.center == (0.3, -0.3) and g.size == pytest.approx((0.184, 0.184))
    with pytest.raises(ValueError):
        shrink_goal(Rect((0, 0), (0.02, 0.2)), 0.01)


def test_failure_words_cover_every_stop_reason():
    """止まった理由ごとに、子どもに分かる言葉で言う（2026-10-22、「うまくできなかった」だけでなく理由を伝えるため）。
    ランタイムが出す理由（FieldStop の最初の引数）を、ソースから全部拾って、言葉があることを確かめる。"""
    import re
    from pathlib import Path
    from toy_sweeper.field import kid_words as kw
    from toy_sweeper.task_manager import TidyInstruction
    src = "".join(p.read_text() for p in (Path(__file__).parents[1] / "src" / "toy_sweeper" / "field").glob("*.py"))
    reasons = set(re.findall(r'FieldStop\("([a-z_]+)"', src))
    reasons |= {f"{k}_{n}" for k in ("no", "stale") for n in ("camera", "perception")}   # f"no_{name}" / f"stale_{name}"
    reasons -= {"stop_request", "finished", "already_done"}   # already_done は失敗ではない（kw.already_done）
    ins = TidyInstruction(("lego",), ("green",), (), "bottom")
    for r in sorted(reasons):
        s = kw.failed(r, ins)
        assert s.endswith("もういちど いってね") and s != kw.FAILED, r
    assert kw.failed("no_targets", ins) == "ごめんね、みどりの レゴが みつからないよ。もういちど いってね"


def test_kid_words_done_with_given_up():
    """あきらめた対象があったときの完了の言葉（2026-10-23）。"""
    from toy_sweeper.field import kid_words as kw
    from toy_sweeper.task_manager import TidyInstruction
    ins = TidyInstruction(("lego",), ("red",), (), "top_right")
    assert kw.done(ins, n_given_up=1) == "できたよ！ でも あかい レゴを 1こ みつけられなかった"
    assert kw.done(ins, n_given_up=0) == kw.done(ins)


def test_session_records_one_video_per_instruction(cfg, fcfg, tmp_path):
    """recorder を渡すと、指示を受けて動き始めてから終わるまでを 1 本の動画にする（2026-10-25、config/field.yaml の record）。"""
    import cv2
    from toy_sweeper.field.record import TaskRecorder
    s, _ = _session(cfg, fcfg, {"赤を右上": {"ok": True, "instruction": _ins("red", "top_right")}})
    s.rig.use_truth_perception()
    rec = TaskRecorder(s.rig.camera.frames, tmp_path, fps=fcfg.sim.camera_fps, hold_after_s=0.3)
    s.attach_recorder(rec)

    async def go():
        async with s.rig.running() as watch:
            tasks = [asyncio.create_task(s.serve(watch)), asyncio.create_task(rec.run())]
            assert (await s.say("赤を右上"))["ok"]
            while not s.results or s.busy:
                await asyncio.sleep(0.1)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.run(asyncio.wait_for(go(), 120))
    assert s.results[0]["result"] == "success"
    files = sorted(p for p in tmp_path.glob("*.mp4") if not p.stem.endswith("_raw"))
    assert len(files) == 1 and not rec.recording
    assert files[0].with_name(files[0].stem + "_raw.mp4").exists()   # 重ね描きなしも（2026-10-28）
    assert int(cv2.VideoCapture(str(files[0])).get(cv2.CAP_PROP_FRAME_COUNT)) >= 5


def test_unexpected_error_ends_the_task_not_the_app(cfg, fcfg):
    """思わぬ例外（実機: マーカーの読み違いで Observation が値域の外）では、その指示だけを終えて理由を言い、次の指示を待つ
    （2026-10-26、アプリごと終わり、実機もシミュレーションも使えなくなった）。ロボットはランタイムの後始末で止まる。"""
    s, _ = _session(cfg, fcfg, {"赤を右上": {"ok": True, "instruction": _ins("red", "top_right")}})
    s.rig.use_truth_perception()

    async def broken(rt, get_instruction, watch):
        raise ValueError("Observation が値域の外（試験）")
    s.rig.run_instruction = broken

    async def go():
        async with s.rig.running() as watch:
            server = asyncio.create_task(s.serve(watch))
            assert (await s.say("赤を右上"))["ok"]
            while not s.results:
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.1)
            assert not server.done() and not s.busy           # アプリ（セッション）は続いている
            server.cancel()
            await asyncio.gather(server, return_exceptions=True)
    asyncio.run(asyncio.wait_for(go(), 60))
    assert s.results[0]["result"] == "error"
    assert s.events[-1].text == "ごめんね、なかで エラーが おきちゃった。もういちど いってね"


def test_already_done_words():
    from toy_sweeper.field import kid_words as kw
    from toy_sweeper.task_manager import TidyInstruction
    assert kw.already_done(TidyInstruction(("lego",), ("red",), (), "top_right")) == "あかい レゴは もう みぎうえに かたづいてるよ"
