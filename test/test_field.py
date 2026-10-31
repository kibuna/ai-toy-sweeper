"""現場の非同期ランタイム（toy_sweeper.field）の単体テスト。実時間で動くもの（SimWorld）は 1 秒程度に抑える。"""
import asyncio
import dataclasses
import math
import subprocess
import sys
import time

import numpy as np
import pytest

from toy_sweeper.config import ConfigError, load_all
from toy_sweeper.field.config import load_field_cfg, validate_field_cfg
from toy_sweeper.field.hardware import (BleRobotLink, VlmObjectIdentifier, encode_command, real_homography,
                                        wheel_motor_values)
from toy_sweeper.field.interfaces import (CommandExpired, DriveCommand, FieldStop, Frame, GripperCommand, LinkError,
                                          LinkStatus, StopCommand, TrackView)
from toy_sweeper.field.latest import Latest
from toy_sweeper.field.runtime import FIELD_STAGE, Assignment, CommandGate, Controller, Supervisor, TrackedState
from toy_sweeper.field.sim import SimRobotLink, SimWorld
from toy_sweeper.field.voice_bridge import VoiceBridge
from toy_sweeper.geometry import Pose2D
from toy_sweeper.perception import chroma_distance, load_perception_cfg
from toy_sweeper.sim2d import ObjectSpec
from toy_sweeper.spaces import Action, GripperState, ObjectType
from toy_sweeper.task import EpisodeSpec, goal_rects
from toy_sweeper.task_manager import TidyInstruction


@pytest.fixture(scope="module")
def cfg():
    return load_all()


@pytest.fixture(scope="module")
def fcfg(cfg):
    return load_field_cfg(cfg)


# ---------- Latest ----------
def test_latest_get_set_seq_age():
    async def body():
        x: Latest[int] = Latest("x")
        assert x.seq == 0
        with pytest.raises(LookupError):
            x.get()
        with pytest.raises(LookupError):
            x.age()
        now = time.monotonic()
        a = x.set(10, now - 0.5)
        assert (a.value, a.seq) == (10, 1) and x.get() == a
        assert 0.5 <= x.age() < 0.6
        assert x.age(now) == pytest.approx(0.5)
        x.set(11, now)
        assert x.get().value == 11 and x.seq == 2
        with pytest.raises(ValueError):   # 時刻が戻る値は受け付けない
            x.set(12, now - 1.0)
    asyncio.run(body())


def test_latest_wait_newer_returns_only_newest():
    async def body():
        x: Latest[str] = Latest("x")

        async def producer():
            await asyncio.sleep(0.02)
            for v in ("a", "b", "c"):      # 受け手が起きる前に 3 回置き換わる
                x.set(v, time.monotonic())
        asyncio.get_running_loop().create_task(producer())
        got = await x.wait_newer(0, 1.0)
        assert got.value == "c" and got.seq == 3   # 待ち行列ではないので、最新だけを受け取る
        with pytest.raises(TimeoutError):
            await x.wait_newer(3, 0.05)
    asyncio.run(body())


# ---------- 設定 ----------
def test_field_cfg_validation(cfg, fcfg):
    bad = dataclasses.replace(fcfg, control=dataclasses.replace(fcfg.control, command_ttl_s=0.3))
    with pytest.raises(ConfigError, match="command_ttl_s"):
        validate_field_cfg(bad, cfg)   # 0.25 + 0.05 + 0.1 より短いと、毎周期の指令の間でロボットが止まる
    bad = dataclasses.replace(fcfg, safety=dataclasses.replace(fcfg.safety, max_obs_age_s=0.15))
    with pytest.raises(ConfigError, match="max_obs_age_s"):
        validate_field_cfg(bad, cfg)


# ---------- 安全監視 ----------
class FakeLink:
    def __init__(self):
        self.connected = True
        self.sent = []

    async def connect(self):
        pass

    async def send(self, cmd):
        self.sent.append(cmd)

    def status(self):
        return LinkStatus(self.connected, "ok" if self.connected else "切断")

    async def close(self):
        pass


def _frame(stamp, seq=1):
    return Frame(np.zeros((2, 2, 3), np.uint8), stamp, seq)


def test_supervisor_stop_conditions(fcfg):
    async def body():
        now = time.monotonic()
        frames, tracked, link = Latest("camera"), Latest("tracked"), FakeLink()
        sup = Supervisor(fcfg, frames, tracked, link, start=now)
        assert sup.check(now) is None   # 開始直後はまだフレームがなくてよい
        assert sup.check(now + fcfg.safety.startup_timeout_s + 0.1).reason == "no_camera"
        frames.set(_frame(now), now)
        tracked.set("ts", now)
        assert sup.check(now + 0.1) is None
        assert sup.check(now + fcfg.safety.max_frame_age_s + 0.01).reason == "stale_camera"
        t1 = now + fcfg.safety.max_frame_age_s + 0.5
        ttl = fcfg.control.command_ttl_s
        frames.set(_frame(t1, 2), t1)
        assert sup.check(t1).reason == "stale_perception"   # カメラは新しいが検出が止まった
        tracked.set("ts", t1)
        sup.controlling, sup.last_command = True, t1 - ttl - 0.1
        assert sup.check(t1).reason == "control_stalled"     # 指令の期限を過ぎても次が出ない
        sup.last_command = t1 - 0.1
        assert sup.check(t1) is None
        link.connected = False
        assert sup.check(t1).reason == "link_lost"
        sup.stop_requested = "keyboard"
        assert sup.check(t1).reason == "stop_request"        # 停止の要求がいちばん先
    asyncio.run(body())


def test_command_gate_blocks_after_trip():
    async def body():
        link = FakeLink()
        gate = CommandGate(link)
        now = time.monotonic()
        await gate.send(DriveCommand(gate.next_seq(), now, 0.5, 0.08, 0.0))
        first = gate.trip(FieldStop("stale_camera", "test"))
        assert gate.trip(FieldStop("other", "x")) is first   # 最初の理由が残る
        with pytest.raises(FieldStop, match="stale_camera"):
            await gate.send(DriveCommand(gate.next_seq(), now, 0.5, 0.08, 0.0))
        with pytest.raises(FieldStop):
            await gate.send(GripperCommand(gate.next_seq(), now, 0.5, True))
        await gate.stop()
        assert [type(c).__name__ for c in link.sent] == ["DriveCommand", "StopCommand"]
        assert [c.seq for c in link.sent] == [1, 4]
    asyncio.run(body())


# ---------- 制御（観測 → Action、完了の判定） ----------
class FixedPolicy:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def reset(self):
        pass

    def predict(self, obs):
        self.calls += 1
        return int(self.action)


def _assignment(cfg, target_pose, other_pose=None):
    goal = goal_rects(cfg.sim)["top_right"]
    objs = [ObjectSpec("t", ObjectType.LEGO, target_pose, "red")]
    if other_pose is not None:
        objs.append(ObjectSpec("o", ObjectType.LEGO, other_pose, "blue"))
    ep = EpisodeSpec(FIELD_STAGE, Pose2D(0, 0, 0), tuple(objs), ("t",), None, "top_right", goal)
    ins = TidyInstruction(("lego",), ("red",), (), "top_right")
    return Assignment(ep, {"t": 1, "o": 2} if other_pose is not None else {"t": 1}, ins, {"t": RED})


RED, BLACK = (220.0, 50.0, 50.0), (40.0, 40.0, 40.0)


def _fp(cfg):
    return load_field_cfg(cfg).perception


def _tracked(tracks: dict[int, Pose2D], robot=Pose2D(-0.2, -0.2, 0.0), rgb=None, missed=None):
    rgb, missed = rgb or {}, missed or {}
    views = {tid: TrackView(tid, p, (0.032, 0.016), rgb.get(tid, RED), missed.get(tid, 0)) for tid, p in tracks.items()}
    return TrackedState(_frame(time.monotonic()), robot, views, 0.0)


def test_controller_completion_requires_open_gripper(cfg):
    pcfg = load_perception_cfg()
    pol = FixedPolicy(Action.FORWARD)
    a = _assignment(cfg, Pose2D(0.0, 0.0, 0.0))
    ctl = Controller(cfg, pcfg, pol, a, _fp(cfg))
    # ゴールの外: Policy の Action
    d = ctl.decide(_tracked({1: Pose2D(0.0, 0.0, 0.0)}))
    assert not d.done and d.action == Action.FORWARD and d.snapshot.n_remaining == 1
    inside = Pose2D(0.3, 0.3, 0.0)   # top_right の中心
    # 爪が閉じている: まず開く（挟んだ物体の追跡は位置がずれるので、その位置で完了にしない）
    ctl.commit(Action.CLOSE_GRIPPER)
    d = ctl.decide(_tracked({1: inside}))
    assert not d.done and d.action == Action.OPEN_GRIPPER
    ctl.commit(d.action)
    # 開いた後の新しいフレームでも入っていれば完了
    calls = pol.calls
    d = ctl.decide(_tracked({1: inside}))
    assert d.done and d.action is None and pol.calls == calls


def test_controller_no_completion_on_held_position(cfg):
    """見失っている（missed > 0）対象の追跡は最後の位置のまま。その位置がゴールの中でも完了にしない（爪を開いていれば
    下がって確かめる。test_controller_backs_off_when_target_unseen_after_opening）。"""
    pol = FixedPolicy(Action.FORWARD)
    ctl = Controller(cfg, load_perception_cfg(), pol, _assignment(cfg, Pose2D(0, 0, 0)), _fp(cfg))
    d = ctl.decide(_tracked({1: Pose2D(0.3, 0.3, 0.0)}, missed={1: 3}))
    assert not d.done and d.action == Action.BACKWARD
    d = ctl.decide(_tracked({1: Pose2D(0.3, 0.3, 0.0)}))
    assert d.done


def test_controller_detects_track_swap_by_color(cfg):
    """対象の追跡が別の色の物体に移ったら止める（Stage 6 で、赤い対象の追跡が隣の黒い LEGO に移った例）。色は明るさを除いた
    色度で比べ、上限を超えたのが identity_persist_frames 回続いたら止める（2026-10-19、Gazebo の色の記録から）。"""
    from toy_sweeper.perception import chromaticity
    fp = _fp(cfg)
    ctl = Controller(cfg, load_perception_cfg(), FixedPolicy(Action.FORWARD), _assignment(cfg, Pose2D(0, 0, 0)), fp)
    dark = (110.0, 25.0, 25.0)       # 同じ赤の半分の明るさ（影）: 色度は同じなので取り違えにしない
    d = ctl.decide(_tracked({1: Pose2D(0, 0, 0)}, rgb={1: dark}))
    assert d.color_drift == pytest.approx(255 * math.dist(chromaticity(dark), chromaticity(RED)), abs=1e-6) and d.color_drift < 1e-6
    ctl.decide(_tracked({1: Pose2D(0, 0, 0)}, rgb={1: BLACK}, missed={1: 2}))      # 見失っている間は色を見ない
    for _ in range(fp.identity_persist_frames - 1):
        ctl.decide(_tracked({1: Pose2D(0, 0, 0)}, rgb={1: BLACK}))
    ctl.decide(_tracked({1: Pose2D(0, 0, 0)}, rgb={1: RED}))                       # 続かなければ数え直す
    for _ in range(fp.identity_persist_frames - 1):
        ctl.decide(_tracked({1: Pose2D(0, 0, 0)}, rgb={1: BLACK}))
    with pytest.raises(FieldStop) as e:
        ctl.decide(_tracked({1: Pose2D(0, 0, 0)}, rgb={1: BLACK}))
    assert e.value.reason == "target_identity"


def test_controller_skips_identity_check_near_closed_jaws(cfg):
    """爪を閉じていて対象が爪の近く（identity_skip_closed_radius_m 以内）にあるときは色を見ない（閉じた爪の画素が物体の塊に
    混ざって色度が 70 まで変わった。2026-10-19、Gazebo の記録）。開けば見る。"""
    from toy_sweeper.robot_model import gripper_center_x
    fp = _fp(cfg)
    robot = Pose2D(-0.2, -0.2, 0.0)
    near = Pose2D(-0.2 + gripper_center_x(cfg.sim), -0.2, 0.0)
    ctl = Controller(cfg, load_perception_cfg(), FixedPolicy(Action.FORWARD), _assignment(cfg, near), fp)
    ctl.commit(Action.CLOSE_GRIPPER)
    for _ in range(fp.identity_persist_frames + 2):
        d = ctl.decide(_tracked({1: near}, robot=robot, rgb={1: BLACK}))
    assert d.color_drift == 0.0
    ctl.commit(Action.OPEN_GRIPPER)     # 開けば、爪の間（identity_skip_open_radius_m）より外では見る
    mid = Pose2D(near.x + 0.05, near.y, 0.0)
    with pytest.raises(FieldStop):
        for _ in range(fp.identity_persist_frames):
            ctl.decide(_tracked({1: mid}, robot=robot, rgb={1: BLACK}))


def test_controller_target_lost_and_extra_tracks(cfg):
    pcfg = load_perception_cfg()
    ctl = Controller(cfg, pcfg, FixedPolicy(Action.STOP), _assignment(cfg, Pose2D(0, 0, 0), Pose2D(0.1, 0, 0)), _fp(cfg))
    # 名前のない追跡（同定の後に出た塊。実機では爪の中の物のかけら・爪の陰）は観測に入れない（2026-10-30、
    # 実機で名前のない塊を障害物にすると、ほとんどが誤りだった）。同定した対象外（追跡 2）だけが障害物
    d = ctl.decide(_tracked({1: Pose2D(0, 0, 0), 2: Pose2D(0.1, 0, 0), 7: Pose2D(-0.1, 0.1, 0)}))
    assert len(d.snapshot.obstacles) == 1
    # 対象外の追跡が消えても続ける（見えていないだけ）
    d = ctl.decide(_tracked({1: Pose2D(0, 0, 0)}))
    assert len(d.snapshot.obstacles) == 0
    # 対象の追跡が消えたら止める
    with pytest.raises(FieldStop) as e:
        ctl.decide(_tracked({2: Pose2D(0.1, 0, 0)}))
    assert e.value.reason == "target_lost"


def test_controller_commands(cfg):
    ctl = Controller(cfg, load_perception_cfg(), FixedPolicy(Action.STOP), _assignment(cfg, Pose2D(0, 0, 0)), _fp(cfg))
    seq = iter(range(1, 100))
    cmds = ctl.commands(Action.TURN_LEFT, 5.0, 0.5, lambda: next(seq))
    assert cmds == [DriveCommand(1, 5.0, 0.5, 0.0, cfg.sim.robot.angular_speed_radps)]
    cmds = ctl.commands(Action.CLOSE_GRIPPER, 6.0, 0.5, lambda: next(seq))
    assert cmds == [DriveCommand(2, 6.0, 0.5, 0.0, 0.0), GripperCommand(3, 6.0, 0.5, True)]


# ---------- 実時間の SimWorld ----------
def _open_board(cfg):
    goal = goal_rects(cfg.sim)["top_right"]
    obj = ObjectSpec("obj0", ObjectType.LEGO, Pose2D(0.25, -0.25, 0.0), "red")
    return EpisodeSpec(1, Pose2D(-0.1, 0.0, 0.0), (obj,), ("obj0",), None, "top_right", goal)


def test_sim_world_ttl_watchdog_and_latency(cfg, fcfg):
    """指令は command_latency_s 後に効き、最後に効いた指令から ttl_s で止まる（micro:bit のウォッチドッグの代わり）。"""
    async def body():
        world = SimWorld(cfg, fcfg.sim, _open_board(cfg), 0)
        task = asyncio.get_running_loop().create_task(world.run())
        while world.time is None:
            await asyncio.sleep(0.005)
        x0 = world.robot_pose().x
        v, ttl = cfg.sim.robot.linear_speed_mps, 0.3
        world.submit(DriveCommand(1, time.monotonic(), ttl, v, 0.0))
        await asyncio.sleep(0.8)
        x1 = world.robot_pose().x
        await asyncio.sleep(0.2)
        x2 = world.robot_pose().x
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return world, x0, x1, x2, v, ttl
    world, x0, x1, x2, v, ttl = asyncio.run(body())
    assert world.watchdog_stops == 1
    assert x1 - x0 == pytest.approx(v * ttl, abs=0.004)    # ttl の間だけ進んだ（加速の遅れ・刻みの丸めを許す）
    assert abs(x2 - x1) < 1e-4                              # その後は止まっている
    (ap,) = world.applied
    delay = ap.applied - ap.received
    assert fcfg.sim.command_latency_s - 1e-6 <= delay <= fcfg.sim.command_latency_s + 3 * world.dt
    assert world.max_lag_s < fcfg.sim.max_realtime_lag_s


def test_sim_world_gripper_holds_robot(cfg, fcfg):
    """グリッパーの指令の後、actuation_time_s の間は走行の指令が効かない（Sim2D.actuate_gripper と同じ）。"""
    async def body():
        world = SimWorld(cfg, fcfg.sim, _open_board(cfg), 0)
        task = asyncio.get_running_loop().create_task(world.run())
        while world.time is None:
            await asyncio.sleep(0.005)
        x0 = world.robot_pose().x
        now = time.monotonic()
        world.submit(GripperCommand(1, now, 1.0, True))
        world.submit(DriveCommand(2, now, 1.0, cfg.sim.robot.linear_speed_mps, 0.0))
        await asyncio.sleep(fcfg.sim.command_latency_s + cfg.sim.gripper.actuation_time_s - 0.05)
        x_hold = world.robot_pose().x
        jaw_hold = world.sim.jaw_angle()
        await asyncio.sleep(0.3)
        x_after = world.robot_pose().x
        g = world.sim.gripper_state()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return x0, x_hold, x_after, g, jaw_hold
    x0, x_hold, x_after, g, jaw_hold = asyncio.run(body())
    assert g == GripperState.CLOSED
    assert abs(x_hold - x0) < 1e-4 and x_after - x_hold > 0.01
    # 爪は止まっている間の drive の刻みで回り切っている（サーボの速さで 0.075 s）
    assert jaw_hold == cfg.sim.gripper.jaw_closed_angle_rad


def test_sim_world_grasps_after_jaws_arrive(cfg, fcfg):
    """SimWorld（set_gripper の後、刻みごとの drive で時間を進める）でも、爪が着いた刻みで爪の間の LEGO をつなぐ。"""
    from toy_sweeper.robot_model import gripper_center_x
    goal = goal_rects(cfg.sim)["top_right"]
    obj = ObjectSpec("obj0", ObjectType.LEGO, Pose2D(-0.1 + gripper_center_x(cfg.sim), 0.0, 0.0), "red")
    board = EpisodeSpec(1, Pose2D(-0.1, 0.0, 0.0), (obj,), ("obj0",), None, "top_right", goal)

    async def body():
        world = SimWorld(cfg, fcfg.sim, board, 0)
        task = asyncio.get_running_loop().create_task(world.run())
        while world.time is None:
            await asyncio.sleep(0.005)
        world.submit(GripperCommand(1, time.monotonic(), 1.0, True))
        await asyncio.sleep(fcfg.sim.command_latency_s + cfg.sim.gripper.actuation_time_s)
        held, jaw = world.sim.grasped_object(), world.sim.jaw_angle()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return held, jaw
    held, jaw = asyncio.run(body())
    assert held == "obj0"
    assert cfg.sim.gripper.jaw_closed_angle_rad <= jaw < cfg.sim.gripper.jaw_open_angle_rad


def test_sim_link_refuses_expired_and_disconnected(cfg, fcfg):
    async def body():
        link = SimRobotLink(SimWorld(cfg, fcfg.sim, _open_board(cfg), 0))
        await link.connect()
        with pytest.raises(CommandExpired):
            await link.send(DriveCommand(1, time.monotonic() - 1.0, 0.5, 0.08, 0.0))
        await link.send(StopCommand(2, time.monotonic() - 10.0))   # 停止はいつでも送れる
        link.disconnect()
        assert not link.status().connected
        from toy_sweeper.field.interfaces import LinkError
        with pytest.raises(LinkError):
            await link.send(StopCommand(3, time.monotonic()))
    asyncio.run(body())


# ---------- 実機（機器なしで確かめられる部分） ----------
def test_ble_command_encoding(cfg, fcfg):
    """走行の Action ごとのモーター指令は、実測で決めた表（config/field.yaml の ble.action_motor、動きの計測）。"""
    b, r = fcfg.ble, cfg.sim.robot
    v, w = r.linear_speed_mps, r.angular_speed_radps
    assert wheel_motor_values(v, 0.0, r, b) == tuple(b.action_motor["FORWARD"])
    assert wheel_motor_values(-v, 0.0, r, b) == tuple(b.action_motor["BACKWARD"])
    left, right = wheel_motor_values(0.0, w, r, b)
    assert left == -right and left < 0   # 左回り: 左が後ろ、右が前
    assert wheel_motor_values(0.0, 0.0, r, b) == (0, 0)
    fl, fr = b.action_motor["FORWARD"]
    assert encode_command(DriveCommand(42, 0.0, 0.5, v, 0.0), r, b) == f"M,{fl},{fr},42\n".encode()
    assert encode_command(GripperCommand(43, 0.0, 0.5, False), r, b) == b"G,1,43\n"   # G,1 は開く
    assert encode_command(GripperCommand(44, 0.0, 0.5, True), r, b) == b"G,0,44\n"
    assert encode_command(StopCommand(45, 0.0), r, b) == b"S,45\n"
    with pytest.raises(ValueError, match="Action"):
        wheel_motor_values(0.1, 0.0, r, b)   # 学習の Action の速さでない（勝手に比例させない）
    with pytest.raises(ValueError, match="バイト"):
        encode_command(DriveCommand(10 ** 12, 0.0, 0.5, v, 0.0), r, b)


class FakeBleClient:
    """bleak.BleakClient の代わり（書き込みを記録する）。"""
    address = "fake"

    def __init__(self):
        self.is_connected = True
        self.writes = []

    async def write_gatt_char(self, uuid, data, response):
        self.writes.append((uuid, data, response))


def test_ble_write_line(cfg, fcfg):
    """scripts/check_ble.py が BLE の確かめに使う、生の 1 行の書き込み（doc/real_robot_runbook.md）。"""
    link = BleRobotLink(fcfg.ble, cfg.sim.robot)

    async def body():
        with pytest.raises(LinkError, match="接続していない"):
            await link.write_line("S,1")
        client = FakeBleClient()
        link._client = client
        dt = await link.write_line("M,30,-30,7")
        assert client.writes == [(fcfg.ble.uart_rx_char_uuid, b"M,30,-30,7\n", fcfg.ble.write_with_response)]
        assert dt >= 0.0 and link.n_sent == 1
        for bad in ("S,1\n", "", "M,30,30,1234567890123", "M,３０"):
            with pytest.raises(ValueError):
                await link.write_line(bad)
        assert len(client.writes) == 1
        await link.send(StopCommand(8, 0.0))   # 指令の書き込みも同じ道を通る
        assert client.writes[-1][1] == b"S,8\n" and link.n_sent == 2
    asyncio.run(body())


def test_ble_connect_keeps_live_connection(cfg, fcfg):
    """指示ごとにランタイムが connect を呼ぶ。つながっていれば、つなぎ直さない（micro:bit は 1 つしか接続を受けない）。"""
    link = BleRobotLink(fcfg.ble, cfg.sim.robot)
    client = FakeBleClient()
    link._client = client

    async def body():
        await link.connect()        # 機器を探しに行けば bleak を import して失敗する
        assert link._client is client
    asyncio.run(body())


def test_hardware_module_imports_without_bleak():
    code = ("import sys; import toy_sweeper.field.hardware, toy_sweeper.field.runtime; "
            "assert 'bleak' not in sys.modules, 'bleak を import した'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_real_hooks_are_explicitly_unimplemented():
    """ArucoRobotPose は実装した（test/test_robot_marker.py）。"""
    with pytest.raises(NotImplementedError):
        real_homography()
    with pytest.raises(NotImplementedError):
        VlmObjectIdentifier().identify(_frame(0.0), {})


# ---------- 音声との橋渡し ----------
def test_voice_bridge_second_instruction_stops():
    class FakeRuntime:
        def __init__(self):
            self.stops = []

        def request_stop(self, source):
            self.stops.append(source)

    async def body():
        rt = FakeRuntime()
        br = VoiceBridge(asyncio.get_running_loop(), rt)
        ins = TidyInstruction(("lego",), ("red",), (), "top_right")
        br.on_instruction(ins)
        assert await br.get_instruction() == ins
        br.on_instruction(ins)      # 動いている間の 2 つ目
        await asyncio.sleep(0.01)
        return rt.stops
    assert asyncio.run(body()) == ["new_instruction"]


def test_find_board_rejects_stage9(cfg):
    """現場のランタイムは盤面の置き場の指定（Stage 9、v24）を受け取らない。盤面を探す時点で例外にする。"""
    from toy_sweeper.field.rig import find_board
    with pytest.raises(ValueError, match="Stage 9"):
        find_board(cfg, load_perception_cfg(), 0.01, 9, 0, 10)


def test_controller_no_completion_while_home_object_out(cfg):
    """置き場（zones）から出した対象外が残っていたら完了にしない（2026-10-19、学習の成功の条件 _n_home_out == 0 と
    そろえる。前は現場の完了判定が置き場を見ていなかった）。"""
    from toy_sweeper.geometry import Rect
    goal = goal_rects(cfg.sim)["top_right"]
    zone = goal_rects(cfg.sim)["top_left"]
    objs = (ObjectSpec("t", ObjectType.LEGO, Pose2D(0.0, 0.0, 0.0), "red"),
            ObjectSpec("o", ObjectType.LEGO, Pose2D(-0.3, 0.3, 0.0), "blue"))   # top_left の中から始まる
    ep = EpisodeSpec(FIELD_STAGE, Pose2D(0, 0, 0), objs, ("t",), None, "top_right", goal, (zone,))
    a = Assignment(ep, {"t": 1, "o": 2}, TidyInstruction(("lego",), ("red",), (), "top_right"), {"t": RED})
    pol = FixedPolicy(Action.FORWARD)
    ctl = Controller(cfg, load_perception_cfg(), pol, a, _fp(cfg))
    inside = Pose2D(0.3, 0.3, 0.0)
    d = ctl.decide(_tracked({1: inside, 2: Pose2D(-0.12, 0.3, 0.0)}))   # o は置き場（x −0.4〜−0.2）の外
    assert not d.done and d.action == Action.FORWARD
    d = ctl.decide(_tracked({1: inside, 2: Pose2D(-0.3, 0.3, 0.0)}))
    assert d.done


def test_controller_completes_with_pending_fixes_when_targets_only(cfg):
    """success.require_cleanup: false では、置き場から出た対象外やゴールの中の対象外が残っていても、全対象が入れば完了
    （学習・評価の成功の条件とそろえる）。"""
    goal = goal_rects(cfg.sim)["top_right"]
    zone = goal_rects(cfg.sim)["top_left"]
    objs = (ObjectSpec("t", ObjectType.LEGO, Pose2D(0.0, 0.0, 0.0), "red"),
            ObjectSpec("o", ObjectType.LEGO, Pose2D(-0.3, 0.3, 0.0), "blue"))
    ep = EpisodeSpec(FIELD_STAGE, Pose2D(0, 0, 0), objs, ("t",), None, "top_right", goal, (zone,))
    a = Assignment(ep, {"t": 1, "o": 2}, TidyInstruction(("lego",), ("red",), (), "top_right"), {"t": RED})
    loose = dataclasses.replace(cfg, task=dataclasses.replace(
        cfg.task, success=dataclasses.replace(cfg.task.success, require_cleanup=False)))
    ctl = Controller(loose, load_perception_cfg(), FixedPolicy(Action.FORWARD), a, _fp(loose))
    d = ctl.decide(_tracked({1: Pose2D(0.3, 0.3, 0.0), 2: Pose2D(-0.12, 0.3, 0.0)}))   # o は置き場の外
    assert d.done


def test_controller_no_completion_while_target_color_is_off(cfg):
    """対象の色（色度）が上限を超えているフレームでは、続いた回数が identity_persist_frames に届く前でも完了にしない
    （2026-10-19、盤面 20029: 挟み損ねたのに補った追跡が、爪を開いた後に近くの別の色の物体に移り、その位置で完了にした）。"""
    fp = _fp(cfg)
    pol = FixedPolicy(Action.FORWARD)
    ctl = Controller(cfg, load_perception_cfg(), pol, _assignment(cfg, Pose2D(0.0, 0.0, 0.0)), fp)
    inside = Pose2D(0.3, 0.3, 0.0)
    d = ctl.decide(_tracked({1: inside}, rgb={1: (60.0, 90.0, 200.0)}))    # ゴールの中だが青い
    assert not d.done and d.action == Action.FORWARD
    d = ctl.decide(_tracked({1: inside}))                                 # 色が戻れば完了
    assert d.done


def test_controller_skips_identity_check_between_open_jaws(cfg):
    """爪を開いていても、対象が爪の中心から identity_skip_open_radius_m 以内なら色を見ない（爪と床の画素が混ざる）。"""
    from toy_sweeper.robot_model import gripper_center_x
    fp = _fp(cfg)
    robot = Pose2D(-0.2, -0.2, 0.0)
    near = Pose2D(-0.2 + gripper_center_x(cfg.sim) + 0.01, -0.2, 0.0)
    ctl = Controller(cfg, load_perception_cfg(), FixedPolicy(Action.FORWARD), _assignment(cfg, near), fp)
    for _ in range(fp.identity_persist_frames + 2):
        d = ctl.decide(_tracked({1: near}, robot=robot, rgb={1: BLACK}))
    assert d.color_drift == 0.0
    far = Pose2D(near.x + 0.05, near.y, 0.0)
    with pytest.raises(FieldStop):
        for _ in range(fp.identity_persist_frames):
            ctl.decide(_tracked({1: far}, robot=robot, rgb={1: BLACK}))


def test_controller_backs_off_to_confirm_color_before_completion(cfg):
    """爪を開いて全対象がゴールに入ったように見えても、対象が爪の中心から identity_skip_open_radius_m 以内（色を見ない範囲）
    なら完了にせず、下がって色を確かめる（2026-10-19、盤面 20029: 挟み損ねて空の爪で運び、開いた爪の中の暗い検出に対象の
    追跡が付いて、色を見ないまま完了にした）。completion_confirm_back_steps 回下がっても爪の中にあれば（ロボットに付いて
    くる偽の追跡）、完了にせず Policy に戻す。"""
    from toy_sweeper.robot_model import gripper_center_x
    fp = _fp(cfg)
    pol = FixedPolicy(Action.FORWARD)
    gx = gripper_center_x(cfg.sim)
    robot = Pose2D(0.3 - gx - 0.01, 0.3, 0.0)          # 爪の中心が対象（top_right の中心）の 1 cm 手前
    inside = Pose2D(0.3, 0.3, 0.0)
    ctl = Controller(cfg, load_perception_cfg(), pol, _assignment(cfg, Pose2D(0.0, 0.0, 0.0)), fp)
    d = ctl.decide(_tracked({1: inside}, robot=robot))
    assert not d.done and d.action == Action.BACKWARD
    ctl.commit(d.action)
    far = Pose2D(robot.x - 0.03, robot.y, 0.0)        # 下がった: 対象は爪の中心から 4 cm
    d = ctl.decide(_tracked({1: inside}, robot=far))
    assert d.done
    # 偽の追跡: 下がっても爪の中にある
    ctl = Controller(cfg, load_perception_cfg(), pol, _assignment(cfg, Pose2D(0.0, 0.0, 0.0)), fp)
    step = 0.01      # 下がる歩数（6、2026-10-28）の間、ゴールの中にとどまる刻み
    for k in range(fp.completion_confirm_back_steps):
        r = Pose2D(robot.x - step * k, robot.y, 0.0)
        d = ctl.decide(_tracked({1: Pose2D(inside.x - step * k, inside.y, 0.0)}, robot=r))
        assert not d.done and d.action == Action.BACKWARD
        ctl.commit(d.action)
    r = Pose2D(robot.x - step * fp.completion_confirm_back_steps, robot.y, 0.0)
    d = ctl.decide(_tracked({1: Pose2D(inside.x - step * fp.completion_confirm_back_steps, inside.y, 0.0)}, robot=r))
    assert not d.done and d.action == Action.FORWARD


def test_controller_backs_off_when_target_unseen_after_opening(cfg):
    """爪を開いて全対象がゴールに入ったように見えるが、対象がこのフレームで見えていない（爪の中で隠れている）ときも、
    下がって確かめる（2026-10-19、盤面 20006・20011: 開く → 見えないので完了にしない → 方策がまた閉じる、を 120 回
    繰り返して時間切れ）。爪を閉じているときは今までどおり Policy を続ける。"""
    fp = _fp(cfg)
    pol = FixedPolicy(Action.CLOSE_GRIPPER)
    inside = Pose2D(0.3, 0.3, 0.0)
    ctl = Controller(cfg, load_perception_cfg(), pol, _assignment(cfg, Pose2D(0.0, 0.0, 0.0)), fp)
    d = ctl.decide(_tracked({1: inside}, missed={1: 1}))
    assert not d.done and d.action == Action.BACKWARD
    ctl.commit(d.action)
    d = ctl.decide(_tracked({1: inside}))           # 下がったら見えた
    assert d.done
    ctl = Controller(cfg, load_perception_cfg(), pol, _assignment(cfg, Pose2D(0.0, 0.0, 0.0)), fp)
    ctl.commit(Action.CLOSE_GRIPPER)
    d = ctl.decide(_tracked({1: inside}, missed={1: 1}))
    assert not d.done and d.action == Action.CLOSE_GRIPPER
    # 下がっても見えなければ、その追跡は爪の中に無い（挟み損ねたのに補った偽の追跡）。見失ったとして止め、仕切り直しに
    # 任せる（盤面 20029: Policy に戻すと、閉じる → 偽の追跡を補う → 開く → 見えない、を繰り返して時間切れ）
    ctl = Controller(cfg, load_perception_cfg(), pol, _assignment(cfg, Pose2D(0.0, 0.0, 0.0)), fp)
    for _ in range(fp.completion_confirm_back_steps):
        d = ctl.decide(_tracked({1: inside}, missed={1: 1}))
        assert d.action == Action.BACKWARD
        ctl.commit(d.action)
    with pytest.raises(FieldStop) as e:
        ctl.decide(_tracked({1: inside}, missed={1: 1}))
    assert e.value.reason == "target_lost"


# ---------- 作業領域の大きさ（v30、実機は校正の大きさ） ----------
def test_assignment_uses_area_size(cfg):
    """実機の作業領域（校正の幅・奥行き）を渡すと、ゴールはその大きさへ写した位置（goal_rects_for_area）、盤面に大きさが入る。"""
    from toy_sweeper.field.runtime import TrackedState, make_assignment, shrink_goal
    from toy_sweeper.task import goal_rects_for_area

    class Ident:
        def identify(self, frame, tracks):
            return {1: ObjectSpec("obj1", ObjectType.LEGO, Pose2D(0.0, 0.0, 0.0), "red")}

    ts = TrackedState(_frame(0.0), Pose2D(0.0, -0.2, 0.0),
                      {1: TrackView(1, Pose2D(0.0, 0.0, 0.0), (0.032, 0.016), (200.0, 30.0, 30.0), 0)}, 0.0)
    ins = TidyInstruction(("lego",), ("red",), (), "top_right")
    size = (0.75, 0.83)
    a = make_assignment(cfg, Ident(), ts, ins, (), 0.01, area_size=size)
    assert a.episode.area_size == size
    assert a.episode.goal == shrink_goal(goal_rects_for_area(cfg.sim, size)["top_right"], 0.01)
    b = make_assignment(cfg, Ident(), ts, ins, (), 0.01)   # 渡さなければ今まで（sim.work_area）
    assert b.episode.area_size is None and b.episode.goal == shrink_goal(goal_rects(cfg.sim)["top_right"], 0.01)


def test_assignment_identifies_only_steady_tracks(cfg):
    """同定（VLM）に出すのは、min_hits フレーム以上見え続けて今も見えている追跡だけ（2026-10-27 実機: 同定し直しで、ロボットの
    爪の先の短命な塊を「青の LEGO」として対象にし、VLM の 1〜2 s の間に消えて仕切り直しをくり返した）。外れた追跡は名前のない
    追跡のまま。"""
    from toy_sweeper.field.runtime import TrackedState, make_assignment
    seen = []

    class Ident:
        def identify(self, frame, tracks):
            seen.append(sorted(tracks))
            return {1: ObjectSpec("obj1", ObjectType.LEGO, Pose2D(0.0, 0.0, 0.0), "red")}
    ts = TrackedState(_frame(0.0), Pose2D(0.0, -0.2, 0.0),
                      {1: TrackView(1, Pose2D(0.0, 0.0, 0.0), (0.032, 0.016), (200.0, 30.0, 30.0), 0, hits=30),
                       2: TrackView(2, Pose2D(0.1, 0.0, 0.0), (0.03, 0.02), (80.0, 60.0, 50.0), 0, hits=3),
                       3: TrackView(3, Pose2D(-0.1, 0.0, 0.0), (0.03, 0.02), (80.0, 60.0, 50.0), 2, hits=40)}, 0.0)
    ins = TidyInstruction(("lego",), ("red",), (), "top_right")
    a = make_assignment(cfg, Ident(), ts, ins, (), 0.01, min_hits=10)
    assert seen == [[1]] and a.track_of == {"obj1": 1}
    make_assignment(cfg, Ident(), ts, ins, (), 0.01)          # 渡さなければ今まで（全部）
    assert seen[-1] == [1, 2, 3]


def test_control_stall_counts_from_control_start(cfg, fcfg):
    """制御の止まりは、制御を始めた時刻から数える（指示の後の同定が遅くても、最初の指令の前に止めない）。2026-10-22、実機で
    同定に 3.9 s かかり、爪を開く指令から数えて「最後の指令から 3.903 s」で止まった。"""
    from toy_sweeper.field.runtime import Supervisor

    class Link:
        def status(self):
            return LinkStatus(True, "接続中")
    frames, tracked = Latest("camera"), Latest("tracked")
    frames.set(_frame(10.0), 10.0)
    tracked.set(object(), 10.0)
    sup = Supervisor(fcfg, frames, tracked, Link(), start=0.0)
    sup.last_command = 6.0          # 爪を開く指令（その後 4 s 同定していた）
    sup.start_control(10.0)
    assert sup.check(10.1) is None
    t = 10.0 + fcfg.control.command_ttl_s + 0.01
    frames.set(_frame(t), t)
    tracked.set(object(), t)
    assert sup.check(t).reason == "control_stalled"   # 始めてからは今までどおり


def test_persist_supervisor_only_stops_on_request(fcfg):
    """止めずに続ける（実機、control.persist）とき、安全監視は古いフレーム・追跡・通信の切れでは止めない（制御の側で待つ・
    つなぎ直す）。停止の要求では止める。"""
    from toy_sweeper.field.runtime import Supervisor
    f = dataclasses.replace(fcfg, control=dataclasses.replace(fcfg.control, persist=True))

    class Link:
        connected = False

        def status(self):
            return LinkStatus(self.connected, "切れた")
    sup = Supervisor(f, Latest("camera"), Latest("tracked"), Link(), start=0.0)
    assert sup.check(100.0) is None             # フレームも追跡も来ない、通信も切れている
    sup.stop_requested = "とまって"
    assert sup.check(100.0).reason == "stop_request"



def test_sim_world_has_physics_rng(cfg, fcfg):
    """実時間のシミュレーションの物理にも乱数を渡す（挟み損ねの有効な設定で、閉じたときに例外になった、2026-10-22）。"""
    w = SimWorld(cfg, fcfg.sim, _open_board(cfg), 3)
    assert w.sim.rng is not None


def test_sim_world_pause_does_not_fall_behind(cfg, fcfg):
    """止めている間（実機を動かしている間のシミュレーション）は物理を進めず、再開しても実時間から遅れたとみなさない（2026-10-22、
    実機と同じアプリで動かしたシミュレーションの物理がカメラの処理に押されて 0.106 s 遅れ、アプリごと止まった）。"""
    w = SimWorld(cfg, fcfg.sim, _open_board(cfg), 0)

    async def body():
        t = asyncio.create_task(w.run())
        await asyncio.sleep(0.2)
        w.pause()
        t_paused = w.time
        await asyncio.sleep(0.5)                         # 止めている間（上限 0.1 s より長い）
        assert w.time == t_paused
        w.resume()
        await asyncio.sleep(0.2)
        assert w.time > t_paused and not t.done()        # 遅れで止まっていない
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
    asyncio.run(body())


def test_camera_reopens_after_read_failure(fcfg, monkeypatch):
    """カメラから一時的に読めないときは、開き直して続ける（2026-10-22、1 回読めずアプリごと止まった）。"""
    import numpy as np
    from toy_sweeper.field import hardware as H
    reads = iter([False, True, True])

    class Cap:
        def read(self):
            return next(reads, True), np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass
    opened = []
    monkeypatch.setattr(H.OpenCVCamera, "_open", lambda self: opened.append(1) or Cap())
    monkeypatch.setattr(H, "CAMERA_REOPEN_WAIT_S", 0.0)
    cam = H.OpenCVCamera(fcfg.camera)

    async def body():
        t = asyncio.create_task(cam.run())
        await cam.frames.wait_newer(1, 2.0)        # 開き直した後に 2 枚来る
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
    asyncio.run(body())
    assert len(opened) == 2


def test_sim_world_idle_lag_resyncs(cfg, fcfg, monkeypatch):
    """動いていない（指令が無い）ときの実時間からの遅れ（MacBook のスリープからの復帰など）は、数え直して続ける。動いている間の遅れは
    これまでどおり止める（2026-10-25、3 日スリープして復帰したときに 0.348 s 遅れ、待っていただけのアプリが終わった）。"""
    import time as T
    from toy_sweeper.field import sim as S
    w = SimWorld(cfg, fcfg.sim, _open_board(cfg), 0)
    real = T.monotonic
    jump = [0.0]
    monkeypatch.setattr(S.time, "monotonic", lambda: real() + jump[0])

    async def body():
        t = asyncio.create_task(w.run())
        await asyncio.sleep(0.1)
        jump[0] = 1.0                                   # スリープからの復帰（時計が飛ぶ）
        await asyncio.sleep(0.2)
        assert not t.done() and w.idle_resyncs >= 1
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
    asyncio.run(body())


def test_policy_stochastic_from_config():
    """現場（scripts/field_app.py）の行動の選び方は設定で決める（2026-10-25、確率的な選び方を
    実機で試せるように）。2026-10-30 から既定は決定的（旋回の切り返しが確率的に引いた
    行動で増えていないかを実機で確かめる。config/field.yaml の根拠）。"""
    import dataclasses
    import importlib.util
    from pathlib import Path
    cfg = load_all()
    f = load_field_cfg(cfg)
    assert f.control.policy_stochastic is False
    spec = importlib.util.spec_from_file_location("field_app", Path(__file__).resolve().parents[1] / "scripts/field_app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    import toy_sweeper.policies as pol
    made = []
    orig = pol.PPOPolicy
    pol.PPOPolicy = lambda model, deterministic, cfg: made.append(deterministic)
    try:
        app.make_policy(Path("m.zip"), cfg, dataclasses.replace(f, control=dataclasses.replace(f.control, policy_stochastic=True)))
        app.make_policy(Path("m.zip"), cfg, dataclasses.replace(f, control=dataclasses.replace(f.control, policy_stochastic=False)))
    finally:
        pol.PPOPolicy = orig
    assert made == [False, True]


def test_camera_index_by_name():
    """カメラの番号は名前（config/field.yaml の camera.device_name）から、開くたびに引く（2026-10-27、毎回番号を
    確かめる手間をなくすため）。OpenCV（AVFoundation）の番号は、カメラの一覧を固有の ID で
    並べた順（2026-10-27 実測: 一覧の 0 = C920・1 = 内蔵・2 = iPhone のとき、ID 順で 1 番の C920 が 10 fps を受け付け、0 番の iPhone
    は 30 fps を返した）。どのカメラも開かずに決まる。"""
    import pytest
    from toy_sweeper.field.hardware import camera_index
    devs = [("0x1130000046d08e5", "HD Pro Webcam C920"), ("6C707041-05AC", "MacBook Airカメラ"),
            ("04F11748-959D", "iPhoneのカメラ")]
    assert camera_index("HD Pro Webcam C920", devs) == 1
    assert camera_index("HD Pro Webcam C920", devs[:2]) == 0          # iPhone が無ければ番号が変わる
    with pytest.raises(RuntimeError, match="見つからない"):
        camera_index("HD Pro Webcam C920", devs[1:])                  # 抜けている
    with pytest.raises(RuntimeError, match="2 台"):
        camera_index("HD Pro Webcam C920", devs + [("0x2", "HD Pro Webcam C920")])


def test_gazebo_command_uses_training_overrides(tmp_path):
    """3090 の Gazebo（talk_sim.py）に、方策を学習したときの設定の差し替え（run_info.json の overrides）を渡す（2026-10-27、
    渡さずに起動して「observation.edge_distance: corners で学習したが、今の設定は center」で落ちた）。記録が無ければ例外。"""
    import importlib.util
    import json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("field_app", Path(__file__).resolve().parents[1] / "scripts/field_app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    run = app.ROOT / "checkpoints" / "_test_gazebo_cmd"
    run.mkdir(parents=True, exist_ok=True)
    try:
        (run / "run_info.json").write_text(json.dumps({"overrides": ["config/experiments/a.yaml", "config/experiments/b.yaml"]}))
        cmd = app.gazebo_command(run / "final.zip", True, 7, 20000)
        assert "--model checkpoints/_test_gazebo_cmd/final.zip --stochastic" in cmd
        assert "--override config/experiments/a.yaml config/experiments/b.yaml" in cmd
        (run / "run_info.json").unlink()
        with pytest.raises(FileNotFoundError):
            app.gazebo_command(run / "final.zip", True, 7, 20000)
    finally:
        for f in run.iterdir():
            f.unlink()
        run.rmdir()


def test_perception_rejects_impossible_yaw_jump(cfg):
    """ロボットの向きが、その間に回れる量を大きく超えて跳んだ（実機: マーカーの向きの読み違い）フレームは姿勢が取れなかったものと
    して扱い、直前の指令からの見積もりで埋める（2026-10-26、実機で向きが 1 フレームで約 150° 跳び、直前 1 歩の回転が Observation の
    値域の外になってアプリごと止まった）。同じ跳びが JUMP_ACCEPT_FRAMES 続いたら本当に向きが変わった（持ち上げて置き直した）とみなす。"""
    import numpy as np
    from toy_sweeper.field import runtime as R
    from toy_sweeper.field.interfaces import Frame
    from toy_sweeper.geometry import Pose2D
    from toy_sweeper.perception import load_perception_cfg
    yaws = iter([0.0, 0.02, 2.6, 0.04] + [2.6] * R.JUMP_ACCEPT_FRAMES)

    class Pose:
        def estimate(self, frame):
            return Pose2D(0.0, 0.0, next(yaws))
    p = R.Perception(cfg, load_perception_cfg(), None, Pose(), None, lambda t: None, R.TimingStats(),
                     detector=lambda f, r, j: [], predict=lambda pose, t0, t1: pose, predict_max_s=0.5,
                     max_yaw_rate_radps=cfg.sim.robot.angular_speed_radps)
    img = np.zeros((4, 4, 3), np.uint8)
    got = [p._process(Frame(img, 0.1 * i, i), R.GripperState.OPEN)[0].yaw for i in range(4 + R.JUMP_ACCEPT_FRAMES)]
    assert got[:4] == [0.0, 0.02, 0.02, 0.04]                       # 跳んだフレームは直前の姿勢から見積もる
    assert got[-1] == 2.6                                           # 続いたら受け入れる


def test_controller_opens_after_missed_grasp(cfg):
    """爪を閉じていて、主対象が見えていて、学習の挟み損ね（弾かれて kick_m の上限まで滑る）より遠いときは、Policy より先に爪を
    開く（2026-10-27 実機: 挟み損ねた緑の LEGO が爪の後ろ 12 cm に残り、閉じた爪の Policy は前進 100% で空のままゴールへ
    向かった。学習で見たことのない場面）。弾かれた範囲の内なら Policy に任せる（学習で見た場面）。"""
    import math
    from toy_sweeper.robot_model import gripper_center_x
    robot = Pose2D(-0.2, -0.2, 0.0)
    gc = robot.transform_point((gripper_center_x(cfg.sim), 0.0))
    reach = cfg.sim.sim2d.grasp_fail.kick_m[1] + max(o.size_m[0] for o in cfg.sim.objects.values()) / 2
    far = Pose2D(gc[0] - (reach + 0.03), gc[1], 0.0)          # 爪の後ろ、学習で弾かれる範囲の外
    near = Pose2D(gc[0] + 0.03, gc[1] + 0.01, 0.0)            # 爪の前 3 cm（学習で弾かれる範囲の内）
    for pose, want in ((far, Action.OPEN_GRIPPER), (near, Action.FORWARD)):
        ctl = Controller(cfg, load_perception_cfg(), FixedPolicy(Action.FORWARD), _assignment(cfg, pose), _fp(cfg))
        ctl.gripper = GripperState.CLOSED
        assert ctl.decide(_tracked({1: pose}, robot=robot)).action == want, (pose, math.dist(pose.xy, gc))
    # 開いていれば Policy のまま
    ctl = Controller(cfg, load_perception_cfg(), FixedPolicy(Action.FORWARD), _assignment(cfg, far), _fp(cfg))
    assert ctl.decide(_tracked({1: far}, robot=robot)).action == Action.FORWARD


def test_to_now_moves_robot_and_held_objects(cfg):
    """観測の古さを埋める（2026-10-27 実機: 方策の見る姿勢が平均 0.2 s 古く、学習では 0。旋回のたびに 50〜90° 行き過ぎて左右に
    振れた）。写真の時刻の姿勢を今の時刻まで進め、閉じた爪の間の物はロボットといっしょに動かす。床の物はそのまま。"""
    import math
    from toy_sweeper.field.runtime import to_now
    from toy_sweeper.robot_model import gripper_center_x
    robot = Pose2D(0.0, 0.0, 0.0)
    gc = robot.transform_point((gripper_center_x(cfg.sim), 0.0))
    held, floor = Pose2D(gc[0], gc[1], 0.0), Pose2D(0.2, 0.2, 0.0)
    ts = _tracked({1: held, 2: floor}, robot=robot)
    turned = Pose2D(0.0, 0.0, math.pi / 2)
    out = to_now(ts, turned, GripperState.CLOSED, cfg)
    assert out.robot == turned and out.tracks[2].pose == floor
    assert out.tracks[1].pose.x == pytest.approx(0.0, abs=1e-9) and out.tracks[1].pose.y == pytest.approx(gc[0])
    assert out.tracks[1].pose.yaw == pytest.approx(math.pi / 2)
    out = to_now(ts, turned, GripperState.OPEN, cfg)      # 開いた爪では運ばない
    assert out.tracks[1].pose == held


def test_command_lead_is_recent_send_time():
    """先読みの終点は「今」ではなく「今 + 指令が効くまで」。指令の送り（BLE、実機で平均 57〜120 ms）の最近の平均を足す
    （2026-10-28、Fable のレビュー: 青の本は送りが 120 ms で 1 周期ぶん遅れて効き、旋回が行き過ぎた）。まだ送っていなければ 0。"""
    from toy_sweeper.field.runtime import TimingStats, command_lead_s
    st = TimingStats()
    assert command_lead_s(st) == 0.0
    for v in [0.5] * 10 + [0.1] * 16:          # 古い値は見ない（最近の 16 回）
        st.add("send_s", v)
    assert command_lead_s(st) == pytest.approx(0.1)


def test_identity_count_restarts_after_stop(cfg):
    """色が変わったとして止めた（仕切り直し）後は、続いたフレーム数を 0 から数え直す（2026-10-30 実機 11:18:59〜11:19:02:
    数が 3・4・5・6 と残り、仕切り直しのたびに次のフレームですぐまた仕切り直して、1 秒おきに 4 回下がった）。"""
    fp = _fp(cfg)
    robot = Pose2D(-0.2, -0.2, 0.0)
    far = Pose2D(0.2, 0.2, 0.0)
    ctl = Controller(cfg, load_perception_cfg(), FixedPolicy(Action.FORWARD), _assignment(cfg, far), fp)
    with pytest.raises(FieldStop):
        for _ in range(fp.identity_persist_frames):
            ctl.decide(_tracked({1: far}, robot=robot, rgb={1: BLACK}))
    for _ in range(fp.identity_persist_frames - 1):
        ctl.decide(_tracked({1: far}, robot=robot, rgb={1: BLACK}))
    with pytest.raises(FieldStop):
        ctl.decide(_tracked({1: far}, robot=robot, rgb={1: BLACK}))


def test_controller_treats_impossible_one_step_motion_as_pose_jump(cfg):
    """1 周期の間の姿勢の差が Observation の値域を超える（1 歩では動けない）ときは、マーカーの読み違い・待った後の跳びとして
    「直前 1 歩の動き」を 0 にして続ける（2026-10-30 実機 12:44:37: 同定し直しの後、1 周期で向きが −103° 変わり、Observation の
    値域の外で指示ごとエラーで終わった）。ふつうの 1 歩の動きはそのまま渡す。"""
    pcfg = load_perception_cfg()
    far = Pose2D(0.2, 0.2, 0.0)
    ctl = Controller(cfg, pcfg, FixedPolicy(Action.FORWARD), _assignment(cfg, far), _fp(cfg))
    r0 = Pose2D(-0.2, -0.2, 0.0)
    ctl.decide(_tracked({1: far}, robot=r0))
    small = Pose2D(-0.2, -0.2, 0.1)
    d = ctl.decide(_tracked({1: far}, robot=small))
    assert d.snapshot.robot_motion[2] == pytest.approx(0.1)
    jump = Pose2D(-0.19, -0.21, 0.1 - 1.79)
    d = ctl.decide(_tracked({1: far}, robot=jump))
    assert d.snapshot.robot_motion == (0.0, 0.0, 0.0) and ctl.motion_jumps == 1


def test_step_toward_center():
    """マーカーが読めないまま記録した動きを戻し終えたら、最後に見えた姿勢から作業領域の中心（原点）へ近づく向きに前進か後退
    する（2026-10-30 実機 13:04: 奥の隅で画面の外へ出て、6 歩戻っても見えず止まって待ち続けた）。"""
    from toy_sweeper.field.runtime import step_toward_center
    assert step_toward_center(Pose2D(-0.3, 0.27, math.radians(77))) == Action.BACKWARD   # 奥の隅で奥を向いている
    assert step_toward_center(Pose2D(-0.3, 0.27, math.radians(-100))) == Action.FORWARD  # 手前を向いている
    assert step_toward_center(Pose2D(0.3, 0.0, 0.0)) == Action.BACKWARD
    assert step_toward_center(Pose2D(0.3, 0.0, math.pi)) == Action.FORWARD


def test_predict_pose_with_motion_model():
    """姿勢の先読み（FieldRuntime._predict_pose）に実機の動き方（遅れ・時定数・倍率、2026-10-30 に telemetry.jsonl から割り出した）
    を入れる。理想（遅れ 0・惰性なし・倍率 1）なら今までどおり、指令はその場から効く。"""
    from types import SimpleNamespace
    from toy_sweeper.field.runtime import FieldRuntime, IDEAL_MOTION, MotionModel
    rt = SimpleNamespace(f=SimpleNamespace(control=SimpleNamespace(command_ttl_s=0.5)), motion_model=IDEAL_MOTION,
                         _drive_log=[(10.0, 0.0, 2.0), (10.125, 0.0, 0.0)])
    pred = lambda t1: FieldRuntime._predict_pose(rt, Pose2D(0.0, 0.0, 0.0), 10.0, t1)
    assert pred(10.1).yaw == pytest.approx(0.2)
    assert pred(10.5).yaw == pytest.approx(0.25)
    rt.motion_model = MotionModel(delay_s=0.12, lag_s=0.08, gain_linear=1.0, gain_angular=1.1)
    assert abs(pred(10.1).yaw) < 1e-3                             # 遅れの間は動かない
    assert pred(10.25).yaw < 0.25 * 1.1 * 0.8                     # 止めた後も惰性で回り続けている
    assert pred(11.0).yaw == pytest.approx(0.25 * 1.1, abs=0.005)  # 回った量は倍率 × 指令
    rt._drive_log = [(9.0, 0.1, 0.0), (9.4, 0.1, 0.0), (9.8, 0.1, 0.0)]   # t0 より前から続く前進（速さは追いついている）
    p = FieldRuntime._predict_pose(rt, Pose2D(0.0, 0.0, 0.0), 9.5, 9.7)
    assert p.x == pytest.approx(0.1 * 0.2, abs=0.002)


def test_identity_allows_same_object_color_shift_far_from_others(cfg):
    """同じ物の色が置き場所で上限を少し超えて変わっても、ほかの同定した物の色に近づいていなければ止めない（2026-10-30 実機:
    ゴールに置いた赤い LEGO が、同定したときから 16 変わり（上限 14）、ほかの物とは 49 以上離れていたのに、入れ替わりとみなして
    仕切り直しをくり返した）。上限の 2 倍を超える変化（黒など）は今までどおり止める。"""
    a = _assignment(cfg, Pose2D(0.0, 0.0, 0.0), Pose2D(0.2, 0.2, 0.0))
    a.ref_rgb["o"] = (60.0, 60.0, 200.0)
    ctl = Controller(cfg, load_perception_cfg(), FixedPolicy(Action.FORWARD), a, _fp(cfg))
    fp = _fp(cfg)
    shifted = (220.0, 64.0, 50.0)       # RED から色度 11.8（試験の設定の上限 8 と 16 の間）、ほかの物（青）からは遠い
    assert fp.identity_chroma_tolerance < chroma_distance(shifted, RED) < 2 * fp.identity_chroma_tolerance
    tracks = {1: Pose2D(0.0, 0.0, 0.0), 2: Pose2D(0.2, 0.2, 0.0)}
    for _ in range(fp.identity_persist_frames + 2):
        ctl.decide(_tracked(tracks, rgb={1: shifted, 2: a.ref_rgb["o"]}))
    with pytest.raises(FieldStop):
        for _ in range(fp.identity_persist_frames):
            ctl.decide(_tracked(tracks, rgb={1: BLACK, 2: a.ref_rgb["o"]}))


def test_runtime_feeds_and_uses_online_motion_gain():
    """送った走行の指令と、マーカーで読めた姿勢（見積もった姿勢は渡さない）を倍率の見積もりに渡し、先読みの動き方を
    見積もりの今の値にする（2026-10-30、電池で速さが ±3 割変わる）。"""
    from types import SimpleNamespace
    from toy_sweeper.field.runtime import FieldRuntime, MotionModel
    got = []
    est = SimpleNamespace(add_send=lambda *a: got.append(("send", a)), add_pose=lambda *a: got.append(("pose", a)),
                          model=lambda: MotionModel(0.075, 0.1, 1.3, 1.25))
    rt = SimpleNamespace(motion_gain=est, motion_model=MotionModel(0.075, 0.1, 1.1, 1.0), _drive_log=[],
                         telemetry=SimpleNamespace(publish=lambda kind, d: None))
    FieldRuntime._note_pose(rt, {"stamp": 1.0, "seq": 3, "x": 0.1, "y": 0.2, "yaw": 0.3, "measured": True})
    FieldRuntime._note_pose(rt, {"stamp": 1.1, "seq": 4, "x": 0.1, "y": 0.2, "yaw": 0.3, "measured": False})
    FieldRuntime._note_drive(rt, 2.0, 0.145, 0.0)
    assert got == [("pose", (1.0, 0.1, 0.2, 0.3)), ("send", (2.0, 0.145, 0.0))] and rt._drive_log == [(2.0, 0.145, 0.0)]
    FieldRuntime._refresh_motion(rt)
    assert rt.motion_model == MotionModel(0.075, 0.1, 1.3, 1.25)
