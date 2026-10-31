import math
import sys

import pytest

from toy_sweeper.config import load_all
from toy_sweeper.geometry import Pose2D
from toy_sweeper.robot_model import front_x, gripper_center_x
from toy_sweeper.sim2d import ObjectSpec, Sim2D
from toy_sweeper.spaces import GripperState, ObjectType
from pathlib import Path

TEST_CFG = Path(__file__).resolve().parent / "configs"   # テストだけで使う設定の差し替え


@pytest.fixture
def cfg():
    return load_all()


@pytest.fixture
def sim(cfg):
    return Sim2D(cfg.sim)


# 学習中の例外の記録（test/data/crash_*.txt）は 3090（Linux）で作られ、乱数から引き直した盤面とビット単位で比べる。MacBook
# （macOS arm64）では、同じ乱数の列でも位置・角度の最後の 1 桁（1e-17 ほど）が違って一致しない（2026-10-22。数学関数の
# 実装の違いと見ているが、どの関数かは未確認）。学習の物理は Linux で確かめるので、Linux 以外では飛ばす。
_crash_replay_linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="crash_*.txt は Linux で作った記録で、盤面をビット単位で比べる（macOS では最後の 1 桁が違う）")


def _lego(pose, name="lego0"):
    return ObjectSpec(name=name, type=ObjectType.LEGO, pose=pose, color="red")


def test_forward_and_turn(sim, cfg):
    sim.reset(Pose2D(0, 0, 0), [], GripperState.OPEN)
    sim.drive(cfg.sim.robot.linear_speed_mps, 0.0, 1.0)
    p = sim.robot_pose()
    # 駆動力の上限（タイヤの摩擦）による発進の加速時間（約 8 ms）の分だけ短い
    assert math.isclose(p.x, cfg.sim.robot.linear_speed_mps, rel_tol=0.02) and abs(p.y) < 1e-6
    sim.drive(0.0, 1.0, 0.5)
    assert math.isclose(sim.robot_pose().yaw, 0.5, rel_tol=0.03)


def test_duration_must_be_multiple_of_dt(sim):
    sim.reset(Pose2D(0, 0, 0), [], GripperState.OPEN)
    with pytest.raises(ValueError):
        sim.drive(0.1, 0.0, 0.001)


def test_untouched_object_stays(sim):
    sim.reset(Pose2D(0, 0, 0), [_lego(Pose2D(0.2, 0.2, 0.3))], GripperState.OPEN)
    sim.drive(0.0, 0.0, 1.0)
    p = sim.object_pose("lego0")
    assert abs(p.x - 0.2) < 1e-6 and abs(p.y - 0.2) < 1e-6 and abs(p.yaw - 0.3) < 1e-6


def test_push_moves_object_forward_and_it_stops(sim, cfg):
    # 開いた爪の間に入るよう正面に置き、前進して押す
    x0 = gripper_center_x(cfg.sim)
    sim.reset(Pose2D(0, 0, 0), [_lego(Pose2D(x0, 0.0, math.pi / 2))], GripperState.OPEN)
    sim.drive(cfg.sim.robot.linear_speed_mps, 0.0, 2.0)
    p = sim.object_pose("lego0")
    assert p.x > x0 + 0.1, p
    assert abs(p.y) < 0.02
    # 止めると摩擦で物体も止まる
    sim.drive(0.0, 0.0, 0.25)
    before = sim.object_pose("lego0")
    sim.drive(0.0, 0.0, 0.5)
    after = sim.object_pose("lego0")
    assert math.hypot(after.x - before.x, after.y - before.y) < 1e-4


def test_grasp_carry_release(sim, cfg):
    x0 = gripper_center_x(cfg.sim)
    sim.reset(Pose2D(0, 0, 0), [_lego(Pose2D(x0, 0.0, 0.0))], GripperState.OPEN)
    sim.actuate_gripper(close=True)
    assert sim.gripper_state() == GripperState.CLOSED
    assert sim.grasped_object() == "lego0"
    sim.drive(-cfg.sim.robot.linear_speed_mps, 0.0, 1.0)   # 後退しても付いてくる
    p = sim.object_pose("lego0")
    assert p.x < x0 - 0.05
    # 爪は LEGO の幅（長辺 32 mm が前後、短辺 16 mm が左右）に触れる角度で止まり、閉じ切らない
    assert cfg.sim.gripper.jaw_closed_angle_rad <= sim.jaw_angle() < cfg.sim.gripper.jaw_open_angle_rad
    sim.actuate_gripper(close=False)
    assert sim.gripper_state() == GripperState.OPEN and sim.grasped_object() is None
    sim.drive(-cfg.sim.robot.linear_speed_mps, 0.0, 1.0)   # 放した後は付いてこない
    q = sim.object_pose("lego0")
    assert abs(q.x - p.x) < 0.01


def test_close_without_object(sim):
    sim.reset(Pose2D(0, 0, 0), [], GripperState.OPEN)
    sim.actuate_gripper(close=True)
    assert sim.gripper_state() == GripperState.CLOSED and sim.grasped_object() is None


def test_reset_is_clean(sim, cfg):
    sim.reset(Pose2D(0, 0, 0), [_lego(Pose2D(0.1, 0, 0))], GripperState.OPEN)
    sim.drive(cfg.sim.robot.linear_speed_mps, 0.0, 1.0)
    sim.reset(Pose2D(0.1, 0.1, 1.0), [_lego(Pose2D(-0.1, 0, 0))], GripperState.CLOSED)
    assert sim.robot_pose() == Pose2D(0.1, 0.1, 1.0)
    assert sim.object_pose("lego0").x == pytest.approx(-0.1)
    assert sim.gripper_state() == GripperState.CLOSED


def test_turn_in_place_pivots_about_axle(sim, cfg):
    # 旋回中心は車軸中点。車体中心は車軸から前方にずれているので、その場旋回で車体中心は円を描く
    from toy_sweeper.robot_model import body_center_x
    sim.reset(Pose2D(0, 0, 0), [], GripperState.OPEN)
    sim.drive(0.0, 1.0, 1.0)
    p = sim.robot_pose()
    assert abs(p.x) < 1e-3 and abs(p.y) < 1e-3
    assert body_center_x(cfg.sim) == pytest.approx(-cfg.sim.robot.axle_offset_x_m)



def test_toy_car_rolls_along_but_not_across(cfg):
    # ミニカーは前後に転がりやすく、横には滑りにくい
    from toy_sweeper.sim2d import ObjectSpec as OS
    dists = {}
    for name, yaw in (("along", 0.0), ("across", math.pi / 2)):
        sim = Sim2D(cfg.sim)
        sim.reset(Pose2D(-0.2, 0, 0), [OS("car", ObjectType.TOY_CAR, Pose2D(0.0, 0.0, yaw), "blue")], GripperState.OPEN)
        sim._objects["car"][0].velocity = (0.3, 0.0)   # 同じ初速で x 方向へ弾く
        sim.drive(0.0, 0.0, 2.0)
        dists[name] = sim.object_pose("car").x
    assert dists["along"] > 5 * dists["across"], dists


def test_robot_cannot_tunnel_object_through_wall(sim, cfg):
    half = cfg.sim.work_area.size_m[0] / 2
    x0 = gripper_center_x(cfg.sim)
    sim.reset(Pose2D(half - x0 - 0.1, 0.0, 0.0), [_lego(Pose2D(half - 0.1, 0.0, 0.0))], GripperState.CLOSED)
    sim.drive(cfg.sim.robot.linear_speed_mps, 0.0, 5.0)
    o = sim.object_pose("lego0")
    assert o.x < half and sim.robot_pose().x < half


def test_grasped_object_stays_inside_wall(sim, cfg):
    # 挟んだ LEGO を壁に押し付けても、壁にめり込まない（2026-10-16、学習中に壁の外へ出て例外になった）
    half = cfg.sim.work_area.size_m[0] / 2
    x0 = gripper_center_x(cfg.sim)
    sim.reset(Pose2D(0.2, 0.0, 0.0), [_lego(Pose2D(0.2 + x0, 0.0, 0.0))], GripperState.OPEN)
    sim.actuate_gripper(close=True)
    assert sim.grasped_object() == "lego0"
    sim.drive(cfg.sim.robot.linear_speed_mps, 0.0, 3.0)
    o = sim.object_pose("lego0")
    size = cfg.sim.objects["lego"].size_m
    assert o.x + size[0] / 2 < half + 0.002, o   # めり込みは 2 mm 未満


def test_open_jaws_wedge_lego_against_wall(sim, cfg):
    # 開いた爪の V 字と壁の間に LEGO（長辺を壁と平行）を挟んで押し続けても、めり込まない
    # （2026-10-16、学習中に LEGO が壁へ 8 mm めり込み、中心が壁の外に出た状況の再現）
    half = cfg.sim.work_area.size_m[0] / 2
    x0 = gripper_center_x(cfg.sim)
    sim.reset(Pose2D(half - x0 - 0.06, 0.0, 0.0), [_lego(Pose2D(half - 0.05, 0.0, math.pi / 2))], GripperState.OPEN)
    sim.drive(cfg.sim.robot.linear_speed_mps, 0.0, 3.0)
    o = sim.object_pose("lego0")
    half_x = cfg.sim.objects["lego"].size_m[1] / 2
    assert o.x + half_x < half + 0.002, o


def test_chassis_contacts_count_body_not_jaws(sim, cfg):
    """車体の形（アームは除く）に触れた物体を数える（v19）。開いたアームの間で押した物体は数えない。"""
    gx = gripper_center_x(cfg.sim)
    half_w = cfg.sim.robot.body_size_m[1] / 2
    # 車体の前面の右の角の前にミニカー（アームの外）、アームの間に LEGO
    car = ObjectSpec("car", ObjectType.TOY_CAR, Pose2D(front_x(cfg.sim) + 0.02, -(half_w - 0.008), 0.0), "red")
    lego = ObjectSpec("lego", ObjectType.LEGO, Pose2D(gx + 0.005, 0.0, math.pi / 2), "blue")
    sim.reset(Pose2D(0.0, 0.0, 0.0), [car, lego], GripperState.OPEN)
    sim.drive(cfg.sim.robot.linear_speed_mps, 0.0, 0.5)
    assert sim.chassis_contacts() == {"car"}
    sim.drive(0.0, 0.0, 0.25)            # 動かなければ触れない（1 Action ごとに数え直す）
    assert sim.object_pose("lego").x > gx  # LEGO はアームで押されている


def _grasp_cfg(cfg, **kw):
    import dataclasses
    gf = dataclasses.replace(cfg.sim.sim2d.grasp_fail, enabled=True, **kw)
    return dataclasses.replace(cfg.sim, sim2d=dataclasses.replace(cfg.sim.sim2d, grasp_fail=gf))


def test_grasp_failure_kicks_object(cfg):
    """挟み損ね（v21）: 失敗の確率 1 なら、爪の間の LEGO でもつながず、爪の中心から弾く。0 なら中央の LEGO はつかむ。"""
    import numpy as np
    gx = gripper_center_x(cfg.sim)
    lego = ObjectSpec("lego", ObjectType.LEGO, Pose2D(gx, 0.004, math.pi / 2), "red")
    sim = Sim2D(_grasp_cfg(cfg, base_prob=1.0, kick_m=(0.03, 0.03)))
    sim.rng = np.random.default_rng(0)
    sim.reset(Pose2D(0.0, 0.0, 0.0), [lego], GripperState.OPEN)
    sim.actuate_gripper(True)
    assert sim.grasped_object() is None
    p = sim.object_pose("lego")
    assert math.dist(p.xy, (gx, 0.004)) > 0.02        # 3 cm 滑る速さで斜め前に弾かれた
    assert p.x > gx                                    # 前へ
    ok = Sim2D(_grasp_cfg(cfg, base_prob=0.0))
    ok.rng = np.random.default_rng(0)
    ok.reset(Pose2D(0.0, 0.0, 0.0), [ObjectSpec("lego", ObjectType.LEGO, Pose2D(gx, 0.0, math.pi / 2), "red")],
             GripperState.OPEN)
    ok.actuate_gripper(True)
    assert ok.grasped_object() == "lego"


def test_grasp_failure_needs_rng(cfg):
    sim = Sim2D(_grasp_cfg(cfg))
    sim.reset(Pose2D(0.0, 0.0, 0.0), [ObjectSpec("lego", ObjectType.LEGO, Pose2D(gripper_center_x(cfg.sim), 0.0, 0.0), "red")],
              GripperState.OPEN)
    with pytest.raises(RuntimeError):
        sim.actuate_gripper(True)


# ---------- 爪をサーボの速さで回す ----------

def _step_jaws(sim, n):
    """物理の刻みを 1 つずつ n 回進め、各刻みの後の爪の角度を返す。"""
    out = []
    for _ in range(n):
        sim.drive(0.0, 0.0, sim.dt)
        out.append(sim.jaw_angle())
    return out


def test_jaw_opens_at_servo_speed(sim, cfg):
    """開閉の指令で爪の角度はすぐには変わらず、1 刻みに jaw_speed_radps × dt ずつ回り、最後は目標の角度ちょうどで止まる。"""
    g = cfg.sim.gripper
    step = g.jaw_speed_radps * sim.dt
    n = math.ceil((g.jaw_open_angle_rad - g.jaw_closed_angle_rad) / step)
    assert n == 9                                    # 0.716 rad / 0.0873 rad = 8.2 → 9 刻み（0.075 s）
    sim.reset(Pose2D(0, 0, 0), [], GripperState.CLOSED)
    sim.set_gripper(False)
    assert sim.gripper_state() == GripperState.OPEN   # 状態は指令した状態
    assert sim.jaw_angle() == g.jaw_closed_angle_rad  # 角度はまだ動いていない
    angles = _step_jaws(sim, n + 2)
    for k in range(n - 1):
        assert angles[k] == pytest.approx(g.jaw_closed_angle_rad + (k + 1) * step, abs=1e-12), k
    assert angles[n - 1:] == [g.jaw_open_angle_rad] * 3
    # 閉じるのも同じ速さ
    sim.set_gripper(True)
    angles = _step_jaws(sim, n + 1)
    assert angles[0] == pytest.approx(g.jaw_open_angle_rad - step, abs=1e-12)
    assert angles[n - 2] > g.jaw_closed_angle_rad and angles[n - 1:] == [g.jaw_closed_angle_rad] * 2


def test_grasp_attaches_when_jaws_arrive(sim, cfg):
    """爪の間の中央の LEGO: 閉じ始めにはまだつながず、触れる角度に着いた刻みでつなぐ。爪はその角度で止まる。"""
    from toy_sweeper.robot_model import angle_touching
    g = cfg.sim.gripper
    x0 = gripper_center_x(cfg.sim)
    sim.reset(Pose2D(0, 0, 0), [_lego(Pose2D(x0, 0.0, 0.0))], GripperState.OPEN)
    sim.set_gripper(True)
    assert sim.gripper_state() == GripperState.CLOSED and sim.grasped_object() is None
    touch = angle_touching(cfg.sim, x0, cfg.sim.objects["lego"].size_m[1] / 2)
    n = math.ceil((g.jaw_open_angle_rad - touch) / (g.jaw_speed_radps * sim.dt))
    _step_jaws(sim, n - 1)
    assert sim.grasped_object() is None and sim.jaw_angle() > touch
    _step_jaws(sim, 1)
    assert sim.grasped_object() == "lego0"
    # 着いた時点の物体の位置で決め直した角度（閉じる爪に触れて LEGO が少し動くので、閉じ始めの角度とはわずかに違う）
    assert sim.jaw_angle() == pytest.approx(touch, abs=1e-3)
    a = sim.jaw_angle()
    _step_jaws(sim, 5)
    assert sim.jaw_angle() == a                                  # 触れた角度のまま
    # 開くときは開き始めに離す
    sim.set_gripper(False)
    assert sim.grasped_object() is None and sim.jaw_angle() == a


def test_target_pushed_out_while_closing_is_not_attached(sim, cfg):
    """閉じている間に対象が爪の間から出たら、着いてもつながず、閉じ切る。"""
    g = cfg.sim.gripper
    x0 = gripper_center_x(cfg.sim)
    sim.reset(Pose2D(0, 0, 0), [_lego(Pose2D(x0, 0.0, 0.0))], GripperState.OPEN)
    sim.set_gripper(True)
    sim._objects["lego0"][0].position = (x0 + 0.08, 0.0)      # 閉じ始めの後に、爪の先より前へ動かす
    sim.drive(0.0, 0.0, g.actuation_time_s)
    assert sim.grasped_object() is None
    assert sim.jaw_angle() == g.jaw_closed_angle_rad
    assert sim.gripper_state() == GripperState.CLOSED


def test_actuate_gripper_raises_when_jaws_cannot_finish(cfg):
    """actuation_time_s の間に回り切らない速さは、設定の検査と actuate_gripper の両方で例外にする（黙って途中で止めない）。"""
    import dataclasses
    from toy_sweeper.config import ConfigError, validate
    slow = dataclasses.replace(cfg.sim, gripper=dataclasses.replace(cfg.sim.gripper, jaw_speed_radps=1.0))
    with pytest.raises(ConfigError, match="jaw_speed_radps"):
        validate(dataclasses.replace(cfg, sim=slow))
    sim = Sim2D(slow)
    sim.reset(Pose2D(0, 0, 0), [], GripperState.CLOSED)
    with pytest.raises(RuntimeError, match="actuation_time_s"):
        sim.actuate_gripper(False)


def _max_penetration(sim) -> float:
    """ロボットの形と物体の接触の、最も深いめり込み（m、正の値）。"""
    out = [0.0]

    def f(arb):
        out.extend(-p.distance for p in arb.contact_point_set.points)
    sim._robot.each_arbiter(f)
    return max(out)


def test_failed_grasp_does_not_bury_jaws_in_lego(cfg):
    """挟み損ねて閉じ切る爪と、爪の間の斜めの LEGO（下の再生の 236 歩目の盤面を、ロボット座標で同じに置いたもの）。
    爪を一瞬で閉じていたときは、爪が LEGO に 16 mm めり込んだまま抜けず（1 s 後も 16 mm）、閉じ込められた LEGO が
    300 歩後の開閉でロボットを回した。サーボの速さで閉じると、爪は LEGO を押して整え、めり込みは解ける。"""
    import numpy as np
    sim = Sim2D(_grasp_cfg(cfg, base_prob=1.0, kick_m=(0.001, 0.001)))
    sim.rng = np.random.default_rng(0)
    gx = gripper_center_x(cfg.sim)
    sim.reset(Pose2D(0.0, 0.0, 0.0), [_lego(Pose2D(gx - 0.0085, 0.0071, -2.218))], GripperState.OPEN)
    sim.actuate_gripper(True)
    assert sim.grasped_object() is None and sim.jaw_angle() == cfg.sim.gripper.jaw_closed_angle_rad
    sim.drive(0.0, 0.0, 1.0)
    assert _max_penetration(sim) < 0.002


def test_replay_open_gripper_spin_case(cfg):
    """2026-10-18 の評価（v20gb、不完全さあり、Stage 7 Seed 10063）の行動列の再生。爪を一瞬で開いていたときは、
    566・586・589 歩目の OPEN で 0.4〜0.6 rad 回り、600 歩目で観測の値域を超えて例外になった（236 歩目の挟み損ねで
    一瞬で閉じた爪が LEGO に 16 mm めり込み、閉じ込めたまま運んでいた。上の試験）。test/data/jaw_spin_replay_acts.txt"""
    from pathlib import Path
    from toy_sweeper.env import ToySweeperEnv
    from toy_sweeper.spaces import Action
    root = Path(__file__).resolve().parents[1]
    c = load_all(overrides=(root / "config/experiments/set_encoder.yaml", root / "config/experiments/imperfect.yaml"))
    acts = [int(a) for a in (Path(__file__).parent / "data" / "jaw_spin_replay_acts.txt").read_text().split(",")]
    env = ToySweeperEnv(c, stage=7)
    env.reset(seed=10063)
    worst = 0.0
    for a in acts:
        y0 = env.backend.robot_pose().yaw
        _, _, term, trunc, _ = env.step(a)
        if a in (Action.OPEN_GRIPPER, Action.CLOSE_GRIPPER):
            worst = max(worst, abs(math.remainder(env.backend.robot_pose().yaw - y0, math.tau)))
        if term or trunc:
            break
    assert worst < 0.05, worst


def _deepest_robot_contact(sim) -> float:
    """ロボット（車体・爪）と物体の接触の最も深いめり込み（m、正）。"""
    out = [0.0]
    def f(arb):
        a, b = arb.shapes
        other = b if a.body is sim._robot else a
        if other in sim._name_of and arb.contact_point_set.points:
            out.append(-min(p.distance for p in arb.contact_point_set.points))
    sim._robot.each_arbiter(f)
    return max(out)


@_crash_replay_linux_only
@pytest.mark.parametrize("name", ["crash_v25_s2", "crash_v27_s0"])
def test_jaw_stall_crash_replays_run_through(name):
    """爪が止まる物理（jaw_stall）で、学習中の例外の盤面と行動の列が最後まで例外なく進み、ロボットが大きく回らない。
    crash_v27_s0: 壁際の LEGO に爪が少しずつ食い込み、LEGO が壁の中へ押し込まれて作業領域の外に出た（爪が触れている物体の
    壁へのめり込みも、止まる判定の深さに入れて直した）。"""
    import math
    from pathlib import Path
    from toy_sweeper.crash_replay import parse, reset_like_crash
    from toy_sweeper.env import ToySweeperEnv
    root = Path(__file__).resolve().parents[1]
    text = (root / "test" / "data" / f"{name}.txt").read_text()
    exp = root / "config" / "experiments"
    cfg = load_all(overrides=tuple(exp / f for f in ("set_encoder.yaml", "chassis03.yaml", "imperfect.yaml", "gpu_ppo.yaml",
                                                    "cleanup.yaml", "corner_edge.yaml", "cleanup_guidance.yaml")) + (TEST_CFG / "jaw_stall.yaml",))
    spec, acts = parse(text)
    env = ToySweeperEnv(cfg, stage=spec.stage)
    assert reset_like_crash(env, text)
    spin = 0.0
    for a in acts:
        y0 = env.backend.robot_pose().yaw
        _, _, term, trunc, _ = env.step(a)
        spin = max(spin, abs(math.remainder(env.backend.robot_pose().yaw - y0, math.tau)))
        if term or trunc:
            break
    assert spin < 0.4


@_crash_replay_linux_only
def test_jaw_stall_prevents_embedding_from_crash_replay():
    """爪がふさがれたら止まる（sim2d.jaw_stall、2026-10-19）。v25 s2 の学習中の例外（test/data/crash_v25_s2.txt）: 作業領域の
    角で閉じた爪が逃げ場の無い LEGO に 9 mm めり込み、80 歩後に開いたときロボットが 1 歩で 0.96 rad 回った。止まる爪なら
    同じ行動の列で、めり込みは浅く、例外にならない。無効なら今までどおり（同じ歩で例外）。"""
    import dataclasses
    import math
    from pathlib import Path
    from toy_sweeper.crash_replay import parse, reset_like_crash
    from toy_sweeper.env import ToySweeperEnv
    root = Path(__file__).resolve().parents[1]
    text = (root / "test" / "data" / "crash_v25_s2.txt").read_text()
    exp = root / "config" / "experiments"
    base = load_all(overrides=tuple(exp / f for f in ("set_encoder.yaml", "chassis03.yaml", "imperfect.yaml", "gpu_ppo.yaml",
                                                     "cleanup.yaml", "corner_edge.yaml", "seed2.yaml")))
    spec, acts = parse(text)
    env = ToySweeperEnv(base, stage=spec.stage)
    assert reset_like_crash(env, text)
    with pytest.raises(ValueError, match="値域"):
        for a in acts:
            env.step(a)
    st = dataclasses.replace(base.sim.sim2d.jaw_stall, enabled=True)
    on = dataclasses.replace(base, sim=dataclasses.replace(base.sim, sim2d=dataclasses.replace(base.sim.sim2d, jaw_stall=st)))
    env = ToySweeperEnv(on, stage=spec.stage)
    assert reset_like_crash(env, text)
    deepest, spin = 0.0, 0.0
    for a in acts:
        y0 = env.backend.robot_pose().yaw
        _, _, term, trunc, _ = env.step(a)
        deepest = max(deepest, _deepest_robot_contact(env.backend))
        spin = max(spin, abs(math.remainder(env.backend.robot_pose().yaw - y0, math.tau)))
        if term or trunc:
            break
    assert deepest < 0.006 and spin < 0.4


def _lag_distance(v: float, tau: float, dt: float, n: int, v0: float = 0.0) -> tuple[float, float]:
    """v += (v_cmd − v) × dt / τ を n 刻み回したときの進む距離と最後の速さ。pymunk は位置を 1 刻み前の速さで進める
    （惰性なしの発進が約 1 刻み遅れるのと同じ。test_forward_and_turn）。"""
    x, u = 0.0, v0
    for _ in range(n):
        x += u * dt
        u += (v - u) * dt / tau
    return x, u


def test_motor_lag_follows_command_with_time_constant(sim, cfg):
    """v36: motor_lag_s = τ なら、車輪の速さは刻みごとに v += (v_cmd − v) × dt / τ で追いつく。止めた後も惰性で進む。"""
    dt, v, tau = sim.dt, cfg.sim.robot.linear_speed_mps, 0.1
    n = round(0.125 / dt)
    sim.reset(Pose2D(0, 0, 0), [], GripperState.OPEN)
    sim.motor_lag_s = tau
    sim.drive(v, 0.0, 0.125)
    want, u = _lag_distance(v, tau, dt, n)
    assert sim.robot_pose().x == pytest.approx(want, rel=0.01)
    assert sim._robot.velocity.length == pytest.approx(u, rel=1e-6)
    assert want < 0.6 * v * 0.125      # 惰性なしの距離よりはっきり短い
    x1 = sim.robot_pose().x
    sim.drive(0.0, 0.0, 0.125)        # 止めた後の惰性
    coast, _ = _lag_distance(0.0, tau, dt, n, v0=u)
    assert sim.robot_pose().x - x1 == pytest.approx(coast, rel=0.01) and coast > 0.003


def test_motor_lag_turn_coasts_and_reset_clears_speed(sim, cfg):
    tau, w = 0.1, cfg.sim.robot.angular_speed_radps
    sim.reset(Pose2D(0, 0, 0), [], GripperState.OPEN)
    sim.motor_lag_s = tau
    sim.drive(0.0, w, 0.25)
    y1 = sim.robot_pose().yaw
    sim.drive(0.0, 0.0, 0.125)
    assert sim.robot_pose().yaw > y1 + 0.05          # 旋回も惰性で回り続ける
    sim.reset(Pose2D(0, 0, 0), [], GripperState.OPEN)
    sim.drive(0.0, 0.0, 0.125)                       # reset で速さは 0 に戻る
    assert sim.robot_pose() == Pose2D(0, 0, 0)

