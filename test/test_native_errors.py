"""Rust 版（native/）の失敗の扱い。

- C（Chipmunk）から呼ばれる Rust の関数（床の摩擦、車体の接触）の失敗は、C を越えて巻き戻さずに記録し、drive が
  例外にする（黙って続けない）。
- Chipmunk の assert（cpMessage）は、Chipmunk の文に加えて、どの env の何をしていたか（reset / step、Stage、歩数、
  乱数の状態、行動の列）を stderr に書く。強い assert（cpAssertHard）は pymunk と同じく abort する（別のプロセスで
  確かめる）。警告は書いて続ける。
- VecEnv の step / reset の途中の失敗は一部の env だけ進んだ半端な状態にするので、以後の呼び出しは例外にする。
"""
import dataclasses
import inspect
import re
import signal
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import toy_sweeper_native as nat
from toy_sweeper.config import load_all
from toy_sweeper.native import NativeVecEnv, cfg_json, seed_state
from toy_sweeper.spaces import N_ACTIONS

NO_CONTEXT = "\tnative の状況: env、reset / step の外\n"


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=300)


# ---------- C から呼ばれる関数の失敗 ----------

def _sim_with_object_behind():
    """ロボットの真後ろに物体を 1 つ置く（後ろへ走ると車体の後ろの面に当たる。爪は前にしかない）。"""
    cfg = load_all()
    body_center_x = nat._robot_model(cfg_json(cfg), 0.0, 0.0, 0.0)[0]
    rear = body_center_x - cfg.sim.robot.body_size_m[0] / 2
    lx = cfg.sim.objects["lego"].size_m[0]
    sim = nat.Sim2D(cfg_json(cfg))
    sim.reset((0.0, 0.0, 0.0), [("a", 0, rear - 0.02 - lx / 2, 0.0, 0.0)], 0)
    return sim, cfg.task.control.action_duration_s


def test_contact_callback_error_stops_drive():
    sim, t = _sim_with_object_behind()
    sim.drive(-0.2, 0.0, 2 * t)
    assert sim.chassis_contacts() == ["a"]    # 壊さなければ車体の接触として数える
    sim, t = _sim_with_object_behind()
    sim._set_shape_user_data_for_test("a", 99)
    with pytest.raises(RuntimeError, match="車体の接触: 形の user data の物体の番号 98 が範囲外（物体 1 個）"):
        sim.drive(-0.2, 0.0, 2 * t)


def test_floor_friction_callback_error_stops_drive():
    sim, t = _sim_with_object_behind()
    sim._clear_body_user_data_for_test("a")
    with pytest.raises(RuntimeError, match="床の摩擦: 物体の体に床の摩擦の値（user data）がない"):
        sim.drive(0.0, 0.0, t)


# ---------- Chipmunk の assert の報告 ----------

def test_warning_prints_and_continues():
    r = _run("""
        import toy_sweeper_native as nat
        nat._cp_warning_for_test("テストの警告 %d")
        print("続いた")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "続いた\n"
    assert "Chipmunk warning: テストの警告 %d\n\tFailed condition: test_condition\n\tSource:sim2d.rs:0\n" in r.stderr
    assert NO_CONTEXT in r.stderr


def test_context_is_cleared_after_step():
    rs = NativeVecEnv(load_all(), 2, 0, 1, 2)
    rs.reset()
    rs.step(np.array([1, 6]))
    assert nat._cp_context_text() == NO_CONTEXT


def _with(cfg, part, **kw):
    """cfg.sim.<part> の項目を変える。"""
    sim = cfg.sim
    return dataclasses.replace(cfg, sim=dataclasses.replace(sim, **{part: dataclasses.replace(getattr(sim, part), **kw)}))


# 別のプロセスの台本でも同じ _with を使う
_WITH = "import dataclasses\n" + inspect.getsource(_with)


def test_hard_assert_in_reset_aborts_with_env_context():
    """負の摩擦（Chipmunk の cpAssertHard）。VecEnv の reset の途中で止まり、どの env の reset か書く。"""
    r = _run(_WITH + """
from toy_sweeper.config import load_all
from toy_sweeper.native import NativeVecEnv
cfg = _with(load_all(), "contact", robot_object_friction=-0.3)
NativeVecEnv(cfg, 3, 0, 1, 2).reset()
print("続いた")
""")
    assert r.returncode == -signal.SIGABRT, (r.returncode, r.stderr)
    assert "続いた" not in r.stdout
    err = r.stderr
    assert "Aborting due to Chipmunk error: Friction must be positive.\n\tFailed condition: friction >= 0.0f\n" in err
    m = re.search(r"\tnative の状況: VecEnv の env (\d+)、reset の途中（Stage 2、この step の前までに 0 歩、行動 なし）\n", err)
    assert m, err
    i = int(m.group(1))
    assert f"\treset の始めの乱数の状態 (state, inc, 32 ビットの残り): {seed_state(0 + i)!r}\n" in err, err
    assert "\t学習の設定 (Stage, 復習の Stage, 復習の確率): (2, [], 0.0)\n" in err
    assert "\tこのエピソードの行動: []\n" in err


def test_hard_assert_in_step_aborts_with_env_context():
    """負の保持力（つかんだときの cpConstraintSetMaxForce の cpAssertHard）。step の途中で止まり、どの env の何歩目の
    どの行動か、そのエピソードの行動の列を書く。"""
    cfg = load_all()
    # 同じ Seed・同じ乱数の行動で、最初につかむ歩を探す（保持力はつかむまで効かないので、そこまでは同じ動き）
    env = nat.Env(cfg_json(cfg), 1)
    env.reset(seed_state(0))
    rng = np.random.default_rng(0)
    episode = []
    for _ in range(3000):
        a = int(rng.integers(N_ACTIONS))
        _, _, term, trunc, _ = env.step(a)
        if env.grasped_object() is not None:
            break
        episode.append(a)
        if term or trunc:
            env.reset()
            episode = []
    else:
        raise AssertionError("つかむ場面がない")
    r = _run(_WITH + f"""
import numpy as np
from toy_sweeper.config import load_all
from toy_sweeper.native import NativeVecEnv
cfg = _with(load_all(), "gripper", grip_holding_force_n=-1.0)
rs = NativeVecEnv(cfg, 1, 0, 1, 1)
rs.reset()
rng = np.random.default_rng(0)
for _ in range(3000):
    rs.step(np.array([rng.integers({N_ACTIONS})]))
print("続いた")
""")
    assert r.returncode == -signal.SIGABRT, (r.returncode, r.stderr)
    err = r.stderr
    assert "Aborting due to Chipmunk error: maxForce must be positive.\n" in err, err
    assert (f"\tnative の状況: VecEnv の env 0、step の途中（Stage 1、この step の前までに {len(episode)} 歩、"
            f"行動 {a}）\n") in err, err
    assert f"\tこのエピソードの行動: {episode!r}\n" in err, err
    assert "\tロボット (x, y, yaw)（step の始め。reset では初期位置）: (" in err


# ---------- VecEnv の失敗の後 ----------

def _noisy(noise):
    """観測の誤差を大きくして、どれかの env がいずれ観測の値域の外で失敗するようにする。"""
    return _with(load_all(), "real_world_effects", obs_position_noise_m=noise)


def test_vecenv_is_poisoned_after_error():
    rs = NativeVecEnv(_noisy(0.02), 8, 0, 2, 2)
    rs.reset()
    rng = np.random.default_rng(0)
    for k in range(200):
        try:
            rs.step(rng.integers(N_ACTIONS, size=8))
        except ValueError as e:
            first = str(e)
            break
    else:
        raise AssertionError("失敗しなかった")
    assert k >= 1 and re.match(r"env \d+: Observation が値域", first), (k, first)
    head = re.escape(first.splitlines()[0])
    for call in (lambda: rs.step(np.zeros(8, dtype=np.int64)), rs.reset,
                 lambda: rs.env_method("set_curriculum", 3, (), 0.0)):
        with pytest.raises(RuntimeError, match=f"VecEnv は前の失敗で止まっている.*最初の失敗: {head}"):
            call()


def test_invalid_action_does_not_poison():
    """どの env も進める前に分かる失敗（行動の値）は、何も変えずに例外にする。"""
    cfg = load_all()
    rs, ref = NativeVecEnv(cfg, 4, 3, 2, 2), NativeVecEnv(cfg, 4, 3, 2, 2)
    rs.reset(), ref.reset()
    with pytest.raises(ValueError, match="env 2: 9 is not a valid Action"):
        rs.step(np.array([1, 1, 9, 1]))
    a = np.array([1, 3, 6, 0])
    o, r, d, _ = rs.step(a)
    po, pr, pd, _ = ref.step(a)
    assert o.tobytes() == po.tobytes() and r.tobytes() == pr.tobytes() and np.array_equal(d, pd)
    assert rs._native.rng_states() == ref._native.rng_states()


def test_python_side_error_poisons():
    """Rust の step が通った後に Python 側（info を作る途中）で失敗しても、以後は進めない。"""
    rs = NativeVecEnv(load_all(), 2, 0, 1, 2)
    rs.reset()

    def broken(out):
        raise KeyError("壊した")
    rs._step_result = broken
    with pytest.raises(KeyError, match="壊した"):
        rs.step(np.array([1, 1]))
    with pytest.raises(RuntimeError, match="NativeVecEnv は前の失敗で止まっている.*壊した"):
        rs.step(np.array([1, 1]))
    with pytest.raises(RuntimeError, match="止まっている"):
        rs.reset()


def test_diagnostics_has_rng_state_and_curriculum_for_replay():
    """例外に付ける状況に reset の直前の乱数の状態と Curriculum を入れる（2026-10-19、v24 の学習中の例外が、不完全さの
    乱数が分からず盤面と行動の列だけでは再生できなかった）。その状態から reset し直すと同じ盤面・同じ歩みになる。"""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cfg = load_all(overrides=(root / "config/experiments/imperfect.yaml", root / "config/experiments/cleanup.yaml"))
    a = nat.Env(cfg_json(cfg), 9)
    a.set_curriculum(9, [0, 1, 2, 4, 5, 6, 7, 8], 0.3)
    a.reset(seed_state(3))
    rng = np.random.default_rng(1)
    for _ in range(5):          # 何回か終わらせて、Seed なしの reset（乱数の続き）を通す
        while True:
            *_, term, trunc, _ = a.step(int(rng.integers(N_ACTIONS)))
            if term or trunc:
                break
        a.reset(None)
    d = a.diagnostics()
    m = re.search(r"rng_at_reset: Some\(\((\d+), (\d+), (None|Some\((\d+)\))\)\)", d)
    assert m, d
    state = (int(m.group(1)), int(m.group(2)), None if m.group(3) == "None" else int(m.group(4)))
    assert "curriculum: stage 9, previous [0, 1, 2, 4, 5, 6, 7, 8], rehearsal 0.3" in d
    acts = [int(rng.integers(N_ACTIONS)) for _ in range(40)]
    b = nat.Env(cfg_json(cfg), 9)
    b.set_curriculum(9, [0, 1, 2, 4, 5, 6, 7, 8], 0.3)
    a2 = nat.Env(cfg_json(cfg), 9)          # 同じ状態から 2 つの env を始めて、同じ歩みになるか
    a2.set_curriculum(9, [0, 1, 2, 4, 5, 6, 7, 8], 0.3)
    oa, _ = a2.reset(state)
    ob, _ = b.reset(state)
    assert np.array_equal(oa, ob)
    assert a2.diagnostics().split("\nsteps:")[0] == d.split("\nsteps:")[0]   # 同じ盤面
    for act in acts:
        ra, rb = a2.step(act), b.step(act)
        assert np.array_equal(ra[0], rb[0]) and ra[1] == rb[1]
        if ra[2] or ra[3]:
            break
