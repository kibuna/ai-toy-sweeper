"""Rust 版の VecEnv（native/src/vecenv.rs と toy_sweeper.native.NativeVecEnv）が、SB3 の
make_vec_env(lambda: ToySweeperEnv(cfg, stage), seed=...) + Monitor(info_keywords=("is_success",)) と同じ観測・報酬・
終わり方・終わったときの観測・info になる。

乱数の行動で数百歩進め、途中で set_curriculum（一部の env だけ、復習あり）も呼ぶ。DummyVecEnv は報酬を float32 に
丸めるので、報酬は float32 にして比べ、f64 のままの値は Monitor のエピソードの報酬の和（episode["r"]）で比べる。
"""
import dataclasses
import time
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from toy_sweeper.config import load_all
from toy_sweeper.env import ToySweeperEnv
from toy_sweeper.native import NativeVecEnv
from toy_sweeper.spaces import N_ACTIONS

EXP = Path(__file__).resolve().parents[1] / "config" / "experiments"
N_ENVS = 8
# info の鍵のうち、終わっていない env でも比べるもの（SB3 と train.py が読む）
ALWAYS = ("is_success", "stage", "TimeLimit.truncated")


def _short(cfg, max_steps):
    """エピソードの終わりを多く通すため、短いエピソードにする。"""
    ep = dataclasses.replace(cfg.task.episode, max_steps=max_steps, max_steps_per_extra_target=10)
    return dataclasses.replace(cfg, task=dataclasses.replace(cfg.task, episode=ep))


def _no_walls(cfg):
    return dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, walls=dataclasses.replace(cfg.sim.walls,
                                                                                               enabled=False)))


def _pair(cfg, stage, seed, threads=4):
    py = make_vec_env(lambda: ToySweeperEnv(cfg, stage), n_envs=N_ENVS, seed=seed, vec_env_cls=DummyVecEnv,
                      monitor_kwargs={"info_keywords": ("is_success",)})
    rs = NativeVecEnv(cfg, N_ENVS, seed, threads, stage)
    return py, rs


def _same_info(pi, ri, where):
    pi, ri = dict(pi), dict(ri)
    for d in (pi, ri):
        d.pop("terminal_observation", None)
        if "episode" in d:
            d["episode"] = {k: v for k, v in d["episode"].items() if k != "t"}   # 経過時間は比べない
    assert list(pi) == list(ri), (where, list(pi), list(ri))
    assert repr(pi) == repr(ri), (where, pi, ri)
    for k, v in pi.items():
        assert type(ri[k]) is type(v), (where, k)


def _compare(py, rs, steps, seed, schedule, where):
    po, ro = py.reset(), rs.reset()
    assert np.array_equal(po.view(np.uint32), ro.view(np.uint32)), where
    assert py.reset_infos == rs.reset_infos, where
    rng = np.random.default_rng(seed)
    cover = {"done": 0, "truncated": 0, "terminated": 0, "success": 0}
    for k in range(steps):
        if k in schedule:
            args, idx = schedule[k]
            py.env_method("set_curriculum", *args, indices=idx)
            assert rs.env_method("set_curriculum", *args, indices=idx) == [None] * (N_ENVS if idx is None else len(idx))
        a = rng.integers(N_ACTIONS, size=N_ENVS)
        po, pr, pd, pinfos = py.step(a)
        ro, rr, rd, rinfos = rs.step(a)
        w = (*where, k)
        assert ro.dtype == np.float32 and rr.dtype == np.float64 and rd.dtype == bool, w
        assert np.array_equal(po.view(np.uint32), ro.view(np.uint32)), w
        assert np.array_equal(pr.view(np.uint32), rr.astype(np.float32).view(np.uint32)), (w, pr, rr)
        assert np.array_equal(pd, rd), w
        for i in range(N_ENVS):
            pi, ri = pinfos[i], rinfos[i]
            if pd[i]:
                assert np.array_equal(pi["terminal_observation"].view(np.uint32), ri["terminal_observation"].view(np.uint32)), (w, i)
                _same_info(pi, ri, (*w, i))
                cover["done"] += 1
                cover["truncated"] += pi["TimeLimit.truncated"]
                cover["terminated"] += not pi["TimeLimit.truncated"]
                cover["success"] += pi["is_success"]
            else:
                assert {k: pi[k] for k in ALWAYS} == ri, (w, i)
        assert py.reset_infos == rs.reset_infos, w
    return cover


@pytest.mark.parametrize("name", ["base", "imperfect_no_walls", "cleanup"])
def test_vecenv_matches_sb3_dummy(name):
    if name == "base":
        cfg, stage, steps = _short(load_all(), 40), 2, 400
        schedule = {150: ((5, (0, 1, 2, 3, 4), 0.5), None), 300: ((8, (), 0.0), [1, 4, 6])}
    elif name == "cleanup":   # Stage 9（v24）を、config/experiments/cleanup.yaml の Curriculum と同じ復習の形で
        cfg, stage, steps = _short(load_all(overrides=(EXP / "imperfect.yaml", EXP / "cleanup.yaml")), 30), 9, 300
        schedule = {100: ((9, (0, 1, 2, 4, 5, 6, 7, 8), 0.3), [0, 2, 5])}
    else:
        cfg, stage, steps = _short(_no_walls(load_all(overrides=(EXP / "imperfect.yaml",))), 30), 8, 300
        schedule = {100: ((1, (0,), 1.0), [0, 1, 2, 3]), 200: ((6, (5, 7), 0.5), None)}
    py, rs = _pair(cfg, stage, seed=123)
    cover = _compare(py, rs, steps, 9, schedule, (name,))
    print(f"\n{name}: {cover}")
    assert cover["truncated"] >= 10 and cover["done"] >= 40, cover
    if name == "imperfect_no_walls":   # 壁なしでは外に出て終わる（cleanup は壁ありで乱数の行動なので、成功でもほぼ終わらない）
        assert cover["terminated"] >= 1, cover


def test_reset_again_and_seed():
    """2 回目の reset（Seed なし）は env の乱数のまま。seed() の後の reset は新しい Seed（SB3 の VecEnv.seed と同じ）。"""
    cfg = _short(load_all(), 20)
    py, rs = _pair(cfg, 1, seed=5)
    cover = _compare(py, rs, 30, 1, {}, ("first",))
    cover = _compare(py, rs, 30, 2, {}, ("again",))
    py.seed(77)
    rs.seed(77)
    cover = _compare(py, rs, 30, 3, {}, ("reseed",))
    assert cover["done"] >= 1


def test_sb3_api():
    """SB3 が読むもの: render_mode、Monitor で包んであるか、知らない属性・関数は例外。PPO で学習が回る。"""
    cfg = _short(load_all(), 20)
    rs = NativeVecEnv(cfg, 4, 0, 2, 1)
    assert rs.get_attr("render_mode") == [None] * 4
    assert rs.env_is_wrapped(Monitor) == [True] * 4
    with pytest.raises(AttributeError):
        rs.get_attr("np_random")
    with pytest.raises(AttributeError):
        rs.set_attr("stage", 3)
    with pytest.raises(AttributeError):
        rs.env_method("render")
    with pytest.raises(ValueError, match="Stage"):
        rs.env_method("set_curriculum", 10, (), 0.0)
    model = PPO("MlpPolicy", rs, n_steps=32, batch_size=64, n_epochs=1, device="cpu", seed=0)
    model.learn(total_timesteps=256)
    assert len(model.ep_info_buffer) > 0
    assert all(set(e) == {"r", "l", "t", "is_success"} for e in model.ep_info_buffer)
    rs.close()


def test_error_names_the_env():
    """どれかの env が失敗したら、その env の番号を付けて例外にする（飛ばして続けない）。"""
    cfg = load_all()
    rs = NativeVecEnv(cfg, 4, 0, 2, 1)
    rs.reset()
    with pytest.raises(ValueError, match="env 2: .*not a valid Action"):
        rs.step(np.array([0, 1, 9, 0]))
    with pytest.raises(ValueError, match="整数"):
        rs.step(np.array([0.0, 1.0, 2.0, 0.0]))
    with pytest.raises(ValueError, match="threads"):
        NativeVecEnv(cfg, 4, 0, 0, 1)


@pytest.mark.parametrize("stage", [8, 9])
def test_rewards_match_python_envs_as_f64(stage):
    """報酬を f64 のビットで比べる（DummyVecEnv は float32 に丸めるので、Python の env を 1 つずつ直接進める）。
    終わったら Seed なしで reset する（SB3 の VecEnv と同じ）。Monitor の episode["r"] は Python の sum() と比べる。"""
    cfg, n, seed = _short(load_all(overrides=(EXP / "imperfect.yaml",)), 30), 4, 11
    envs = [ToySweeperEnv(cfg, stage) for _ in range(n)]
    po = [e.reset(seed=seed + i)[0] for i, e in enumerate(envs)]
    rs = NativeVecEnv(cfg, n, seed, 3, stage)
    ro = rs.reset()
    assert np.stack(po).tobytes() == ro.tobytes()
    rng = np.random.default_rng(4)
    rewards = [[] for _ in range(n)]
    done = 0
    for k in range(250):
        a = rng.integers(N_ACTIONS, size=n)
        ro, rr, rd, rinfos = rs.step(a)
        assert rr.dtype == np.float64
        for i, e in enumerate(envs):
            o, r, term, trunc, info = e.step(int(a[i]))
            rewards[i].append(r)
            assert float(rr[i]).hex() == float(r).hex(), (k, i, rr[i], r)
            assert bool(rd[i]) == (term or trunc), (k, i)
            if term or trunc:
                assert rinfos[i]["terminal_observation"].tobytes() == o.tobytes(), (k, i)
                assert rinfos[i]["episode"]["r"] == round(sum(rewards[i]), 6), (k, i)
                rewards[i] = []
                done += 1
                o, _ = e.reset()
            assert ro[i].tobytes() == o.tobytes(), (k, i)
    assert done >= 10, done


def _record(threads, cfg, n, steps):
    """threads 本で進めた観測・報酬・終わり・info・乱数の状態の記録（比べる形）。"""
    rs = NativeVecEnv(cfg, n, 7, threads, 5)
    rng = np.random.default_rng(0)
    rec = [rs.reset().tobytes(), repr(rs.reset_infos)]
    for k in range(steps):
        if k == steps // 3:
            rs.env_method("set_curriculum", 8, (2, 5), 0.5)
        o, r, d, infos = rs.step(rng.integers(N_ACTIONS, size=n))
        canon = []
        for info in infos:
            info = dict(info)
            if "terminal_observation" in info:
                info["terminal_observation"] = info["terminal_observation"].tobytes()
            if "episode" in info:
                info["episode"] = {k: v for k, v in info["episode"].items() if k != "t"}   # 経過時間は比べない
            canon.append(info)
        rec.append((o.tobytes(), r.tobytes(), d.tobytes(), repr(canon), repr(rs.reset_infos)))
    rec.append((rs._native.rng_states(), rs._native.stages()))
    return rec


def test_thread_count_does_not_change_results():
    """スレッドの数（1、3、8）を変えても、出力と乱数の状態がビット単位で同じ（env ごとに自分の乱数と行だけ使う）。"""
    cfg = _short(load_all(overrides=(EXP / "imperfect.yaml",)), 25)
    runs = {t: _record(t, cfg, 16, 150) for t in (1, 3, 8)}
    assert sum(r[2].count(b"\x01") for r in runs[1][2:-1]) >= 20   # 終わりも通る
    assert runs[1] == runs[3] == runs[8]


def test_step_returns_fresh_arrays():
    """step と reset は毎回新しい配列を返す（SB3 は前の観測を step の後にも使うので、上書きしてはいけない）。"""
    rs = NativeVecEnv(_short(load_all(), 20), 4, 0, 2, 2)
    o0 = rs.reset()
    keep0 = o0.copy()
    a = np.array([1, 3, 6, 2])
    o1, r1, d1, i1 = rs.step(a)
    keep1 = (o1.copy(), r1.copy(), d1.copy())
    o2, r2, d2, _ = rs.step(a)
    assert not np.array_equal(o1, o2)    # 動いたので観測は変わる
    for x, y in ((o0, o1), (o1, o2), (r1, r2), (d1, d2)):
        assert not np.shares_memory(x, y)
    assert np.array_equal(o0, keep0)
    assert np.array_equal(o1, keep1[0]) and np.array_equal(r1, keep1[1]) and np.array_equal(d1, keep1[2])


def test_speed():
    """Stage 8（v21 の不完全さあり）で、Python 版の env 1 つと Rust 版の VecEnv の 1 秒あたりの歩数。"""
    cfg = load_all(overrides=(EXP / "imperfect.yaml",))
    py = ToySweeperEnv(cfg, 8)
    py.reset(seed=0)
    rng = np.random.default_rng(0)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        _, _, term, trunc, _ = py.step(int(rng.integers(N_ACTIONS)))
        n += 1
        if term or trunc:
            py.reset()
    py_rate = n / (time.perf_counter() - t0)
    out = {}
    for threads in (1, 8):
        rs = NativeVecEnv(cfg, 256, 0, threads, 8)
        rs.reset()
        k, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 2.0:
            rs.step(rng.integers(N_ACTIONS, size=256))
            k += 1
        out[threads] = 256 * k / (time.perf_counter() - t0)
    print(f"\n1 秒あたりの歩数（Stage 8、imperfect）: Python 1 env {py_rate:.0f}、"
          + "、".join(f"Rust 256 env {t} スレッド {r:.0f}" for t, r in out.items()))
    assert out[1] > py_rate


def test_disjoint_seeds_do_not_overlap_between_train_seeds():
    """disjoint_seeds: train_seed 0 と 1 の env の最初の Seed が重ならない（rl.training.disjoint_env_seeds）。"""
    from toy_sweeper.config import load_all
    from toy_sweeper.native import NativeVecEnv, seed_state
    cfg = load_all()
    a, b = (NativeVecEnv(cfg, 8, s, 2, 7, disjoint_seeds=True) for s in (0, 1))
    assert a._seeds == list(range(8)) and b._seeds == list(range(8, 16))
    import pytest
    with pytest.raises(ValueError):
        a.seed(None)
