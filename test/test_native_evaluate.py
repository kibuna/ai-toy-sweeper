"""評価（scripts/evaluate_policies.py --env native）が、Python 版の env での評価と同じ結果になる。

同じ Seed・Stage・Policy・設定の差分で、1 エピソードごとの結果（float はビット単位）を比べる。確率的な PPO の乱数
（torch の大域の乱数）は Policy の中で引くので、env を替えても同じ数列になる（評価の後の torch の乱数の状態も比べる）。
"""
import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from toy_sweeper.config import load_all
from toy_sweeper.env import ToySweeperEnv
from toy_sweeper.native import NativeEnv
from toy_sweeper.policies import MixedHeuristicPolicy
from toy_sweeper.spaces import N_ACTIONS
from toy_sweeper.task import STAGES
from toy_sweeper.train import build_model

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "config" / "experiments"
SCRIPT = ROOT / "scripts" / "evaluate_policies.py"


def _script():
    spec = importlib.util.spec_from_file_location("evaluate_policies", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EP = _script()


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """学習していない小さな PPO（set_encoder の設定。重みは build_model の Seed で決まる）。"""
    cfg = load_all(overrides=(EXP / "set_encoder.yaml",))
    m = build_model(cfg, DummyVecEnv([lambda: ToySweeperEnv(cfg, stage=1)]))
    path = tmp_path_factory.mktemp("ppo") / "m.zip"
    m.save(path)
    # PPOPolicy が学習の記録の観測の設定を確かめる（policies.check_observation_config）
    (path.parent / "config_snapshot").mkdir()
    (path.parent / "config_snapshot" / "task.yaml").write_text("observation:\n  edge_distance: center\n")
    return path


@pytest.fixture(autouse=True)
def _keep_torch_threads():
    # eval_chunk は torch のスレッド数を 1 にする。ほかの試験の速さに響かないよう戻す
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def _canon(results):
    out = []
    for r in results:
        d = dataclasses.asdict(r)
        out.append({k: (v.hex() if isinstance(v, float) else (type(v).__name__, v)) for k, v in d.items()})
    return out


def _chunk(env, stage, name, seeds, model=None, stochastic=False, overrides=()):
    torch.manual_seed(123)
    res, stats = EP.eval_chunk((stage, name, list(seeds), model, stochastic, list(overrides), [], None, env))
    return _canon(res), stats, torch.get_rng_state()


CASES = [
    ("random", False, ()),
    ("heuristic", False, ()),
    ("carry", False, ()),
    ("mixed", False, ()),
    ("mixed", False, (EXP / "imperfect.yaml",)),
    ("ppo", False, (EXP / "set_encoder.yaml",)),
    ("ppo", True, (EXP / "set_encoder.yaml",)),
    ("ppo", True, (EXP / "set_encoder.yaml", EXP / "imperfect.yaml")),
]


@pytest.mark.parametrize("stage", [0, 4, 8, 9])
@pytest.mark.parametrize("name,stochastic,overrides", CASES)
def test_eval_chunk_matches_python(name, stochastic, overrides, stage, model):
    seeds = range(10000 + 5 * stage, 10000 + 5 * stage + 2)
    m = model if name == "ppo" else None
    py, pst, prng = _chunk("python", stage, name, seeds, m, stochastic, overrides)
    rs, rst, rrng = _chunk("native", stage, name, seeds, m, stochastic, overrides)
    assert rs == py
    assert rst == pst == []
    assert torch.equal(rrng, prng)


def test_progress_distance_matches_python():
    """評価の最後のゴールまでの距離（env._progress_distance()）が、毎歩ビット単位で同じ。"""
    for cfg in (load_all(), load_all(overrides=(EXP / "imperfect.yaml",))):
        for stage in STAGES:
            py, rs = ToySweeperEnv(cfg, stage), NativeEnv(cfg, stage)
            po, _ = py.reset(seed=300 + stage)
            ro, _ = rs.reset(seed=300 + stage)
            policy, rng = MixedHeuristicPolicy(cfg), np.random.default_rng(stage)
            for k in range(120):
                assert rs._progress_distance().hex() == py._progress_distance().hex(), (stage, k)
                a = int(rng.integers(N_ACTIONS)) if rng.random() < 0.3 else policy.predict(po)
                po, _, pterm, ptrunc, _ = py.step(a)
                ro, _, rterm, rtrunc, _ = rs.step(a)
                assert np.array_equal(po.view(np.uint32), ro.view(np.uint32)) and (pterm, ptrunc) == (rterm, rtrunc)
                if pterm or ptrunc:
                    po, _ = py.reset()
                    ro, _ = rs.reset()


def test_native_env_errors():
    cfg = load_all()
    env = NativeEnv(cfg, 1)
    with pytest.raises(RuntimeError, match="reset"):
        env._progress_distance()
    with pytest.raises(ValueError, match="Seed"):
        env.reset()          # 最初の reset は Seed が要る（Python 版は OS の乱数で Seed を決めるので同じにできない）
    with pytest.raises(ValueError, match="options"):
        env.reset(seed=0, options={"episode": None})
    env.reset(seed=0)
    with pytest.raises(NotImplementedError):
        env.render()


def test_unsupported_with_native_raises():
    """planner は env の中身（盤面・観測した姿勢）を読むので native では評価できない。Gazebo も native にはない。"""
    with pytest.raises(ValueError, match="planner"):
        EP.eval_chunk((1, "planner", [10000], None, False, [], [], None, "native"))
    with pytest.raises(ValueError, match="sim2d"):
        EP.eval_chunk((1, "heuristic", [10000], None, False, [], [], object(), "native"))
    with pytest.raises(ValueError, match="env"):
        EP.eval_chunk((1, "heuristic", [10000], None, False, [], [], None, "rust"))
    for backend, policy in (("sim2d", "planner"), ("gazebo", "heuristic")):
        argv = ["--backend", backend, "--policies", policy, "--stages", "1", "--n", "1", "--out", "/nonexistent/x",
                "--env", "native"]
        p = subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True)
        assert p.returncode == 2 and "native" in p.stderr, p.stderr


def test_cli_files_match_python(tmp_path, model):
    """スクリプト全体（--jobs で Seed を分ける）: episodes.csv が同じ、summary.json は env と時間の項目のほかは同じ。"""
    common = ["--backend", "sim2d", "--stages", "1", "6", "--policies", "mixed", "ppo", "--stochastic", "--n", "4",
              "--jobs", "2", "--model", str(model), "--override", str(EXP / "set_encoder.yaml")]
    out = {}
    for env in ("python", "native"):
        p = subprocess.run([sys.executable, str(SCRIPT), *common, "--env", env, "--out", str(tmp_path / env)],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        out[env] = tmp_path / env
    assert (out["python"] / "episodes.csv").read_bytes() == (out["native"] / "episodes.csv").read_bytes()
    sp, sn = (json.loads((out[e] / "summary.json").read_text()) for e in ("python", "native"))
    assert (sp["meta"].pop("env"), sn["meta"].pop("env")) == ("python", "native")
    assert sp["meta"] == sn["meta"]
    for s in (*sp["results"].values(), *sn["results"].values()):
        s.pop("wall_s")
    # NaN（成功のない平均ステップ）があるので文字列で比べる
    assert json.dumps(sp["results"]) == json.dumps(sn["results"])
