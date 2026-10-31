import pytest

from toy_sweeper.config import load_all
from toy_sweeper.env import ToySweeperEnv
from toy_sweeper.evaluate import evaluate, summarize
from toy_sweeper.policies import HeuristicPolicy, RandomPolicy


@pytest.fixture(scope="module")
def cfg():
    # Heuristic の実装のデグレ検出用。ゴールは 4 エリア（goal_mode: zones）で測る。
    # random では端の物体を反対側へ押す、押すだけでは解けない盤面が混ざる
    import dataclasses
    c = load_all()
    return dataclasses.replace(c, task=dataclasses.replace(
        c.task, sampling=dataclasses.replace(c.task.sampling, goal_mode="zones")))


# Seed 50000〜50019、goal_mode: zones（四隅）での実測（2026-10-16、壁あり、v6）: Stage 0 1.00, 1 1.00, 2 0.85, 3 0.55。
# 押すだけの Heuristic は、押し始める位置が壁の向こうになる盤面で壁に押し付けたまま時間切れになる（四隅のゴールで増えた）。
# デグレ検出のため、実測から 0.1 下げた値を下限にする
HEURISTIC_FLOOR = {0: 0.9, 1: 0.9, 2: 0.75, 3: 0.45}


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_heuristic_solves_2d(cfg, stage):
    env = ToySweeperEnv(cfg, stage=stage)
    s = summarize(evaluate(env, HeuristicPolicy(cfg), range(50000, 50020)))
    assert s["success_rate"] >= HEURISTIC_FLOOR[stage], s


def test_random_is_poor(cfg):
    env = ToySweeperEnv(cfg, stage=2)
    s = summarize(evaluate(env, RandomPolicy(0), range(50000, 50010)))
    assert s["success_rate"] <= 0.3


def test_check_observation_config(tmp_path):
    """学習の記録と今の設定で observation.edge_distance が違えば止める（v25）。記録に無ければ center（v25 より前）。"""
    import dataclasses
    from toy_sweeper.config import load_all
    from toy_sweeper.policies import check_observation_config
    cfg = load_all()
    corners = dataclasses.replace(cfg, task=dataclasses.replace(
        cfg.task, observation=dataclasses.replace(cfg.task.observation, edge_distance="corners")))
    model = tmp_path / "final.zip"
    with pytest.raises(FileNotFoundError, match="学習の記録"):
        check_observation_config(model, cfg)
    (tmp_path / "config_snapshot").mkdir()
    snap = tmp_path / "config_snapshot" / "task.yaml"
    snap.write_text("observation:\n  n_obstacle_slots: 6\n")
    check_observation_config(model, cfg)
    with pytest.raises(ValueError, match="edge_distance"):
        check_observation_config(model, corners)
    snap.write_text("observation:\n  edge_distance: corners\n")
    check_observation_config(model, corners)
    with pytest.raises(ValueError, match="edge_distance"):
        check_observation_config(model, cfg)


def test_evaluate_records_leftovers_at_end():
    """評価の記録に、終わったときのゴールの中の対象外・置き場から出た対象外の数（v34）。全対象が入れば成功
    （require_cleanup: false）で、成功のうち残った物のある割合を summary に出す。"""
    import dataclasses
    from toy_sweeper.evaluate import EpisodeResult
    from toy_sweeper.spaces import Action

    class Stop:
        def reset(self):
            pass

        def predict(self, obs):
            return int(Action.STOP)
    c = load_all()
    t = c.task
    loose = dataclasses.replace(c, task=dataclasses.replace(
        t, success=dataclasses.replace(t.success, require_cleanup=False),
        sampling=dataclasses.replace(t.sampling, stage9_mistake_probability=1.0, stage9_all_delivered_probability=1.0)))
    res = evaluate(ToySweeperEnv(loose, stage=9), Stop(), range(5))
    assert all(r.success and r.steps == 1 and r.final_foreign_in_goal + r.final_home_out >= 1 for r in res)
    s = summarize(res)
    assert s["success_with_leftover_rate"] == 1.0
    ok = dataclasses.replace(res[0], final_foreign_in_goal=0, final_home_out=0)
    assert summarize([ok, *res[1:2]])["success_with_leftover_rate"] == 0.5
    assert "final_foreign_in_goal" in EpisodeResult.__dataclass_fields__
