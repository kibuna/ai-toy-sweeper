import dataclasses
import math

import numpy as np
import pytest

from toy_sweeper.config import load_all
from toy_sweeper.geometry import Pose2D, dist
from toy_sweeper.robot_model import gripper_center_x, robot_radius
from toy_sweeper.spaces import OBJECT_TYPE_NAMES, ObjectType
from toy_sweeper.task import STAGES, TaskSampler, approach_point, goal_rects, gripper_point
from pathlib import Path

TEST_CFG = Path(__file__).resolve().parent / "configs"   # テストだけで使う設定の差し替え


@pytest.fixture(scope="module")
def cfg():
    return load_all()


def _zones(cfg):
    return dataclasses.replace(cfg.task, sampling=dataclasses.replace(cfg.task.sampling, goal_mode="zones"))


@pytest.mark.parametrize("stage", STAGES)
def test_samples_are_valid(cfg, stage):
    sampler = TaskSampler(cfg.sim, _zones(cfg))
    rng = np.random.default_rng(0)
    half = cfg.sim.work_area.size_m[0] / 2
    m = cfg.task.sampling.robot_margin_m
    goals = goal_rects(cfg.sim)
    seen_goals = set()
    for _ in range(300):
        ep = sampler.sample(stage, rng)
        seen_goals.add(ep.goal_name)
        assert ep.goal == goals[ep.goal_name]
        assert abs(ep.robot.x) <= half - m + 1e-9 and abs(ep.robot.y) <= half - m + 1e-9
        if stage == 0:
            assert not ep.target_names and not ep.objects and ep.reach_point is not None
            continue
        target = next(o for o in ep.objects if o.name == ep.target_names[0])
        # Stage 9 で対象を全部ゴールに置き直した盤面（置き場の指定が付く）のほかは、対象の中心はゴールの外
        assert ep.goal.contains(target.pose.xy) == (stage == 9 and target.name in dict(ep.home_overrides))
        if stage in (1, 2, 3):
            assert len(ep.objects) == 1 and target.type == ObjectType.LEGO
        if stage == 4:
            assert len(ep.objects) == 1 and target.type == ObjectType.TOY_CAR
        if stage == 5:
            lo, hi = cfg.task.sampling.stage5_edge_distance_m
            edge = half - max(abs(target.pose.x), abs(target.pose.y))
            assert len(ep.objects) == 1 and lo - 1e-9 <= edge <= hi + 1e-9
        if stage == 6:
            lo, hi = cfg.task.sampling.clutter_range
            assert lo + 1 <= len(ep.objects) <= hi + 1
        if stage == 7:
            tl, th = cfg.task.sampling.stage7_targets
            targets = [o for o in ep.objects if o.name in ep.target_names]
            assert tl <= len(targets) <= th and all(o.type == ObjectType.LEGO for o in targets)
            assert len({o.color for o in targets}) == 1
            for o in ep.objects:
                if o.name not in ep.target_names:   # 対象外は色か種類が違う
                    assert o.type != ObjectType.LEGO or o.color != targets[0].color
    assert seen_goals == set(goals)  # 4 つのゴールがすべて出る


def test_stage6_no_overlap(cfg):
    sampler = TaskSampler(cfg.sim, cfg.task)
    rng = np.random.default_rng(3)
    r = {t: math.hypot(*cfg.sim.objects[OBJECT_TYPE_NAMES[t]].size_m[:2]) / 2 for t in ObjectType}
    types = set()
    for _ in range(200):
        ep = sampler.sample(6, rng)
        for i, a in enumerate(ep.objects):
            types.add(a.type)
            assert dist(ep.robot.xy, a.pose.xy) >= robot_radius(cfg.sim) + r[a.type]
            for b in ep.objects[i + 1:]:
                if b.pose.yaw == a.pose.yaw:
                    # 対象に接して置いた物体（同じ向き）: 対象の座標系で長方形が重ならない
                    c, s = math.cos(a.pose.yaw), math.sin(a.pose.yaw)
                    dx, dy = b.pose.x - a.pose.x, b.pose.y - a.pose.y
                    lx, ly = abs(c * dx + s * dy), abs(-s * dx + c * dy)
                    sa = cfg.sim.objects[OBJECT_TYPE_NAMES[a.type]].size_m
                    sb = cfg.sim.objects[OBJECT_TYPE_NAMES[b.type]].size_m
                    assert lx >= (sa[0] + sb[0]) / 2 - 1e-9 or ly >= (sa[1] + sb[1]) / 2 - 1e-9
                else:
                    assert dist(a.pose.xy, b.pose.xy) >= r[a.type] + r[b.type]
    assert types == {ObjectType.LEGO, ObjectType.TOY_CAR}


def test_stage1_robot_faces_goal_behind_object(cfg):
    sampler = TaskSampler(cfg.sim, cfg.task)
    rng = np.random.default_rng(1)
    for _ in range(100):
        ep = sampler.sample(1, rng)
        o = ep.objects[0].pose
        d = math.atan2(ep.goal.center[1] - o.y, ep.goal.center[0] - o.x)
        assert abs(math.remainder(ep.robot.yaw - d, 2 * math.pi)) <= cfg.task.sampling.stage1_yaw_noise_rad + 1e-9
        assert dist(ep.robot.xy, ep.goal.center) > dist(o.xy, ep.goal.center)


@pytest.mark.parametrize("stage", [2, 4])
def test_rear_half_plane(cfg, stage):
    sampler = TaskSampler(cfg.sim, cfg.task)
    rng = np.random.default_rng(2)
    for _ in range(200):
        ep = sampler.sample(stage, rng)
        o = ep.objects[0].pose
        dx, dy = ep.goal.center[0] - o.x, ep.goal.center[1] - o.y
        rx, ry = ep.robot.x - o.x, ep.robot.y - o.y
        assert dx * rx + dy * ry <= 1e-9


def test_same_seed_same_episode(cfg):
    sampler = TaskSampler(cfg.sim, cfg.task)
    assert sampler.sample(6, np.random.default_rng(5)) == sampler.sample(6, np.random.default_rng(5))


def test_impossible_sampling_raises(cfg):
    bad = dataclasses.replace(cfg.task, sampling=dataclasses.replace(cfg.task.sampling, robot_margin_m=0.39))
    with pytest.raises(RuntimeError):
        TaskSampler(cfg.sim, bad).sample(2, np.random.default_rng(0))


def test_unknown_stage_raises(cfg):
    with pytest.raises(ValueError):
        TaskSampler(cfg.sim, cfg.task).sample(99, np.random.default_rng(0))


def test_stage5_cannot_push_from_behind(cfg):
    # 物体の後ろ（ゴールと反対側）の押し始める位置は、ロボット中心が作業領域の外になる
    from toy_sweeper.robot_model import gripper_center_x
    sampler = TaskSampler(cfg.sim, cfg.task)
    rng = np.random.default_rng(11)
    half = cfg.sim.work_area.size_m[0] / 2
    for _ in range(100):
        ep = sampler.sample(5, rng)
        o = ep.objects[0].pose
        dx, dy = ep.goal.center[0] - o.x, ep.goal.center[1] - o.y
        n = math.hypot(dx, dy)
        behind = (o.x - dx / n * gripper_center_x(cfg.sim), o.y - dy / n * gripper_center_x(cfg.sim))
        assert max(abs(behind[0]), abs(behind[1])) > half - cfg.task.sampling.stage5_edge_distance_m[1] - 1e-9


def test_points(cfg):
    gp = gripper_point(cfg.sim, Pose2D(0, 0, 0))
    assert gp[0] == pytest.approx(gripper_center_x(cfg.sim))
    assert approach_point((0.0, 0.0), (1.0, 0.0), 0.04) == pytest.approx((-0.04, 0.0))


def test_random_goal_mode(cfg):
    task = dataclasses.replace(cfg.task, sampling=dataclasses.replace(cfg.task.sampling, goal_mode="random"))
    sampler = TaskSampler(cfg.sim, task)
    rng = np.random.default_rng(7)
    half = cfg.sim.work_area.size_m[0] / 2
    lo, hi = cfg.task.sampling.random_goal_size_m
    sizes = []
    for _ in range(200):
        ep = sampler.sample(6, rng)
        g = ep.goal
        assert ep.goal_name == "random"
        assert lo <= g.size[0] <= hi and lo <= g.size[1] <= hi
        assert abs(g.center[0]) + g.size[0] / 2 <= half + 1e-9 and abs(g.center[1]) + g.size[1] / 2 <= half + 1e-9
        assert not g.contains(ep.objects[0].pose.xy)
        sizes.append(g.size[0])
    assert max(sizes) - min(sizes) > 0.1


def test_stage6_clusters(cfg):
    # 一定の割合で、対象に接する物体（隙間 5 mm 以下）がある
    sampler = TaskSampler(cfg.sim, cfg.task)
    rng = np.random.default_rng(21)
    touching = 0
    n = 300
    for _ in range(n):
        ep = sampler.sample(6, rng)
        t = ep.objects[0]
        tsz = cfg.sim.objects[OBJECT_TYPE_NAMES[t.type]].size_m[:2]
        for o in ep.objects[1:]:
            osz = cfg.sim.objects[OBJECT_TYPE_NAMES[o.type]].size_m[:2]
            # 同じ向きで、対象の座標系で見て辺どうしの隙間が 5 mm 以下
            lx, ly = (lambda dx, dy, c, s: (c * dx + s * dy, -s * dx + c * dy))(
                o.pose.x - t.pose.x, o.pose.y - t.pose.y, math.cos(t.pose.yaw), math.sin(t.pose.yaw))
            if abs(o.pose.yaw - t.pose.yaw) < 1e-9 and (
                    abs(abs(lx) - (tsz[0] + osz[0]) / 2) <= 0.005 + 1e-9 or abs(abs(ly) - (tsz[1] + osz[1]) / 2) <= 0.005 + 1e-9):
                touching += 1
                break
    p = cfg.task.sampling.cluster_probability
    assert 0.5 * p < touching / n < 1.5 * p, touching / n


@pytest.mark.parametrize("mode", ["zones", "random"])
def test_stage8_home_zones(cfg, mode):
    """Stage 8（v18）: 他の置き場に物体が入っていて、対象と自由な対象外はゴールにも置き場にもかからない。"""
    from toy_sweeper.geometry import box_corners, signed_distance_to_rect
    task = dataclasses.replace(cfg.task, sampling=dataclasses.replace(cfg.task.sampling, goal_mode=mode))
    sampler = TaskSampler(cfg.sim, task)
    sp = cfg.task.sampling
    rng = np.random.default_rng(1)
    n_free = []
    for _ in range(100):
        ep = sampler.sample(8, rng)
        assert sp.stage8_zones[0] <= len(ep.zones) <= sp.stage8_zones[1]
        if mode == "zones":
            assert all(z in goal_rects(cfg.sim).values() and z != ep.goal for z in ep.zones)
        rects = (ep.goal, *ep.zones)
        for i, a in enumerate(rects):   # ゴールと置き場は重ならない
            for b in rects[i + 1:]:
                assert abs(a.center[0] - b.center[0]) >= (a.size[0] + b.size[0]) / 2 - 1e-9 or \
                    abs(a.center[1] - b.center[1]) >= (a.size[1] + b.size[1]) / 2 - 1e-9
        targets = [o for o in ep.objects if o.name in ep.target_names]
        assert sp.stage7_targets[0] <= len(targets) <= sp.stage7_targets[1]
        free = 0
        for o in ep.objects:
            corners = box_corners(o.pose, cfg.sim.objects[OBJECT_TYPE_NAMES[o.type]].size_m[:2])
            homes = [z for z in ep.zones if z.contains_all(corners)]
            if o.name in ep.target_names:
                assert not ep.goal.contains(o.pose.xy)   # ゴールについては Stage 7 と同じ（中心が外）
                assert all(min(signed_distance_to_rect(c, z) for c in corners) >= sp.goal_clearance_m for z in ep.zones)
            elif not homes:   # 置き場の外の対象外（Stage 7 と同じく、ゴールの中に丸ごと入っていてもよい）
                free += 1
                if not ep.goal.contains_all(corners):
                    assert all(min(signed_distance_to_rect(c, z) for c in corners) >= sp.goal_clearance_m for z in rects)
        for z in ep.zones:   # どの置き場にも物体がある
            lo, hi = sp.stage8_zone_objects
            k = sum(z.contains_all(box_corners(o.pose, cfg.sim.objects[OBJECT_TYPE_NAMES[o.type]].size_m[:2]))
                    for o in ep.objects)
            assert lo <= k <= hi
        n_free.append(free)
    assert min(n_free) >= sp.stage8_clutter[0] and max(n_free) <= sp.stage8_clutter[1]


# ---------- Stage 9（後片付けから始まる盤面、v24） ----------
def _stage9_task(cfg, mode, **kw):
    return dataclasses.replace(cfg.task, sampling=dataclasses.replace(cfg.task.sampling, goal_mode=mode, **kw))


def _corners(cfg, o):
    from toy_sweeper.geometry import box_corners
    return box_corners(o.pose, cfg.sim.objects[OBJECT_TYPE_NAMES[o.type]].size_m[:2])


def test_stage9_without_mistakes_is_stage8(cfg):
    """やり残しの確率 0 なら、Stage 9 の盤面は同じ Seed の Stage 8 の盤面そのまま（stage だけ 9、置き場の指定なし）。"""
    for mode in ("zones", "random"):
        sampler = TaskSampler(cfg.sim, _stage9_task(cfg, mode, stage9_mistake_probability=0.0))
        for seed in range(100):
            ep8 = sampler.sample(8, np.random.default_rng(seed))
            ep9 = sampler.sample(9, np.random.default_rng(seed))
            assert ep9 == dataclasses.replace(ep8, stage=9) and ep9.home_overrides == ()


@pytest.mark.parametrize("mode", ["zones", "random"])
def test_stage9_mistakes(cfg, mode):
    """やり残しのある盤面: 元の Stage 8 の盤面から、選んだ物体だけが条件どおりに置き直され、置き場の指定が付く。"""
    import copy
    from toy_sweeper.geometry import signed_distance_to_rect
    from toy_sweeper.task import start_home_index
    task = _stage9_task(cfg, mode, stage9_mistake_probability=1.0)
    sampler = TaskSampler(cfg.sim, task)
    sp = task.sampling
    half = cfg.sim.work_area.size_m[0] / 2
    r = {t: math.hypot(*cfg.sim.objects[OBJECT_TYPE_NAMES[t]].size_m[:2]) / 2 for t in ObjectType}
    rng = np.random.default_rng(9)
    n_boards = n_delivered = 0
    kinds = {"in": 0, "out": 0}
    counts = set()
    while n_boards < 300:
        base_rng = copy.deepcopy(rng)
        ep = sampler._try_stage9(rng)
        base = sampler._try_stage8(base_rng)
        if ep is None:
            continue
        assert base is not None
        n_boards += 1
        assert ep.stage == 9 and ep.robot == base.robot and ep.goal == base.goal and ep.zones == base.zones
        assert ep.target_names == base.target_names and [o.name for o in ep.objects] == [o.name for o in base.objects]
        over = dict(ep.home_overrides)
        assert len(over) == len(ep.home_overrides)               # 同じ物体を 2 回選ばない
        rects = (ep.goal, *ep.zones)
        moved_targets = [n for n in ep.target_names if n in over]
        assert moved_targets in ([], list(ep.target_names))      # 対象は全部置き直すか、全部そのまま
        n_delivered += bool(moved_targets)
        mistakes = [n for n in over if n not in ep.target_names]
        assert 1 <= len(mistakes) <= 2
        counts.add(len(mistakes))
        before = {o.name: o for o in base.objects}
        for o in ep.objects:
            b = before[o.name]
            assert (o.type, o.color) == (b.type, b.color)
            if o.name not in over:
                assert o == b
                continue
            m = sp.object_margin_m
            assert abs(o.pose.x) <= half - m and abs(o.pose.y) <= half - m
            home0 = start_home_index(cfg.sim, base, b)
            if o.name in ep.target_names:
                assert over[o.name] == -1 and ep.goal.contains_all(_corners(cfg, o))
            elif over[o.name] == -1:                               # 入れてしまった: 置き場の無い対象外をゴールの中へ
                kinds["in"] += 1
                assert home0 == -1 and ep.goal.contains_all(_corners(cfg, o))
            else:                                                   # 出してしまった: 置き場の縁の外へ
                kinds["out"] += 1
                assert over[o.name] == home0 >= 1
                zone = rects[home0]
                lo, hi = sp.stage9_home_out_m
                assert lo <= signed_distance_to_rect(o.pose.xy, zone) <= hi
                for z in rects:                                     # 今のゴールと他の置き場にかからない
                    if z is not zone:
                        assert min(signed_distance_to_rect(c, z) for c in _corners(cfg, o)) >= sp.goal_clearance_m
            # 置き直した物体は、他の物体ともロボットとも重ならない
            for q in ep.objects:
                if q.name != o.name:
                    assert dist(o.pose.xy, q.pose.xy) >= r[o.type] + r[q.type] + sp.object_gap_m
            assert dist(o.pose.xy, ep.robot.xy) >= robot_radius(cfg.sim) + r[o.type] + sp.object_gap_m
    assert counts == {1, 2} and kinds["in"] > 30 and kinds["out"] > 30, (counts, kinds)
    assert 0.3 < n_delivered / n_boards < 0.7, n_delivered


@pytest.mark.parametrize("mode", ["zones", "random"])
def test_stage9_wedged(cfg, mode):
    """挟まった対象外（v35）: 対象を全部ゴールに置き直した盤面で、入れてしまった対象外は、届けた対象のどれかに接して
    （同じ向きで、前後か左右に隙間 cluster_gap_m）、4 隅がゴールの中、作業領域の余白の内側、ロボットと重ならない。"""
    task = _stage9_task(cfg, mode, stage9_mistake_probability=1.0, stage9_all_delivered_probability=1.0,
                        stage9_wedged_probability=1.0)
    sampler = TaskSampler(cfg.sim, task)
    sp = task.sampling
    size = {t: cfg.sim.objects[OBJECT_TYPE_NAMES[t]].size_m[:2] for t in ObjectType}
    r = {t: math.hypot(*size[t]) / 2 for t in ObjectType}
    half = cfg.sim.work_area.size_m[0] / 2
    rng = np.random.default_rng(11)
    n_in = 0
    for _ in range(200):
        ep = sampler.sample(9, rng)
        objs = {o.name: o for o in ep.objects}
        targets = [objs[n] for n in ep.target_names]
        for n, h in ep.home_overrides:
            if h != -1 or n in ep.target_names:
                continue
            o = objs[n]
            n_in += 1
            assert ep.goal.contains_all(_corners(cfg, o))
            m = sp.object_margin_m
            assert abs(o.pose.x) <= half - m and abs(o.pose.y) <= half - m
            assert dist(o.pose.xy, ep.robot.xy) >= robot_radius(cfg.sim) + r[o.type] + sp.object_gap_m

            def touches(t):
                if o.pose.yaw != t.pose.yaw:
                    return False
                c, s_ = math.cos(-t.pose.yaw), math.sin(-t.pose.yaw)
                dx, dy = o.pose.x - t.pose.x, o.pose.y - t.pose.y
                local = (c * dx - s_ * dy, s_ * dx + c * dy)
                ts, ns = size[t.type], size[o.type]
                for a in (0, 1):
                    gap = abs(local[a]) - (ts[a] + ns[a]) / 2
                    if sp.cluster_gap_m[0] - 1e-9 <= gap <= sp.cluster_gap_m[1] + 1e-9 and abs(local[1 - a]) <= ts[1 - a] / 2 + 1e-9:
                        return True
                return False
            assert any(touches(t) for t in targets), (n, o)
    assert n_in > 50, n_in


def test_stage9_wedged_needs_delivered_boards(cfg):
    """対象を置き直さない盤面（stage9_all_delivered_probability 0）では、挟まった対象外の確率は使わない（乱数も引かない）。"""
    for mode in ("zones", "random"):
        a = TaskSampler(cfg.sim, _stage9_task(cfg, mode, stage9_mistake_probability=1.0,
                                              stage9_all_delivered_probability=0.0, stage9_wedged_probability=0.0))
        b = TaskSampler(cfg.sim, _stage9_task(cfg, mode, stage9_mistake_probability=1.0,
                                              stage9_all_delivered_probability=0.0, stage9_wedged_probability=1.0))
        for seed in range(50):
            assert a.sample(9, np.random.default_rng(seed)) == b.sample(9, np.random.default_rng(seed))


def test_stage9_mistake_rate(cfg):
    """やり残しは確率 stage9_mistake_probability（盤面の引き直しがあるので目安）。"""
    sampler = TaskSampler(cfg.sim, cfg.task)
    rng = np.random.default_rng(4)
    n = 400
    with_mistakes = sum(bool(sampler.sample(9, rng).home_overrides) for _ in range(n))
    p = cfg.task.sampling.stage9_mistake_probability
    assert abs(with_mistakes / n - p) < 0.1, with_mistakes


def test_start_home_index_follows_overrides(cfg):
    """置き場の番号: 指定があればそれ、無ければ開始の位置（今のゴールを最初に調べる）。範囲外の指定は例外。"""
    from toy_sweeper.geometry import Rect
    from toy_sweeper.sim2d import ObjectSpec
    from toy_sweeper.task import EpisodeSpec, start_home_index
    goal, home = Rect((0.3, 0.0), (0.2, 0.3)), Rect((-0.3, 0.3), (0.2, 0.2))
    a = ObjectSpec("a", ObjectType.LEGO, Pose2D(0.3, 0.0, 0.0), "red")      # ゴールの中
    b = ObjectSpec("b", ObjectType.LEGO, Pose2D(-0.3, 0.3, 0.0), "blue")    # 置き場の中
    c = ObjectSpec("c", ObjectType.LEGO, Pose2D(0.0, -0.3, 0.0), "blue")    # どこにもない
    ep = EpisodeSpec(9, Pose2D(0.0, 0.0, 0.0), (a, b, c), ("c",), None, "test", goal, (home,))
    assert [start_home_index(cfg.sim, ep, o) for o in (a, b, c)] == [0, 1, -1]
    ep2 = dataclasses.replace(ep, home_overrides=(("a", -1), ("c", 1)))
    assert [start_home_index(cfg.sim, ep2, o) for o in (a, b, c)] == [-1, 1, 1]
    for bad in (2, -2):
        with pytest.raises(ValueError, match="置き場の番号"):
            start_home_index(cfg.sim, dataclasses.replace(ep, home_overrides=(("a", bad),)), a)


@pytest.mark.parametrize("stage", [7, 8])
def test_multi_target_toy_cars(stage):
    """v28: multi_target_toy_car_probability > 0 なら、Stage 7・8 の対象にミニカーが混ざり（同じ色）、対象外は対象の色を
    使わない（指示「その色のもの全部」が 1 つに決まる）。0 なら今までどおり対象は LEGO だけ。"""
    import numpy as np
    from pathlib import Path
    from toy_sweeper.config import load_all
    from toy_sweeper.spaces import ObjectType
    from toy_sweeper.task import TaskSampler
    exp = Path(__file__).resolve().parents[1] / "config" / "experiments"
    cfg = load_all(overrides=(exp / "multi_toy_car.yaml",))
    sampler = TaskSampler(cfg.sim, cfg.task)
    n_car, n_all = 0, 0
    for seed in range(200):
        ep = sampler.sample(stage, np.random.default_rng(seed))
        tg = [o for o in ep.objects if o.name in ep.target_names]
        color = tg[0].color
        assert all(o.color == color for o in tg)
        assert all(o.color != color for o in ep.objects if o.name not in ep.target_names)
        n_car += sum(o.type == ObjectType.TOY_CAR for o in tg)
        n_all += len(tg)
    assert 0.4 < n_car / n_all < 0.6, n_car / n_all
    base = TaskSampler(load_all().sim, load_all().task)
    for seed in range(50):
        ep = base.sample(stage, np.random.default_rng(seed))
        assert all(o.type == ObjectType.LEGO for o in ep.objects if o.name in ep.target_names)


@pytest.mark.parametrize("stage", [7, 8])
def test_target_group(stage):
    """v29: target_group_probability > 0 なら、Stage 7・8 の 2 個目からの対象の一部が、それまでの対象の近く（外接円の
    隙間が target_group_gap_m の範囲、接しない）に置かれる。0（既定）ならそうした配置は偶然の分だけ。"""
    import math
    import numpy as np
    from pathlib import Path
    from toy_sweeper.config import load_all
    from toy_sweeper.task import TaskSampler
    exp = Path(__file__).resolve().parents[1] / "config" / "experiments"

    def grouped_fraction(cfg):
        sampler = TaskSampler(cfg.sim, cfg.task)
        lo, hi = cfg.task.sampling.target_group_gap_m
        n_g, n_all = 0, 0
        for seed in range(200):
            ep = sampler.sample(stage, np.random.default_rng(seed))
            tg = [o for o in ep.objects if o.name in ep.target_names]
            for i, o in enumerate(tg[1:], start=1):
                gaps = [math.dist(o.pose.xy, q.pose.xy) - sampler.obj_r[o.type] - sampler.obj_r[q.type] for q in tg[:i]]
                n_g += any(lo - 1e-9 <= g <= hi + 1e-9 for g in gaps)
                n_all += 1
        return n_g / n_all

    assert load_all().task.sampling.target_group_probability == 0.0
    grouped = grouped_fraction(load_all(overrides=(exp / "multi_toy_car.yaml", TEST_CFG / "target_group.yaml")))
    base = grouped_fraction(load_all(overrides=(exp / "multi_toy_car.yaml",)))
    assert grouped > 0.2 and base < 0.1, (grouped, base)


def _free_area_cfg(*extra):
    from pathlib import Path
    from toy_sweeper.config import load_all
    exp = Path(__file__).resolve().parents[1] / "config" / "experiments"
    return load_all(overrides=(exp / "free_area.yaml", *(TEST_CFG / e for e in extra)))


def test_free_area_boards():
    """v30: work_area_size_range_m を指定すると、盤面ごとに幅と奥行きを別々に引き（EpisodeSpec.area_size）、
    その大きさの中に今までと同じ規則で置く。既定（[]）は sim.work_area の固定で area_size は None。"""
    import numpy as np
    from toy_sweeper.config import load_all
    from toy_sweeper.task import STAGES, TaskSampler
    cfg = _free_area_cfg()
    (wlo, whi), (dlo, dhi) = cfg.task.sampling.work_area_size_range_m
    sampler = TaskSampler(cfg.sim, cfg.task)
    sp = cfg.task.sampling
    ws, ds = [], []
    for stage in STAGES:
        for seed in range(40):
            ep = sampler.sample(stage, np.random.default_rng(seed))
            w, d = ep.area_size
            assert wlo <= w <= whi and dlo <= d <= dhi
            ws.append(w)
            ds.append(d)
            hx, hy = w / 2, d / 2
            assert abs(ep.robot.x) <= hx - sp.robot_margin_m + 1e-9 and abs(ep.robot.y) <= hy - sp.robot_margin_m + 1e-9
            for o in ep.objects:
                assert abs(o.pose.x) < hx and abs(o.pose.y) < hy, (stage, seed, o)
            for g in (ep.goal, *ep.zones):
                assert abs(g.center[0]) + g.size[0] / 2 <= hx + 1e-9 and abs(g.center[1]) + g.size[1] / 2 <= hy + 1e-9
    ws, ds = np.array(ws), np.array(ds)
    assert ws.min() < wlo + 0.05 and ws.max() > whi - 0.05 and ds.min() < dlo + 0.05 and ds.max() > dhi - 0.05
    assert np.corrcoef(ws, ds)[0, 1] < 0.2      # 幅と奥行きは別々
    base = load_all()
    assert base.task.sampling.work_area_size_range_m == ()
    assert TaskSampler(base.sim, base.task).sample(7, np.random.default_rng(0)).area_size is None


def test_goal_rects_for_area():
    """v30: 名前つきのゴールを、近い縁からの距離を保って大きさの違う作業領域へ写す（中心 0 の軸は 0 のまま）。"""
    import pytest
    from toy_sweeper.config import load_all
    from toy_sweeper.task import goal_rects, goal_rects_for_area
    sim = load_all().sim
    assert goal_rects_for_area(sim, tuple(sim.work_area.size_m)) == goal_rects(sim)
    g = goal_rects_for_area(sim, (0.6, 1.0))
    assert g["top_right"].center == pytest.approx((0.2, 0.4)) and g["top_right"].size == (0.2, 0.2)
    assert g["bottom_left"].center == pytest.approx((-0.2, -0.4))
    assert g["top"].center == pytest.approx((0.0, 0.4)) and g["right"].center == pytest.approx((0.2, 0.0))


def test_free_area_zones_mode():
    """v30: goal_mode zones では、盤面の大きさに写した名前つきのゴールを使う（角と辺に接する）。"""
    import numpy as np
    from toy_sweeper.task import TaskSampler, goal_rects_for_area
    cfg = _free_area_cfg("goals_zones.yaml")
    sampler = TaskSampler(cfg.sim, cfg.task)
    for seed in range(30):
        ep = sampler.sample(8, np.random.default_rng(seed))
        zones = goal_rects_for_area(cfg.sim, ep.area_size)
        assert ep.goal == zones[ep.goal_name]
        assert all(z in zones.values() for z in ep.zones)


def test_max_objects_bounds_every_stage(cfg):
    """max_objects（Gazebo の World の物体の枠の数）は、どの Stage の盤面の物体数も上回らない（2026-10-23、v18 以降の
    Stage 8 は 15 を超え、World に置けずに止まった）。"""
    from toy_sweeper.task import max_objects
    sp = cfg.task.sampling
    assert max_objects(cfg.task) == max(1 + sp.clutter_range[1], sp.stage7_targets[1] + sp.stage7_clutter[1],
                                        sp.stage8_zones[1] * sp.stage8_zone_objects[1] + sp.stage7_targets[1]
                                        + sp.stage8_clutter[1])
    sampler = TaskSampler(cfg.sim, cfg.task)
    rng = np.random.default_rng(5)
    for stage in STAGES:
        n = max(len(sampler.sample(stage, rng).objects) for _ in range(100))
        assert n <= max_objects(cfg.task), (stage, n)
