"""学習中の例外（Rust 版の env の表示: 盤面、行動の列、reset の直前の乱数の状態、Curriculum）を Python 版の env で再生する
（2026-10-19。物理の回帰テスト test/test_sim2d.py が使う）。"""
from __future__ import annotations

import re

import numpy as np

from .geometry import Pose2D, Rect
from .sim2d import ObjectSpec
from .spaces import ObjectType
from .task import EpisodeSpec

F = r"(-?[0-9.e+-]+)"


def _pose(s: str) -> Pose2D:
    m = re.fullmatch(rf"Pose2D \{{ x: {F}, y: {F}, yaw: {F} \}}", s.strip())
    return Pose2D(*map(float, m.groups()))


def _rect(s: str) -> Rect:
    m = re.fullmatch(rf"Rect \{{ center: \({F}, {F}\), size: \({F}, {F}\) \}}", s.strip())
    a = list(map(float, m.groups()))
    return Rect((a[0], a[1]), (a[2], a[3]))


def parse(text: str) -> tuple[EpisodeSpec, list[int]]:
    """表示から (盤面, 行動の列)。"""
    ep = next(l for l in text.splitlines() if l.startswith("episode: "))
    acts = next(l for l in text.splitlines() if l.startswith("actions: "))
    stage = int(re.search(r"stage: (\d+)", ep).group(1))
    robot = _pose(re.search(r"robot: (Pose2D \{[^}]*\})", ep).group(1))
    objs = [ObjectSpec(n, ObjectType.LEGO if t == "Lego" else ObjectType.TOY_CAR, _pose(p), c) for n, t, p, c in
            re.findall(r'ObjectSpec \{ name: "(\w+)", otype: (\w+), pose: (Pose2D \{[^}]*\}), color: "(\w+)" \}', ep)]
    targets = tuple(re.findall(r'"(\w+)"', re.search(r"target_names: \[([^\]]*)\]", ep).group(1)))
    goal_name = re.search(r'goal_name: "(\w+)"', ep).group(1)
    goal = _rect(re.search(r"goal: (Rect \{[^}]*\})", ep).group(1))
    zones = tuple(_rect(z) for z in re.findall(r"Rect \{[^}]*\}", re.search(r"zones: \[(.*?)\], home_overrides", ep).group(1)))
    over = tuple((n, int(i)) for n, i in re.findall(r'\("(\w+)", (-?\d+)\)', re.search(r"home_overrides: \[(.*?)\]", ep).group(1)))
    return EpisodeSpec(stage, robot, tuple(objs), targets, None, goal_name, goal, zones, over), \
        [int(a) for a in re.findall(r"\d+", acts.split(":", 1)[1])]


def rng_and_curriculum(text: str):
    """表示の reset の直前の乱数の状態 (state, inc, uinteger か None) と Curriculum (stage, previous, rehearsal)。無ければ None。"""
    m = re.search(r"rng_at_reset: Some\(\((\d+), (\d+), (None|Some\((\d+)\))\)\)", text)
    if not m:
        return None
    c = re.search(r"curriculum: stage (\d+), previous \[([\d, ]*)\], rehearsal ([\d.e-]+)", text)
    rng = (int(m.group(1)), int(m.group(2)), None if m.group(3) == "None" else int(m.group(4)))
    cur = (int(c.group(1)), tuple(int(x) for x in re.findall(r"\d+", c.group(2))), float(c.group(3)))
    return rng, cur


def reset_like_crash(env, text: str) -> bool:
    """env を表示と同じ状態から reset する。乱数の状態があれば、Curriculum をそろえて同じ乱数から盤面を引き直し（不完全さの
    乱数まで同じ）、表示の盤面と同じことを確かめて True。無ければ盤面を渡して Seed 0 で reset し False。"""
    spec, _ = parse(text)
    rc = rng_and_curriculum(text)
    if rc is None:
        env.reset(seed=0, options={"episode": spec})
        return False
    (state, inc, uint), cur = rc
    env.set_curriculum(*cur)
    env.reset(seed=0)           # 乱数を作ってから状態を入れ替える
    bg = np.random.PCG64()
    st = bg.state
    st["state"] = {"state": state, "inc": inc}
    st["has_uint32"], st["uinteger"] = (0, 0) if uint is None else (1, uint)
    bg.state = st
    env.np_random = np.random.Generator(bg)
    env.reset()
    if env.episode != spec:
        raise RuntimeError(f"乱数の状態から引いた盤面が表示と違う:\n{env.episode}\n{spec}")
    return True
