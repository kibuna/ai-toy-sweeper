"""比較用の Policy。どれも Observation だけを見て Action を返す。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AllCfg
from .geometry import wrap_angle
from .robot_model import front_x, gripper_center_x, jaw_inner_half_gap
from .spaces import N_ACTIONS, OBJECT_TYPE_NAMES, Action, GripperState, ObjectType, ObservationBuilder


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def predict(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(N_ACTIONS))


@dataclass(frozen=True)
class HeuristicParams:
    """Heuristic の調整値。2D シミュレータ Stage 1〜3 の成功率を見て手で決めた。"""
    heading_tol_rad: float = 0.3       # 旋回ステップ 14° の約 1.2 倍。これより向きがずれていたら回る
    push_heading_tol_rad: float = 0.25  # 押している最中にゴール方向へ向け直す閾値
    waypoint_tol_m: float = 0.025      # 中継点に着いたとみなす距離（1 前進 2 cm より少し大きい）
    staging_clearance_m: float = 0.04  # 押し始める点を、押す姿勢からさらに後ろへ下げる距離
    side_offset_m: float = 0.12        # 物体の横を回り込む中継点の距離（車体半対角 6 cm の 2 倍）


class HeuristicPolicy:
    """物体の後ろ（ゴールと反対側）へ回り込み、ゴール方向へ押す。毎ステップ再計画する。"""
    name = "heuristic"

    def __init__(self, cfg: AllCfg, params: HeuristicParams = HeuristicParams()):
        self.cfg = cfg
        self.p = params
        self.dec = ObservationBuilder(cfg.sim, cfg.task)
        self.front = front_x(cfg.sim)
        self.grip_x = gripper_center_x(cfg.sim)
        self.open_angle = cfg.sim.gripper.jaw_open_angle_rad

    def reset(self) -> None:
        pass

    def _go_to(self, r, yaw, w) -> int:
        e = wrap_angle(math.atan2(w[1] - r[1], w[0] - r[0]) - yaw)
        if abs(e) > self.p.heading_tol_rad:
            return Action.TURN_LEFT if e > 0 else Action.TURN_RIGHT
        return Action.FORWARD

    def predict(self, obs: np.ndarray) -> int:
        d = self.dec.decode(obs)
        r, yaw, o, g = d["robot_xy"], d["robot_yaw"], d["target_xy"], d["goal_center"]
        if d["target_type"] is None:  # Stage 0: 指定点へ行く
            return int(self._go_to(r, yaw, o))

        push_dir = math.atan2(g[1] - o[1], g[0] - o[0])
        u = np.array([math.cos(push_dir), math.sin(push_dir)])
        # 物体のロボット座標
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly = c * (o[0] - r[0]) + s * (o[1] - r[1]), -s * (o[0] - r[0]) + c * (o[1] - r[1])
        in_cage = 0.0 < lx and abs(ly) < 0.8 * max(jaw_inner_half_gap(self.cfg.sim, self.open_angle, lx), 0.0)
        if in_cage and abs(wrap_angle(push_dir - yaw)) < math.pi / 2:
            # 押している: ロボットからゴールへ向かう方向に舵を取る
            e = wrap_angle(math.atan2(g[1] - r[1], g[0] - r[0]) - yaw)
            if abs(e) > self.p.push_heading_tol_rad:
                return int(Action.TURN_LEFT if e > 0 else Action.TURN_RIGHT)
            return int(Action.FORWARD)

        # 押し始める点（物体の後ろ）
        stage_pt = o - (self.grip_x + self.p.staging_clearance_m) * u
        rel = r - o
        if float(rel @ u) > -(self.front * 0.5):
            # ロボットが物体の横〜ゴール側にいる: 物体の横の中継点を経由する
            perp = np.array([-u[1], u[0]])
            side = 1.0 if float(rel @ perp) >= 0 else -1.0
            w = o + side * self.p.side_offset_m * perp - 0.5 * self.p.side_offset_m * u
            return int(self._go_to(r, yaw, w))
        if np.linalg.norm(stage_pt - r) > self.p.waypoint_tol_m and abs(float((r - stage_pt) @ np.array([-u[1], u[0]]))) > 0.01:
            return int(self._go_to(r, yaw, stage_pt))
        # 後ろに着いた: 物体へ向いて進む
        return int(self._go_to(r, yaw, o))


class CarryHeuristicPolicy:
    """挟んで運ぶ台本: 主対象へ向かい、爪の中央に来たら閉じ、ゴールへ運んで開く。毎ステップ再計画する。

    盤面が解けるかの確認と、模倣による初期化のための基準。
    """
    name = "carry"
    AIM_TOL_RAD = 0.15      # 物体へ向く角度の許容（旋回 1 回 14° の約 0.6 倍）
    GRASP_TOL_M = 0.01      # 物体が爪の中央からこれ以内なら閉じる
    HOLD_TOL_M = 0.015      # 閉じた爪の中央からこれ以内に物体があれば、持っているとみなす

    def __init__(self, cfg: AllCfg):
        self.cfg = cfg
        self.h = HeuristicPolicy(cfg)
        self.front = front_x(cfg.sim)
        self.grip_x = gripper_center_x(cfg.sim)
        self.half_len = {t: cfg.sim.objects[n].size_m[0] / 2 for t, n in OBJECT_TYPE_NAMES.items()}

    def reset(self) -> None:
        pass

    def _grasp_x(self, ttype) -> float:
        """物体中心をここ（ロボット座標の前方位置）に置けば挟める。長い物体は後ろ端が車体に当たらない位置まで前へ。"""
        return max(self.grip_x, self.front + self.half_len[ttype] + 0.003)

    def predict(self, obs: np.ndarray) -> int:
        d = self.h.dec.decode(obs)
        r, yaw, o, g = d["robot_xy"], d["robot_yaw"], d["target_xy"], d["goal_center"]
        if d["target_type"] is None:
            return int(self.h._go_to(r, yaw, o))
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly = c * (o[0] - r[0]) + s * (o[1] - r[1]), -s * (o[0] - r[0]) + c * (o[1] - r[1])
        gx = self._grasp_x(d["target_type"])
        closed = d["gripper"] == GripperState.CLOSED
        held = closed and abs(lx - gx) < self.HOLD_TOL_M + (gx - self.grip_x) and abs(ly) < self.HOLD_TOL_M
        gs = d["goal_size"]
        in_goal = abs(o[0] - g[0]) < gs[0] / 2 - 0.03 and abs(o[1] - g[1]) < gs[1] / 2 - 0.03
        if held:
            return int(Action.OPEN_GRIPPER if in_goal else self.h._go_to(r, yaw, g))
        if closed:
            return int(Action.OPEN_GRIPPER)
        if abs(lx - gx) < self.GRASP_TOL_M and abs(ly) < self.GRASP_TOL_M:
            return int(Action.CLOSE_GRIPPER)
        e = wrap_angle(math.atan2(o[1] - r[1], o[0] - r[0]) - yaw)
        if abs(e) > self.AIM_TOL_RAD:
            return int(Action.TURN_LEFT if e > 0 else Action.TURN_RIGHT)
        return int(Action.FORWARD if lx > gx else Action.BACKWARD)


class MixedHeuristicPolicy:
    """台本の組み合わせ: LEGO は挟んで運ぶ（CarryHeuristicPolicy）、ミニカーは押す（HeuristicPolicy）。

    ミニカー（幅 30 mm）は爪の根元（内側の隙間 約 11.5 mm）より奥に入らず、2D の把持の判定（中心が梁の間）を満たせない。
    前後に転がるので押して運ぶ方が向く。模倣による初期化の手本に使う。
    """
    name = "mixed"

    def __init__(self, cfg: AllCfg):
        self.carry, self.push = CarryHeuristicPolicy(cfg), HeuristicPolicy(cfg)
        self.dec = self.push.dec

    def reset(self) -> None:
        pass

    def predict(self, obs: np.ndarray) -> int:
        t = self.dec.decode(obs)["target_type"]
        return (self.push if t in (None, ObjectType.TOY_CAR) else self.carry).predict(obs)


def check_observation_config(model: Path, cfg: AllCfg) -> None:
    """学習のときと観測の値の意味が同じ設定か（v25）。次元が同じで意味だけ違う設定（edge_distance）は、違っても
    例外にならず黙って別物を測るので、学習の記録（checkpoints/<run>/config_snapshot/task.yaml）と比べて止める。
    v25 より前の記録には edge_distance が無く、そのときのコードは中心で測っていたので center とみなす。"""
    import yaml
    snap = model.parent / "config_snapshot" / "task.yaml"
    if not snap.is_file():
        raise FileNotFoundError(f"モデルの学習の記録が無い（観測の設定を確かめられない）: {snap}")
    trained = (yaml.safe_load(snap.read_text()).get("observation") or {}).get("edge_distance", "center")
    now = cfg.task.observation.edge_distance
    if trained != now:
        raise ValueError(f"{model} は observation.edge_distance: {trained} で学習したが、今の設定は {now}"
                         f"（config/experiments/corner_edge.yaml の付け忘れ・付け間違い）")


class PPOPolicy:
    name = "ppo"

    def __init__(self, path: Path, deterministic: bool = True, *, cfg: AllCfg):
        from stable_baselines3 import PPO
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        check_observation_config(Path(path), cfg)
        self.model = PPO.load(path, device="cpu")
        self.deterministic = deterministic

    def reset(self) -> None:
        pass

    def predict(self, obs: np.ndarray) -> int:
        a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(a)
