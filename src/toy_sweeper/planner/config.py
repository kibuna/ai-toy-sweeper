"""config/planner.yaml の読み込み（不正なら ConfigError、既定値で代用しない）。値の根拠は yaml のコメント。"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pathlib import Path

import yaml

from ..config import CONFIG_DIR, ConfigError, _merge, load_yaml, strict_dataclass_from_dict


@dataclass(frozen=True)
class GridCfg:
    resolution_m: float
    heading_bins: int


@dataclass(frozen=True)
class SearchCfg:
    cell_m: float
    heuristic_weights: tuple[float, ...]
    max_expansions: tuple[int, ...]
    backward_extra_cost: float
    soft_cost: float
    overlap_cost: float
    lateral_shift_max_m: float
    heuristic_clearance_m: float
    lateral_shift_turns: tuple[int, ...]


@dataclass(frozen=True)
class MarginsCfg:
    hard_margin_m: float
    soft_margin_m: float
    tail_margin_m: float


@dataclass(frozen=True)
class HeldCfg:
    lateral_tol_m: float


@dataclass(frozen=True)
class GraspCfg:
    approach_lateral_tol_m: float
    close_x_inset_m: float
    close_lateral_tol_m: float
    approach_extra_m: float
    max_failures: int
    neighbor_gap_m: float
    neighbor_max_displacement_m: float
    max_push_m: float
    toy_car_first: bool
    toy_car_max_rel_angle_rad: float


@dataclass(frozen=True)
class DropCfg:
    goal_edge_margin_m: float
    object_clearance_m: tuple[float, ...]
    edge_band_m: float


@dataclass(frozen=True)
class ForeignCfg:
    out_clearance_m: float
    max_relocation_m: float
    wall_margin_m: float


@dataclass(frozen=True)
class PushCfg:
    min_rel_angle_rad: float
    disc_x_m: float
    disc_extra_m: float
    goal_margin_m: float
    approach_extra_m: float
    sweep_step_m: float
    max_push_m: float
    contact_x_window_m: tuple[float, ...]
    contact_lateral_tol_m: float
    approach_lateral_tol_m: float
    settled_rel_angle_rad: tuple[float, ...]
    via_lengths_m: tuple[float, ...]
    via_rel_angle_extra_rad: float
    max_via: int
    pivot_offset_m: float
    pivot_steps: int
    pivot_min_rel_angle_rad: float
    max_pivot: int


@dataclass(frozen=True)
class ReplanCfg:
    pose_tol_m: float
    yaw_tol_rad: float
    object_move_tol_m: float


@dataclass(frozen=True)
class RecoveryCfg:
    blocked_retry_steps: int
    stuck_steps: int
    stuck_motion_m: float


@dataclass(frozen=True)
class PlannerCfg:
    grid: GridCfg
    search: SearchCfg
    margins: MarginsCfg
    held: HeldCfg
    grasp: GraspCfg
    drop: DropCfg
    foreign: ForeignCfg
    push: PushCfg
    replan: ReplanCfg
    recovery: RecoveryCfg


def validate_planner_cfg(c: PlannerCfg) -> None:
    def check(cond: bool, msg: str):
        if not cond:
            raise ConfigError(f"planner: {msg}")
    check(c.grid.resolution_m > 0 and c.grid.heading_bins >= 8, "grid の値が不正")
    check(c.search.cell_m > 0 and len(c.search.heuristic_weights) >= 1
          and len(c.search.heuristic_weights) == len(c.search.max_expansions)
          and all(wt >= 1.0 for wt in c.search.heuristic_weights) and all(n > 0 for n in c.search.max_expansions),
          "search の値が不正（heuristic_weights と max_expansions は同じ長さ、重みは 1 以上）")
    check(min(c.margins.hard_margin_m, c.margins.soft_margin_m, c.margins.tail_margin_m) >= 0, "margins は 0 以上")
    check(c.grasp.close_x_inset_m >= 0 and c.grasp.max_failures >= 1, "grasp の値が不正")
    check(len(c.push.contact_x_window_m) == 2 and c.push.contact_x_window_m[0] < c.push.contact_x_window_m[1],
          "push.contact_x_window_m は [下限, 上限]")
    check(len(c.drop.object_clearance_m) >= 1
          and list(c.drop.object_clearance_m) == sorted(c.drop.object_clearance_m, reverse=True),
          "drop.object_clearance_m は広い順に 1 個以上")
    check(0 < c.push.min_rel_angle_rad <= math.pi / 2, "push.min_rel_angle_rad は (0, π/2]")
    check(len(c.push.settled_rel_angle_rad) == 2 and 0 < c.push.settled_rel_angle_rad[0] < c.push.settled_rel_angle_rad[1]
          <= math.pi / 2, "push.settled_rel_angle_rad は 0 < 下限 < 上限 ≤ π/2")
    check(len(c.push.via_lengths_m) >= 1 and list(c.push.via_lengths_m) == sorted(c.push.via_lengths_m)
          and c.push.via_lengths_m[0] > 0, "push.via_lengths_m は短い順の正の値")
    check(c.push.max_via >= 0, "push.max_via は 0 以上")
    check(c.recovery.blocked_retry_steps > 0 and c.recovery.stuck_steps > 0, "recovery のステップ数は正")


def load_planner_cfg(config_dir=CONFIG_DIR, overrides: tuple[Path, ...] = ()) -> PlannerCfg:
    """config/planner.yaml に、overrides（トップレベルが planner: の差分の YAML、比較実験用）を順に重ねて読む。"""
    d = load_yaml("planner.yaml", config_dir)
    for path in overrides:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"上書き設定ファイルがない: {path}")
        over = yaml.safe_load(path.read_text())
        if not isinstance(over, dict) or set(over) != {"planner"}:
            raise ConfigError(f"{path}: トップレベルは planner だけ")
        d = _merge(d, over["planner"], path.name)
    cfg = strict_dataclass_from_dict(PlannerCfg, d, "planner")
    validate_planner_cfg(cfg)
    return cfg
