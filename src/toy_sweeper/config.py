"""config/*.yaml を厳密に読み込む。

欠けたキー、未知のキー、型違いはすべて ConfigError にする（既定値で代用しない）。
"""
from __future__ import annotations

import dataclasses
import math
import types
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class ConfigError(ValueError):
    pass


def _convert(tp, value, where: str):
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return strict_dataclass_from_dict(tp, value, where)
    if origin in (list, tuple):
        if not isinstance(value, list):
            raise ConfigError(f"{where}: list が必要だが {type(value).__name__}")
        (inner,) = typing.get_args(tp)[:1]
        return tuple(_convert(inner, v, f"{where}[{i}]") for i, v in enumerate(value))
    if origin in (dict,):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: dict が必要")
        _, vt = typing.get_args(tp)
        return {k: _convert(vt, v, f"{where}.{k}") for k, v in value.items()}
    if origin in (typing.Union, types.UnionType):
        raise ConfigError(f"{where}: Union 型は設定に使わない")
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: float が必要だが {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: int が必要だが {value!r}")
        return value
    if tp is bool or tp is str:
        if not isinstance(value, tp):
            raise ConfigError(f"{where}: {tp.__name__} が必要だが {value!r}")
        return value
    raise ConfigError(f"{where}: 未対応の型 {tp}")


def strict_dataclass_from_dict(cls, data, where: str = ""):
    if not isinstance(data, dict):
        raise ConfigError(f"{where or cls.__name__}: dict が必要だが {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    missing = [n for n in names if n not in data]
    unknown = [k for k in data if k not in names]
    if missing:
        raise ConfigError(f"{where or cls.__name__}: missing keys {missing}")
    if unknown:
        raise ConfigError(f"{where or cls.__name__}: unknown keys {unknown}")
    return cls(**{n: _convert(hints[n], data[n], f"{where}.{n}" if where else n) for n in names})


# ---- config/sim.yaml ----
@dataclass(frozen=True)
class WorkAreaCfg:
    size_m: tuple[float, ...]


@dataclass(frozen=True)
class WallsCfg:
    enabled: bool
    height_m: float
    thickness_m: float
    friction: float
    # 壁なしのとき、ロボットか物体の中心が作業領域の縁から外へこの距離を超えたら失敗（v30。0 は v19nw と同じ）
    out_of_area_margin_m: float


@dataclass(frozen=True)
class GoalZoneCfg:
    center_m: tuple[float, ...]
    size_m: tuple[float, ...]


@dataclass(frozen=True)
class RobotCfg:
    body_size_m: tuple[float, ...]
    mass_kg: float
    wheel_radius_m: float
    wheel_separation_m: float
    axle_offset_x_m: float
    chassis_bottom_z_m: float
    skirt_bottom_z_m: float
    caster_radius_m: float
    wheel_width_m: float
    wheel_mass_kg: float
    wheel_floor_friction: float
    linear_speed_mps: float
    angular_speed_radps: float


@dataclass(frozen=True)
class GripperCfg:
    pivot_x_from_front_m: float
    pivot_half_spacing_m: float
    jaw_length_m: float
    jaw_beam_start_m: float
    jaw_lateral_offset_m: float
    jaw_width_m: float
    jaw_closed_angle_rad: float
    jaw_open_angle_rad: float
    actuation_time_s: float
    grip_holding_force_n: float
    jaw_speed_radps: float
    jaw_height_m: float
    jaw_bottom_z_m: float
    jaw_mass_kg: float


@dataclass(frozen=True)
class ObjectCfg:
    size_m: tuple[float, ...]
    mass_kg: float
    floor_friction_xy: tuple[float, ...]


@dataclass(frozen=True)
class ContactCfg:
    robot_object_friction: float


@dataclass(frozen=True)
class RealWorldEffectsCfg:
    command_latency_s: tuple[float, ...]
    wheel_gain: tuple[float, ...]
    # 左右共通の速さの倍率 [下限, 上限]（v31、電池の減りなど）。下限 == 上限なら乱数を引かない
    speed_gain: tuple[float, ...]
    # 車輪の速さが指令に追いつく時定数 [下限, 上限]（v36、旋回の惰性）。下限 == 上限なら乱数を引かない
    motor_lag_s: tuple[float, ...]
    obs_position_noise_m: float
    obs_yaw_noise_rad: float


@dataclass(frozen=True)
class GraspFailCfg:
    enabled: bool
    base_prob: float             # ずれが 0 でも失敗する確率
    offset_tol_m: float          # 横のずれがこれ以上で必ず失敗
    kick_m: tuple[float, ...]    # 失敗したときに物体が滑る距離 [下限, 上限]


@dataclass(frozen=True)
class JawStallCfg:
    """爪がふさがれたら止まる（2026-10-19。config/sim.yaml の説明）。"""
    enabled: bool
    depth_m: float


@dataclass(frozen=True)
class Sim2DCfg:
    physics_dt_s: float
    collision_slop_m: float
    wall_collision_thickness_m: float
    solver_iterations: int
    grasp_fail: GraspFailCfg
    jaw_stall: JawStallCfg


@dataclass(frozen=True)
class GazeboCfg:
    physics_step_s: float
    floor_friction: float
    pose_publish_hz: float
    jaw_p_gain: float
    jaw_d_gain: float
    jaw_max_torque_nm: float
    camera_pose: tuple[float, ...]
    camera_hfov_rad: float
    camera_image_px: tuple[int, ...]
    segmentation_camera: bool   # 物体ごとのマスクを撮るカメラも置く（学習する検出器のデータ）
    # 照明と床の色（2026-10-19、sim2real のため条件を変えて試す。config/sim.yaml の説明）
    light_diffuse: tuple[float, ...]
    light_direction: tuple[float, ...]
    light_cast_shadows: bool
    scene_ambient: tuple[float, ...]
    floor_rgb: tuple[float, ...]
    work_area_rgb: tuple[float, ...]
    parking_xy_m: tuple[float, ...]
    toy_car_wheel_radius_m: float
    toy_car_wheel_width_m: float
    toy_car_axle_x_frac: float
    toy_car_wheel_mass_kg: float
    toy_car_wheel_damping: float


@dataclass(frozen=True)
class SimCfg:
    work_area: WorkAreaCfg
    walls: WallsCfg
    goal_zones: dict[str, GoalZoneCfg]
    robot: RobotCfg
    gripper: GripperCfg
    objects: dict[str, ObjectCfg]
    contact: ContactCfg
    real_world_effects: RealWorldEffectsCfg
    sim2d: Sim2DCfg
    gazebo: GazeboCfg


# ---- config/task.yaml ----
@dataclass(frozen=True)
class ControlCfg:
    action_duration_s: float


@dataclass(frozen=True)
class EpisodeCfg:
    max_steps: int
    max_steps_per_extra_target: int


@dataclass(frozen=True)
class SuccessCfg:
    reach_tolerance_m: float
    require_cleanup: bool


@dataclass(frozen=True)
class SamplingCfg:
    object_margin_m: float
    robot_margin_m: float
    stage1_front_gap_m: tuple[float, ...]
    stage1_yaw_noise_rad: float
    robot_object_distance_m: tuple[float, ...]
    stage2_max_offset_rad: float
    stage5_edge_distance_m: tuple[float, ...]
    stage5_robot_distance_m: tuple[float, ...]
    stage5_yaw_noise_rad: float
    stage5_goal_max_angle_rad: float
    clutter_range: tuple[int, ...]
    cluster_probability: float
    cluster_max_neighbors: int
    cluster_gap_m: tuple[float, ...]
    toy_car_probability: float
    multi_target_toy_car_probability: float   # Stage 7・8 の対象の 1 個ずつがミニカーになる確率（v28）
    target_group_probability: float   # Stage 7・8 の 2 個目からの対象を、それまでの対象の近くに置く確率（v29）
    target_group_gap_m: tuple[float, ...]   # 近くに置くときの外接円どうしの隙間 [最小, 最大]（v29）
    # 盤面ごとの作業領域の [[幅の下限, 上限], [奥行きの下限, 上限]]（v30）。空は sim.work_area の固定
    work_area_size_range_m: tuple[tuple[float, ...], ...]
    object_gap_m: float
    goal_clearance_m: float
    goal_mode: str
    random_goal_size_m: tuple[float, ...]
    stage7_targets: tuple[int, ...]
    stage7_clutter: tuple[int, ...]
    stage8_zones: tuple[int, ...]
    stage8_zone_objects: tuple[int, ...]
    stage8_clutter: tuple[int, ...]
    stage9_mistake_probability: float
    stage9_all_delivered_probability: float
    stage9_home_out_m: tuple[float, ...]
    stage9_wedged_probability: float
    max_tries: int


@dataclass(frozen=True)
class OcclusionCfg:
    enabled: bool
    camera_xyz_m: tuple[float, ...]   # 仮想のカメラの位置（床の座標）
    robot_height_m: float             # 車体の陰を作る高さ
    margin_m: float                   # 車体の外形に足す余白
    object_mid_height_m: float        # 物体の高さの半分（陰の判定に使う点の高さ）
    held_radius_m: float              # 閉じた爪の中心からこの距離以内の物体は爪に隠れる


@dataclass(frozen=True)
class ObservationCfg:
    n_obstacle_slots: int
    n_target_slots: int
    held_lateral_tolerance_m: float
    occlusion: OcclusionCfg
    # 縁までの符号付き距離の測り方（v25）: "center"（中心）か "corners"（置き場を持つ物体と対象は外形の 4 隅のうち
    # いちばん外の角。置き場を持たない対象外は中心のまま）
    edge_distance: str
    # v26: 全対象が入った後、主対象を直すべき対象外にし、頭のゴールの欄をその行き先にする。頭に仕事の 3 次元を足す
    cleanup_primary: bool
    # v30: 頭の終わりに作業領域の半分の幅・奥行き（position_scale_m で割る）の 2 次元を足す
    area_size: bool
    # 位置を割る尺度 [x, y]（v30 で sim.work_area から分けた。今までの 0.8 m 四方の半分）
    position_scale_m: tuple[float, ...]

    def __post_init__(self):
        if self.edge_distance not in ("center", "corners"):
            raise ValueError(f"task.observation.edge_distance は center か corners: {self.edge_distance}")


@dataclass(frozen=True)
class RewardCfg:
    progress_per_m: float
    approach_per_m: float
    approach_offset_m: float
    success_bonus: float
    delivery_bonus: float
    foreign_in_goal_penalty: float
    failure_penalty: float
    non_target_per_m: float
    non_target_tolerance_m: float
    non_target_failure: bool
    chassis_contact_penalty: float
    non_target_near_goal_per_m: float
    non_target_near_goal_band_m: float
    redundant_gripper_penalty: float
    gripper_toggle_penalty: float   # 状態の変わる開閉 1 回あたり（v28、既定 0）
    time_penalty: float
    shaping_discount: float
    home_out_penalty: float
    leftover_penalty: float
    # v26: 全対象が入った後の近づく項を直すべき対象外へ、運ぶ項に対象外の「直すべき距離」を足す
    cleanup_shaping: bool

    def __post_init__(self):
        if not 0.0 < self.shaping_discount <= 1.0:
            raise ValueError(f"reward.shaping_discount は (0, 1]: {self.shaping_discount}")


@dataclass(frozen=True)
class TaskCfg:
    control: ControlCfg
    episode: EpisodeCfg
    observation: ObservationCfg
    success: SuccessCfg
    sampling: SamplingCfg
    reward: RewardCfg


# ---- config/rl.yaml ----
@dataclass(frozen=True)
class PPOCfg:
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    net_arch: tuple[int, ...]
    features: str                  # "flat"（Observation をそのまま全結合へ）か "set"（toy_sweeper.set_encoder）
    slot_hidden: tuple[int, ...]   # features: set のときの物体ごとの MLP φ の幅
    lr_schedule: str               # "constant" か "linear"（learning_rate から 0 へ、1 回の learn の中で直線に下げる）

    def __post_init__(self):
        if self.features not in ("flat", "set"):
            raise ValueError(f"rl.ppo.features は flat か set: {self.features}")
        if self.lr_schedule not in ("constant", "linear"):
            raise ValueError(f"rl.ppo.lr_schedule は constant か linear: {self.lr_schedule}")


@dataclass(frozen=True)
class TrainingCfg:
    n_envs: int
    train_seed: int
    total_timesteps: int
    device: str
    torch_threads: int
    env_backend: str      # python か native（Rust 版、native/）
    env_threads: int
    disjoint_env_seeds: bool   # native の env i の最初の Seed を train_seed × n_envs + i にする
    fast_ppo: bool        # GPU の rollout buffer、分布の検査を外して反復ごとに NaN の検査（結果は同じ）
    checkpoint_every_steps: int   # 途中のチェックポイントを保存する間隔（全 env の合計のステップ）


@dataclass(frozen=True)
class BCCfg:
    epochs: int
    batch_size: int
    learning_rate: float
    value_epochs: int


@dataclass(frozen=True)
class CurriculumCfg:
    stages: tuple[int, ...]
    promote_success_rate: tuple[float, ...]
    window_episodes: int
    rehearsal_probability: float


@dataclass(frozen=True)
class EvaluationCfg:
    seed_start: int
    n_episodes: int
    test_seed_start: int
    test_n_episodes: int


@dataclass(frozen=True)
class RLCfg:
    ppo: PPOCfg
    training: TrainingCfg
    bc: BCCfg
    curriculum: CurriculumCfg
    evaluation: EvaluationCfg


@dataclass(frozen=True)
class AllCfg:
    sim: SimCfg
    task: TaskCfg
    rl: RLCfg


def load_yaml(name: str, config_dir: Path = CONFIG_DIR) -> dict:
    path = config_dir / name
    if not path.is_file():
        raise ConfigError(f"設定ファイルがない: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def _merge(base: dict, over: dict, where: str) -> dict:
    """over を base に重ねる。base にないキーは ConfigError（打ち間違いを黙って無視しない）。リストは置き換え。"""
    out = dict(base)
    for k, v in over.items():
        if k not in base:
            raise ConfigError(f"上書き設定の {where}.{k} は元の設定にない")
        out[k] = _merge(base[k], v, f"{where}.{k}") if isinstance(v, dict) and isinstance(base[k], dict) else v
    return out


def load_effective_dicts(config_dir: Path = CONFIG_DIR, overrides: tuple[Path, ...] = ()) -> dict[str, dict]:
    """sim / task / rl の YAML を読み、overrides（config/experiments/*.yaml、差分だけを書く）を順に重ねる。"""
    d = {"sim": load_yaml("sim.yaml", config_dir), "task": load_yaml("task.yaml", config_dir),
         "rl": load_yaml("rl.yaml", config_dir)}
    for path in overrides:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"上書き設定ファイルがない: {path}")
        over = yaml.safe_load(path.read_text())
        if not isinstance(over, dict) or not set(over) <= set(d):
            raise ConfigError(f"{path}: トップレベルは sim / task / rl のどれか")
        d = _merge(d, over, path.name)
    return d


def load_all(config_dir: Path = CONFIG_DIR, overrides: tuple[Path, ...] = ()) -> AllCfg:
    d = load_effective_dicts(config_dir, overrides)
    cfg = AllCfg(
        sim=strict_dataclass_from_dict(SimCfg, d["sim"], "sim"),
        task=strict_dataclass_from_dict(TaskCfg, d["task"], "task"),
        rl=strict_dataclass_from_dict(RLCfg, d["rl"], "rl"),
    )
    validate(cfg)
    return cfg


def validate(cfg: AllCfg) -> None:
    """値どうしの整合を検査する。矛盾があれば ConfigError。"""
    def check(cond: bool, msg: str):
        if not cond:
            raise ConfigError(msg)

    s, t, r = cfg.sim, cfg.task, cfg.rl
    check(len(s.work_area.size_m) == 2, "work_area.size_m は 2 要素")
    check(len(s.robot.body_size_m) == 3, "robot.body_size_m は 3 要素")
    hx, hy = s.work_area.size_m[0] / 2, s.work_area.size_m[1] / 2
    check(len(s.goal_zones) >= 1, "goal_zones が空")
    for name, gz in s.goal_zones.items():
        check(len(gz.center_m) == 2 and len(gz.size_m) == 2, f"goal_zones.{name} は 2 要素")
        (gx, gy), (gw, gh) = gz.center_m, gz.size_m
        check(-hx <= gx - gw / 2 and gx + gw / 2 <= hx and -hy <= gy - gh / 2 and gy + gh / 2 <= hy,
              f"goal_zones.{name} が作業領域からはみ出している")
    check(set(s.objects) == {"lego", "toy_car"}, f"objects は lego と toy_car: {list(s.objects)}")
    for name, o in s.objects.items():
        check(len(o.size_m) == 3 and len(o.floor_friction_xy) == 2, f"objects.{name} の要素数")
    check(0 <= s.gripper.jaw_closed_angle_rad < s.gripper.jaw_open_angle_rad < 1.5, "爪の角度は 0 <= 閉 < 開 < 1.5 rad")
    lowest = min(o.size_m[2] for o in s.objects.values())
    check(s.robot.skirt_bottom_z_m < lowest,
          f"robot.skirt_bottom_z_m {s.robot.skirt_bottom_z_m} が物体の高さ {lowest} 以上。物体が車体の下に潜るので "
          "2D シミュレータの前提（平面の接触）が成り立たない")
    check(s.gripper.jaw_bottom_z_m < lowest, "爪の下端が物体より高く、爪で物体を押せない")
    check(s.robot.skirt_bottom_z_m <= s.robot.chassis_bottom_z_m, "スカート下端は車体底面以下")
    check(s.gripper.actuation_time_s > 0 and t.control.action_duration_s > 0, "時間は正")
    # 爪は物理の刻みごとに jaw_speed_radps で回る（sim2d.Sim2D）。開閉 1 回（actuation_time_s）の間に回り切る
    g, dt = s.gripper, s.sim2d.physics_dt_s
    check(g.jaw_speed_radps > 0, "gripper.jaw_speed_radps は正")
    n_jaw = math.ceil((g.jaw_open_angle_rad - g.jaw_closed_angle_rad) / (g.jaw_speed_radps * dt))
    check(n_jaw * dt <= g.actuation_time_s,
          f"gripper.jaw_speed_radps {g.jaw_speed_radps} では爪の開閉に {n_jaw * dt:.3f} s かかり、"
          f"actuation_time_s {g.actuation_time_s} s を超える")
    sp = t.sampling
    for rng in (sp.stage1_front_gap_m, sp.robot_object_distance_m):
        check(len(rng) == 2 and rng[0] < rng[1], f"範囲は [lo, hi] で lo < hi: {rng}")
    rwe = s.real_world_effects
    for name, rng in (("command_latency_s", rwe.command_latency_s), ("wheel_gain", rwe.wheel_gain)):
        check(len(rng) == 2 and rng[0] <= rng[1], f"real_world_effects.{name} は [下限, 上限]")
    check(0.0 <= rwe.command_latency_s[0] and rwe.command_latency_s[1] < t.control.action_duration_s,
          "command_latency_s は 0 以上、1 Action 未満")
    check(rwe.wheel_gain[0] > 0, "wheel_gain は正")
    check(len(rwe.speed_gain) == 2 and 0 < rwe.speed_gain[0] <= rwe.speed_gain[1],
          "real_world_effects.speed_gain は [下限, 上限] で 0 < 下限 <= 上限")
    ml = rwe.motor_lag_s
    check(len(ml) == 2 and 0 <= ml[0] <= ml[1] and (ml[1] == 0 or ml[0] >= s.sim2d.physics_dt_s),
          f"real_world_effects.motor_lag_s {ml} は [下限, 上限] で 0 <= 下限 <= 上限、上限 > 0 なら下限 >= physics_dt_s"
          f"（{s.sim2d.physics_dt_s}。dt / τ > 1 だと行き過ぎて振れる）")
    check(rwe.obs_position_noise_m >= 0 and rwe.obs_yaw_noise_rad >= 0, "観測ノイズは 0 以上")
    check(len(sp.clutter_range) == 2 and 0 <= sp.clutter_range[0] <= sp.clutter_range[1], "clutter_range は [lo, hi]")
    for name in ("stage8_zones", "stage8_zone_objects", "stage8_clutter"):
        lo_hi = getattr(sp, name)
        check(len(lo_hi) == 2 and 0 <= lo_hi[0] <= lo_hi[1], f"{name} は [lo, hi]")
    check(sp.stage8_zones[1] <= len(s.goal_zones) - 1,
          "stage8_zones の上限は goal_zones の数 − 1（goal_mode: zones では今のゴール以外のゴールが置き場）")
    check(0.0 <= sp.toy_car_probability <= 1.0, "toy_car_probability は 0〜1")
    check(0.0 <= sp.multi_target_toy_car_probability <= 1.0, "multi_target_toy_car_probability は 0〜1")
    check(0.0 <= sp.target_group_probability <= 1.0, "target_group_probability は 0〜1")
    check(len(sp.target_group_gap_m) == 2 and sp.object_gap_m <= sp.target_group_gap_m[0] <= sp.target_group_gap_m[1],
          f"target_group_gap_m は [下限, 上限] で object_gap_m <= 下限 <= 上限: {sp.target_group_gap_m}")
    for name in ("stage9_mistake_probability", "stage9_all_delivered_probability", "stage9_wedged_probability"):
        check(0.0 <= getattr(sp, name) <= 1.0, f"{name} は 0〜1")
    check(len(sp.stage9_home_out_m) == 2 and 0.0 < sp.stage9_home_out_m[0] <= sp.stage9_home_out_m[1],
          f"stage9_home_out_m は [下限, 上限] で 0 < 下限 <= 上限（中心が置き場の縁の外）: {sp.stage9_home_out_m}")
    check(sp.goal_mode in ("zones", "random"), f"goal_mode は zones か random: {sp.goal_mode}")
    check(len(sp.random_goal_size_m) == 2 and 0 < sp.random_goal_size_m[0] <= sp.random_goal_size_m[1] < min(hx, hy) * 2,
          "random_goal_size_m は [下限, 上限] で作業領域より小さい")
    check(s.walls.out_of_area_margin_m >= 0.0, "walls.out_of_area_margin_m は 0 以上")
    # 失敗に届く（壁なしで外へ出られる、または対象外を動かしすぎると失敗）とき、失敗で終わらせる方が止まり続ける
    # （価値 −time_penalty / (1 − γ)）より得にならない。v30（2026-10-22）は時間の罰 0.1・γ 0.99 で止まり続け −10、
    # 失敗 −5 で、方策がわざと外へ出て終わらせるようになった（成功率 0.9 → 0.2）
    rw = t.reward
    if not s.walls.enabled or rw.non_target_failure:
        stay = rw.time_penalty / (1 - r.ppo.gamma)
        check(rw.failure_penalty >= stay,
              f"失敗に届く設定で failure_penalty {rw.failure_penalty} が止まり続けの価値 {stay:.2f}（time_penalty / (1 − γ)）"
              "より軽い。失敗で終わらせる方が得になる")
    # v34: 終わりに残った直すべき物の減点（leftover_penalty）は、成功を避けて止まり続ける方が得にならない大きさにする。
    # 残りが 3 個（2026-10-22 の時間切れの調べで、ゴールの対象外の最大）でも、成功 − 3 × 減点が止まり続けの価値より上
    check(rw.leftover_penalty >= 0.0, "reward.leftover_penalty は 0 以上")
    if rw.leftover_penalty > 0:
        stay = rw.time_penalty / (1 - r.ppo.gamma)
        check(rw.success_bonus - 3 * rw.leftover_penalty > -stay,
              f"leftover_penalty {rw.leftover_penalty} が重すぎる: 3 個残した成功 {rw.success_bonus - 3 * rw.leftover_penalty:.2f} が"
              f"止まり続けの価値 {-stay:.2f} より低く、対象を入れ切らずに止まる方が得になる")
    ar = sp.work_area_size_range_m
    if ar:
        check(len(ar) == 2 and all(len(a) == 2 and 0 < a[0] <= a[1] for a in ar),
              f"work_area_size_range_m は [] か [[幅の下限, 上限], [奥行きの下限, 上限]]: {ar}")
        check(not s.walls.enabled, "work_area_size_range_m は壁なし（walls.enabled: false）のときだけ（壁は盤面ごとに作り直さない）")
        check(sp.random_goal_size_m[1] < min(ar[0][0], ar[1][0]),
              "random_goal_size_m の上限は、いちばん狭い作業領域（work_area_size_range_m の下限）より小さい")
        for name, gz in s.goal_zones.items():
            check(gz.size_m[0] <= ar[0][0] and gz.size_m[1] <= ar[1][0], f"goal_zones.{name} がいちばん狭い作業領域より大きい")
    check(len(t.observation.position_scale_m) == 2 and min(t.observation.position_scale_m) > 0,
          "task.observation.position_scale_m は正の 2 要素")
    check(t.observation.n_obstacle_slots >= 1, "n_obstacle_slots は 1 以上")
    check(len(r.curriculum.stages) == len(r.curriculum.promote_success_rate), "curriculum の長さ不一致")
    check(r.ppo.batch_size <= r.ppo.n_steps * r.training.n_envs, "batch_size が rollout より大きい")
    check(r.training.env_backend in ("python", "native"), f"rl.training.env_backend は python か native: {r.training.env_backend}")
    check(r.training.env_threads >= 1, "rl.training.env_threads は 1 以上")
    check(not r.training.fast_ppo or r.training.device == "cuda", "rl.training.fast_ppo は device: cuda のときだけ")
    check(r.training.checkpoint_every_steps >= r.training.n_envs, "rl.training.checkpoint_every_steps は n_envs 以上")
    check((r.ppo.n_steps * r.training.n_envs) % r.ppo.batch_size == 0,
          "rollout サイズ (n_steps*n_envs) が batch_size で割り切れない")
