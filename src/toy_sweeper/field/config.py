"""config/field.yaml の読み込みと検査（不正なら ConfigError、既定値で代用しない）。"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import CONFIG_DIR, AllCfg, ConfigError, load_yaml, strict_dataclass_from_dict


@dataclass(frozen=True)
class FieldControlCfg:
    command_ttl_s: float
    max_tick_lateness_s: float
    torch_threads: int
    persist: bool
    policy_stochastic: bool   # PPO の行動を確率的に選ぶ（評価と同じ。config/field.yaml）
    gripper_settle_s: float   # 爪の開け閉めを送った後、走らずに待つ時間


@dataclass(frozen=True)
class SafetyCfg:
    check_period_s: float
    max_frame_age_s: float
    max_obs_age_s: float
    stale_obs_hold_s: float
    startup_timeout_s: float


@dataclass(frozen=True)
class FieldPerceptionCfg:
    track_hold_s: float
    identify_min_seen_s: float   # 同定（VLM）に出す追跡の、見え続けた時間の下限
    recover_extra_back_steps: int
    pose_predict_max_s: float
    association_tolerance_m: float
    identity_chroma_tolerance: float      # 取り違えとみなす色度の差（×255）。付け直し・挟み損ねの判定も同じ
    identity_persist_frames: int          # 上限を超えたのがこの回数続いたら取り違えとみなす
    identity_skip_closed_radius_m: float  # 爪を閉じていて対象が爪の中心からこれ以内なら色を見ない
    identity_skip_open_radius_m: float    # 爪の開閉によらず、対象が爪の中心からこれ以内なら色を見ない
    completion_confirm_back_steps: int    # 完了の前に色を確かめるため下がる歩数の上限（field.runtime.Controller）
    goal_margin_m: float      # Policy に渡すゴールと完了の判定のゴールを、縁から内側へ縮める量


@dataclass(frozen=True)
class SimWorldCfg:
    camera_fps: float
    capture_latency_s: float
    command_latency_s: float
    robot_pose_noise_m: float
    robot_yaw_noise_rad: float
    max_realtime_lag_s: float


@dataclass(frozen=True)
class CameraCfg:
    device_name: str
    width_px: int
    height_px: int
    fps: float
    capture_latency_s: float


@dataclass(frozen=True)
class BleCfg:
    device_name_prefix: str
    uart_service_uuid: str
    uart_rx_char_uuid: str
    uart_tx_char_uuid: str
    scan_timeout_s: float
    connect_timeout_s: float
    write_with_response: bool
    max_payload_bytes: int
    action_motor: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class RealCfg:
    robot_mask_height_m: float
    body_x_m: tuple[float, ...]
    body_half_width_m: float
    mount_x_m: float
    mount_height_m: float
    robot_margin_m: float
    motion_delay_s: float        # 姿勢の先読みの動き方（runtime.MotionModel）: 遅れ
    motion_lag_s: float          # 時定数
    motion_gain_linear: float    # 前進の倍率
    motion_gain_angular: float   # 旋回の倍率
    motion_gain_window_pairs: int   # 走りながら倍率を割り出す、新しい組の数（field.motion_gain）
    motion_gain_min_pairs: int      # これだけ組が溜まるまでは motion_gain_* を使う
    close_px: int
    color_split: bool
    merged_max_length_m: float
    track_chroma_gate: float
    identity_chroma_tolerance: float
    persist: bool
    green_lime_split_hue_deg: float
    camera_match_max_diff: float
    detector: str            # 物体の塊の作り方: background（背景との差）/ learned（学習した検出器、config/seg_detector.yaml）


@dataclass(frozen=True)
class RecordCfg:
    tasks: bool             # 指示ごとの動画（field.record.TaskRecorder）を残す
    hold_after_s: float     # 終わった後も記録する秒数


@dataclass(frozen=True)
class FieldCfg:
    control: FieldControlCfg
    safety: SafetyCfg
    perception: FieldPerceptionCfg
    sim: SimWorldCfg
    camera: CameraCfg
    ble: BleCfg
    real: RealCfg
    record: RecordCfg


def validate_field_cfg(f: FieldCfg, cfg: AllCfg) -> None:
    """値どうしの前提を確かめる。崩れていれば ConfigError。"""
    T = cfg.task.control.action_duration_s

    def check(cond: bool, msg: str):
        if not cond:
            raise ConfigError(f"field.yaml: {msg}")

    for name, v in [("control.command_ttl_s", f.control.command_ttl_s),
                    ("control.max_tick_lateness_s", f.control.max_tick_lateness_s),
                    ("safety.check_period_s", f.safety.check_period_s),
                    ("safety.max_frame_age_s", f.safety.max_frame_age_s),
                    ("safety.max_obs_age_s", f.safety.max_obs_age_s),
                    ("safety.startup_timeout_s", f.safety.startup_timeout_s),
                    ("perception.track_hold_s", f.perception.track_hold_s),
                    ("perception.identify_min_seen_s", f.perception.identify_min_seen_s),
                    ("perception.association_tolerance_m", f.perception.association_tolerance_m),
                    ("perception.identity_chroma_tolerance", f.perception.identity_chroma_tolerance),
                    ("perception.identity_persist_frames", f.perception.identity_persist_frames),
                    ("perception.identity_skip_closed_radius_m", f.perception.identity_skip_closed_radius_m),
                    ("perception.identity_skip_open_radius_m", f.perception.identity_skip_open_radius_m),
                    ("perception.completion_confirm_back_steps", f.perception.completion_confirm_back_steps),
                    ("perception.goal_margin_m", f.perception.goal_margin_m),
                    ("sim.camera_fps", f.sim.camera_fps), ("sim.max_realtime_lag_s", f.sim.max_realtime_lag_s),
                    ("camera.fps", f.camera.fps), ("ble.scan_timeout_s", f.ble.scan_timeout_s),
                    ("ble.connect_timeout_s", f.ble.connect_timeout_s)]:
        check(v > 0, f"{name} は正: {v}")
    for name, v in [("sim.capture_latency_s", f.sim.capture_latency_s),
                    ("sim.command_latency_s", f.sim.command_latency_s),
                    ("sim.robot_pose_noise_m", f.sim.robot_pose_noise_m),
                    ("sim.robot_yaw_noise_rad", f.sim.robot_yaw_noise_rad),
                    ("camera.capture_latency_s", f.camera.capture_latency_s)]:
        check(v >= 0, f"{name} は 0 以上: {v}")
    check(f.control.torch_threads >= 1, f"control.torch_threads は 1 以上: {f.control.torch_threads}")
    check(f.control.max_tick_lateness_s < T, f"control.max_tick_lateness_s は 1 周期 {T} s より短い")
    # 指令の間でロボットが止まらないこと（毎周期の指令が、前の指令の期限内に届く）
    need = T + f.sim.command_latency_s + f.control.max_tick_lateness_s
    check(f.control.command_ttl_s > need,
          f"control.command_ttl_s {f.control.command_ttl_s} は 周期 + 指令の遅れ + 周期の遅れの上限 = {need:.3f} s より長くする")
    check(f.safety.check_period_s < f.safety.max_frame_age_s, "safety.check_period_s は max_frame_age_s より短い")
    # 正常に動いているときの観測の古さ（フレームの間隔 + 撮影の遅れ）で止まらないこと
    for where, fps, lat in [("sim", f.sim.camera_fps, f.sim.capture_latency_s),
                            ("camera", f.camera.fps, f.camera.capture_latency_s)]:
        normal = 1.0 / fps + lat
        check(f.safety.max_obs_age_s > normal,
              f"safety.max_obs_age_s {f.safety.max_obs_age_s} が {where} の通常の古さ {normal:.3f} s 以下")
        check(f.safety.max_frame_age_s > normal,
              f"safety.max_frame_age_s {f.safety.max_frame_age_s} が {where} の通常の古さ {normal:.3f} s 以下")
    r = f.real
    check(len(r.body_x_m) == 2 and r.body_x_m[0] < 0 < r.body_x_m[1], f"real.body_x_m は [後ろ（負）, 前（正）]: {r.body_x_m}")
    check(r.body_half_width_m > 0 and r.robot_mask_height_m > 0, f"real の大きさは正: {r}")
    check(0 <= r.robot_margin_m < 0.05, f"real.robot_margin_m は 0〜5 cm: {r.robot_margin_m}")
    check(0 <= r.motion_delay_s < 1 and 0 <= r.motion_lag_s < 1 and 0.5 <= r.motion_gain_linear <= 2
          and 0.5 <= r.motion_gain_angular <= 2, f"real.motion_* の範囲（遅れ・時定数 0〜1 s、倍率 0.5〜2）: {r}")
    check(0 < r.motion_gain_min_pairs <= r.motion_gain_window_pairs,
          f"0 < real.motion_gain_min_pairs <= motion_gain_window_pairs: {r.motion_gain_min_pairs}、{r.motion_gain_window_pairs}")
    check(0 < r.mount_x_m < r.body_x_m[1], f"real.mount_x_m は車軸と車体の前の端の間: {r.mount_x_m}、{r.body_x_m}")
    check(0 < r.mount_height_m <= r.robot_mask_height_m,
          f"real.mount_height_m は 0 より大きく robot_mask_height_m 以下: {r.mount_height_m}、{r.robot_mask_height_m}")
    want = {"FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT"}
    check(set(f.ble.action_motor) == want, f"ble.action_motor の鍵は {sorted(want)}: {sorted(f.ble.action_motor)}")
    check(all(len(m) == 2 and all(isinstance(x, int) and -100 <= x <= 100 for x in m) for m in f.ble.action_motor.values()),
          f"ble.action_motor の値は -100〜100 の整数の [左, 右]: {f.ble.action_motor}")
    check(f.ble.max_payload_bytes >= 1, f"ble.max_payload_bytes は 1 以上: {f.ble.max_payload_bytes}")


def load_field_cfg(cfg: AllCfg, config_dir=CONFIG_DIR) -> FieldCfg:
    f = strict_dataclass_from_dict(FieldCfg, load_yaml("field.yaml", config_dir), "field")
    validate_field_cfg(f, cfg)
    return f
