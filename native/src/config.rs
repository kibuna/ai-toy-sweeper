//! Python の設定（toy_sweeper.config の SimCfg と TaskCfg）を JSON から読む。JSON は toy_sweeper.native.cfg_json が作る。
//!
//! 項目は Python のデータクラスと同じ名前・同じ並び。知らない項目があれば読む時点で例外にする（deny_unknown_fields）。
//! Python 側に設定を足したのに Rust 側が知らないまま学習する（黙って無視する）のを防ぐ。
//! derived: Python で計算して渡す値（numpy の総和の順序に依る値をビット単位で同じにするため）。
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkAreaCfg {
    pub size_m: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WallsCfg {
    pub enabled: bool,
    pub height_m: f64,
    pub thickness_m: f64,
    pub friction: f64,
    /// v30: 壁なしのとき、中心が作業領域の縁から外へこの距離を超えたら失敗（0 は v19nw）
    pub out_of_area_margin_m: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalZoneCfg {
    pub center_m: Vec<f64>,
    pub size_m: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotCfg {
    pub body_size_m: Vec<f64>,
    pub mass_kg: f64,
    pub wheel_radius_m: f64,
    pub wheel_separation_m: f64,
    pub axle_offset_x_m: f64,
    pub chassis_bottom_z_m: f64,
    pub skirt_bottom_z_m: f64,
    pub caster_radius_m: f64,
    pub wheel_width_m: f64,
    pub wheel_mass_kg: f64,
    pub wheel_floor_friction: f64,
    pub linear_speed_mps: f64,
    pub angular_speed_radps: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GripperCfg {
    pub pivot_x_from_front_m: f64,
    pub pivot_half_spacing_m: f64,
    pub jaw_length_m: f64,
    pub jaw_beam_start_m: f64,
    pub jaw_lateral_offset_m: f64,
    pub jaw_width_m: f64,
    pub jaw_closed_angle_rad: f64,
    pub jaw_open_angle_rad: f64,
    pub actuation_time_s: f64,
    pub grip_holding_force_n: f64,
    pub jaw_speed_radps: f64,
    pub jaw_height_m: f64,
    pub jaw_bottom_z_m: f64,
    pub jaw_mass_kg: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectCfg {
    pub size_m: Vec<f64>,
    pub mass_kg: f64,
    pub floor_friction_xy: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactCfg {
    pub robot_object_friction: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealWorldEffectsCfg {
    pub command_latency_s: Vec<f64>,
    pub wheel_gain: Vec<f64>,
    /// v31: 左右共通の速さの倍率 [下限, 上限]。下限 == 上限なら乱数を引かない
    pub speed_gain: Vec<f64>,
    /// v36: 車輪の速さが指令に追いつく時定数 [下限, 上限]（旋回の惰性）。下限 == 上限なら乱数を引かない
    pub motor_lag_s: Vec<f64>,
    pub obs_position_noise_m: f64,
    pub obs_yaw_noise_rad: f64,
}

/// sim2d.jaw_stall（2026-10-19）
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JawStallCfg {
    pub enabled: bool,
    pub depth_m: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraspFailCfg {
    pub enabled: bool,
    pub base_prob: f64,
    pub offset_tol_m: f64,
    pub kick_m: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sim2DCfg {
    pub physics_dt_s: f64,
    pub collision_slop_m: f64,
    pub wall_collision_thickness_m: f64,
    pub solver_iterations: i64,
    pub grasp_fail: GraspFailCfg,
    pub jaw_stall: JawStallCfg,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimCfg {
    pub work_area: WorkAreaCfg,
    pub walls: WallsCfg,
    pub goal_zones: BTreeMap<String, GoalZoneCfg>,
    pub robot: RobotCfg,
    pub gripper: GripperCfg,
    pub objects: BTreeMap<String, ObjectCfg>,
    pub contact: ContactCfg,
    pub real_world_effects: RealWorldEffectsCfg,
    pub sim2d: Sim2DCfg,
    /// Gazebo の設定（2D では使わない）
    pub gazebo: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlCfg {
    pub action_duration_s: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpisodeCfg {
    pub max_steps: i64,
    pub max_steps_per_extra_target: i64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuccessCfg {
    pub reach_tolerance_m: f64,
    pub require_cleanup: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingCfg {
    pub object_margin_m: f64,
    pub robot_margin_m: f64,
    pub stage1_front_gap_m: Vec<f64>,
    pub stage1_yaw_noise_rad: f64,
    pub robot_object_distance_m: Vec<f64>,
    pub stage2_max_offset_rad: f64,
    pub stage5_edge_distance_m: Vec<f64>,
    pub stage5_robot_distance_m: Vec<f64>,
    pub stage5_yaw_noise_rad: f64,
    pub stage5_goal_max_angle_rad: f64,
    pub clutter_range: Vec<i64>,
    pub cluster_probability: f64,
    pub cluster_max_neighbors: i64,
    pub cluster_gap_m: Vec<f64>,
    pub toy_car_probability: f64,
    pub multi_target_toy_car_probability: f64,
    pub target_group_probability: f64,
    pub target_group_gap_m: Vec<f64>,
    /// v30: 盤面ごとの作業領域の [[幅の下限, 上限], [奥行きの下限, 上限]]。空は sim.work_area の固定
    pub work_area_size_range_m: Vec<Vec<f64>>,
    pub object_gap_m: f64,
    pub goal_clearance_m: f64,
    pub goal_mode: String,
    pub random_goal_size_m: Vec<f64>,
    pub stage7_targets: Vec<i64>,
    pub stage7_clutter: Vec<i64>,
    pub stage8_zones: Vec<i64>,
    pub stage8_zone_objects: Vec<i64>,
    pub stage8_clutter: Vec<i64>,
    pub stage9_mistake_probability: f64,
    pub stage9_all_delivered_probability: f64,
    pub stage9_home_out_m: Vec<f64>,
    pub stage9_wedged_probability: f64,
    pub max_tries: i64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OcclusionCfg {
    pub enabled: bool,
    pub camera_xyz_m: Vec<f64>,
    pub robot_height_m: f64,
    pub margin_m: f64,
    pub object_mid_height_m: f64,
    pub held_radius_m: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationCfg {
    pub n_obstacle_slots: i64,
    pub n_target_slots: i64,
    pub held_lateral_tolerance_m: f64,
    pub occlusion: OcclusionCfg,
    /// v25: "center" か "corners"（spaces.ObservationBuilder._edge_distance）
    pub edge_distance: String,
    /// v26: 全対象が入った後、主対象を直すべき対象外に（snapshot.build_snapshot）。頭に仕事の 3 次元
    pub cleanup_primary: bool,
    /// v30: 頭の終わりに作業領域の半分の幅・奥行き / position_scale_m の 2 次元
    pub area_size: bool,
    /// v30: 位置を割る尺度 [x, y]（sim.work_area から分けた）
    pub position_scale_m: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardCfg {
    pub progress_per_m: f64,
    pub approach_per_m: f64,
    pub approach_offset_m: f64,
    pub success_bonus: f64,
    pub delivery_bonus: f64,
    pub foreign_in_goal_penalty: f64,
    pub failure_penalty: f64,
    pub non_target_per_m: f64,
    pub non_target_tolerance_m: f64,
    pub non_target_failure: bool,
    pub chassis_contact_penalty: f64,
    pub non_target_near_goal_per_m: f64,
    pub non_target_near_goal_band_m: f64,
    pub redundant_gripper_penalty: f64,
    pub gripper_toggle_penalty: f64,
    pub time_penalty: f64,
    pub shaping_discount: f64,
    pub home_out_penalty: f64,
    pub leftover_penalty: f64,
    /// v26: 近づく項を直すべき対象外へ、運ぶ項に直すべき距離（env._approach_distance、_progress_distance）
    pub cleanup_shaping: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCfg {
    pub control: ControlCfg,
    pub episode: EpisodeCfg,
    pub observation: ObservationCfg,
    pub success: SuccessCfg,
    pub sampling: SamplingCfg,
    pub reward: RewardCfg,
}

/// Python で計算して渡す値（toy_sweeper.native.cfg_json）。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedCfg {
    /// 物体の種類（lego, toy_car）→ sim2d.effective_friction_radius（numpy の平均。総和の順序まで同じにするため Python で計算）
    pub effective_friction_radius: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cfg {
    pub sim: SimCfg,
    pub task: TaskCfg,
    pub derived: DerivedCfg,
}

/// 物体の種類（spaces.ObjectType と同じ番号）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectType {
    Lego = 0,
    ToyCar = 1,
}

impl ObjectType {
    pub fn name(self) -> &'static str {
        match self {
            ObjectType::Lego => "lego",
            ObjectType::ToyCar => "toy_car",
        }
    }
    pub fn from_index(i: i64) -> ObjectType {
        match i {
            0 => ObjectType::Lego,
            1 => ObjectType::ToyCar,
            _ => panic!("未知の物体の種類: {i}"),
        }
    }
}

impl Cfg {
    pub fn from_json(s: &str) -> Result<Cfg, String> {
        let c: Cfg = serde_json::from_str(s).map_err(|e| format!("native の設定が読めない: {e}"))?;
        c.validate()?;
        Ok(c)
    }

    /// Rust 版が前提にしている形を確かめる（Python の config.validate は済んでいる前提）。
    fn validate(&self) -> Result<(), String> {
        let s = &self.sim;
        let chk = |ok: bool, msg: &str| if ok { Ok(()) } else { Err(format!("native の設定: {msg}")) };
        chk(s.work_area.size_m.len() == 2, "work_area.size_m は 2 つ")?;
        chk(s.robot.body_size_m.len() >= 2, "robot.body_size_m は 2 つ以上")?;
        for name in ["lego", "toy_car"] {
            let o = s.objects.get(name).ok_or(format!("native の設定: objects.{name} がない"))?;
            chk(o.size_m.len() >= 2 && o.floor_friction_xy.len() == 2, "objects の size_m と floor_friction_xy")?;
            chk(self.derived.effective_friction_radius.contains_key(name), "derived.effective_friction_radius")?;
        }
        chk(s.objects.len() == 2, "objects は lego と toy_car だけ")?;
        chk(s.real_world_effects.command_latency_s.len() == 2 && s.real_world_effects.wheel_gain.len() == 2
                && s.real_world_effects.speed_gain.len() == 2 && s.real_world_effects.motor_lag_s.len() == 2,
            "real_world_effects の範囲は [下限, 上限]")?;
        chk(s.sim2d.grasp_fail.kick_m.len() == 2, "sim2d.grasp_fail.kick_m は [下限, 上限]")?;
        chk(self.task.observation.occlusion.camera_xyz_m.len() == 3, "occlusion.camera_xyz_m は 3 つ")?;
        let ed = self.task.observation.edge_distance.as_str();
        chk(ed == "center" || ed == "corners", "observation.edge_distance は center か corners")?;
        let gm = self.task.sampling.goal_mode.as_str();
        chk(gm == "zones" || gm == "random", "sampling.goal_mode は zones か random")?;
        chk(self.task.sampling.stage9_home_out_m.len() == 2, "sampling.stage9_home_out_m は [下限, 上限]")?;
        chk(self.task.sampling.target_group_gap_m.len() == 2, "sampling.target_group_gap_m は [下限, 上限]")?;
        let ar = &self.task.sampling.work_area_size_range_m;
        chk(ar.is_empty() || (ar.len() == 2 && ar.iter().all(|a| a.len() == 2)),
            "sampling.work_area_size_range_m は [] か [[幅の下限, 上限], [奥行きの下限, 上限]]")?;
        chk(ar.is_empty() || !s.walls.enabled, "sampling.work_area_size_range_m は壁なしのときだけ")?;
        chk(self.task.observation.position_scale_m.len() == 2, "observation.position_scale_m は 2 つ")?;
        Ok(())
    }

    pub fn object(&self, t: ObjectType) -> &ObjectCfg {
        &self.sim.objects[t.name()]
    }

    pub fn friction_radius(&self, t: ObjectType) -> f64 {
        self.derived.effective_friction_radius[t.name()]
    }
}
