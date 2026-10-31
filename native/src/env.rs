//! toy_sweeper.env.ToySweeperEnv の Rust 版（物理は Sim2D だけ）。観測は spaces.ObservationBuilder、
//! snapshot.build_snapshot と同じ計算。
//!
//! Python 版と合わせたところ:
//! - 乱数は env の np_random 1 つ（Python の Sim2D の挟み損ねも同じ乱数）。引く順も同じ: reset では復習の Stage
//!   （random、integers）→ 盤面 → 遅延 → 車輪の利得 2 つ → 観測ノイズ。step では挟み損ね → 観測ノイズ。
//! - 浮動小数点の演算は Python と同じ順。float の sum() は CPython 3.12 の Neumaier の総和（PySum）。
//! - 観測は f64 で計算してから f32 に丸める（np.asarray(v, dtype=float32) と同じ）。
//! - 失敗は黙って落とさない: 壁があるのに外に出たら State（RuntimeError）、観測が値域の外なら Invalid（ValueError）。
//!   どちらも盤面と行動の列（_diagnostics）を付ける。
use crate::config::{Cfg, ObjectType};
use crate::geometry::{box_corners, dist, hypot, py_max, signed_distance_to_rect, wrap_angle, Pose2D, Rect};
use crate::rng::NpRng;
use crate::robot_model::{body_center_x, front_x};
use crate::sampler::{approach_point, gripper_point, start_home_index, EpisodeSpec, TaskSampler, STAGES};
use crate::sim2d::{GripperState, ObjectSpec, Sim2D, SimError};
use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// 指令遅延を丸める刻み（env.LATENCY_GRID_S）
pub const LATENCY_GRID_S: f64 = 0.025;
pub const N_ACTIONS: usize = 7;
/// spaces.Action
pub const OPEN_GRIPPER: u8 = 5;
pub const CLOSE_GRIPPER: u8 = 6;
/// spaces.BASE_DIM / EXTRA_DIM / SLOT_DIM / OBS_BOUND
pub const BASE_DIM: usize = 38;
pub const EXTRA_DIM: usize = 1;
pub const SLOT_DIM: usize = 12;
/// spaces.MODE_DIM（v26、cleanup_primary のときだけ頭に足す）
pub const MODE_DIM: usize = 3;
/// spaces.AREA_DIM（v30、area_size のときだけ頭の終わりに足す）
pub const AREA_DIM: usize = 2;

/// spaces.TaskMode（v26）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskMode {
    Deliver = 0,
    Return = 1,
    Remove = 2,
}
pub const OBS_BOUND: f32 = 3.0;
/// info["reward_terms"] の鍵（Python の dict と同じ順）。前の 7 つがポテンシャルの項
pub const REWARD_TERM_NAMES: [&str; 14] = [
    "progress",
    "approach",
    "delivery",
    "foreign_in_goal",
    "non_target",
    "non_target_near_goal",
    "home_out",
    "success",
    "failure",
    "redundant_gripper",
    "chassis_contact",
    "time",
    "gripper_toggle",
    "leftover",
];
pub const N_TERMS: usize = REWARD_TERM_NAMES.len();
const N_PHI: usize = 7;

/// env の失敗。Invalid は Python の ValueError、State は RuntimeError
#[derive(Debug, Clone, PartialEq)]
pub enum EnvError {
    Invalid(String),
    State(String),
}

impl EnvError {
    pub fn message(&self) -> &str {
        match self {
            EnvError::Invalid(m) | EnvError::State(m) => m,
        }
    }
    /// メッセージの前に文を足す（種類はそのまま）
    pub fn prefixed(self, p: &str) -> EnvError {
        match self {
            EnvError::Invalid(m) => EnvError::Invalid(format!("{p}{m}")),
            EnvError::State(m) => EnvError::State(format!("{p}{m}")),
        }
    }
}

impl std::fmt::Display for EnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl From<SimError> for EnvError {
    fn from(e: SimError) -> EnvError {
        match e {
            SimError::Invalid(m) => EnvError::Invalid(m),
            SimError::State(m) => EnvError::State(m),
        }
    }
}

impl From<EnvError> for PyErr {
    fn from(e: EnvError) -> PyErr {
        match e {
            EnvError::Invalid(m) => PyValueError::new_err(m),
            EnvError::State(m) => PyRuntimeError::new_err(m),
        }
    }
}

/// panic（盤面が引けない、など）も EnvError::State にする（Python は RuntimeError / ValueError を投げるところ）
pub fn guard<T>(f: impl FnOnce() -> Result<T, EnvError>) -> Result<T, EnvError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(p) => {
            let msg = if let Some(s) = p.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = p.downcast_ref::<String>() {
                s.clone()
            } else {
                "panic".to_string()
            };
            Err(EnvError::State(format!("native の env が止まった: {msg}")))
        }
    }
}

/// CPython 3.12 の sum()（float の Neumaier の総和）。0 から始めて 1 つずつ足す
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PySum {
    s: f64,
    c: f64,
}

impl PySum {
    pub fn add(&mut self, x: f64) {
        let t = self.s + x;
        if self.s.abs() >= x.abs() {
            self.c += (self.s - t) + x;
        } else {
            self.c += (x - t) + self.s;
        }
        self.s = t;
    }
    pub fn value(&self) -> f64 {
        if self.c != 0.0 && self.c.is_finite() {
            self.s + self.c
        } else {
            self.s
        }
    }
    pub fn of(xs: impl IntoIterator<Item = f64>) -> f64 {
        let mut s = PySum::default();
        for x in xs {
            s.add(x);
        }
        s.value()
    }
}

/// 失敗の種類（info["failure"]）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    OutOfArea = 0,
    NonTargetMoved = 1,
}

impl Failure {
    pub fn name(self) -> &'static str {
        match self {
            Failure::OutOfArea => "out_of_area",
            Failure::NonTargetMoved => "non_target_moved",
        }
    }
}

/// step の info（Python の dict と同じ中身）
#[derive(Debug, Clone, PartialEq)]
pub struct StepInfo {
    pub is_success: bool,
    pub failure: Option<Failure>,
    pub stage: i64,
    pub reward_terms: [f64; N_TERMS],
    pub steps: i64,
    pub n_delivered: i64,
    pub n_targets: i64,
    pub n_foreign_in_goal: i64,
    pub n_home_out: i64,
    pub n_chassis_contact: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepOut {
    pub obs: Vec<f32>,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info: StepInfo,
}

/// spaces.SlotObject
#[derive(Debug, Clone, Copy)]
struct SlotObject {
    pose: Pose2D,
    otype: ObjectType,
    displacement: (f64, f64),
    home: Option<Rect>,
}

/// spaces.WorldSnapshot（obs_age_s は学習では常に 0）
#[derive(Debug, Clone)]
struct WorldSnapshot {
    robot: Pose2D,
    target: Pose2D,
    target_type: Option<ObjectType>,
    goal: Rect,
    gripper: GripperState,
    prev_action: u8,
    obs_age_s: f64,
    other_targets: Vec<SlotObject>,
    obstacles: Vec<SlotObject>,
    n_remaining: i64,
    robot_motion: (f64, f64, f64),
    time_left: f64,
    target_held: bool,
    /// v26: 主対象の行き先（頭のゴールの欄）といまの仕事（Stage 0 は None）
    destination: Rect,
    task_mode: Option<TaskMode>,
    /// v30: 盤面の作業領域の (幅, 奥行き)
    area_size: (f64, f64),
}

/// spaces.ObservationBuilder の定数
struct ObsConsts {
    half: (f64, f64),
    size: (f64, f64),
    diag: f64,
    action_duration: f64,
    n_target_slots: usize,
    n_obstacle_slots: usize,
    max_targets: i64,
    step_m: f64,
    step_rad: f64,
    dim: usize,
    /// edge_distance: corners（v25）
    corners: bool,
    /// cleanup_primary（v26）
    cleanup: bool,
    /// area_size（v30）
    area: bool,
    /// 観測の値域（spaces.obs_bound、v30。今までの設定では OBS_BOUND）。numpy と同じく f32 で比べる
    bound: f32,
}

/// spaces.obs_bound: ±OBS_BOUND を、いちばん広い作業領域（壁なしなら外へ出てよい分を足す）と尺度の比で広げる
fn obs_bound(cfg: &Cfg) -> f32 {
    let (sim, task) = (&cfg.sim, &cfg.task);
    let ar = &task.sampling.work_area_size_range_m;
    let mut ext = if ar.is_empty() { sim.work_area.size_m[0].max(sim.work_area.size_m[1]) } else { ar[0][1].max(ar[1][1]) };
    if !sim.walls.enabled {
        ext += 2.0 * sim.walls.out_of_area_margin_m;
    }
    let ps = &task.observation.position_scale_m;
    (OBS_BOUND as f64 * (ext / (2.0 * ps[0].min(ps[1]))).max(1.0)) as f32
}

impl ObsConsts {
    fn new(cfg: &Cfg) -> ObsConsts {
        let (sim, task) = (&cfg.sim, &cfg.task);
        // 位置の尺度（v30 で sim.work_area から分けた。spaces.ObservationBuilder と同じ）
        let ps = &task.observation.position_scale_m;
        let (sx, sy) = (2.0 * ps[0], 2.0 * ps[1]);
        let nt = usize::try_from(task.observation.n_target_slots).expect("n_target_slots が負");
        let no = usize::try_from(task.observation.n_obstacle_slots).expect("n_obstacle_slots が負");
        assert!(task.sampling.stage7_targets.len() == 2, "sampling.stage7_targets は [最小, 最大]");
        let head_dim = BASE_DIM + EXTRA_DIM + if task.observation.cleanup_primary { MODE_DIM } else { 0 }
            + if task.observation.area_size { AREA_DIM } else { 0 };
        ObsConsts {
            half: (ps[0], ps[1]),
            size: (sx, sy),
            diag: hypot(sx, sy),
            action_duration: task.control.action_duration_s,
            n_target_slots: nt,
            n_obstacle_slots: no,
            max_targets: task.sampling.stage7_targets[1],
            step_m: sim.robot.linear_speed_mps * task.control.action_duration_s,
            step_rad: sim.robot.angular_speed_radps * task.control.action_duration_s,
            dim: head_dim + SLOT_DIM * (nt + no),
            corners: task.observation.edge_distance == "corners",
            cleanup: task.observation.cleanup_primary,
            area: task.observation.area_size,
            bound: obs_bound(cfg),
        }
    }
}

/// 1 エピソードの状態（reset で作る）
struct Episode {
    ep: EpisodeSpec,
    types: Vec<ObjectType>,
    is_target: Vec<bool>,
    /// 対象の物体の番号（target_names の順）
    targets: Vec<usize>,
    /// 置き場（snapshot.home_zone。対象も含めて全物体。Python の _homes は対象外だけ）
    homes: Vec<Option<Rect>>,
    started_in_goal: Vec<bool>,
    last_seen: Vec<Option<Pose2D>>,
    latency_s: f64,
    wheel_gain: (f64, f64),
    applied_cmd: (f64, f64),
    prev_action: u8,
    history: Vec<u8>,
    steps: i64,
    last_primary: Option<usize>,
    last_obs_robot: Option<Pose2D>,
    phi: [f64; N_PHI],
    max_steps: i64,
}

pub struct Env {
    cfg: Cfg,
    sim: Sim2D,
    sampler: TaskSampler,
    oc: ObsConsts,
    area: Rect,
    stage: i64,
    previous: Vec<i64>,
    rehearsal: f64,
    rng: Option<NpRng>,
    e: Option<Episode>,
    /// 観測した物体の姿勢（毎回使い回す）
    obs_poses: Vec<Pose2D>,
    /// 今のエピソードの reset の始めの乱数の状態（Chipmunk の assert の報告に付ける。盤面を渡したときは None）
    rng_at_reset: Option<RngState>,
}

fn obj_size(cfg: &Cfg, t: ObjectType) -> (f64, f64) {
    let s = &cfg.object(t).size_m;
    (s[0], s[1])
}

/// snapshot.inside_goal
fn inside_goal(cfg: &Cfg, goal: &Rect, t: ObjectType, pose: &Pose2D) -> bool {
    goal.contains_all(&box_corners(pose, obj_size(cfg, t)))
}

/// task.approach_point（ゴール中心と一致したら Python と同じく ValueError）
fn approach_pt(obj: (f64, f64), goal: (f64, f64), offset: f64) -> Result<(f64, f64), EnvError> {
    if hypot(goal.0 - obj.0, goal.1 - obj.1) == 0.0 {
        return Err(EnvError::Invalid("物体がゴール中心と一致していて押す方向が決まらない".into()));
    }
    Ok(approach_point(obj, goal, offset))
}

/// snapshot.fix_destination_xy（v26）: 戻すは置き場の中心、出すはゴールの中心から物体の向こう側
fn fix_destination_xy(goal: &Rect, home: Option<&Rect>, pose: &Pose2D) -> (f64, f64) {
    match home {
        Some(h) => h.center,
        None => (2.0 * pose.x - goal.center.0, 2.0 * pose.y - goal.center.1),
    }
}

/// Python の list の repr 風（['robot', 'obj3']）
fn py_list_repr(names: &[&str]) -> String {
    let v: Vec<String> = names.iter().map(|n| format!("'{n}'")).collect();
    format!("[{}]", v.join(", "))
}

fn check_stage(s: i64) -> Result<(), EnvError> {
    if STAGES.contains(&s) {
        Ok(())
    } else {
        Err(EnvError::Invalid(format!("未定義の Stage: {s}")))
    }
}

impl Env {
    pub fn new(cfg: Cfg, stage: i64) -> Result<Env, EnvError> {
        let sim = Sim2D::new(&cfg);
        let sampler = TaskSampler::new(&cfg);
        let oc = ObsConsts::new(&cfg);
        let area = Rect { center: (0.0, 0.0), size: (cfg.sim.work_area.size_m[0], cfg.sim.work_area.size_m[1]) };
        let mut env = Env {
            cfg,
            sim,
            sampler,
            oc,
            area,
            stage: 0,
            previous: Vec::new(),
            rehearsal: 0.0,
            rng: None,
            e: None,
            obs_poses: Vec::new(),
            rng_at_reset: None,
        };
        env.set_curriculum(stage, &[], 0.0)?;
        Ok(env)
    }

    pub fn obs_dim(&self) -> usize {
        self.oc.dim
    }

    pub fn cfg(&self) -> &Cfg {
        &self.cfg
    }

    /// 学習用: 確率 p で previous の Stage（一様）、それ以外は stage の盤面を出す（env.set_curriculum）。
    pub fn set_curriculum(&mut self, stage: i64, previous: &[i64], p: f64) -> Result<(), EnvError> {
        check_stage(stage)?;
        for &s in previous {
            check_stage(s)?;
        }
        if !(0.0..=1.0).contains(&p) || (p > 0.0 && previous.is_empty()) {
            return Err(EnvError::Invalid(format!("復習の設定が不正: previous={previous:?}, p={p}")));
        }
        self.stage = stage;
        self.previous = previous.to_vec();
        self.rehearsal = p;
        Ok(())
    }

    pub fn stage(&self) -> i64 {
        self.stage
    }

    /// 今のエピソードの Stage
    pub fn episode_stage(&self) -> Option<i64> {
        self.e.as_ref().map(|e| e.ep.stage)
    }

    pub fn episode(&self) -> Option<&EpisodeSpec> {
        self.e.as_ref().map(|e| &e.ep)
    }

    pub fn rng_full_state(&self) -> Option<(u128, u128, Option<u32>)> {
        self.rng.as_ref().map(|r| {
            let (s, i) = r.state();
            (s, i, r.uint32_buffer())
        })
    }

    /// 乱数を入れ替える（次の reset_into から使う）
    pub fn set_rng(&mut self, rng: NpRng) {
        self.rng = Some(rng);
    }

    pub fn sim(&self) -> &Sim2D {
        &self.sim
    }

    pub fn latency_s(&self) -> Option<f64> {
        self.e.as_ref().map(|e| e.latency_s)
    }

    pub fn wheel_gain(&self) -> Option<(f64, f64)> {
        self.e.as_ref().map(|e| e.wheel_gain)
    }

    // ---------- reset ----------
    /// 乱数を入れ替えて盤面を引く（Python の reset(seed=...)。rng は seeding.np_random(seed) と同じ状態）。
    pub fn reset(&mut self, rng: NpRng) -> Result<Vec<f32>, EnvError> {
        self.rng = Some(rng);
        let mut out = vec![0.0f32; self.oc.dim];
        self.reset_into(&mut out)?;
        Ok(out)
    }

    /// 今の乱数のまま盤面を引く（Python の reset()、Seed なし）。観測を out に書く。
    pub fn reset_into(&mut self, out: &mut [f32]) -> Result<(), EnvError> {
        let rng = self.rng.as_mut().ok_or_else(|| EnvError::State("乱数がない（reset(rng) が先）".into()))?;
        let (state, inc) = rng.state();
        self.rng_at_reset = Some((state, inc, rng.uint32_buffer()));
        let mut stage = self.stage;
        if !self.previous.is_empty() && rng.random() < self.rehearsal {
            stage = self.previous[rng.integers(self.previous.len() as i64) as usize];
        }
        let ep = self.sampler.sample(stage, rng);
        self.start_episode(ep, out)
    }

    /// 盤面を渡して始める（Python の reset(options={"episode": ep})）。rng は盤面を引いた後の np_random の状態。
    pub fn reset_with_episode(&mut self, ep: EpisodeSpec, rng: NpRng) -> Result<Vec<f32>, EnvError> {
        self.rng = Some(rng);
        self.rng_at_reset = None;
        let mut out = vec![0.0f32; self.oc.dim];
        self.start_episode(ep, &mut out)?;
        Ok(out)
    }

    fn start_episode(&mut self, ep: EpisodeSpec, out: &mut [f32]) -> Result<(), EnvError> {
        check_stage(ep.stage)?;
        let n = ep.objects.len();
        let mut targets = Vec::with_capacity(ep.target_names.len());
        for t in &ep.target_names {
            let i = ep.objects.iter().position(|o| &o.name == t)
                .ok_or_else(|| EnvError::Invalid(format!("対象 {t:?} が盤面の物体にない")))?;
            if targets.contains(&i) {
                return Err(EnvError::Invalid(format!("対象 {t:?} が重複している")));
            }
            targets.push(i);
        }
        if targets.is_empty() && ep.reach_point.is_none() {
            return Err(EnvError::Invalid("対象のない盤面に reach_point がない".into()));
        }
        let mut is_target = vec![false; n];
        for &i in &targets {
            is_target[i] = true;
        }
        let types: Vec<ObjectType> = ep.objects.iter().map(|o| o.otype).collect();
        // 置き場の指定（v24）の検査（env.reset と同じ）
        for (k, (name, _)) in ep.home_overrides.iter().enumerate() {
            if !ep.objects.iter().any(|o| &o.name == name) {
                return Err(EnvError::Invalid(format!("置き場の指定の物体 {name:?} が盤面の物体にない")));
            }
            if ep.home_overrides[..k].iter().any(|(n, _)| n == name) {
                return Err(EnvError::Invalid(format!("置き場の指定に重複がある: {name:?}")));
            }
        }
        // 置き場（v18、v24）: 指定があればそれ、無ければ開始時に 4 隅が入っている今のゴールか zones（task.start_home_index）
        let home_idx: Vec<i64> = ep.objects.iter()
            .map(|o| start_home_index(&ep, o, obj_size(&self.cfg, o.otype)))
            .collect::<Result<_, _>>()
            .map_err(EnvError::Invalid)?;
        let homes: Vec<Option<Rect>> = home_idx.iter()
            .map(|&i| match i {
                -1 => None,
                0 => Some(ep.goal),
                _ => Some(ep.zones[(i - 1) as usize]),
            })
            .collect();
        // もともとゴールの中 = 置き場が今のゴール（指定が無ければ「開始時に 4 隅がゴールの中」と同じ）
        let started_in_goal: Vec<bool> = home_idx.iter().map(|&i| i == 0).collect();
        // 盤面の作業領域（v30）。壁は盤面ごとに作り直さないので、壁ありでは sim.work_area と同じ大きさだけ
        let wa = &self.cfg.sim.work_area.size_m;
        if self.oc.area && ep.area_size.is_none() {
            return Err(EnvError::Invalid("observation.area_size が有効なのに、盤面に作業領域の大きさが無い".into()));
        }
        if let Some(a) = ep.area_size {
            if self.cfg.sim.walls.enabled && a != (wa[0], wa[1]) {
                return Err(EnvError::Invalid(format!("壁ありで、盤面の作業領域 {a:?} が sim.work_area と違う")));
            }
        }
        self.area = Rect { center: (0.0, 0.0), size: ep.area(&self.cfg.sim) };
        let _cp = self.cp_context("reset", ep.stage, 0, None, ep.robot, &[]);
        self.sim.reset(ep.robot, &ep.objects, GripperState::Open)?;
        let rng = self.rng.as_mut().ok_or_else(|| EnvError::State("乱数がない".into()))?;
        // 実機の誤差をエピソードごとに引く（盤面の後）
        let e = &self.cfg.sim.real_world_effects;
        let latency_s = (rng.uniform(e.command_latency_s[0], e.command_latency_s[1]) / LATENCY_GRID_S).round_ties_even()
            * LATENCY_GRID_S;
        let gl = rng.uniform(e.wheel_gain[0], e.wheel_gain[1]);
        let gr = rng.uniform(e.wheel_gain[0], e.wheel_gain[1]);
        // 左右共通の速さの倍率（v31、env.reset と同じ）。幅 0 なら乱数を引かない
        let (slo, shi) = (e.speed_gain[0], e.speed_gain[1]);
        let sg = if slo != shi { rng.uniform(slo, shi) } else { slo };
        let (gl, gr) = (gl * sg, gr * sg);
        // 車輪の速さが指令に追いつく時定数（v36、env.reset と同じ）。幅 0 なら乱数を引かない
        let (mlo, mhi) = (e.motor_lag_s[0], e.motor_lag_s[1]);
        let tau = if mlo != mhi { rng.uniform(mlo, mhi) } else { mlo };
        self.sim.motor_lag_s = tau;
        let nt = targets.len().max(1) as i64;
        let ec = &self.cfg.task.episode;
        let max_steps = ec.max_steps + ec.max_steps_per_extra_target * (nt - 1);
        let last_primary = targets.first().copied();
        self.e = Some(Episode {
            ep,
            types,
            is_target,
            targets,
            homes,
            started_in_goal,
            last_seen: vec![None; n],
            latency_s,
            wheel_gain: (gl, gr),
            applied_cmd: (0.0, 0.0),
            prev_action: 0,
            history: Vec::new(),
            steps: 0,
            last_primary,
            last_obs_robot: None,
            phi: [0.0; N_PHI],
            max_steps,
        });
        let phi = self.potentials()?;
        self.e.as_mut().unwrap().phi = phi;
        let snap = self.snapshot()?;
        self.build_obs(&snap, out)
    }

    fn ep(&self) -> &Episode {
        self.e.as_ref().expect("reset() が先")
    }

    pub fn max_steps(&self) -> i64 {
        self.ep().max_steps
    }

    // ---------- 観測 ----------
    fn noisy(&mut self, p: Pose2D) -> Pose2D {
        let e = &self.cfg.sim.real_world_effects;
        if e.obs_position_noise_m == 0.0 && e.obs_yaw_noise_rad == 0.0 {
            return p;
        }
        let rng = self.rng.as_mut().expect("乱数がない");
        // Generator.normal(size=3) は normal() を 3 回引くのと同じ
        let n = [rng.normal(), rng.normal(), rng.normal()];
        Pose2D::new(p.x + e.obs_position_noise_m * n[0], p.y + e.obs_position_noise_m * n[1], p.yaw + e.obs_yaw_noise_rad * n[2])
    }

    /// 仮想の斜め上のカメラから見て、車体の陰か閉じた爪の間にある（env._occluded）
    fn occluded(&self, robot: &Pose2D, p: &Pose2D) -> bool {
        let oc = &self.cfg.task.observation.occlusion;
        let sim = &self.cfg.sim;
        if self.sim.gripper_state() == GripperState::Closed && dist(gripper_point(sim, robot), p.xy()) < oc.held_radius_m {
            return true;
        }
        let (cx, cy, cz) = (oc.camera_xyz_m[0], oc.camera_xyz_m[1], oc.camera_xyz_m[2]);
        let (bx, by) = (sim.robot.body_size_m[0] / 2.0 + oc.margin_m, sim.robot.body_size_m[1] / 2.0 + oc.margin_m);
        let bc = body_center_x(sim);
        let (c, s) = (robot.yaw.cos(), robot.yaw.sin());
        let zo = oc.object_mid_height_m;
        // np.linspace(zo, h, 4): k * step + start（k = 0, 1, 2）と stop。step = (stop - start) / 3
        let step = (oc.robot_height_m - zo) / 3.0;
        let zs = [0.0 * step + zo, 1.0 * step + zo, 2.0 * step + zo, oc.robot_height_m];
        for z in zs {
            let t = (cz - z) / (cz - zo);
            let (x, y) = (cx + t * (p.x - cx), cy + t * (p.y - cy));
            let (lx, ly) = (c * (x - robot.x) + s * (y - robot.y), -s * (x - robot.x) + c * (y - robot.y));
            if (lx - bc).abs() <= bx && ly.abs() <= by {
                return true;
            }
        }
        false
    }

    /// env._snapshot と snapshot_from と build_snapshot
    fn snapshot(&mut self) -> Result<WorldSnapshot, EnvError> {
        let n = self.sim.n_objects();
        let mut poses = std::mem::take(&mut self.obs_poses);
        poses.clear();
        for i in 0..n {
            let p = self.sim.object_pose(i);
            let q = self.noisy(p);
            poses.push(q);
        }
        let rp = self.sim.robot_pose();
        let robot = self.noisy(rp);
        if self.cfg.task.observation.occlusion.enabled {
            // 隠れた物体は最後に見えた位置のまま。一度も見えていなければ今の位置
            for i in 0..n {
                let p = poses[i];
                let occ = self.occluded(&robot, &p);
                let e = self.e.as_mut().unwrap();
                match (occ, e.last_seen[i]) {
                    (true, Some(q)) => poses[i] = q,
                    _ => e.last_seen[i] = Some(p),
                }
            }
        }
        let e = self.e.as_mut().unwrap();
        // snapshot_from: 直前 1 ステップの移動は、前回の観測したロボットの姿勢との差
        let prev = e.last_obs_robot.unwrap_or(robot);
        let (c, sn) = (prev.yaw.cos(), prev.yaw.sin());
        let (dx, dy) = (robot.x - prev.x, robot.y - prev.y);
        let motion = (c * dx + sn * dy, -sn * dx + c * dy, wrap_angle(robot.yaw - prev.yaw));
        e.last_obs_robot = Some(robot);
        let time_left = (e.max_steps - e.steps) as f64 / e.max_steps as f64;
        let r = self.build_snapshot(&poses, robot, motion, time_left);
        self.obs_poses = poses;
        let (snap, primary) = r?;
        self.e.as_mut().unwrap().last_primary = primary;
        Ok(snap)
    }

    fn approach_distance_obs(&self, goal: &Rect, robot: &Pose2D, target: &Pose2D) -> Result<f64, EnvError> {
        let ap = approach_pt(target.xy(), goal.center, self.cfg.task.reward.approach_offset_m)?;
        Ok(dist(gripper_point(&self.cfg.sim, robot), ap))
    }

    /// snapshot.pending_fixes（v26）: 戻す（置き場を持ち 4 隅が置き場の外）の物体の順、次に出す（置き場を持たず中心が
    /// 今のゴールの中）の物体の順
    fn pending_fixes(&self, poses: &[Pose2D]) -> Vec<usize> {
        let e = self.ep();
        let n = e.ep.objects.len();
        let ret = (0..n).filter(|&i| !e.is_target[i])
            .filter(|&i| matches!(&e.homes[i], Some(h) if !inside_goal(&self.cfg, h, e.types[i], &poses[i])));
        let rem = (0..n).filter(|&i| !e.is_target[i] && e.homes[i].is_none() && e.ep.goal.contains(poses[i].xy()));
        ret.chain(rem).collect()
    }

    /// snapshot.fix_approach_distance（v26）
    fn fix_approach_distance(&self, i: usize, robot: &Pose2D, pose: &Pose2D) -> Result<f64, EnvError> {
        let e = self.ep();
        let dest = fix_destination_xy(&e.ep.goal, e.homes[i].as_ref(), pose);
        let ap = approach_pt(pose.xy(), dest, self.cfg.task.reward.approach_offset_m)?;
        Ok(dist(gripper_point(&self.cfg.sim, robot), ap))
    }

    /// env._fix_distance（v26）: 置き場の物体は角のはみ出し、置き場の無い物体は中心のゴールの中への深さ
    fn fix_distance(&self, i: usize, pose: &Pose2D) -> f64 {
        let e = self.ep();
        match &e.homes[i] {
            Some(h) => {
                let corners = box_corners(pose, obj_size(&self.cfg, e.types[i]));
                let mut m = py_max(0.0, signed_distance_to_rect(corners[0], h));
                for c in &corners[1..] {
                    let d = py_max(0.0, signed_distance_to_rect(*c, h));
                    if d > m {
                        m = d;
                    }
                }
                m
            }
            None => py_max(0.0, -signed_distance_to_rect(pose.xy(), &e.ep.goal)),
        }
    }

    fn slot_object(&self, i: usize, pose: Pose2D) -> SlotObject {
        let e = self.ep();
        let o = &e.ep.objects[i];
        SlotObject { pose, otype: o.otype, displacement: (pose.x - o.pose.x, pose.y - o.pose.y), home: e.homes[i] }
    }

    /// snapshot.target_held
    fn target_held(&self, robot: &Pose2D, gripper: GripperState, target: &Pose2D) -> bool {
        if gripper != GripperState::Closed {
            return false;
        }
        let (c, s) = (robot.yaw.cos(), robot.yaw.sin());
        let (dx, dy) = (target.x - robot.x, target.y - robot.y);
        let (lx, ly) = (c * dx + s * dy, -s * dx + c * dy);
        let g = &self.cfg.sim.gripper;
        let x0 = front_x(&self.cfg.sim) + g.pivot_x_from_front_m;
        ly.abs() <= self.cfg.task.observation.held_lateral_tolerance_m && x0 <= lx && lx <= x0 + g.jaw_length_m
    }

    fn build_snapshot(&self, poses: &[Pose2D], robot: Pose2D, motion: (f64, f64, f64), time_left: f64)
        -> Result<(WorldSnapshot, Option<usize>), EnvError> {
        let e = self.ep();
        let ep = &e.ep;
        if !(0.0..=1.0).contains(&time_left) {
            return Err(EnvError::Invalid(format!("time_left は 0〜1: {time_left}")));
        }
        let gripper = self.sim.gripper_state();
        let mut snap = WorldSnapshot {
            robot,
            target: robot,
            target_type: None,
            goal: ep.goal,
            gripper,
            prev_action: e.prev_action,
            obs_age_s: 0.0,
            other_targets: Vec::new(),
            obstacles: Vec::new(),
            n_remaining: 0,
            robot_motion: motion,
            time_left,
            target_held: false,
            destination: ep.goal,
            task_mode: None,
            area_size: ep.area(&self.cfg.sim),
        };
        if e.targets.is_empty() {
            // Stage 0: 指定点
            let rp = ep.reach_point.expect("start_episode が確かめた");
            snap.target = Pose2D::new(rp.0, rp.1, 0.0);
            return Ok((snap, None));
        }
        let last = e.last_primary.expect("対象のある盤面の主対象");
        // 主対象: ゴールに入っていない対象のうち、接近点が最も近いもの（最初の最小）。全部入ったら前の主対象
        let mut primary = None;
        let mut best = f64::INFINITY;
        let mut remaining = 0;
        for &i in &e.targets {
            if !inside_goal(&self.cfg, &ep.goal, e.types[i], &poses[i]) {
                remaining += 1;
                let d = self.approach_distance_obs(&ep.goal, &robot, &poses[i])?;
                if primary.is_none() || d < best {
                    primary = Some(i);
                    best = d;
                }
            }
        }
        let primary = primary.unwrap_or(last);
        // v26: 全対象が入っていて直すべき対象外があれば、この観測の主対象をそのうち接近点の最も近いもの（最初の最小）に
        let mut head = primary;
        snap.task_mode = Some(TaskMode::Deliver);
        if self.cfg.task.observation.cleanup_primary && remaining == 0 {
            let mut best: Option<(usize, f64)> = None;
            for i in self.pending_fixes(poses) {
                let d = self.fix_approach_distance(i, &robot, &poses[i])?;
                if best.is_none_or(|(_, b)| d < b) {
                    best = Some((i, d));
                }
            }
            if let Some((i, _)) = best {
                head = i;
                match e.homes[i] {
                    Some(h) => {
                        snap.task_mode = Some(TaskMode::Return);
                        snap.destination = h;
                    }
                    None => snap.task_mode = Some(TaskMode::Remove),
                }
            }
        }
        for &i in &e.targets {
            if i != head {
                snap.other_targets.push(self.slot_object(i, poses[i]));
            }
        }
        for i in 0..ep.objects.len() {
            if !e.is_target[i] && i != head {
                snap.obstacles.push(self.slot_object(i, poses[i]));
            }
        }
        snap.target = poses[head];
        snap.target_type = Some(e.types[head]);
        snap.n_remaining = remaining;
        snap.target_held = self.target_held(&robot, gripper, &poses[head]);
        Ok((snap, Some(primary)))
    }

    /// ObservationBuilder._edge_distance: 縁までの符号付き距離。by_corners なら 4 隅のうち最大（Python の max: 最初の最大）
    fn edge_distance(&self, p: &Pose2D, t: ObjectType, zone: &Rect, by_corners: bool) -> f64 {
        if !by_corners {
            return signed_distance_to_rect(p.xy(), zone);
        }
        let corners = box_corners(p, obj_size(&self.cfg, t));
        let mut m = signed_distance_to_rect(corners[0], zone);
        for c in &corners[1..] {
            let d = signed_distance_to_rect(*c, zone);
            if d > m {
                m = d;
            }
        }
        m
    }

    /// ObservationBuilder._slot_features
    fn slot_features(&self, r: &Pose2D, goal: &Rect, objs: &[SlotObject], n_slots: usize, is_target: bool, v: &mut Vec<f64>) {
        let (c, s) = (r.yaw.cos(), r.yaw.sin());
        let oc = &self.oc;
        let mut rel: Vec<(f64, f64, f64, f64, ObjectType, f64, f64, f64, bool)> = objs.iter().map(|o| {
            let p = o.pose;
            let (dx, dy) = (p.x - r.x, p.y - r.y);
            let (mx, my) = o.displacement;
            let zone = o.home.as_ref().unwrap_or(goal);
            let sd = self.edge_distance(&p, o.otype, zone, oc.corners && (is_target || o.home.is_some()));
            (hypot(dx, dy), c * dx + s * dy, -s * dx + c * dy, wrap_angle(p.yaw - r.yaw), o.otype,
             c * mx + s * my, -s * mx + c * my, sd, o.home.is_some())
        }).collect();
        // list.sort は安定（NaN は後の値域の確かめで止まる）
        rel.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(_, lx, ly, a, t, mx, my, sd, sig) in rel.iter().take(n_slots) {
            let t = t as usize;
            v.extend_from_slice(&[lx / oc.half.0, ly / oc.half.1, (2.0 * a).sin(), (2.0 * a).cos(),
                                  if t == 0 { 1.0 } else { 0.0 }, if t == 1 { 1.0 } else { 0.0 },
                                  mx / oc.diag, my / oc.diag, sd / oc.half.0, if sig { 1.0 } else { 0.0 },
                                  if is_target { 1.0 } else { 0.0 }, 1.0]);
        }
        let pad = SLOT_DIM * (n_slots - rel.len().min(n_slots));
        v.extend(std::iter::repeat(0.0).take(pad));
    }

    /// ObservationBuilder.build: f64 で計算して f32 に丸め、値域 ±3 の外なら Invalid
    fn build_obs(&self, s: &WorldSnapshot, out: &mut [f32]) -> Result<(), EnvError> {
        let oc = &self.oc;
        if !oc.cleanup && (!matches!(s.task_mode, None | Some(TaskMode::Deliver)) || s.destination != s.goal) {
            return Err(EnvError::Invalid(format!("cleanup_primary が無効なのに、運ぶ以外の仕事か今のゴール以外の行き先: {s:?}")));
        }
        // 頭のゴールの欄は行き先（v26）。スロットの縁までの距離（置き場を持たない物体）は今のゴール
        let (r, t, g) = (&s.robot, &s.target, &s.destination);
        let (dx, dy) = (t.x - r.x, t.y - r.y);
        let rel_bearing = wrap_angle(dy.atan2(dx) - r.yaw);
        let (gx, gy) = (g.center.0 - t.x, g.center.1 - t.y);
        let goal_bearing = gy.atan2(gx);
        let mut v: Vec<f64> = Vec::with_capacity(oc.dim);
        v.extend_from_slice(&[
            r.x / oc.half.0, r.y / oc.half.1,
            r.yaw.sin(), r.yaw.cos(),
            t.x / oc.half.0, t.y / oc.half.1,
            (2.0 * t.yaw).sin(), (2.0 * t.yaw).cos(),
            g.center.0 / oc.half.0, g.center.1 / oc.half.1,
            g.size.0 / oc.size.0, g.size.1 / oc.size.1,
            hypot(dx, dy) / oc.diag, rel_bearing.sin(), rel_bearing.cos(),
            hypot(gx, gy) / oc.diag, goal_bearing.sin(), goal_bearing.cos(),
        ]);
        let tt = s.target_type.map(|t| t as usize);
        v.extend((0..2).map(|k| if tt == Some(k) { 1.0 } else { 0.0 }));
        v.extend((0..4).map(|k| if s.gripper as usize == k { 1.0 } else { 0.0 }));
        v.extend((0..N_ACTIONS).map(|k| if s.prev_action as usize == k { 1.0 } else { 0.0 }));
        v.extend_from_slice(&[
            s.obs_age_s / oc.action_duration,
            s.robot_motion.0 / oc.step_m, s.robot_motion.1 / oc.step_m, s.robot_motion.2 / oc.step_rad,
            s.time_left,
            match s.target_type {
                // 出す（v26）は「中心がゴールの外」が条件なので中心で測る
                Some(tt) => self.edge_distance(t, tt, g, oc.corners && s.task_mode != Some(TaskMode::Remove)),
                None => signed_distance_to_rect(t.xy(), g),
            } / oc.half.0,
            if s.target_held { 1.0 } else { 0.0 },
            s.n_remaining as f64 / oc.max_targets as f64,
        ]);
        if oc.cleanup {
            let m = s.task_mode.map(|m| m as usize);
            v.extend((0..MODE_DIM).map(|k| if m == Some(k) { 1.0 } else { 0.0 }));
        }
        if oc.area {
            v.extend_from_slice(&[s.area_size.0 / 2.0 / oc.half.0, s.area_size.1 / 2.0 / oc.half.1]);
        }
        self.slot_features(r, &s.goal, &s.other_targets, oc.n_target_slots, true, &mut v);
        self.slot_features(r, &s.goal, &s.obstacles, oc.n_obstacle_slots, false, &mut v);
        assert_eq!(v.len(), oc.dim, "Observation の次元");
        assert_eq!(out.len(), oc.dim, "Observation の書き先の次元");
        let mut bad = None;
        for (k, (o, &x)) in out.iter_mut().zip(v.iter()).enumerate() {
            *o = x as f32;
            if bad.is_none() && (!o.is_finite() || o.abs() > oc.bound) {
                bad = Some(k);
            }
        }
        if let Some(k) = bad {
            return Err(EnvError::Invalid(format!("Observation が値域 ±{:.1} の外（{k} 番目 = {}）: {s:?}", oc.bound, out[k])));
        }
        Ok(())
    }

    // ---------- 判定とポテンシャル ----------
    fn entry_distance(&self, i: usize, pose: &Pose2D) -> f64 {
        let e = self.ep();
        let corners = box_corners(pose, obj_size(&self.cfg, e.types[i]));
        let mut m = py_max(0.0, signed_distance_to_rect(corners[0], &e.ep.goal));
        for c in &corners[1..] {
            let d = py_max(0.0, signed_distance_to_rect(*c, &e.ep.goal));
            if d > m {
                m = d;
            }
        }
        m
    }

    fn progress_distance(&self) -> f64 {
        let e = self.ep();
        if e.targets.is_empty() {
            return dist(gripper_point(&self.cfg.sim, &self.sim.robot_pose()), e.ep.reach_point.unwrap());
        }
        let d = PySum::of(e.targets.iter().map(|&i| self.entry_distance(i, &self.sim.object_pose(i))));
        if !self.cfg.task.reward.cleanup_shaping {
            return d;
        }
        d + PySum::of((0..e.ep.objects.len()).filter(|&i| !e.is_target[i]).map(|i| self.fix_distance(i, &self.sim.object_pose(i))))
    }

    fn approach_distance(&self) -> Result<f64, EnvError> {
        let e = self.ep();
        let robot = self.sim.robot_pose();
        let mut m: Option<f64> = None;
        for &i in &e.targets {
            let p = self.sim.object_pose(i);
            if inside_goal(&self.cfg, &e.ep.goal, e.types[i], &p) {
                continue;
            }
            let d = self.approach_distance_obs(&e.ep.goal, &robot, &p)?;
            if m.is_none_or(|x| d < x) {
                m = Some(d);
            }
        }
        if m.is_some() || e.targets.is_empty() || !self.cfg.task.reward.cleanup_shaping {
            return Ok(m.unwrap_or(0.0));
        }
        // v26: 全対象配達済みなら、直すべき対象外の接近点までの距離の最小
        let poses: Vec<Pose2D> = (0..e.ep.objects.len()).map(|i| self.sim.object_pose(i)).collect();
        for i in self.pending_fixes(&poses) {
            let d = self.fix_approach_distance(i, &robot, &poses[i])?;
            if m.is_none_or(|x| d < x) {
                m = Some(d);
            }
        }
        Ok(m.unwrap_or(0.0))
    }

    fn n_delivered(&self) -> i64 {
        let e = self.ep();
        e.targets.iter().filter(|&&i| inside_goal(&self.cfg, &e.ep.goal, e.types[i], &self.sim.object_pose(i))).count() as i64
    }

    fn n_foreign_in_goal(&self) -> i64 {
        let e = self.ep();
        (0..e.ep.objects.len())
            .filter(|&i| !e.is_target[i] && !e.started_in_goal[i] && e.ep.goal.contains(self.sim.object_pose(i).xy()))
            .count() as i64
    }

    /// 対象外ごとの移動の量（置き場を持つ物体は置き場の外へはみ出した距離）。物体の順
    fn non_target_displacements(&self) -> impl Iterator<Item = f64> + '_ {
        let e = self.ep();
        (0..e.ep.objects.len()).filter(|&i| !e.is_target[i]).map(move |i| {
            let now = self.sim.object_pose(i);
            match &e.homes[i] {
                Some(h) => py_max(0.0, signed_distance_to_rect(now.xy(), h)),
                None => dist(now.xy(), e.ep.objects[i].pose.xy()),
            }
        })
    }

    fn n_home_out(&self) -> i64 {
        let e = self.ep();
        (0..e.ep.objects.len())
            .filter(|&i| !e.is_target[i])
            .filter(|&i| matches!(&e.homes[i], Some(h) if !inside_goal(&self.cfg, h, e.types[i], &self.sim.object_pose(i))))
            .count() as i64
    }

    fn non_target_near_goal(&self) -> f64 {
        let e = self.ep();
        let band = self.cfg.task.reward.non_target_near_goal_band_m;
        PySum::of((0..e.ep.objects.len()).filter(|&i| !e.is_target[i] && !e.started_in_goal[i]).map(|i| {
            py_max(0.0, band - py_max(0.0, signed_distance_to_rect(self.sim.object_pose(i).xy(), &e.ep.goal)))
        }))
    }

    /// ポテンシャル Φ の各項（REWARD_TERM_NAMES の前の 7 つの順）
    fn potentials(&self) -> Result<[f64; N_PHI], EnvError> {
        let rw = &self.cfg.task.reward;
        let has_targets = !self.ep().targets.is_empty();
        Ok([
            -rw.progress_per_m * self.progress_distance(),
            -rw.approach_per_m * self.approach_distance()?,
            rw.delivery_bonus * (if has_targets { self.n_delivered() } else { 0 }) as f64,
            -rw.foreign_in_goal_penalty * self.n_foreign_in_goal() as f64,
            -rw.non_target_per_m * PySum::of(self.non_target_displacements()),
            -rw.non_target_near_goal_per_m * self.non_target_near_goal(),
            -rw.home_out_penalty * self.n_home_out() as f64,
        ])
    }

    fn is_success(&self) -> bool {
        let e = self.ep();
        if e.targets.is_empty() {
            return self.progress_distance() <= self.cfg.task.success.reach_tolerance_m;
        }
        if !self.cfg.task.success.require_cleanup {
            return self.n_delivered() == e.targets.len() as i64;
        }
        self.n_delivered() == e.targets.len() as i64 && self.n_foreign_in_goal() == 0 && self.n_home_out() == 0
    }

    fn failure(&self) -> Result<Option<Failure>, EnvError> {
        let e = self.ep();
        let mut outside: Vec<&str> = Vec::new();
        if !self.area.contains(self.sim.robot_pose().xy()) {
            outside.push("robot");
        }
        for i in 0..e.ep.objects.len() {
            if !self.area.contains(self.sim.object_pose(i).xy()) {
                outside.push(&e.ep.objects[i].name);
            }
        }
        if !outside.is_empty() {
            // 壁ありでは物理モデルの不具合。黙って失敗扱いにせず止める
            if self.cfg.sim.walls.enabled {
                return Err(EnvError::State(format!("壁があるのに {} が作業領域の外にいる\n{}", py_list_repr(&outside),
                                                   self.diagnostics())));
            }
            // v30: 縁から out_of_area_margin_m までは失敗にしない
            let m = self.cfg.sim.walls.out_of_area_margin_m;
            let outer = Rect { center: self.area.center, size: (self.area.size.0 + 2.0 * m, self.area.size.1 + 2.0 * m) };
            if !outer.contains(self.sim.robot_pose().xy())
                || (0..e.ep.objects.len()).any(|i| !outer.contains(self.sim.object_pose(i).xy()))
            {
                return Ok(Some(Failure::OutOfArea));
            }
        }
        let rw = &self.cfg.task.reward;
        if rw.non_target_failure && self.non_target_displacements().any(|d| d > rw.non_target_tolerance_m) {
            return Ok(Some(Failure::NonTargetMoved));
        }
        Ok(None)
    }

    /// 物理の不具合を調べるための状況（env._diagnostics と同じ項目）
    pub fn diagnostics(&self) -> String {
        let Some(e) = self.e.as_ref() else {
            return "episode: None".into();
        };
        let mut lines = vec![
            format!("episode: {:?}", e.ep),
            format!("steps: {}", e.steps),
            format!("actions: {:?}", e.history),
            format!("robot: {:?}", self.sim.robot_pose()),
            format!("gripper: {:?}", self.sim.gripper_state()),
            format!("jaw_angle: {}", self.sim.jaw_angle()),
            format!("grasped_object: {:?}", self.sim.grasped_object().map(|i| e.ep.objects[i].name.clone())),
            format!("latency_s: {}, wheel_gain: {:?}", e.latency_s, e.wheel_gain),
            // 再生用: reset の直前の乱数の状態（盤面を外から渡したときは None）と Curriculum。
            // 不完全さ（挟み損ねなど）は乱数で決まるので、盤面と行動の列だけでは再生できない（2026-10-19）
            format!("rng_at_reset: {:?}", self.rng_at_reset),
            format!("curriculum: stage {}, previous {:?}, rehearsal {}", self.stage, self.previous, self.rehearsal),
        ];
        for (i, o) in e.ep.objects.iter().enumerate() {
            lines.push(format!("{}: {:?}", o.name, self.sim.object_pose(i)));
        }
        lines.join("\n")
    }

    /// Chipmunk の assert の報告（cp::context_text）に付ける状況を入れる。返り値を持っている間だけ有効
    fn cp_context(&self, phase: &'static str, stage: i64, steps: i64, action: Option<i64>, robot: Pose2D, actions: &[u8])
        -> crate::cp::ContextGuard {
        crate::cp::enter_context(|c| {
            c.phase = phase;
            c.stage = stage;
            c.steps = steps;
            c.action = action;
            c.robot = (robot.x, robot.y, robot.yaw);
            c.curriculum.0 = self.stage;
            c.curriculum.1.clone_from(&self.previous);
            c.curriculum.2 = self.rehearsal;
            c.rng_at_reset = self.rng_at_reset;
            c.actions.clear();
            c.actions.extend_from_slice(actions);
        })
    }

    /// 左右のモーター利得を掛けた実際の (v, ω)（env._wheel_effects）
    fn wheel_effects(&self, v: f64, w: f64) -> (f64, f64) {
        let b = self.cfg.sim.robot.wheel_separation_m;
        let (gl, gr) = self.ep().wheel_gain;
        let (vl, vr) = ((v - w * b / 2.0) * gl, (v + w * b / 2.0) * gr);
        ((vl + vr) / 2.0, (vr - vl) / b)
    }

    /// spaces.drive_velocity
    fn drive_velocity(&self, a: u8) -> (f64, f64) {
        let r = &self.cfg.sim.robot;
        match a {
            0 => (0.0, 0.0),
            1 => (r.linear_speed_mps, 0.0),
            2 => (-r.linear_speed_mps, 0.0),
            3 => (0.0, r.angular_speed_radps),
            4 => (0.0, -r.angular_speed_radps),
            _ => unreachable!("走行の Action でない: {a}"),
        }
    }

    // ---------- step ----------
    pub fn step(&mut self, action: i64) -> Result<StepOut, EnvError> {
        let mut obs = vec![0.0f32; self.oc.dim];
        let (reward, terminated, truncated, info) = self.step_into(action, &mut obs)?;
        Ok(StepOut { obs, reward, terminated, truncated, info })
    }

    /// step の観測を out に書き、(reward, terminated, truncated, info) を返す
    pub fn step_into(&mut self, action: i64, out: &mut [f32]) -> Result<(f64, bool, bool, StepInfo), EnvError> {
        if self.e.is_none() {
            return Err(EnvError::State("reset() が先".into()));
        }
        if !(0..N_ACTIONS as i64).contains(&action) {
            return Err(EnvError::Invalid(format!("{action} is not a valid Action")));
        }
        let a = action as u8;
        let _cp = {
            let e = self.ep();
            self.cp_context("step", e.ep.stage, e.steps, Some(action), self.sim.robot_pose(), &e.history)
        };
        let rw = self.cfg.task.reward.clone();
        let before_gripper = self.sim.gripper_state();
        let mut redundant = false;
        // 状態の変わる開閉（v28 の gripper_toggle_penalty）
        let toggled = (a == OPEN_GRIPPER || a == CLOSE_GRIPPER)
            && before_gripper != if a == CLOSE_GRIPPER { GripperState::Closed } else { GripperState::Open };
        let t_step = self.cfg.task.control.action_duration_s;
        let (latency, applied) = {
            let e = self.ep();
            (e.latency_s, e.applied_cmd)
        };
        if a == OPEN_GRIPPER || a == CLOSE_GRIPPER {
            let want = if a == CLOSE_GRIPPER { GripperState::Closed } else { GripperState::Open };
            redundant = before_gripper == want;
            if latency > 0.0 && !(applied.0 == 0.0 && applied.1 == 0.0) {
                self.sim.drive(applied.0, applied.1, latency)?; // 停止が届くまで前の指令が続く
            }
            self.sim.actuate_gripper(a == CLOSE_GRIPPER, self.rng.as_mut())?;
            self.e.as_mut().unwrap().applied_cmd = (0.0, 0.0);
        } else {
            let (v, w) = self.drive_velocity(a);
            let cmd = self.wheel_effects(v, w);
            if latency > 0.0 {
                self.sim.drive(applied.0, applied.1, latency)?;
            }
            self.sim.drive(cmd.0, cmd.1, t_step - latency)?;
            self.e.as_mut().unwrap().applied_cmd = cmd;
        }
        // 車体の形に触れた対象の数（v19）。減点が 0 なら数えない
        let n_chassis = if rw.chassis_contact_penalty > 0.0 {
            let e = self.ep();
            self.sim.chassis_contacts().into_iter().filter(|&i| e.is_target[i]).count() as i64
        } else {
            0
        };
        {
            let e = self.e.as_mut().unwrap();
            e.prev_action = a;
            e.history.push(a);
            e.steps += 1;
        }
        let failure = self.failure()?;
        let success = failure.is_none() && self.is_success();
        let terminated = success || failure.is_some();
        // 形のつく項: kΦ(s′) − Φ(s)。終端では Φ(s′) = 0
        let k_phi = rw.shaping_discount;
        let phi_new = self.potentials()?;
        let phi_old = self.ep().phi;
        let mut terms = [0.0f64; N_TERMS];
        for k in 0..N_PHI {
            terms[k] = (if terminated { 0.0 } else { k_phi * phi_new[k] }) - phi_old[k];
        }
        terms[7] = if success { rw.success_bonus } else { 0.0 };
        terms[8] = if failure.is_some() { -rw.failure_penalty } else { 0.0 };
        terms[9] = if redundant { -rw.redundant_gripper_penalty } else { 0.0 };
        terms[10] = -rw.chassis_contact_penalty * n_chassis as f64;
        terms[11] = -rw.time_penalty;
        terms[12] = if toggled { -rw.gripper_toggle_penalty } else { 0.0 };
        // v34: 終わったときに残っている直すべき物（ポテンシャルの foreign_in_goal・home_out は終端で戻るので）
        terms[13] = if terminated && rw.leftover_penalty > 0.0 {
            -rw.leftover_penalty * (self.n_foreign_in_goal() + self.n_home_out()) as f64
        } else {
            0.0
        };
        self.e.as_mut().unwrap().phi = phi_new;
        let reward = PySum::of(terms.iter().copied());
        let (steps, max_steps, stage, n_targets) = {
            let e = self.ep();
            (e.steps, e.max_steps, e.ep.stage, e.targets.len() as i64)
        };
        let truncated = !terminated && steps >= max_steps;
        let info = StepInfo {
            is_success: success,
            failure,
            stage,
            reward_terms: terms,
            steps,
            n_delivered: if n_targets > 0 { self.n_delivered() } else { 0 },
            n_targets,
            n_foreign_in_goal: self.n_foreign_in_goal(),
            n_home_out: self.n_home_out(),
            n_chassis_contact: n_chassis,
        };
        let r = self.snapshot().and_then(|s| self.build_obs(&s, out));
        if let Err(e) = r {
            // 値域の外は物理の不具合のことが多い。盤面と行動の列を付ける
            return Err(match e {
                EnvError::Invalid(m) => EnvError::Invalid(format!("{m}\n{}", self.diagnostics())),
                other => other,
            });
        }
        Ok((reward, terminated, truncated, info))
    }
}

// ---------- Python から ----------

pub type RngState = (u128, u128, Option<u32>);

pub fn rng_from(s: RngState) -> NpRng {
    NpRng::from_full_state(s.0, s.1, s.2)
}

fn get<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<T> {
    d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("盤面の dict に {k} がない")))?.extract()
}

/// 盤面の dict（TaskSampler.sample と同じ形、toy_sweeper.native.episode_dict が作る）から EpisodeSpec
pub fn episode_from_dict(d: &Bound<'_, PyDict>) -> PyResult<EpisodeSpec> {
    let known = ["stage", "robot", "objects", "target_names", "reach_point", "goal_name", "goal", "zones", "home_overrides",
                 "area_size"];
    for k in d.keys() {
        let k: String = k.extract()?;
        if !known.contains(&k.as_str()) {
            return Err(PyValueError::new_err(format!("盤面の dict の未知の鍵: {k}")));
        }
    }
    let robot: (f64, f64, f64) = get(d, "robot")?;
    let objs: Vec<(String, i64, f64, f64, f64, String)> = get(d, "objects")?;
    let mut objects = Vec::with_capacity(objs.len());
    for (name, t, x, y, yaw, color) in objs {
        if t != 0 && t != 1 {
            return Err(PyValueError::new_err(format!("未知の物体の種類: {t}")));
        }
        objects.push(ObjectSpec { name, otype: ObjectType::from_index(t), pose: Pose2D::new(x, y, yaw), color });
    }
    let rect = |r: ((f64, f64), (f64, f64))| Rect { center: r.0, size: r.1 };
    let zones: Vec<((f64, f64), (f64, f64))> = get(d, "zones")?;
    Ok(EpisodeSpec {
        stage: get(d, "stage")?,
        robot: Pose2D::new(robot.0, robot.1, robot.2),
        objects,
        target_names: get(d, "target_names")?,
        reach_point: get(d, "reach_point")?,
        goal_name: get(d, "goal_name")?,
        goal: rect(get(d, "goal")?),
        zones: zones.into_iter().map(rect).collect(),
        home_overrides: get(d, "home_overrides")?,
        area_size: get(d, "area_size")?,
    })
}

/// info を Python の env と同じ dict にする（鍵の順も同じ）
pub fn info_dict<'py>(py: Python<'py>, info: &StepInfo) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("is_success", info.is_success)?;
    d.set_item("failure", info.failure.map(|f| f.name()))?;
    d.set_item("stage", info.stage)?;
    let terms = PyDict::new(py);
    for (k, v) in REWARD_TERM_NAMES.iter().zip(info.reward_terms.iter()) {
        terms.set_item(k, v)?;
    }
    d.set_item("reward_terms", terms)?;
    d.set_item("steps", info.steps)?;
    d.set_item("n_delivered", info.n_delivered)?;
    d.set_item("n_targets", info.n_targets)?;
    d.set_item("n_foreign_in_goal", info.n_foreign_in_goal)?;
    d.set_item("n_home_out", info.n_home_out)?;
    d.set_item("n_chassis_contact", info.n_chassis_contact)?;
    Ok(d)
}

/// ToySweeperEnv と同じ動きの env（試験 test/test_native_env.py が Python 版と突き合わせる）
#[pyclass(name = "Env", module = "toy_sweeper_native")]
pub struct PyEnv {
    inner: Env,
}

#[pymethods]
impl PyEnv {
    #[new]
    fn py_new(cfg_json: &str, stage: i64) -> PyResult<Self> {
        let cfg = Cfg::from_json(cfg_json).map_err(PyValueError::new_err)?;
        Ok(PyEnv { inner: guard(|| Env::new(cfg, stage))? })
    }

    #[pyo3(signature = (stage, previous, rehearsal_probability))]
    fn set_curriculum(&mut self, stage: i64, previous: Vec<i64>, rehearsal_probability: f64) -> PyResult<()> {
        Ok(self.inner.set_curriculum(stage, &previous, rehearsal_probability)?)
    }

    #[getter]
    fn obs_dim(&self) -> usize {
        self.inner.obs_dim()
    }

    /// rng_state = (state, inc, 32 ビットの残りか None)。Python の reset(seed=s) なら seeding.np_random(s) の状態、
    /// None なら今の乱数のまま（Python の reset()）
    #[pyo3(signature = (rng_state=None))]
    fn reset<'py>(&mut self, py: Python<'py>, rng_state: Option<RngState>)
        -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyDict>)> {
        let env = &mut self.inner;
        let obs = guard(|| match rng_state {
            Some(s) => env.reset(rng_from(s)),
            None => {
                let mut out = vec![0.0f32; env.obs_dim()];
                env.reset_into(&mut out).map(|_| out)
            }
        })?;
        self.reset_out(py, obs)
    }

    /// episode: 盤面の dict。rng_state: 盤面を引いた後の np_random の状態
    fn reset_with_episode<'py>(&mut self, py: Python<'py>, episode: &Bound<'py, PyDict>, rng_state: RngState)
        -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyDict>)> {
        let ep = episode_from_dict(episode)?;
        let env = &mut self.inner;
        let obs = guard(|| env.reset_with_episode(ep, rng_from(rng_state)))?;
        self.reset_out(py, obs)
    }

    fn step<'py>(&mut self, py: Python<'py>, action: i64)
        -> PyResult<(Bound<'py, PyArray1<f32>>, f64, bool, bool, Bound<'py, PyDict>)> {
        let env = &mut self.inner;
        let o = guard(|| env.step(action))?;
        Ok((PyArray1::from_vec(py, o.obs), o.reward, o.terminated, o.truncated, info_dict(py, &o.info)?))
    }

    fn rng_state(&self) -> Option<RngState> {
        self.inner.rng_full_state()
    }

    fn max_steps(&self) -> PyResult<i64> {
        self.ready()?;
        Ok(self.inner.max_steps())
    }

    #[getter]
    fn latency_s(&self) -> Option<f64> {
        self.inner.latency_s()
    }

    #[getter]
    fn wheel_gain(&self) -> Option<(f64, f64)> {
        self.inner.wheel_gain()
    }

    /// 今のエピソードの物体の名前（盤面の順）
    fn object_names(&self) -> PyResult<Vec<String>> {
        self.ready()?;
        Ok(self.inner.episode().unwrap().objects.iter().map(|o| o.name.clone()).collect())
    }

    fn robot_pose(&self) -> PyResult<(f64, f64, f64)> {
        self.ready()?;
        let p = self.inner.sim().robot_pose();
        Ok((p.x, p.y, p.yaw))
    }

    fn object_pose(&self, i: usize) -> PyResult<(f64, f64, f64)> {
        self.ready()?;
        if i >= self.inner.sim().n_objects() {
            return Err(PyValueError::new_err(format!("物体の番号 {i} が範囲外")));
        }
        let p = self.inner.sim().object_pose(i);
        Ok((p.x, p.y, p.yaw))
    }

    fn grasped_object(&self) -> PyResult<Option<String>> {
        self.ready()?;
        let s = self.inner.sim();
        Ok(s.grasped_object().map(|i| s.object_spec(i).name.clone()))
    }

    fn diagnostics(&self) -> String {
        self.inner.diagnostics()
    }

    /// Python の env._progress_distance()（評価の最後のゴールまでの距離、evaluate.run_episode が読む）
    fn progress_distance(&self) -> PyResult<f64> {
        self.ready()?;
        Ok(self.inner.progress_distance())
    }
}

impl PyEnv {
    fn ready(&self) -> PyResult<()> {
        if self.inner.episode().is_none() {
            return Err(PyRuntimeError::new_err("reset() が先"));
        }
        Ok(())
    }

    fn reset_out<'py>(&self, py: Python<'py>, obs: Vec<f32>) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyDict>)> {
        let d = PyDict::new(py);
        d.set_item("stage", self.inner.episode_stage().unwrap())?;
        Ok((PyArray1::from_vec(py, obs), d))
    }
}

/// 試験用: CPython 3.12 の sum() と同じ総和
#[pyfunction]
fn _py_sum(xs: Vec<f64>) -> f64 {
    PySum::of(xs)
}

pub fn py_register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnv>()?;
    m.add_function(wrap_pyfunction!(_py_sum, m)?)?;
    m.add("REWARD_TERM_NAMES", PyList::new(m.py(), REWARD_TERM_NAMES)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_sum_is_neumaier() {
        assert_eq!(PySum::of([1e16, 1.0, -1e16]), 1.0);
        assert_eq!(PySum::of([]), 0.0);
        assert_eq!(PySum::of([-0.0]).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn latency_rounds_ties_to_even() {
        // Python の round(0.0375 / 0.025) は 2（1.5 → 2）、round(2.5) は 2
        assert_eq!((0.0625f64 / LATENCY_GRID_S).round_ties_even(), 2.0);
        assert_eq!(2.5f64.round_ties_even(), 2.0);
    }
}
