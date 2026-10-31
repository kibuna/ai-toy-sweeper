//! Stage ごとの盤面生成（toy_sweeper.task.TaskSampler と同じ盤面を、同じ乱数の引き方で作る）。
//!
//! 乱数は task.py と同じ順に、同じ方法（random / uniform / integers / choice）で引く。浮動小数点の演算の順序も同じにする。
//! 違いは test/test_native_sampler.py が Python 版と突き合わせて見つける。
use crate::config::{Cfg, ObjectType, SamplingCfg, SimCfg};
use crate::geometry::{box_corners, dist, hypot, ieee_remainder, signed_distance_to_rect, Pose2D, Rect, PI};
use crate::rng::NpRng;
use crate::robot_model::{front_x, gripper_center_x, robot_radius};
use crate::sim2d::ObjectSpec;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub const STAGES: [i64; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// task.COLORS と同じ並び
pub const COLORS: [&str; 10] = ["red", "blue", "yellow", "green", "white", "black", "lime", "light_blue", "pink", "orange"];

#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeSpec {
    pub stage: i64,
    pub robot: Pose2D,
    pub objects: Vec<ObjectSpec>,
    /// 対象の名前（Python と同じ順）。Stage 0 は空
    pub target_names: Vec<String>,
    /// Stage 0 のみ
    pub reach_point: Option<(f64, f64)>,
    pub goal_name: String,
    pub goal: Rect,
    pub zones: Vec<Rect>,
    /// 置き場の指定（v24、task.EpisodeSpec.home_overrides）: (物体名, 番号)。0 = 今のゴール、1〜 = zones[番号 − 1]、−1 = なし
    pub home_overrides: Vec<(String, i64)>,
    /// v30: 作業領域の (幅, 奥行き)（task.EpisodeSpec.area_size）。None は sim.work_area.size_m
    pub area_size: Option<(f64, f64)>,
}

impl EpisodeSpec {
    /// task.episode_area: 盤面の作業領域の (幅, 奥行き)
    pub fn area(&self, sim: &SimCfg) -> (f64, f64) {
        self.area_size.unwrap_or((sim.work_area.size_m[0], sim.work_area.size_m[1]))
    }
}

/// task.goal_rects_for_area: 名前つきのゴール（sim.work_area での goal_zones、名前の順）を大きさ size の作業領域へ写す。
/// 軸ごとに近い縁からの距離を保つ。中心 0 の軸は 0 のまま
pub fn goals_for_area(sim: &SimCfg, goals: &[(String, Rect)], size: (f64, f64)) -> Vec<(String, Rect)> {
    let reanchor = |v: f64, r: f64, n: f64| if v == 0.0 { 0.0 } else { (n / 2.0 - (r / 2.0 - v.abs())).copysign(v) };
    let (rx, ry) = (sim.work_area.size_m[0], sim.work_area.size_m[1]);
    goals
        .iter()
        .map(|(k, g)| {
            (k.clone(), Rect { center: (reanchor(g.center.0, rx, size.0), reanchor(g.center.1, ry, size.1)), size: g.size })
        })
        .collect()
}

/// 物体の置き場の番号（task.start_home_index）。指定があればそれ（範囲外は Err）。無ければ開始時に外形の 4 隅が入っている
/// 長方形を、今のゴール（0）、zones（1〜）の順に調べた最初のもの。どこにも入っていなければ −1。
pub fn start_home_index(ep: &EpisodeSpec, o: &ObjectSpec, size: (f64, f64)) -> Result<i64, String> {
    if let Some((_, i)) = ep.home_overrides.iter().find(|(n, _)| n == &o.name) {
        if !(-1 <= *i && *i <= ep.zones.len() as i64) {
            return Err(format!("置き場の番号 {i} が範囲外（−1〜{}）: {}", ep.zones.len(), o.name));
        }
        return Ok(*i);
    }
    let corners = box_corners(&o.pose, size);
    Ok(std::iter::once(&ep.goal)
        .chain(ep.zones.iter())
        .position(|z| z.contains_all(&corners))
        .map_or(-1, |i| i as i64))
}

/// 爪の間の中心（task.gripper_point）。
pub fn gripper_point(sim: &SimCfg, robot: &Pose2D) -> (f64, f64) {
    robot.transform_point((gripper_center_x(sim), 0.0))
}

/// 物体中心からゴールと反対方向へ offset 離れた点（task.approach_point）。
pub fn approach_point(obj: (f64, f64), goal: (f64, f64), offset: f64) -> (f64, f64) {
    let (dx, dy) = (goal.0 - obj.0, goal.1 - obj.1);
    let n = hypot(dx, dy);
    if n == 0.0 {
        panic!("物体がゴール中心と一致していて押す方向が決まらない");
    }
    (obj.0 - offset * dx / n, obj.1 - offset * dy / n)
}

fn check_stage(stage: i64) -> Result<(), String> {
    if STAGES.contains(&stage) {
        Ok(())
    } else {
        Err(format!("未定義の Stage: {stage}（定義済み: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)）"))
    }
}

/// rng.uniform(*v)（v は [下限, 上限]）
fn uni(rng: &mut NpRng, v: &[f64]) -> f64 {
    rng.uniform(v[0], v[1])
}

/// rng.integers(v[0], v[1] + 1)
fn int_incl(rng: &mut NpRng, v: &[i64]) -> i64 {
    rng.integers_range(v[0], v[1] + 1)
}

fn pick_color(rng: &mut NpRng, pool: &[&'static str]) -> &'static str {
    pool[rng.integers(pool.len() as i64) as usize]
}

fn name(k: i64) -> String {
    format!("obj{k}")
}

/// Python の min(iterable)（最初の最小）で、長方形までの符号付き距離の最小
fn min_sd(corners: &[(f64, f64); 4], r: &Rect) -> f64 {
    let mut m = signed_distance_to_rect(corners[0], r);
    for &c in &corners[1..] {
        let d = signed_distance_to_rect(c, r);
        if d < m {
            m = d;
        }
    }
    m
}

#[derive(Clone)]
pub struct TaskSampler {
    sp: SamplingCfg,
    /// goals_for_area の基にする sim（work_area と goal_zones）
    sim: SimCfg,
    zones_mode: bool,
    /// goal_zones を名前の順（Python の sorted）に
    goals: Vec<(String, Rect)>,
    half: (f64, f64),
    robot_r: f64,
    front_x: f64,
    /// 物体の種類ごと（ObjectType の番号）の外接円の半径と、外形の (x, y)
    obj_r: [f64; 2],
    obj_size: [[f64; 2]; 2],
}

impl TaskSampler {
    pub fn new(cfg: &Cfg) -> Self {
        let sim = &cfg.sim;
        let sp = cfg.task.sampling.clone();
        for (k, v) in [
            ("stage1_front_gap_m", &sp.stage1_front_gap_m),
            ("robot_object_distance_m", &sp.robot_object_distance_m),
            ("stage5_edge_distance_m", &sp.stage5_edge_distance_m),
            ("stage5_robot_distance_m", &sp.stage5_robot_distance_m),
            ("cluster_gap_m", &sp.cluster_gap_m),
            ("random_goal_size_m", &sp.random_goal_size_m),
        ] {
            assert!(v.len() == 2, "sampling.{k} は [下限, 上限]（{v:?}）");
        }
        for (k, v) in [
            ("clutter_range", &sp.clutter_range),
            ("stage7_targets", &sp.stage7_targets),
            ("stage7_clutter", &sp.stage7_clutter),
            ("stage8_zones", &sp.stage8_zones),
            ("stage8_zone_objects", &sp.stage8_zone_objects),
            ("stage8_clutter", &sp.stage8_clutter),
        ] {
            assert!(v.len() == 2, "sampling.{k} は [最小, 最大]（{v:?}）");
        }
        let goals = sim
            .goal_zones
            .iter()
            .map(|(k, z)| {
                assert!(z.center_m.len() == 2 && z.size_m.len() == 2, "goal_zones.{k} の center_m と size_m は 2 つ");
                (k.clone(), Rect { center: (z.center_m[0], z.center_m[1]), size: (z.size_m[0], z.size_m[1]) })
            })
            .collect();
        let size = |t: ObjectType| {
            let s = &cfg.object(t).size_m;
            [s[0], s[1]]
        };
        let (sl, sc) = (size(ObjectType::Lego), size(ObjectType::ToyCar));
        TaskSampler {
            zones_mode: sp.goal_mode == "zones",
            sp,
            sim: sim.clone(),
            goals,
            half: (sim.work_area.size_m[0] / 2.0, sim.work_area.size_m[1] / 2.0),
            robot_r: robot_radius(sim),
            front_x: front_x(sim),
            obj_r: [hypot(sl[0], sl[1]) / 2.0, hypot(sc[0], sc[1]) / 2.0],
            obj_size: [sl, sc],
        }
    }

    pub fn sample(&self, stage: i64, rng: &mut NpRng) -> EpisodeSpec {
        if let Err(e) = check_stage(stage) {
            panic!("{e}");
        }
        // v30: 盤面ごとの作業領域（幅 → 奥行きの順）。この盤面だけ half と名前つきのゴールを置き換えた写しで引く
        let ar = &self.sp.work_area_size_range_m;
        let (s, area) = if ar.is_empty() {
            (None, None)
        } else {
            let a = (uni(rng, &ar[0]), uni(rng, &ar[1]));
            let mut c = self.clone();
            c.half = (a.0 / 2.0, a.1 / 2.0);
            c.goals = goals_for_area(&self.sim, &self.goals, a);
            (Some(c), Some(a))
        };
        let s = s.as_ref().unwrap_or(self);
        for _ in 0..self.sp.max_tries {
            if let Some(mut ep) = s.try_(stage, rng) {
                ep.area_size = area;
                return ep;
            }
        }
        panic!("Stage {stage}: {} 回試しても有効な盤面が引けない。設定を見直すこと", self.sp.max_tries);
    }

    // ---------- 部品 ----------
    fn r(&self, t: ObjectType) -> f64 {
        self.obj_r[t as usize]
    }

    fn robot_ok(&self, x: f64, y: f64) -> bool {
        let m = self.sp.robot_margin_m;
        x.abs() <= self.half.0 - m && y.abs() <= self.half.1 - m
    }

    fn xy(&self, rng: &mut NpRng) -> (f64, f64) {
        let m = self.sp.object_margin_m;
        let x = rng.uniform(-self.half.0 + m, self.half.0 - m);
        let y = rng.uniform(-self.half.1 + m, self.half.1 - m);
        (x, y)
    }

    fn otype(&self, rng: &mut NpRng) -> ObjectType {
        if rng.random() < self.sp.toy_car_probability {
            ObjectType::ToyCar
        } else {
            ObjectType::Lego
        }
    }

    /// task._target_type: Stage 7・8 の対象の種類（v28）。multi_target_toy_car_probability が 0 なら LEGO（乱数を引かない）
    fn target_type(&self, rng: &mut NpRng) -> ObjectType {
        let p = self.sp.multi_target_toy_car_probability;
        if p > 0.0 && rng.random() < p {
            ObjectType::ToyCar
        } else {
            ObjectType::Lego
        }
    }

    /// task._group: Stage 7・8 の対象を近くに置くか（v29）。target_group_probability が 0 なら false（乱数を引かない）
    fn group(&self, rng: &mut NpRng) -> bool {
        let p = self.sp.target_group_probability;
        p > 0.0 && rng.random() < p
    }

    /// task._near: 対象の近く（接しない）に置く（v29）。中心の距離 = 外接円の半径の和 + target_group_gap_m、方向と向きは一様。
    /// 作業領域の余白の外か、他の物体と重なれば None
    fn near(
        &self,
        rng: &mut NpRng,
        base: &ObjectSpec,
        objects: &[ObjectSpec],
        name: String,
        t: ObjectType,
        color: &str,
    ) -> Option<ObjectSpec> {
        let a = rng.uniform(-PI, PI);
        let d = self.r(base.otype) + self.r(t) + uni(rng, &self.sp.target_group_gap_m);
        let (x, y) = (base.pose.x + d * a.cos(), base.pose.y + d * a.sin());
        let m = self.sp.object_margin_m;
        if x.abs() > self.half.0 - m || y.abs() > self.half.1 - m {
            return None;
        }
        if objects.iter().any(|q| dist((x, y), q.pose.xy()) < self.r(t) + self.r(q.otype) + self.sp.object_gap_m) {
            return None;
        }
        Some(ObjectSpec { name, otype: t, pose: Pose2D::new(x, y, rng.uniform(-PI, PI)), color: color.to_string() })
    }

    /// task._clutter_colors: 対象外の色。LEGO は対象の色以外。ミニカーはどの色でもよいが、対象にミニカーが混ざりうる
    /// とき（v28）は対象の色以外
    fn clutter_color(&self, rng: &mut NpRng, t: ObjectType, others: &[&'static str]) -> &'static str {
        if t == ObjectType::Lego || self.sp.multi_target_toy_car_probability > 0.0 {
            pick_color(rng, others)
        } else {
            pick_color(rng, &COLORS)
        }
    }

    /// ロボットの中心を作業領域に一様に置く（Stage 6・7・8）
    fn robot_xy(&self, rng: &mut NpRng) -> (f64, f64) {
        let m = self.sp.robot_margin_m;
        let rx = rng.uniform(-self.half.0 + m, self.half.0 - m);
        let ry = rng.uniform(-self.half.1 + m, self.half.1 - m);
        (rx, ry)
    }

    /// ロボットが物体に重ならない（外接円の間に object_gap_m 以上）
    fn robot_clear(&self, p: (f64, f64), objects: &[ObjectSpec]) -> bool {
        objects.iter().all(|q| !(dist(p, q.pose.xy()) < self.robot_r + self.r(q.otype) + self.sp.object_gap_m))
    }

    /// 他の物体と重ならない（外接円の間に object_gap_m 以上）
    fn spaced(&self, p: (f64, f64), t: ObjectType, objects: &[ObjectSpec]) -> bool {
        objects.iter().all(|q| dist(p, q.pose.xy()) >= self.r(t) + self.r(q.otype) + self.sp.object_gap_m)
    }

    fn robot_near(&self, rng: &mut NpRng, o: (f64, f64), to_goal: f64, stage: i64) -> Option<Pose2D> {
        let sp = &self.sp;
        let (rx, ry, ryaw);
        if stage == 1 {
            let r = self.front_x + uni(rng, &sp.stage1_front_gap_m);
            rx = o.0 - r * to_goal.cos();
            ry = o.1 - r * to_goal.sin();
            ryaw = to_goal + rng.uniform(-sp.stage1_yaw_noise_rad, sp.stage1_yaw_noise_rad);
        } else {
            let u = if stage == 2 || stage == 4 {
                to_goal + PI + rng.uniform(-sp.stage2_max_offset_rad, sp.stage2_max_offset_rad)
            } else {
                rng.uniform(-PI, PI)
            };
            let r = uni(rng, &sp.robot_object_distance_m);
            rx = o.0 + r * u.cos();
            ry = o.1 + r * u.sin();
            ryaw = rng.uniform(-PI, PI);
        }
        if !self.robot_ok(rx, ry) {
            return None;
        }
        Some(Pose2D::new(rx, ry, ieee_remainder(ryaw, 2.0 * PI)))
    }

    /// 対象外の置き方の条件（v9）: 4 隅がゴール内か、外形の 4 隅がゴールから goal_clearance_m 以上離れている。
    fn clear_of_goal(&self, t: ObjectType, pose: &Pose2D, goal: &Rect) -> bool {
        let s = self.obj_size[t as usize];
        let corners = box_corners(pose, (s[0], s[1]));
        if goal.contains_all(&corners) {
            return true;
        }
        min_sd(&corners, goal) >= self.sp.goal_clearance_m
    }

    /// 外形の 4 隅が長方形から goal_clearance_m 以上離れている。
    fn clear_of(&self, t: ObjectType, pose: &Pose2D, rect: &Rect) -> bool {
        let s = self.obj_size[t as usize];
        min_sd(&box_corners(pose, (s[0], s[1])), rect) >= self.sp.goal_clearance_m
    }

    /// 対象に接する物体（同じ向きで、対象の前後か左右に隙間 cluster_gap_m で並べる）。otype / color がなければ乱数で決める。
    fn neighbor(
        &self,
        rng: &mut NpRng,
        target: &ObjectSpec,
        objects: &[ObjectSpec],
        name: String,
        otype: Option<ObjectType>,
        color: Option<&str>,
    ) -> Option<ObjectSpec> {
        let t = match otype {
            Some(t) => t,
            None => self.otype(rng),
        };
        let ts = self.obj_size[target.otype as usize];
        let ns = self.obj_size[t as usize];
        let axis = rng.integers(2) as usize; // 0: 対象の x（長辺）方向に並べる, 1: y 方向
        let sign = if rng.random() < 0.5 { 1.0 } else { -1.0 };
        let gap = uni(rng, &self.sp.cluster_gap_m);
        let along = sign * (ts[axis] / 2.0 + ns[axis] / 2.0 + gap);
        let other = 1 - axis;
        let shift = rng.uniform(-ts[other] / 2.0, ts[other] / 2.0);
        let local = if axis == 0 { (along, shift) } else { (shift, along) };
        let (x, y) = target.pose.transform_point(local);
        let m = self.sp.object_margin_m;
        if x.abs() > self.half.0 - m || y.abs() > self.half.1 - m {
            return None;
        }
        if objects
            .iter()
            .filter(|q| q.name != target.name)
            .any(|q| dist((x, y), q.pose.xy()) < self.r(t) + self.r(q.otype))
        {
            return None;
        }
        let c = match color {
            Some(c) => c,
            None => pick_color(rng, &COLORS),
        };
        Some(ObjectSpec { name, otype: t, pose: Pose2D::new(x, y, target.pose.yaw), color: c.to_string() })
    }

    // ---------- 盤面 ----------
    /// ゴール。名前付きなら goals の番号、乱数の長方形なら None（名前は "random"）
    fn goal(&self, rng: &mut NpRng) -> (Option<usize>, Rect) {
        if self.zones_mode {
            let i = rng.integers(self.goals.len() as i64) as usize;
            return (Some(i), self.goals[i].1);
        }
        let w = uni(rng, &self.sp.random_goal_size_m);
        let h = uni(rng, &self.sp.random_goal_size_m);
        let cx = rng.uniform(-self.half.0 + w / 2.0, self.half.0 - w / 2.0);
        let cy = rng.uniform(-self.half.1 + h / 2.0, self.half.1 - h / 2.0);
        (None, Rect { center: (cx, cy), size: (w, h) })
    }

    fn goal_name(&self, g: Option<usize>) -> String {
        match g {
            Some(i) => self.goals[i].0.clone(),
            None => "random".to_string(),
        }
    }

    /// 壁際（作業領域の端の近く）の LEGO を、内側のゴールへ。
    fn try_stage5(&self, rng: &mut NpRng) -> Option<EpisodeSpec> {
        let sp = &self.sp;
        let side = rng.integers(4) as usize; // 0: 右, 1: 上, 2: 左, 3: 下 の端
        let inward = [(-1.0f64, 0.0f64), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0)][side];
        let d = uni(rng, &sp.stage5_edge_distance_m);
        let along = rng.uniform(-self.half.0 + sp.object_margin_m, self.half.0 - sp.object_margin_m);
        let o = if side == 0 || side == 2 {
            (-inward.0 * (self.half.0 - d), along)
        } else {
            (along, -inward.1 * (self.half.1 - d))
        };
        let (gi, goal) = self.goal(rng);
        let (gx, gy) = (goal.center.0 - o.0, goal.center.1 - o.1);
        if goal.contains(o) || (gx * inward.0 + gy * inward.1) < sp.stage5_goal_max_angle_rad.cos() * hypot(gx, gy) {
            return None;
        }
        let r = uni(rng, &sp.stage5_robot_distance_m);
        let (rx, ry) = (o.0 + inward.0 * r, o.1 + inward.1 * r);
        if !self.robot_ok(rx, ry) {
            return None;
        }
        let yaw = (-inward.1).atan2(-inward.0) + rng.uniform(-sp.stage5_yaw_noise_rad, sp.stage5_yaw_noise_rad);
        let oyaw = rng.uniform(-PI, PI);
        let c = pick_color(rng, &COLORS);
        let target = ObjectSpec { name: name(0), otype: ObjectType::Lego, pose: Pose2D::new(o.0, o.1, oyaw), color: c.to_string() };
        Some(EpisodeSpec {
            stage: 5,
            robot: Pose2D::new(rx, ry, ieee_remainder(yaw, 2.0 * PI)),
            objects: vec![target],
            target_names: vec![name(0)],
            reach_point: None,
            goal_name: self.goal_name(gi),
            goal,
            zones: vec![],
            home_overrides: vec![],
            area_size: None,
        })
    }

    /// 作業領域内の、他の物体と重ならない位置に置く。ok（種類と姿勢 → bool）を渡すと、それも満たす位置を探す。
    fn place(
        &self,
        rng: &mut NpRng,
        objects: &[ObjectSpec],
        t: ObjectType,
        name: String,
        color: &str,
        ok: Option<&dyn Fn(ObjectType, &Pose2D) -> bool>,
    ) -> Option<ObjectSpec> {
        for _ in 0..self.sp.max_tries {
            let p = self.xy(rng);
            if self.spaced(p, t, objects) {
                let pose = Pose2D::new(p.0, p.1, rng.uniform(-PI, PI));
                if ok.is_none_or(|f| f(t, &pose)) {
                    return Some(ObjectSpec { name, otype: t, pose, color: color.to_string() });
                }
            }
        }
        None
    }

    /// 同じ色の LEGO を 2〜5 個まとめて運ぶ（v7）。対象外は色か種類が違う。
    fn try_stage7(&self, rng: &mut NpRng) -> Option<EpisodeSpec> {
        let sp = &self.sp;
        let (gi, goal) = self.goal(rng);
        let color = pick_color(rng, &COLORS);
        let n_t = int_incl(rng, &sp.stage7_targets);
        let n_o = int_incl(rng, &sp.stage7_clutter);
        let mut objects: Vec<ObjectSpec> = Vec::new();
        for i in 0..n_t {
            let tt = self.target_type(rng);
            let o = if i > 0 && rng.random() < sp.cluster_probability {
                let b = rng.integers(objects.len() as i64) as usize;
                self.neighbor(rng, &objects[b], &objects, name(i), Some(tt), Some(color))
            } else if i > 0 && self.group(rng) {
                // v29: それまでの対象の近く（接しない）に置く。まとめて押す練習
                let b = rng.integers(objects.len() as i64) as usize;
                self.near(rng, &objects[b], &objects, name(i), tt, color)
            } else {
                self.place(rng, &objects, tt, name(i), color, None)
            };
            match o {
                Some(o) if !goal.contains(o.pose.xy()) => objects.push(o),
                _ => return None,
            }
        }
        let others: Vec<&'static str> = COLORS.iter().copied().filter(|&c| c != color).collect();
        for i in n_t..n_t + n_o {
            let t = self.otype(rng);
            let c = self.clutter_color(rng, t, &others);
            match self.place(rng, &objects, t, name(i), c, None) {
                Some(o) if self.clear_of_goal(o.otype, &o.pose, &goal) => objects.push(o),
                _ => return None,
            }
        }
        let (rx, ry) = self.robot_xy(rng);
        if !self.robot_clear((rx, ry), &objects) {
            return None;
        }
        let robot = Pose2D::new(rx, ry, rng.uniform(-PI, PI));
        Some(EpisodeSpec {
            stage: 7,
            robot,
            objects,
            target_names: (0..n_t).map(name).collect(),
            reach_point: None,
            goal_name: self.goal_name(gi),
            goal,
            zones: vec![],
            home_overrides: vec![],
            area_size: None,
        })
    }

    /// 2 つの長方形の間に goal_clearance_m 以上の隙間がある（x か y のどちらかの方向で）。
    fn rects_apart(&self, a: &Rect, b: &Rect) -> bool {
        let g = self.sp.goal_clearance_m;
        (a.center.0 - b.center.0).abs() >= (a.size.0 + b.size.0) / 2.0 + g
            || (a.center.1 - b.center.1).abs() >= (a.size.1 + b.size.1) / 2.0 + g
    }

    /// Stage 8 の置き場（v18）。zones: 今のゴール以外の名前付きゴールから選ぶ。random: 乱数の長方形。
    fn zones(&self, rng: &mut NpRng, gi: Option<usize>, goal: &Rect) -> Option<Vec<Rect>> {
        let n = int_incl(rng, &self.sp.stage8_zones);
        assert!(n >= 0, "stage8_zones の数が負（{n}）");
        if self.zones_mode {
            let rest: Vec<usize> = (0..self.goals.len()).filter(|&i| Some(i) != gi).collect();
            let mut idx = rng.choice_no_replace(rest.len() as i64, n as usize);
            idx.sort();
            return Some(idx.into_iter().map(|i| self.goals[rest[i as usize]].1).collect());
        }
        let mut zones: Vec<Rect> = Vec::new();
        for _ in 0..n {
            // 置き場ごとに、ゴールと他の置き場から離れた長方形が出るまで引き直す
            let mut found = false;
            for _ in 0..self.sp.max_tries {
                let (_, z) = self.goal(rng);
                if self.rects_apart(&z, goal) && zones.iter().all(|q| self.rects_apart(&z, q)) {
                    zones.push(z);
                    found = true;
                    break;
                }
            }
            if !found {
                return None;
            }
        }
        Some(zones)
    }

    /// 外形の 4 隅が zone に入り、他の物体と重ならない位置に置く。ok を渡すと、それも満たす位置を探す（v24。向きを
    /// 引いた後に判定するので、ok が無いときと乱数の引き方は同じ）。
    fn place_in(
        &self,
        rng: &mut NpRng,
        objects: &[ObjectSpec],
        t: ObjectType,
        name: String,
        color: &str,
        zone: &Rect,
        ok: Option<&dyn Fn(ObjectType, &Pose2D) -> bool>,
    ) -> Option<ObjectSpec> {
        let r = self.r(t);
        let (hx, hy) = (zone.size.0 / 2.0 - r, zone.size.1 / 2.0 - r);
        if hx <= 0.0 || hy <= 0.0 {
            panic!("置き場 {zone:?} が物体（外接円の半径 {r} m）より小さい");
        }
        for _ in 0..self.sp.max_tries {
            let px = zone.center.0 + rng.uniform(-hx, hx);
            let py = zone.center.1 + rng.uniform(-hy, hy);
            if self.spaced((px, py), t, objects) {
                let pose = Pose2D::new(px, py, rng.uniform(-PI, PI));
                if ok.is_none_or(|f| f(t, &pose)) {
                    return Some(ObjectSpec { name, otype: t, pose, color: color.to_string() });
                }
            }
        }
        None
    }

    /// 連続の色分けの 1 回分（v18）: Stage 7 に、前の指示で運んだ物体の入った置き場を足す。
    fn try_stage8(&self, rng: &mut NpRng) -> Option<EpisodeSpec> {
        let sp = &self.sp;
        let (gi, goal) = self.goal(rng);
        let zones = self.zones(rng, gi, &goal)?;
        let color = pick_color(rng, &COLORS);
        let others: Vec<&'static str> = COLORS.iter().copied().filter(|&c| c != color).collect();
        let mut objects: Vec<ObjectSpec> = Vec::new();
        let mut k = 0i64;
        for z in &zones {
            // 置き場の物体: 置き場ごとに 1 色の LEGO（ミニカーは toy_car_probability）
            let zc = pick_color(rng, &others);
            for _ in 0..int_incl(rng, &sp.stage8_zone_objects) {
                let t = self.otype(rng);
                let o = self.place_in(rng, &objects, t, name(k), zc, z, None)?;
                objects.push(o);
                k += 1;
            }
        }
        let target_ok = |t: ObjectType, p: &Pose2D| !goal.contains(p.xy()) && zones.iter().all(|z| self.clear_of(t, p, z));
        let clutter_ok = |t: ObjectType, p: &Pose2D| self.clear_of_goal(t, p, &goal) && zones.iter().all(|z| self.clear_of(t, p, z));

        // 物体は 1 個ずつ、条件を満たす位置が出るまで引き直す。接して置く位置が条件を外れたら、自由な位置に置く
        let n_t = int_incl(rng, &sp.stage7_targets);
        let mut targets: Vec<usize> = Vec::new(); // objects の中の番号
        for i in 0..n_t {
            let mut o = None;
            let tt = self.target_type(rng);
            if i > 0 && rng.random() < sp.cluster_probability {
                let b = targets[rng.integers(targets.len() as i64) as usize];
                o = self.neighbor(rng, &objects[b], &objects, name(k), Some(tt), Some(color));
            } else if i > 0 && self.group(rng) {
                // v29: それまでの対象の近く（接しない）
                let b = targets[rng.integers(targets.len() as i64) as usize];
                o = self.near(rng, &objects[b], &objects, name(k), tt, color);
            }
            if !o.as_ref().is_some_and(|o| target_ok(o.otype, &o.pose)) {
                o = self.place(rng, &objects, tt, name(k), color, Some(&target_ok));
            }
            objects.push(o?);
            targets.push(objects.len() - 1);
            k += 1;
        }
        for _ in 0..int_incl(rng, &sp.stage8_clutter) {
            let t = self.otype(rng);
            let c = self.clutter_color(rng, t, &others);
            let mut o = None;
            if rng.random() < sp.cluster_probability {
                // 対象に接して置く（Stage 6 と同じ。引き離す練習）
                let b = targets[rng.integers(targets.len() as i64) as usize];
                o = self.neighbor(rng, &objects[b], &objects, name(k), Some(t), Some(c));
            }
            if !o.as_ref().is_some_and(|o| clutter_ok(o.otype, &o.pose)) {
                o = self.place(rng, &objects, t, name(k), c, Some(&clutter_ok));
            }
            objects.push(o?);
            k += 1;
        }
        let mut robot_xy = None;
        for _ in 0..sp.max_tries {
            let p = self.robot_xy(rng);
            if objects.iter().all(|q| dist(p, q.pose.xy()) >= self.robot_r + self.r(q.otype) + sp.object_gap_m) {
                robot_xy = Some(p);
                break;
            }
        }
        let (rx, ry) = robot_xy?;
        let robot = Pose2D::new(rx, ry, rng.uniform(-PI, PI));
        let target_names = targets.iter().map(|&i| objects[i].name.clone()).collect();
        Some(EpisodeSpec {
            stage: 8,
            robot,
            objects,
            target_names,
            reach_point: None,
            goal_name: self.goal_name(gi),
            goal,
            zones,
            home_overrides: vec![],
            area_size: None,
        })
    }

    /// ロボットと物体の外接円の間に object_gap_m 以上（Stage 8 のロボットの置き方と同じ式、task._robot_clear）
    fn robot_clear_of(&self, robot: &Pose2D, t: ObjectType, p: &Pose2D) -> bool {
        dist(robot.xy(), p.xy()) >= self.robot_r + self.r(t) + self.sp.object_gap_m
    }

    /// 中心が作業領域の端から object_margin_m 以上内側（task._in_margin）
    fn in_margin(&self, x: f64, y: f64) -> bool {
        let m = self.sp.object_margin_m;
        x.abs() <= self.half.0 - m && y.abs() <= self.half.1 - m
    }

    /// 置き場 zone の縁の外、中心の符号付き距離が stage9_home_out_m の範囲に置き直す（task._place_out）。
    fn place_out(&self, rng: &mut NpRng, objects: &[ObjectSpec], o: &ObjectSpec, zone: &Rect, avoid: &[Rect], robot: &Pose2D)
        -> Option<ObjectSpec> {
        let (lo, hi) = (self.sp.stage9_home_out_m[0], self.sp.stage9_home_out_m[1]);
        let t = o.otype;
        let (wx, wy) = (zone.size.0 / 2.0 + hi, zone.size.1 / 2.0 + hi);
        for _ in 0..self.sp.max_tries {
            let x = zone.center.0 + rng.uniform(-wx, wx);
            let y = zone.center.1 + rng.uniform(-wy, wy);
            let pose = Pose2D::new(x, y, rng.uniform(-PI, PI));
            let sd = signed_distance_to_rect((x, y), zone);
            if lo <= sd
                && sd <= hi
                && self.in_margin(x, y)
                && self.spaced((x, y), t, objects)
                && self.robot_clear_of(robot, t, &pose)
                && avoid.iter().all(|z| self.clear_of(t, &pose, z))
            {
                return Some(ObjectSpec { name: o.name.clone(), otype: t, pose, color: o.color.clone() });
            }
        }
        None
    }

    /// 届けた対象に挟まった対象外（v35、task._wedge）: 毎回 targets から 1 個選び、neighbor でそれに接して置く。4 隅が
    /// goal の中、ok（作業領域の余白の内側、ロボットと重ならない）を満たすまで max_tries 回まで引き直す。
    fn wedge(
        &self,
        rng: &mut NpRng,
        objects: &[ObjectSpec],
        o: &ObjectSpec,
        targets: &[ObjectSpec],
        goal: &Rect,
        ok: &dyn Fn(ObjectType, &Pose2D) -> bool,
    ) -> Option<ObjectSpec> {
        let s = self.obj_size[o.otype as usize];
        for _ in 0..self.sp.max_tries {
            let t = &targets[rng.integers(targets.len() as i64) as usize];
            if let Some(c) = self.neighbor(rng, t, objects, o.name.clone(), Some(o.otype), Some(o.color.as_str())) {
                if goal.contains_all(&box_corners(&c.pose, (s[0], s[1]))) && ok(c.otype, &c.pose) {
                    return Some(c);
                }
            }
        }
        None
    }

    /// 後片付けから始まる盤面（v24、task._try_stage9）。乱数の引き方は task.py の docstring と同じ順。
    fn try_stage9(&self, rng: &mut NpRng) -> Option<EpisodeSpec> {
        let sp = &self.sp;
        let mut ep = self.try_stage8(rng)?;
        ep.stage = 9;
        if !(rng.random() < sp.stage9_mistake_probability) {
            return Some(ep);
        }
        let base_objects = ep.objects.clone();
        let robot = ep.robot;
        let goal = ep.goal;
        let relocated_ok = |t: ObjectType, p: &Pose2D| self.in_margin(p.x, p.y) && self.robot_clear_of(&robot, t, p);
        // 置き直した物体の番号（objects の中）と置き場の番号
        let mut over: Vec<(usize, i64)> = Vec::new();
        let others = |objects: &[ObjectSpec], i: usize| -> Vec<ObjectSpec> {
            objects.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, q)| q.clone()).collect()
        };
        let index = |n: &String| ep.objects.iter().position(|o| &o.name == n).unwrap();
        let target_idx: Vec<usize> = ep.target_names.iter().map(index).collect();
        let mut objects = ep.objects.clone();
        let delivered = rng.random() < sp.stage9_all_delivered_probability;
        if delivered {
            for &i in &target_idx {
                let t = &objects[i];
                let o = self.place_in(rng, &others(&objects, i), t.otype, t.name.clone(), &t.color.clone(), &goal,
                                      Some(&relocated_ok))?;
                objects[i] = o;
                over.push((i, -1));
            }
        }
        // 対象外の置き場の番号（Stage 8 の盤面の開始の位置から、物体の順）
        let home: Vec<(usize, i64)> = (0..base_objects.len())
            .filter(|i| !target_idx.contains(i))
            .map(|i| {
                let o = &base_objects[i];
                (i, start_home_index(&ep, o, (self.obj_size[o.otype as usize][0], self.obj_size[o.otype as usize][1])).unwrap())
            })
            .collect();
        for _ in 0..rng.integers_range(1, 3) {
            let chosen = |i: usize| over.iter().any(|&(j, _)| j == i);
            let ins: Vec<usize> = home.iter().filter(|&&(i, h)| h == -1 && !chosen(i)).map(|&(i, _)| i).collect();
            let outs: Vec<usize> = home.iter().filter(|&&(i, h)| h >= 1 && !chosen(i)).map(|&(i, _)| i).collect();
            if ins.is_empty() && outs.is_empty() {
                return None;
            }
            let put_in = if !ins.is_empty() && !outs.is_empty() { rng.random() < 0.5 } else { !ins.is_empty() };
            let pool = if put_in { &ins } else { &outs };
            let i = pool[rng.integers(pool.len() as i64) as usize];
            let b = objects[i].clone();
            let o = if put_in && delivered && sp.stage9_wedged_probability > 0.0
                && rng.random() < sp.stage9_wedged_probability
            {
                let targets: Vec<ObjectSpec> = target_idx.iter().map(|&j| objects[j].clone()).collect();
                self.wedge(rng, &others(&objects, i), &b, &targets, &goal, &relocated_ok)?
            } else if put_in {
                self.place_in(rng, &others(&objects, i), b.otype, b.name.clone(), &b.color, &goal, Some(&relocated_ok))?
            } else {
                let h = home.iter().find(|&&(j, _)| j == i).unwrap().1;
                let zone = ep.zones[(h - 1) as usize];
                let mut avoid = vec![goal];
                avoid.extend(ep.zones.iter().enumerate().filter(|&(k, _)| k as i64 != h - 1).map(|(_, z)| *z));
                self.place_out(rng, &others(&objects, i), &b, &zone, &avoid, &robot)?
            };
            objects[i] = o;
            let h = if put_in { -1 } else { home.iter().find(|&&(j, _)| j == i).unwrap().1 };
            over.push((i, h));
        }
        over.sort_by_key(|&(i, _)| i); // 物体の順（task.py と同じ）
        ep.home_overrides = over.iter().map(|&(i, h)| (objects[i].name.clone(), h)).collect();
        ep.objects = objects;
        Some(ep)
    }

    fn try_(&self, stage: i64, rng: &mut NpRng) -> Option<EpisodeSpec> {
        match stage {
            9 => return self.try_stage9(rng),
            5 => return self.try_stage5(rng),
            7 => return self.try_stage7(rng),
            8 => return self.try_stage8(rng),
            _ => {}
        }
        let (gi, goal) = self.goal(rng);
        let o = self.xy(rng);
        if goal.contains(o) {
            return None;
        }
        let to_goal = (goal.center.1 - o.1).atan2(goal.center.0 - o.0);
        let ep = |robot: Pose2D, objects: Vec<ObjectSpec>, target_names: Vec<String>, reach_point| EpisodeSpec {
            stage,
            robot,
            objects,
            target_names,
            reach_point,
            goal_name: self.goal_name(gi),
            goal,
            zones: vec![],
            home_overrides: vec![],
            area_size: None,
        };

        if stage == 0 {
            let robot = self.robot_near(rng, o, to_goal, 0)?;
            return Some(ep(robot, vec![], vec![], Some(o)));
        }

        let ttype = match stage {
            1..=3 => ObjectType::Lego,
            4 => ObjectType::ToyCar,
            _ => self.otype(rng), // Stage 6
        };
        let oyaw = rng.uniform(-PI, PI);
        let c = pick_color(rng, &COLORS);
        let mut objects = vec![ObjectSpec { name: name(0), otype: ttype, pose: Pose2D::new(o.0, o.1, oyaw), color: c.to_string() }];

        let robot = if stage != 6 {
            self.robot_near(rng, o, to_goal, stage)?
        } else {
            let sp = &self.sp;
            let n = int_incl(rng, &sp.clutter_range);
            let mut neighbors = 0;
            if rng.random() < sp.cluster_probability {
                neighbors = n.min(rng.integers_range(1, sp.cluster_max_neighbors + 1));
            }
            for i in 1..=neighbors {
                match self.neighbor(rng, &objects[0], &objects, name(i), None, None) {
                    Some(nb) if self.clear_of_goal(nb.otype, &nb.pose, &goal) => objects.push(nb),
                    _ => return None,
                }
            }
            for i in neighbors + 1..=n {
                let mut cand = None;
                for _ in 0..sp.max_tries {
                    let t = self.otype(rng);
                    let p = self.xy(rng);
                    let pose = Pose2D::new(p.0, p.1, rng.uniform(-PI, PI));
                    let c = pick_color(rng, &COLORS);
                    if self.spaced(p, t, &objects) && self.clear_of_goal(t, &pose, &goal) {
                        cand = Some(ObjectSpec { name: name(i), otype: t, pose, color: c.to_string() });
                        break;
                    }
                }
                objects.push(cand?);
            }
            let (rx, ry) = self.robot_xy(rng);
            if !self.robot_clear((rx, ry), &objects) {
                return None;
            }
            Pose2D::new(rx, ry, rng.uniform(-PI, PI))
        };
        Some(ep(robot, objects, vec![name(0)], None))
    }
}

// ---------- Python から ----------
type RngState = (u128, u128, Option<u32>);

fn rng_state(r: &NpRng) -> RngState {
    let (s, i) = r.state();
    (s, i, r.uint32_buffer())
}

fn rect_tuple(r: &Rect) -> ((f64, f64), (f64, f64)) {
    (r.center, r.size)
}

/// TaskSampler（試験 test/test_native_sampler.py が Python 版と突き合わせる）。
#[pyclass(name = "TaskSampler", module = "toy_sweeper_native")]
pub struct PyTaskSampler {
    inner: TaskSampler,
}

#[pymethods]
impl PyTaskSampler {
    #[new]
    fn new(cfg_json: &str) -> PyResult<Self> {
        let c = Cfg::from_json(cfg_json).map_err(PyValueError::new_err)?;
        Ok(Self { inner: TaskSampler::new(&c) })
    }

    /// rng_state = (state, inc, 32 ビットの残りか None)。(盤面の dict, 引いた後の rng_state) を返す。
    fn sample<'py>(&self, py: Python<'py>, stage: i64, rng_state: RngState) -> PyResult<(Bound<'py, PyDict>, RngState)> {
        check_stage(stage).map_err(PyValueError::new_err)?;
        let mut rng = NpRng::from_full_state(rng_state.0, rng_state.1, rng_state.2);
        let ep = self.inner.sample(stage, &mut rng);
        let d = PyDict::new(py);
        d.set_item("stage", ep.stage)?;
        d.set_item("robot", (ep.robot.x, ep.robot.y, ep.robot.yaw))?;
        let objs: Vec<_> = ep
            .objects
            .iter()
            .map(|o| (o.name.clone(), o.otype as i64, o.pose.x, o.pose.y, o.pose.yaw, o.color.clone()))
            .collect();
        d.set_item("objects", objs)?;
        d.set_item("target_names", ep.target_names)?;
        d.set_item("reach_point", ep.reach_point)?;
        d.set_item("goal_name", ep.goal_name)?;
        d.set_item("goal", rect_tuple(&ep.goal))?;
        d.set_item("zones", ep.zones.iter().map(rect_tuple).collect::<Vec<_>>())?;
        d.set_item("home_overrides", ep.home_overrides)?;
        d.set_item("area_size", ep.area_size)?;
        Ok((d, self::rng_state(&rng)))
    }

    /// 試験用: n 枚続けて引き、引いた後の rng_state だけ返す（速さの計測）。
    fn _sample_count(&self, stage: i64, rng_state: RngState, n: usize) -> PyResult<RngState> {
        check_stage(stage).map_err(PyValueError::new_err)?;
        let mut rng = NpRng::from_full_state(rng_state.0, rng_state.1, rng_state.2);
        for _ in 0..n {
            std::hint::black_box(self.inner.sample(stage, &mut rng));
        }
        Ok(self::rng_state(&rng))
    }
}

/// 試験用: gripper_point
#[pyfunction]
fn _gripper_point(cfg_json: &str, x: f64, y: f64, yaw: f64) -> PyResult<(f64, f64)> {
    let c = Cfg::from_json(cfg_json).map_err(PyValueError::new_err)?;
    Ok(gripper_point(&c.sim, &Pose2D::new(x, y, yaw)))
}

/// 試験用: approach_point
#[pyfunction]
fn _approach_point(obj: (f64, f64), goal: (f64, f64), offset: f64) -> (f64, f64) {
    approach_point(obj, goal, offset)
}

pub fn py_register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTaskSampler>()?;
    m.add_function(wrap_pyfunction!(_gripper_point, m)?)?;
    m.add_function(wrap_pyfunction!(_approach_point, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approach_point_basic() {
        assert_eq!(approach_point((0.0, 0.0), (3.0, 4.0), 5.0), (-3.0, -4.0));
    }

    #[test]
    #[should_panic(expected = "ゴール中心と一致")]
    fn approach_point_same_point_panics() {
        approach_point((0.1, 0.2), (0.1, 0.2), 0.05);
    }

    #[test]
    fn unknown_stage_is_error() {
        assert!(check_stage(10).unwrap_err().contains("未定義の Stage: 10"));
        assert!(STAGES.iter().all(|&s| check_stage(s).is_ok()));
    }
}
