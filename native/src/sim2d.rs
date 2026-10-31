//! toy_sweeper.sim2d.Sim2D の Rust 版。同梱の Chipmunk2D（pymunk と同じ C、同じフラグ）を直接呼び、pymunk が呼ぶのと
//! 同じ C の関数を同じ順で呼ぶ（形の追加・削除の順は解く順を変えるので、これも同じにする）。浮動小数点の演算も
//! Python と同じ順にして、軌跡をビット単位で同じにする（test/test_native_sim2d.py）。
//!
//! pymunk と合わせたところ（pymunk 7.3.1 の Python のソースから）:
//! - Space.add は、まず Body を全部、次に Shape / Constraint を引数の順に足す。
//! - Space.static_body は空間に元からある静的な体ではなく、最初に触れたときに作る新しい静的な体（cpBodyNewStatic を
//!   cpSpaceAddBody）。壁があるときだけ、制御用の体と拘束の後に足される。
//! - Poly(body, verts) は cpPolyShapeNew（恒等変換、半径 0）、Poly.create_box は cpBoxShapeNew（半径 0）。
//! - ShapeFilter() は (0, 0xFFFFFFFF, 0xFFFFFFFF)、ShapeFilter(group=1) は (1, 0xFFFFFFFF, 0xFFFFFFFF)。
//! - 床の摩擦は pymunk の velocity_func と同じく cpBodySetVelocityUpdateFunc に渡す（Rust の関数）。速度の読み書きは
//!   pymunk の属性と同じ cpBodyGetVelocity / cpBodySetVelocity など。
//! - 爪は drive の刻みの初め（cpSpaceStep の前）に jaw_speed_radps で回し、形を作り直す（sim2d.py と同じ所、同じ順）。
//! - 外した形と拘束は、pymunk では後でごみ集めが free する。ここではすぐ free する（空間から外した後は空間が触れない）。
use crate::config::{Cfg, ObjectType};
use crate::cp::*;
use crate::geometry::{box_corners, py_max, py_min, Pose2D};
use crate::rng::NpRng;
use crate::robot_model::{angle_touching, body_center_x, jaw_inner_half_gap, jaw_polygons, SIDES};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::os::raw::c_void;
use std::ptr;

pub const GRAVITY: f64 = 9.81;
/// ロボットの形状と把持中の物体はこのグループにして、互いの衝突を切る
const GRASP_GROUP: cpGroup = 1;

/// spaces.GripperState と同じ番号
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GripperState {
    Open = 0,
    Closed = 1,
    Moving = 2,
    Unknown = 3,
}

impl GripperState {
    pub fn from_index(i: i64) -> Option<GripperState> {
        match i {
            0 => Some(GripperState::Open),
            1 => Some(GripperState::Closed),
            2 => Some(GripperState::Moving),
            3 => Some(GripperState::Unknown),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectSpec {
    pub name: String,
    pub otype: ObjectType,
    pub pose: Pose2D,
    /// 描画と VLM 用画像のためだけの情報。物理と Observation には使わない
    pub color: String,
}

/// Sim2D の失敗。Invalid は引数が不正（Python の ValueError）、State は呼ぶ順や状態が不正（RuntimeError）
#[derive(Debug, Clone, PartialEq)]
pub enum SimError {
    Invalid(String),
    State(String),
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimError::Invalid(m) | SimError::State(m) => f.write_str(m),
        }
    }
}

impl From<SimError> for PyErr {
    fn from(e: SimError) -> PyErr {
        match e {
            SimError::Invalid(m) => PyValueError::new_err(m),
            SimError::State(m) => PyRuntimeError::new_err(m),
        }
    }
}

/// 床の摩擦の値（物体の体の user data に置く。Box なので場所は動かない）
struct FloorFriction {
    mu_x: f64,
    mu_y: f64,
    r_eff: f64,
    mass: f64,
    moment: f64,
}

#[inline]
fn toward_zero(v: f64, dv: f64) -> f64 {
    if v.abs() <= dv {
        0.0
    } else {
        v - dv.copysign(v)
    }
}

// ---------- C から呼ばれる関数（cpSpaceStep の中の床の摩擦、cpBodyEachArbiter の中の接触） ----------
// panic は C の枠を越えて巻き戻せない（越えると abort）。中身は panic しない形（番号は確かめてから使う）で書き、
// 念のため catch_unwind でも包む。失敗はこのスレッドの CALLBACK_ERROR に記録し、C から戻った後に drive が Err にする
// （C の途中では止められないので、その刻みは最後まで進む）。

thread_local! {
    static CALLBACK_ERROR: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// 最初の失敗だけ残す。記録できない（スレッドの終わり）ときは、黙って続けずに理由を書いて止める（最後の手段）
fn record_callback_error(msg: String) {
    let ok = CALLBACK_ERROR.try_with(|c| match c.try_borrow_mut() {
        Ok(mut c) => {
            c.get_or_insert(msg.clone());
            true
        }
        Err(_) => false,
    });
    if ok != Ok(true) {
        eprintln!("toy_sweeper_native: C から呼ばれた関数の失敗を記録できないので止める: {msg}");
        std::process::abort();
    }
}

/// C から呼ばれた関数の失敗（あれば）を取り出す
fn take_callback_error() -> Option<String> {
    CALLBACK_ERROR.with(|c| c.borrow_mut().take())
}

/// C から呼ばれる関数の中身を包む（Err と panic を記録する）
fn in_callback(name: &str, f: impl FnOnce() -> Result<(), String>) {
    let msg = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => return,
        Ok(Err(m)) => m,
        Err(p) => match (p.downcast_ref::<&str>(), p.downcast_ref::<String>()) {
            (Some(s), _) => format!("panic: {s}"),
            (_, Some(s)) => format!("panic: {s}"),
            _ => "panic".to_string(),
        },
    };
    record_callback_error(format!("{name}: {msg}"));
}

/// sim2d._floor_friction_velocity_func と同じ計算（同じ順）。重力 0・外力なしの前提で速度の積分はしない。
/// 止まっている物体はすぐ返す。
unsafe extern "C" fn floor_friction_velocity(body: *mut cpBody, _gravity: cpVect, _damping: f64, dt: f64) {
    in_callback("床の摩擦", || {
        let p = cpBodyGetUserData(body) as *const FloorFriction;
        if p.is_null() {
            return Err("物体の体に床の摩擦の値（user data）がない".into());
        }
        let p = &*p;
        let v = cpBodyGetVelocity(body);
        let w = cpBodyGetAngularVelocity(body);
        if v.x == 0.0 && v.y == 0.0 && w == 0.0 {
            return Ok(());
        }
        let a = cpBodyGetAngle(body);
        let (c, s) = (a.cos(), a.sin());
        let (lx, ly) = (c * v.x + s * v.y, -s * v.x + c * v.y);
        let lx = toward_zero(lx, p.mu_x * GRAVITY * dt);
        let ly = toward_zero(ly, p.mu_y * GRAVITY * dt);
        cpBodySetVelocity(body, cpv(c * lx - s * ly, s * lx + c * ly));
        cpBodySetAngularVelocity(body, toward_zero(w, p.mu_y * p.mass * GRAVITY * p.r_eff / p.moment * dt));
        Ok(())
    })
}

struct ContactCtx<'a> {
    chassis: *mut cpShape,
    touched: &'a mut [bool],
}

/// Sim2D._note_chassis_contact: 車体の形に触れた物体を記録する（物体の形の user data は 番号 + 1、他は 0）
unsafe extern "C" fn note_chassis_contact(_body: *mut cpBody, arb: *mut cpArbiter, data: *mut c_void) {
    in_callback("車体の接触", || {
        let ctx = (data as *mut ContactCtx).as_mut().ok_or("接触の記録先（data）がない")?;
        let (mut a, mut b): (*mut cpShape, *mut cpShape) = (ptr::null_mut(), ptr::null_mut());
        cpArbiterGetShapes(arb, &mut a, &mut b);
        if a == ctx.chassis || b == ctx.chassis {
            let other = if a == ctx.chassis { b } else { a };
            let k = cpShapeGetUserData(other) as usize;
            if k != 0 {
                let n = ctx.touched.len();
                let t = ctx.touched.get_mut(k - 1)
                    .ok_or_else(|| format!("形の user data の物体の番号 {} が範囲外（物体 {n} 個）", k - 1))?;
                *t = true;
            }
        }
        Ok(())
    })
}

struct Obj {
    body: *mut cpBody,
    shape: *mut cpShape,
    spec: ObjectSpec,
    _friction: Box<FloorFriction>,
}

/// reset ごとに作り直す物理の世界。持っている C の物はすべてここで free する。
struct World {
    space: *mut cpSpace,
    robot: *mut cpBody,
    control: *mut cpBody,
    static_body: *mut cpBody,
    chassis: *mut cpShape,
    walls: Vec<*mut cpShape>,
    drive_joints: [*mut cpConstraint; 2],
    jaws: Vec<*mut cpShape>,
    objects: Vec<Obj>,
    /// つかんでいる物体の番号と、そのピンと歯車の拘束
    grasp: Option<(usize, [*mut cpConstraint; 2])>,
    /// 直前の drive の間に車体の形（アームは除く）に触れた物体（v19）
    touched: Vec<bool>,
}

// SAFETY（Send）: World は C の物（Space と、その中の体・形・拘束）を自分だけで持ち、ほかの World や Rust の値から
// 指されない。Chipmunk は書き換える大域の状態を持たない（多スレッドの cpHastySpace はビルドしない）。床の摩擦の値は
// World の Obj が持つ Box で、一緒に移る。なので持ち主（VecEnv の 1 つの env）ごと別のスレッドへ移せる。
// SAFETY（Sync）: World を書き換えるのは &mut の Sim2D からだけ（reset、set_jaws、drive、set_gripper、kick、attach、
// detach。C の体の速度や拘束を変える関数はすべて &mut self）。&Sim2D から呼べる関数は C の値を読むだけ
// （cpBodyGetPosition / cpBodyGetAngle / cpBodyWorldToLocal）。なので共有の参照から並んで読んでも安全
// （pyo3 の pyclass が Sync を求める）
unsafe impl Send for World {}
unsafe impl Sync for World {}

impl Drop for World {
    fn drop(&mut self) {
        unsafe {
            // 空間を先に free する（cpSpaceFree は残っている体を起こすので、体はまだ生きている必要がある）
            cpSpaceFree(self.space);
            for &s in self.walls.iter().chain(self.jaws.iter()).chain([self.chassis].iter()) {
                cpShapeFree(s);
            }
            for &c in self.drive_joints.iter() {
                cpConstraintFree(c);
            }
            if let Some((_, cs)) = self.grasp {
                for c in cs {
                    cpConstraintFree(c);
                }
            }
            for o in &self.objects {
                cpShapeFree(o.shape);
                cpBodyFree(o.body);
            }
            cpBodyFree(self.robot);
            cpBodyFree(self.control);
            if !self.static_body.is_null() {
                cpBodyFree(self.static_body);
            }
        }
    }
}

pub struct Sim2D {
    cfg: Cfg,
    dt: f64,
    world: Option<World>,
    /// 今の爪の角度（回っている途中ならその角度。形もこの角度）
    jaw_angle: f64,
    /// 爪が向かっている角度（drive の刻みごとに jaw_speed_radps で近づく）
    jaw_target: f64,
    /// 閉じながら向かっている対象の番号（着いたらつなぐ）
    closing_on: Option<usize>,
    /// sim2d.jaw_stall で止まった刻みがあったか（actuate_gripper の中）
    stalled: bool,
    /// 指令した開閉の状態
    gripper: GripperState,
    pub time_s: f64,
    /// 車輪の速さが指令に追いつく時定数（v36、旋回の惰性）。0 なら指令どおり。env の reset が引いて入れる
    pub motor_lag_s: f64,
    /// 今の (前進 m/s, 旋回 rad/s)。motor_lag_s > 0 のとき指令に追いつく
    wheel_v: (f64, f64),
}

impl Sim2D {
    pub fn new(cfg: &Cfg) -> Sim2D {
        Sim2D {
            cfg: cfg.clone(),
            dt: cfg.sim.sim2d.physics_dt_s,
            world: None,
            jaw_angle: cfg.sim.gripper.jaw_open_angle_rad,
            jaw_target: cfg.sim.gripper.jaw_open_angle_rad,
            closing_on: None,
            stalled: false,
            gripper: GripperState::Unknown,
            time_s: 0.0,
            motor_lag_s: 0.0,
            wheel_v: (0.0, 0.0),
        }
    }

    pub fn cfg(&self) -> &Cfg {
        &self.cfg
    }

    fn w(&self) -> &World {
        self.world.as_ref().expect("Sim2D: reset() が先")
    }

    // ---------- 構築 ----------
    pub fn reset(&mut self, robot: Pose2D, objects: &[ObjectSpec], gripper: GripperState) -> Result<(), SimError> {
        if gripper != GripperState::Open && gripper != GripperState::Closed {
            return Err(SimError::Invalid(format!("初期グリッパー状態は OPEN か CLOSED: {gripper:?}")));
        }
        for (i, o) in objects.iter().enumerate() {
            if objects[..i].iter().any(|p| p.name == o.name) {
                let names: Vec<&str> = objects.iter().map(|o| o.name.as_str()).collect();
                return Err(SimError::Invalid(format!("物体名が重複している: {names:?}")));
            }
        }
        self.world = None; // 前の世界を free する
        let c = &self.cfg.sim;
        unsafe {
            let space = cpSpaceNew();
            cpSpaceSetGravity(space, cpv(0.0, 0.0));
            cpSpaceSetIterations(space, i32::try_from(c.sim2d.solver_iterations).expect("sim2d.solver_iterations が C の int に入らない"));
            cpSpaceSetCollisionSlop(space, c.sim2d.collision_slop_m);

            let (bx, by) = (c.robot.body_size_m[0], c.robot.body_size_m[1]);
            let cx = body_center_x(c);
            let m = c.robot.mass_kg;
            let body = cpBodyNew(m, cpMomentForBox(m, bx, by) + m * cx * cx);
            cpBodySetCenterOfGravity(body, cpv(cx, 0.0));
            // 重心がずれているので、向きを先に決めてから原点（車軸中点）の位置を置く
            cpBodySetAngle(body, robot.yaw);
            cpBodySetPosition(body, cpv(robot.x, robot.y));
            let control = cpBodyNewKinematic();
            cpBodySetPosition(control, cpv(robot.x, robot.y));
            let f_max = c.robot.wheel_floor_friction * m * GRAVITY;
            let pivot = cpPivotJointNew2(control, body, cpv(0.0, 0.0), cpv(0.0, 0.0));
            cpConstraintSetMaxBias(pivot, 0.0);
            cpConstraintSetMaxForce(pivot, f_max);
            let gear = cpGearJointNew(control, body, 0.0, 1.0);
            cpConstraintSetMaxBias(gear, 0.0);
            cpConstraintSetMaxForce(gear, f_max * c.robot.wheel_separation_m / 2.0);
            cpSpaceAddBody(space, control);
            cpSpaceAddConstraint(space, pivot);
            cpSpaceAddConstraint(space, gear);

            let mut static_body = ptr::null_mut();
            let mut walls = Vec::new();
            if c.walls.enabled {
                // pymunk の space.static_body（最初に触れたときに作って足す）
                static_body = cpBodyNewStatic();
                cpSpaceAddBody(space, static_body);
                let (hx, hy) = (c.work_area.size_m[0] / 2.0, c.work_area.size_m[1] / 2.0);
                // すり抜け防止のため実物の壁より厚い（内側の面は同じ）。線分の内側の面が境界に来るよう r だけ外へずらす
                let r = c.sim2d.wall_collision_thickness_m / 2.0;
                let (x, y) = (hx + r, hy + r);
                for (a, b) in [((-x, -y), (x, -y)), ((x, -y), (x, y)), ((x, y), (-x, y)), ((-x, y), (-x, -y))] {
                    let seg = cpSegmentShapeNew(static_body, cpv(a.0, a.1), cpv(b.0, b.1), r);
                    cpShapeSetFriction(seg, c.walls.friction);
                    cpShapeSetElasticity(seg, 0.0);
                    cpSpaceAddShape(space, seg);
                    walls.push(seg);
                }
            }
            let verts = [cpv(cx - bx / 2.0, -by / 2.0), cpv(cx + bx / 2.0, -by / 2.0), cpv(cx + bx / 2.0, by / 2.0),
                         cpv(cx - bx / 2.0, by / 2.0)];
            let chassis = cpPolyShapeNew(body, 4, verts.as_ptr(), TRANSFORM_IDENTITY, 0.0);
            cpShapeSetFriction(chassis, c.contact.robot_object_friction);
            cpShapeSetElasticity(chassis, 0.0);
            cpShapeSetFilter(chassis, shape_filter(GRASP_GROUP));
            cpSpaceAddBody(space, body);
            cpSpaceAddShape(space, chassis);

            let mut objs = Vec::with_capacity(objects.len());
            for (i, spec) in objects.iter().enumerate() {
                let ocfg = self.cfg.object(spec.otype);
                let (sx, sy) = (ocfg.size_m[0], ocfg.size_m[1]);
                let moment = cpMomentForBox(ocfg.mass_kg, sx, sy);
                let ob = cpBodyNew(ocfg.mass_kg, moment);
                cpBodySetPosition(ob, cpv(spec.pose.x, spec.pose.y));
                cpBodySetAngle(ob, spec.pose.yaw);
                let fr = Box::new(FloorFriction {
                    mu_x: ocfg.floor_friction_xy[0],
                    mu_y: ocfg.floor_friction_xy[1],
                    r_eff: self.cfg.friction_radius(spec.otype),
                    mass: ocfg.mass_kg,
                    moment,
                });
                cpBodySetUserData(ob, &*fr as *const FloorFriction as *mut c_void);
                cpBodySetVelocityUpdateFunc(ob, floor_friction_velocity);
                let shape = cpBoxShapeNew(ob, sx, sy, 0.0);
                // 物体側 1.0（Chipmunk は接触摩擦を積で合成する。sim2d.py の説明）
                cpShapeSetFriction(shape, 1.0);
                cpShapeSetElasticity(shape, 0.0);
                cpShapeSetUserData(shape, (i + 1) as *mut c_void);
                cpSpaceAddBody(space, ob);
                cpSpaceAddShape(space, shape);
                objs.push(Obj { body: ob, shape, spec: spec.clone(), _friction: fr });
            }
            self.world = Some(World {
                space,
                robot: body,
                control,
                static_body,
                chassis,
                walls,
                drive_joints: [pivot, gear],
                jaws: Vec::new(),
                touched: vec![false; objs.len()],
                objects: objs,
                grasp: None,
            });
        }
        let g = &self.cfg.sim.gripper;
        let a = if gripper == GripperState::Open { g.jaw_open_angle_rad } else { g.jaw_closed_angle_rad };
        self.set_jaws(a);
        self.jaw_target = self.jaw_angle;
        self.closing_on = None;
        self.stalled = false;
        self.gripper = gripper;
        self.time_s = 0.0;
        self.wheel_v = (0.0, 0.0);
        Ok(())
    }

    fn set_jaws(&mut self, angle: f64) {
        let w = self.world.as_mut().expect("Sim2D: reset() が先");
        let c = &self.cfg.sim;
        unsafe {
            for s in w.jaws.drain(..) {
                cpSpaceRemoveShape(w.space, s);
                cpShapeFree(s);
            }
            for side in SIDES {
                for poly in jaw_polygons(c, side, angle) {
                    let verts = poly.map(|(x, y)| cpv(x, y));
                    let s = cpPolyShapeNew(w.robot, 4, verts.as_ptr(), TRANSFORM_IDENTITY, 0.0);
                    cpShapeSetFriction(s, c.contact.robot_object_friction);
                    cpShapeSetElasticity(s, 0.0);
                    cpShapeSetFilter(s, shape_filter(GRASP_GROUP));
                    cpSpaceAddShape(w.space, s);
                    w.jaws.push(s);
                }
            }
            self.jaw_angle = angle;
            cpSpaceReindexShapesForBody(w.space, w.robot);
        }
    }

    // ---------- 時間発展 ----------
    /// ロボットを (linear m/s, angular rad/s) の指令で duration 秒動かす。motor_lag_s = τ > 0 なら、車輪の速さは刻みごとに
    /// v += (v_cmd − v) × dt / τ で指令に追いつく（v36、sim2d.py と同じ）。
    pub fn drive(&mut self, linear: f64, angular: f64, duration: f64) -> Result<(), SimError> {
        let dt = self.dt;
        self.world.as_mut().ok_or_else(|| SimError::State("Sim2D: reset() が先".into()))?.touched.fill(false);
        let n = n_substeps(dt, duration)?;
        let tau = self.motor_lag_s;
        for _ in 0..n {
            // 爪を 1 刻み分回す（space.step の前。sim2d.py と同じ）
            self.move_jaws();
            self.wheel_v = if tau > 0.0 {
                let (v, w) = self.wheel_v;
                (v + (linear - v) * dt / tau, w + (angular - w) * dt / tau)
            } else {
                (linear, angular)
            };
            let (v, om) = self.wheel_v;
            let w = self.world.as_mut().expect("Sim2D: reset() が先");
            unsafe {
                let a = cpBodyGetAngle(w.robot);
                cpBodySetVelocity(w.control, cpv(v * a.cos(), v * a.sin()));
                cpBodySetAngularVelocity(w.control, om);
                cpSpaceStep(w.space, dt);
                self.time_s += dt;
                let mut ctx = ContactCtx { chassis: w.chassis, touched: &mut w.touched };
                cpBodyEachArbiter(w.robot, note_chassis_contact, &mut ctx as *mut ContactCtx as *mut c_void);
                if let Some(m) = take_callback_error() {
                    return Err(SimError::State(format!("物理の刻みの途中で失敗した（時刻 {} s）: {m}", self.time_s)));
                }
            }
        }
        let w = self.world.as_mut().expect("Sim2D: reset() が先");
        unsafe {
            cpBodySetVelocity(w.control, cpv(0.0, 0.0));
            cpBodySetAngularVelocity(w.control, 0.0);
        }
        Ok(())
    }

    /// 爪を 1 刻み分（jaw_speed_radps × dt、最後は目標ちょうど）目標の角度へ回す（sim2d._move_jaws）。
    /// sim2d._jaw_depth: 今の爪の形が物体・壁にめり込んでいる深さの最大（重なりが無ければ 0）。爪が触れている物体が壁に
    /// めり込んでいれば（直前の刻みの接触）、その深さも入れる
    fn jaw_depth(&self) -> f64 {
        struct Ctx<'a> {
            d: f64,
            world: &'a World,
        }
        unsafe extern "C" fn wall_cb(_b: *mut cpBody, arb: *mut cpArbiter, data: *mut c_void) {
            let ctx = &mut *(data as *mut Ctx);
            let (mut a, mut b): (*mut cpShape, *mut cpShape) = (ptr::null_mut(), ptr::null_mut());
            cpArbiterGetShapes(arb, &mut a, &mut b);
            if ctx.world.walls.contains(&a) || ctx.world.walls.contains(&b) {
                let set = cpArbiterGetContactPointSet(arb);
                for k in 0..set.count as usize {
                    let v = -set.points[k].distance;
                    if v > ctx.d {
                        ctx.d = v;
                    }
                }
            }
        }
        unsafe extern "C" fn cb(s: *mut cpShape, set: *mut cpContactPointSet, data: *mut c_void) {
            let ctx = &mut *(data as *mut Ctx);
            let set = &*set;
            for k in 0..set.count as usize {
                let v = -set.points[k].distance;
                if v > ctx.d {
                    ctx.d = v;
                }
            }
            // 物体の形の user data は 番号 + 1（壁・ロボットは 0）
            let k = cpShapeGetUserData(s) as usize;
            if k != 0 {
                let body = ctx.world.objects[k - 1].body;
                cpBodyEachArbiter(body, wall_cb, data);
            }
        }
        let w = self.world.as_ref().expect("Sim2D: reset() が先");
        let mut ctx = Ctx { d: 0.0, world: w };
        for &s in &w.jaws {
            unsafe {
                cpSpaceShapeQuery(w.space, s, cb, &mut ctx as *mut Ctx as *mut c_void);
            }
        }
        ctx.d
    }

    /// sim2d._turn_jaws: jaw_stall が有効なら、置き直した爪が depth_m より深く、前より深くめり込むときは戻す
    fn turn_jaws(&mut self, angle: f64) {
        let (enabled, depth_m) = (self.cfg.sim.sim2d.jaw_stall.enabled, self.cfg.sim.sim2d.jaw_stall.depth_m);
        if !enabled {
            self.set_jaws(angle);
            return;
        }
        let old = self.jaw_angle;
        let before = self.jaw_depth();
        self.set_jaws(angle);
        let after = self.jaw_depth();
        if after > depth_m && after > before {
            self.set_jaws(old);
            self.stalled = true;
        }
    }

    fn move_jaws(&mut self) {
        if self.jaw_angle == self.jaw_target && self.closing_on.is_none() {
            return;
        }
        let g = &self.cfg.sim.gripper;
        let (step, closed) = (g.jaw_speed_radps * self.dt, g.jaw_closed_angle_rad);
        let d = self.jaw_target - self.jaw_angle;
        if d.abs() > step {
            self.turn_jaws(self.jaw_angle + step.copysign(d));
            return;
        }
        // 着いた。対象に向かっていたなら、まだ爪の間にあればつなぐ（attach がその時点の位置で角度を決め直す）
        let name = self.closing_on.take();
        if let Some(i) = name {
            if self.is_between(i, self.jaw_target) {
                self.attach(i);
                self.jaw_target = self.jaw_angle;
                return;
            }
        }
        self.turn_jaws(self.jaw_target);
        if name.is_some() {
            self.jaw_target = closed; // 対象が爪の間から出た。つながずに閉じ切る
        }
    }

    /// 爪を開閉する。actuation_time_s の間ロボットは止まる。その間に爪が回り切らなければ失敗。
    /// rng は挟み損ね（sim2d.grasp_fail）の乱数（env の乱数）。
    pub fn actuate_gripper(&mut self, close: bool, rng: Option<&mut NpRng>) -> Result<(), SimError> {
        self.set_gripper(close, rng)?;
        self.stalled = false;
        let t = self.cfg.sim.gripper.actuation_time_s;
        self.drive(0.0, 0.0, t)?;
        // jaw_stall でふさがれて止まった刻みがあれば例外にしない（後の drive で回り続ける）
        if !self.stalled && (self.jaw_angle != self.jaw_target || self.closing_on.is_some()) {
            return Err(SimError::State(format!(
                "爪が actuation_time_s {t} s の間に回り切らない（角度 {}、目標 {}）。gripper.jaw_speed_radps を確かめる",
                self.jaw_angle, self.jaw_target)));
        }
        Ok(())
    }

    /// 開閉の指令。目標の角度と把持の判定だけを決め、時間は進めない（角度は drive の刻みの中で回る。sim2d.set_gripper）。
    pub fn set_gripper(&mut self, close: bool, mut rng: Option<&mut NpRng>) -> Result<(), SimError> {
        let grasped = self.world.as_ref().ok_or_else(|| SimError::State("Sim2D: reset() が先".into()))?.grasp.is_some();
        let (closed, open) = (self.cfg.sim.gripper.jaw_closed_angle_rad, self.cfg.sim.gripper.jaw_open_angle_rad);
        if close {
            // つかんでいる・対象へ閉じている途中なら何もしない
            if !grasped && self.closing_on.is_none() {
                let mut target = self.object_between_jaws();
                if let Some(t) = target {
                    if self.grasp_fails(t, rng.as_deref_mut())? {
                        let rng = rng.ok_or_else(|| SimError::State("挟み損ねの乱数が無い".into()))?;
                        self.kick(t, rng);
                        target = None;
                    }
                }
                match target {
                    Some(t) => {
                        self.closing_on = Some(t);
                        self.jaw_target = self.touching_angle(t);
                    }
                    None => self.jaw_target = closed,
                }
            }
            self.gripper = GripperState::Closed;
        } else {
            self.detach();
            self.closing_on = None;
            self.jaw_target = open;
            self.gripper = GripperState::Open;
        }
        Ok(())
    }

    /// 挟み損ね（sim2d.grasp_fail、v21）。無効なら必ずつかむ。
    fn grasp_fails(&self, i: usize, rng: Option<&mut NpRng>) -> Result<bool, SimError> {
        let gf = &self.cfg.sim.sim2d.grasp_fail;
        if !gf.enabled {
            return Ok(false);
        }
        let rng = rng.ok_or_else(|| {
            SimError::State("挟み損ねが有効なのに乱数（Sim2D の rng）が無い。env.reset が渡す".into())
        })?;
        let (_, ly) = self.local(i);
        // ** 2 は Python と同じく libm の pow（x * x と最後の桁が違うことがある）
        let p = gf.base_prob + (1.0 - gf.base_prob) * py_pow(py_min(1.0, ly.abs() / gf.offset_tol_m), 2.0);
        Ok(rng.random() < p)
    }

    /// 挟み損ねた物体を、斜め前（横のずれの側）に、床の摩擦で kick_m だけ滑る速さで弾く（sim2d._kick）。
    /// C の体の速度を書き換えるので &mut self（World の Sync の前提）
    fn kick(&mut self, i: usize, rng: &mut NpRng) {
        let gf = &self.cfg.sim.sim2d.grasp_fail;
        let (_, ly) = self.local(i);
        let w = self.world.as_mut().expect("Sim2D: reset() が先");
        let o = &w.objects[i];
        let d = rng.uniform(gf.kick_m[0], gf.kick_m[1]);
        let mu = self.cfg.object(o.spec.otype).floor_friction_xy[0];
        let speed = (2.0 * mu * GRAVITY * d).sqrt();
        let side = if ly >= 0.0 { 1.0 } else { -1.0 };
        unsafe {
            let a = cpBodyGetAngle(w.robot) + side * rng.uniform(0.2, 0.8); // 前から 11〜46° 横へ
            cpBodySetVelocity(o.body, cpv(speed * a.cos(), speed * a.sin()));
            cpBodySetAngularVelocity(o.body, rng.uniform(-10.0, 10.0));
        }
    }

    /// 物体の中心のロボット座標（sim2d._local）
    fn local(&self, i: usize) -> (f64, f64) {
        let op = self.object_pose(i);
        let rp = self.robot_pose();
        let (c, s) = (rp.yaw.cos(), rp.yaw.sin());
        let (dx, dy) = (op.x - rp.x, op.y - rp.y);
        (c * dx + s * dy, -s * dx + c * dy)
    }

    /// 爪の間にある物体（横のずれが最も小さいもの。同じなら先のもの）
    fn object_between_jaws(&self) -> Option<usize> {
        let (mut best, mut best_d) = (None, f64::INFINITY);
        for i in 0..self.w().objects.len() {
            let (lx, ly) = self.local(i);
            if ly.abs() < jaw_inner_half_gap(&self.cfg.sim, self.jaw_angle, lx) && ly.abs() < best_d {
                best = Some(i);
                best_d = ly.abs();
            }
        }
        best
    }

    /// 物体の中心が、爪の角度 angle のときの爪の間の領域にあるか（object_between_jaws と同じ判定）
    fn is_between(&self, i: usize, angle: f64) -> bool {
        let (lx, ly) = self.local(i);
        ly.abs() < jaw_inner_half_gap(&self.cfg.sim, angle, lx)
    }

    /// 今の物体の位置で、梁の内側がその物体に触れる角度（sim2d._touching_angle）
    fn touching_angle(&self, i: usize) -> f64 {
        let rp = self.robot_pose();
        let otype = self.w().objects[i].spec.otype;
        let size = &self.cfg.object(otype).size_m;
        let corners = box_corners(&self.object_pose(i), (size[0], size[1]));
        let (c, s) = (rp.yaw.cos(), rp.yaw.sin());
        let ly = corners.map(|(x, y)| -s * (x - rp.x) + c * (y - rp.y));
        let (lx, _) = self.local(i);
        // Python の max / min（最初の最大・最小）
        let (mx, mn) = (ly[1..].iter().fold(ly[0], |a, &b| py_max(a, b)), ly[1..].iter().fold(ly[0], |a, &b| py_min(a, b)));
        let half = (mx - mn) / 2.0;
        angle_touching(&self.cfg.sim, lx, half)
    }

    /// 爪をその時点の物体に触れる角度で止め、物体をロボットに固定する
    fn attach(&mut self, i: usize) {
        let otype = self.w().objects[i].spec.otype;
        self.set_jaws(self.touching_angle(i));
        let hold = self.cfg.sim.gripper.grip_holding_force_n;
        let r_eff = self.cfg.friction_radius(otype);
        let w = self.world.as_mut().unwrap();
        let o = &w.objects[i];
        unsafe {
            cpShapeSetFilter(o.shape, shape_filter(GRASP_GROUP));
            let local = cpBodyWorldToLocal(w.robot, cpBodyGetPosition(o.body));
            let pin = cpPivotJointNew2(w.robot, o.body, local, cpv(0.0, 0.0));
            let gear = cpGearJointNew(w.robot, o.body, cpBodyGetAngle(o.body) - cpBodyGetAngle(w.robot), 1.0);
            // 保持力の上限（sim2d.py の説明）
            cpConstraintSetMaxForce(pin, hold);
            cpConstraintSetMaxForce(gear, hold * r_eff);
            cpSpaceAddConstraint(w.space, pin);
            cpSpaceAddConstraint(w.space, gear);
            w.grasp = Some((i, [pin, gear]));
        }
    }

    fn detach(&mut self) {
        let w = self.world.as_mut().unwrap();
        if let Some((i, cs)) = w.grasp.take() {
            unsafe {
                for c in cs {
                    cpSpaceRemoveConstraint(w.space, c);
                    cpConstraintFree(c);
                }
                cpShapeSetFilter(w.objects[i].shape, shape_filter(CP_NO_GROUP));
            }
        }
    }

    // ---------- 状態 ----------
    pub fn robot_pose(&self) -> Pose2D {
        let w = self.w();
        unsafe {
            let p = cpBodyGetPosition(w.robot);
            Pose2D::new(p.x, p.y, cpBodyGetAngle(w.robot))
        }
    }

    /// reset に渡した並びの i 番目の物体の姿勢
    pub fn object_pose(&self, i: usize) -> Pose2D {
        let b = self.w().objects[i].body;
        unsafe {
            let p = cpBodyGetPosition(b);
            Pose2D::new(p.x, p.y, cpBodyGetAngle(b))
        }
    }

    pub fn n_objects(&self) -> usize {
        self.w().objects.len()
    }

    pub fn object_spec(&self, i: usize) -> &ObjectSpec {
        &self.w().objects[i].spec
    }

    pub fn object_index(&self, name: &str) -> Option<usize> {
        self.w().objects.iter().position(|o| o.spec.name == name)
    }

    pub fn gripper_state(&self) -> GripperState {
        self.gripper
    }

    pub fn jaw_angle(&self) -> f64 {
        self.jaw_angle
    }

    pub fn grasped_object(&self) -> Option<usize> {
        self.w().grasp.map(|(i, _)| i)
    }

    /// 直前の drive（actuate_gripper を含む）の間に、車体の形（アームは除く）に触れた物体の番号（小さい順、v19）
    pub fn chassis_contacts(&self) -> Vec<usize> {
        self.w().touched.iter().enumerate().filter(|(_, &t)| t).map(|(i, _)| i).collect()
    }

    /// i 番目の物体が直前の drive の間に車体の形に触れたか
    pub fn chassis_touched(&self, i: usize) -> bool {
        self.w().touched[i]
    }
}

/// duration を刻む回数（Python の round は偶数への丸め）。整数倍でなければ失敗
fn n_substeps(dt: f64, duration: f64) -> Result<usize, SimError> {
    let n = (duration / dt).round_ties_even();
    if !n.is_finite() || n <= 0.0 || (n * dt - duration).abs() > 1e-6 {
        return Err(SimError::Invalid(format!("duration {duration} s は physics_dt_s {dt} の整数倍でない")));
    }
    Ok(n as usize)
}

/// Python の float の ** と同じ（CPython の float_pow は libm の pow を呼ぶ）。指数が定数だと LLVM が x * x に
/// 置き換えるので、black_box で隠す
fn py_pow(x: f64, y: f64) -> f64 {
    extern "C" {
        fn pow(x: f64, y: f64) -> f64;
    }
    unsafe { pow(x, std::hint::black_box(y)) }
}

// ---------- Python から（試験用） ----------

/// 試験用の Sim2D。挟み損ねの乱数は set_rng(state, inc) で入れ（numpy の PCG64 の bit_generator.state と同じ値）、
/// rng_random() で 1 つ引いて Python 側の乱数と比べる（両方から同じ数だけ引くので、以後も同じ流れのまま）。
#[pyclass(name = "Sim2D")]
struct PySim2D {
    sim: Sim2D,
    rng: Option<NpRng>,
}

#[pymethods]
impl PySim2D {
    #[new]
    fn py_new(cfg_json: &str) -> PyResult<Self> {
        let cfg = Cfg::from_json(cfg_json).map_err(PyValueError::new_err)?;
        Ok(PySim2D { sim: Sim2D::new(&cfg), rng: None })
    }

    /// objects: [(name, 種類の番号, x, y, yaw)]、gripper: GripperState の番号
    fn reset(&mut self, robot: (f64, f64, f64), objects: Vec<(String, i64, f64, f64, f64)>, gripper: i64) -> PyResult<()> {
        let g = GripperState::from_index(gripper)
            .ok_or_else(|| PyValueError::new_err(format!("初期グリッパー状態は OPEN か CLOSED: {gripper}")))?;
        let mut specs = Vec::with_capacity(objects.len());
        for (name, t, x, y, yaw) in objects {
            if t != 0 && t != 1 {
                return Err(PyValueError::new_err(format!("未知の物体の種類: {t}")));
            }
            specs.push(ObjectSpec { name, otype: ObjectType::from_index(t), pose: Pose2D::new(x, y, yaw), color: String::new() });
        }
        Ok(self.sim.reset(Pose2D::new(robot.0, robot.1, robot.2), &specs, g)?)
    }

    fn drive(&mut self, linear: f64, angular: f64, duration: f64) -> PyResult<()> {
        Ok(self.sim.drive(linear, angular, duration)?)
    }

    fn actuate_gripper(&mut self, close: bool) -> PyResult<()> {
        Ok(self.sim.actuate_gripper(close, self.rng.as_mut())?)
    }

    fn set_gripper(&mut self, close: bool) -> PyResult<()> {
        Ok(self.sim.set_gripper(close, self.rng.as_mut())?)
    }

    /// v36: 旋回の惰性の時定数（Python の Sim2D.motor_lag_s と同じ）
    fn set_motor_lag(&mut self, tau: f64) {
        self.sim.motor_lag_s = tau;
    }

    fn set_rng(&mut self, state: u128, inc: u128) {
        self.rng = Some(NpRng::from_state(state, inc));
    }

    fn rng_random(&mut self) -> PyResult<f64> {
        let r = self.rng.as_mut().ok_or_else(|| PyRuntimeError::new_err("乱数が無い（set_rng が先）"))?;
        Ok(r.random())
    }

    fn robot_pose(&self) -> PyResult<(f64, f64, f64)> {
        self.ready()?;
        let p = self.sim.robot_pose();
        Ok((p.x, p.y, p.yaw))
    }

    fn object_pose(&self, name: &str) -> PyResult<(f64, f64, f64)> {
        self.ready()?;
        let i = self.sim.object_index(name).ok_or_else(|| PyKeyError::new_err(name.to_string()))?;
        let p = self.sim.object_pose(i);
        Ok((p.x, p.y, p.yaw))
    }

    fn object_names(&self) -> PyResult<Vec<String>> {
        self.ready()?;
        Ok((0..self.sim.n_objects()).map(|i| self.sim.object_spec(i).name.clone()).collect())
    }

    fn gripper_state(&self) -> i64 {
        self.sim.gripper_state() as i64
    }

    fn jaw_angle(&self) -> f64 {
        self.sim.jaw_angle()
    }

    fn grasped_object(&self) -> PyResult<Option<String>> {
        self.ready()?;
        Ok(self.sim.grasped_object().map(|i| self.sim.object_spec(i).name.clone()))
    }

    fn chassis_contacts(&self) -> PyResult<Vec<String>> {
        self.ready()?;
        Ok(self.sim.chassis_contacts().into_iter().map(|i| self.sim.object_spec(i).name.clone()).collect())
    }

    #[getter]
    fn time_s(&self) -> f64 {
        self.sim.time_s
    }

    /// 試験用: 物体の形の user data（物体の番号 + 1）を k にする（C から呼ばれる接触の関数の失敗を起こす）
    fn _set_shape_user_data_for_test(&mut self, name: &str, k: usize) -> PyResult<()> {
        let o = self.obj(name)?;
        unsafe { cpShapeSetUserData(o.shape, k as *mut c_void) };
        Ok(())
    }

    /// 試験用: 物体の体の床の摩擦の値（user data）を外す（C から呼ばれる床の摩擦の関数の失敗を起こす）
    fn _clear_body_user_data_for_test(&mut self, name: &str) -> PyResult<()> {
        let o = self.obj(name)?;
        unsafe { cpBodySetUserData(o.body, ptr::null_mut()) };
        Ok(())
    }
}

impl PySim2D {
    fn ready(&self) -> PyResult<()> {
        if self.sim.world.is_none() {
            return Err(PyRuntimeError::new_err("Sim2D: reset() が先"));
        }
        Ok(())
    }

    fn obj(&mut self, name: &str) -> PyResult<&mut Obj> {
        self.ready()?;
        let i = self.sim.object_index(name).ok_or_else(|| PyKeyError::new_err(name.to_string()))?;
        Ok(&mut self.sim.world.as_mut().unwrap().objects[i])
    }
}

/// 試験用: Chipmunk の警告（cpAssertWarn と同じ cpMessage の呼び方）を出す。表示して戻る（止まらない）
#[pyfunction]
fn _cp_warning_for_test(text: &str) -> PyResult<()> {
    let text = std::ffi::CString::new(text).map_err(|e| PyValueError::new_err(e.to_string()))?;
    unsafe {
        cpMessage(c"test_condition".as_ptr(), c"sim2d.rs".as_ptr(), 0, 0, 0, c"%s".as_ptr(), text.as_ptr());
    }
    Ok(())
}

/// 試験用: cpMessage の後に書く env の状況の文
#[pyfunction]
fn _cp_context_text() -> String {
    crate::cp::context_text()
}

pub fn py_register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySim2D>()?;
    m.add_function(wrap_pyfunction!(_cp_warning_for_test, m)?)?;
    m.add_function(wrap_pyfunction!(_cp_context_text, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toward_zero_matches_python() {
        assert_eq!(toward_zero(0.5, 0.1), 0.4);
        assert_eq!(toward_zero(-0.05, 0.1), 0.0);
        assert_eq!(toward_zero(-0.5, 0.1), -0.4);
    }

    #[test]
    fn callback_panic_and_error_are_recorded() {
        // C から呼ばれる関数の中の panic は外へ出さず、最初の失敗だけ記録する
        in_callback("試験", || panic!("壊れた"));
        in_callback("試験 2", || Err("2 つ目".into()));
        assert_eq!(take_callback_error().as_deref(), Some("試験: panic: 壊れた"));
        assert_eq!(take_callback_error(), None);
        in_callback("試験", || Ok(()));
        assert_eq!(take_callback_error(), None);
    }
}
