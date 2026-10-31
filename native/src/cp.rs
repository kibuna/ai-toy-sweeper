//! 同梱の Chipmunk2D（chipmunk/、pymunk 7.3.1 の Munk2D）の C の関数の宣言（sim2d が使う分だけ）。
//! 型は chipmunk_types.h と同じ（CP_USE_DOUBLES: cpFloat = double、cpGroup = uintptr_t、cpBitmask = unsigned int）。
//!
//! Chipmunk の assert の報告（cpMessage）は src/cp_message.c が定義する（build.rs が CP_OVERRIDE_MESSAGE を付けて
//! chipmunk.c のものを外す。pymunk と同じ）。Chipmunk の文の後に、いま動いている env の状況を stderr に書く。
//! 状況は env.rs（enter_context）と vecenv.rs（enter_env_index）がスレッドごとに入れる。
#![allow(non_camel_case_types, non_snake_case)]

use std::cell::{Cell, RefCell};
use std::io::Write;
use std::os::raw::{c_char, c_int, c_uint, c_void};

pub type cpFloat = f64;
pub type cpGroup = usize;
pub type cpBitmask = c_uint;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct cpVect {
    pub x: cpFloat,
    pub y: cpFloat,
}

pub const fn cpv(x: f64, y: f64) -> cpVect {
    cpVect { x, y }
}

/// chipmunk_types.h: a, b, c, d, tx, ty の順
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cpTransform {
    pub a: cpFloat,
    pub b: cpFloat,
    pub c: cpFloat,
    pub d: cpFloat,
    pub tx: cpFloat,
    pub ty: cpFloat,
}

/// pymunk の Transform.identity()
pub const TRANSFORM_IDENTITY: cpTransform = cpTransform { a: 1.0, b: 0.0, c: 0.0, d: 1.0, tx: 0.0, ty: 0.0 };

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cpShapeFilter {
    pub group: cpGroup,
    pub categories: cpBitmask,
    pub mask: cpBitmask,
}

/// CP_NO_GROUP = (cpGroup)0、CP_ALL_CATEGORIES = ~(cpBitmask)0（pymunk の ShapeFilter の既定 0xFFFFFFFF と同じ）
pub const CP_NO_GROUP: cpGroup = 0;
pub const CP_ALL_CATEGORIES: cpBitmask = !0;

/// pymunk の ShapeFilter(group=g)
pub const fn shape_filter(group: cpGroup) -> cpShapeFilter {
    cpShapeFilter { group, categories: CP_ALL_CATEGORIES, mask: CP_ALL_CATEGORIES }
}

#[repr(C)]
pub struct cpSpace {
    _p: [u8; 0],
}
#[repr(C)]
pub struct cpBody {
    _p: [u8; 0],
}
#[repr(C)]
pub struct cpShape {
    _p: [u8; 0],
}
#[repr(C)]
pub struct cpConstraint {
    _p: [u8; 0],
}
#[repr(C)]
pub struct cpArbiter {
    _p: [u8; 0],
}

/// cpContactPointSet の 1 点（chipmunk/cpArbiter.h）
#[repr(C)]
#[derive(Clone, Copy)]
pub struct cpContactPoint {
    pub point_a: cpVect,
    pub point_b: cpVect,
    pub distance: cpFloat,
}
/// cpContactPointSet（CP_MAX_CONTACTS_PER_ARBITER = 2）
#[repr(C)]
pub struct cpContactPointSet {
    pub count: c_int,
    pub normal: cpVect,
    pub points: [cpContactPoint; 2],
}
pub type cpSpaceShapeQueryFunc = unsafe extern "C" fn(shape: *mut cpShape, points: *mut cpContactPointSet, data: *mut c_void);

pub type cpBodyVelocityFunc = unsafe extern "C" fn(body: *mut cpBody, gravity: cpVect, damping: cpFloat, dt: cpFloat);
pub type cpBodyArbiterIteratorFunc = unsafe extern "C" fn(body: *mut cpBody, arbiter: *mut cpArbiter, data: *mut c_void);

extern "C" {
    pub fn cpMomentForBox(m: cpFloat, width: cpFloat, height: cpFloat) -> cpFloat;

    pub fn cpSpaceNew() -> *mut cpSpace;
    pub fn cpSpaceFree(space: *mut cpSpace);
    pub fn cpSpaceSetIterations(space: *mut cpSpace, iterations: c_int);
    pub fn cpSpaceSetGravity(space: *mut cpSpace, gravity: cpVect);
    pub fn cpSpaceSetCollisionSlop(space: *mut cpSpace, slop: cpFloat);
    pub fn cpSpaceAddShape(space: *mut cpSpace, shape: *mut cpShape) -> *mut cpShape;
    pub fn cpSpaceAddBody(space: *mut cpSpace, body: *mut cpBody) -> *mut cpBody;
    pub fn cpSpaceAddConstraint(space: *mut cpSpace, c: *mut cpConstraint) -> *mut cpConstraint;
    pub fn cpSpaceRemoveShape(space: *mut cpSpace, shape: *mut cpShape);
    pub fn cpSpaceRemoveConstraint(space: *mut cpSpace, c: *mut cpConstraint);
    pub fn cpSpaceReindexShapesForBody(space: *mut cpSpace, body: *mut cpBody);
    pub fn cpSpaceStep(space: *mut cpSpace, dt: cpFloat);
    pub fn cpSpaceShapeQuery(space: *mut cpSpace, shape: *mut cpShape, func: cpSpaceShapeQueryFunc, data: *mut c_void) -> u8;

    pub fn cpBodyNew(mass: cpFloat, moment: cpFloat) -> *mut cpBody;
    pub fn cpBodyNewKinematic() -> *mut cpBody;
    pub fn cpBodyNewStatic() -> *mut cpBody;
    pub fn cpBodyFree(body: *mut cpBody);
    pub fn cpBodyGetPosition(body: *const cpBody) -> cpVect;
    pub fn cpBodySetPosition(body: *mut cpBody, pos: cpVect);
    pub fn cpBodySetCenterOfGravity(body: *mut cpBody, cog: cpVect);
    pub fn cpBodyGetVelocity(body: *const cpBody) -> cpVect;
    pub fn cpBodySetVelocity(body: *mut cpBody, v: cpVect);
    pub fn cpBodyGetAngle(body: *const cpBody) -> cpFloat;
    pub fn cpBodySetAngle(body: *mut cpBody, a: cpFloat);
    pub fn cpBodyGetAngularVelocity(body: *const cpBody) -> cpFloat;
    pub fn cpBodySetAngularVelocity(body: *mut cpBody, w: cpFloat);
    pub fn cpBodyGetUserData(body: *const cpBody) -> *mut c_void;
    pub fn cpBodySetUserData(body: *mut cpBody, data: *mut c_void);
    pub fn cpBodySetVelocityUpdateFunc(body: *mut cpBody, f: cpBodyVelocityFunc);
    pub fn cpBodyWorldToLocal(body: *const cpBody, point: cpVect) -> cpVect;
    pub fn cpBodyEachArbiter(body: *mut cpBody, f: cpBodyArbiterIteratorFunc, data: *mut c_void);

    pub fn cpPolyShapeNew(body: *mut cpBody, count: c_int, verts: *const cpVect, t: cpTransform, radius: cpFloat)
        -> *mut cpShape;
    pub fn cpBoxShapeNew(body: *mut cpBody, width: cpFloat, height: cpFloat, radius: cpFloat) -> *mut cpShape;
    pub fn cpSegmentShapeNew(body: *mut cpBody, a: cpVect, b: cpVect, radius: cpFloat) -> *mut cpShape;
    pub fn cpShapeFree(shape: *mut cpShape);
    pub fn cpShapeSetFriction(shape: *mut cpShape, u: cpFloat);
    pub fn cpShapeSetElasticity(shape: *mut cpShape, e: cpFloat);
    pub fn cpShapeSetFilter(shape: *mut cpShape, f: cpShapeFilter);
    pub fn cpShapeGetUserData(shape: *const cpShape) -> *mut c_void;
    pub fn cpShapeSetUserData(shape: *mut cpShape, data: *mut c_void);

    pub fn cpPivotJointNew2(a: *mut cpBody, b: *mut cpBody, anchor_a: cpVect, anchor_b: cpVect) -> *mut cpConstraint;
    pub fn cpGearJointNew(a: *mut cpBody, b: *mut cpBody, phase: cpFloat, ratio: cpFloat) -> *mut cpConstraint;
    pub fn cpConstraintFree(c: *mut cpConstraint);
    pub fn cpConstraintSetMaxForce(c: *mut cpConstraint, f: cpFloat);
    pub fn cpConstraintSetMaxBias(c: *mut cpConstraint, b: cpFloat);

    pub fn cpArbiterGetShapes(arb: *const cpArbiter, a: *mut *mut cpShape, b: *mut *mut cpShape);
    pub fn cpArbiterGetContactPointSet(arb: *const cpArbiter) -> cpContactPointSet;

    /// src/cp_message.c（Chipmunk の assert の報告。表示だけして戻る）
    pub fn cpMessage(condition: *const c_char, file: *const c_char, line: c_int, is_error: c_int, is_hard_error: c_int,
                     message: *const c_char, ...);
}

// ---------- Chipmunk の assert の報告に付ける env の状況 ----------

/// いま動いている env の状況。env.rs が reset / step の間だけ入れる（入れ子にしない）
#[derive(Debug, Default)]
pub struct CpContext {
    active: bool,
    /// "reset" か "step"
    pub phase: &'static str,
    pub stage: i64,
    /// この step の前までの歩数
    pub steps: i64,
    pub action: Option<i64>,
    /// step の始め（reset では盤面の初期位置）のロボットの姿勢 (x, y, yaw)
    pub robot: (f64, f64, f64),
    /// 学習の設定 (Stage, 復習の Stage, 復習の確率)
    pub curriculum: (i64, Vec<i64>, f64),
    /// このエピソードの reset の始めの乱数の状態 (state, inc, 32 ビットの残り)。盤面を外から渡したときは None
    pub rng_at_reset: Option<(u128, u128, Option<u32>)>,
    /// このエピソードのこれまでの行動
    pub actions: Vec<u8>,
}

thread_local! {
    static CONTEXT: RefCell<CpContext> = RefCell::new(CpContext::default());
    static ENV_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

/// 持っている間だけ状況が有効（drop で外す）
pub struct ContextGuard(());

impl Drop for ContextGuard {
    fn drop(&mut self) {
        // panic 中の drop でも止まらないよう panic しない形で外す（外せないのは表示の中身だけの問題）
        let _ = CONTEXT.try_with(|c| {
            if let Ok(mut c) = c.try_borrow_mut() {
                c.active = false;
            }
        });
    }
}

/// 状況を入れる（f が中身を書く。Vec は使い回すので、毎歩の確保はない）
pub fn enter_context(f: impl FnOnce(&mut CpContext)) -> ContextGuard {
    CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        f(&mut c);
        c.active = true;
    });
    ContextGuard(())
}

/// VecEnv の env の番号（持っている間だけ。drop で前の値に戻す）
pub struct EnvIndexGuard(Option<usize>);

impl Drop for EnvIndexGuard {
    fn drop(&mut self) {
        ENV_INDEX.with(|c| c.set(self.0));
    }
}

pub fn enter_env_index(i: usize) -> EnvIndexGuard {
    EnvIndexGuard(ENV_INDEX.with(|c| c.replace(Some(i))))
}

/// cpMessage の後に書く文（1 行目は "\tnative の状況: "）
pub fn context_text() -> String {
    let who = match ENV_INDEX.try_with(|c| c.get()).ok().flatten() {
        Some(i) => format!("VecEnv の env {i}"),
        None => "env".to_string(),
    };
    let body = CONTEXT.try_with(|c| {
        // 書き込みの途中（同じスレッドで enter_context の中）なら読めない
        let Ok(c) = c.try_borrow() else {
            return Some("状況を書き込みの途中で読めない".to_string());
        };
        if !c.active {
            return None;
        }
        let action = c.action.map_or("なし".to_string(), |a| a.to_string());
        let rng = c.rng_at_reset.map_or("なし（盤面を外から渡した）".to_string(), |r| format!("{r:?}"));
        Some(format!(
            "{} の途中（Stage {}、この step の前までに {} 歩、行動 {action}）\n\
             \tロボット (x, y, yaw)（step の始め。reset では初期位置）: {:?}\n\
             \t学習の設定 (Stage, 復習の Stage, 復習の確率): {:?}\n\
             \treset の始めの乱数の状態 (state, inc, 32 ビットの残り): {rng}\n\
             \tこのエピソードの行動: {:?}",
            c.phase, c.stage, c.steps, c.robot, c.curriculum, c.actions))
    });
    match body {
        Ok(Some(b)) => format!("\tnative の状況: {who}、{b}\n"),
        Ok(None) => format!("\tnative の状況: {who}、reset / step の外\n"),
        Err(_) => "\tnative の状況: 読めない（スレッドの終わり）\n".to_string(),
    }
}

/// src/cp_message.c の cpMessage が呼ぶ。いま動いている env の状況を stderr に書く。
/// C から呼ばれるので panic を外へ出さない。書き込みの失敗は無視する（この後 Chipmunk は abort するので、打てる手がない）
#[no_mangle]
pub extern "C" fn toy_sweeper_cp_print_context() {
    let text = std::panic::catch_unwind(context_text)
        .unwrap_or_else(|_| "\tnative の状況: 表示の途中で panic した\n".to_string());
    let _ = std::io::stderr().write_all(text.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layouts() {
        assert_eq!(std::mem::size_of::<cpVect>(), 16);
        assert_eq!(std::mem::size_of::<cpTransform>(), 48);
        assert_eq!(std::mem::size_of::<cpShapeFilter>(), 16);
        assert_eq!(CP_ALL_CATEGORIES, 0xFFFF_FFFF);
    }

    #[test]
    fn moment_for_box() {
        // cpMomentForBox = m (w^2 + h^2) / 12
        let m = unsafe { cpMomentForBox(12.0, 1.0, 2.0) };
        assert_eq!(m, 12.0 * (1.0 + 4.0) / 12.0);
    }
}
