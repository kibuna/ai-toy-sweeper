//! ロボットの形状（toy_sweeper.robot_model と同じ計算）。ロボット座標系: 原点 = 車軸中点、x 前方、y 左。
use crate::config::SimCfg;
use crate::geometry::hypot;

pub const SIDES: [f64; 2] = [1.0, -1.0]; // 左爪, 右爪

pub fn body_center_x(sim: &SimCfg) -> f64 {
    -sim.robot.axle_offset_x_m
}

pub fn front_x(sim: &SimCfg) -> f64 {
    body_center_x(sim) + sim.robot.body_size_m[0] / 2.0
}

pub fn pivot_xy(sim: &SimCfg, side: f64) -> (f64, f64) {
    let g = &sim.gripper;
    (front_x(sim) + g.pivot_x_from_front_m, side * g.pivot_half_spacing_m)
}

fn jaw_point(sim: &SimCfg, side: f64, angle: f64, d: f64, e: f64) -> (f64, f64) {
    let (px, py) = pivot_xy(sim, side);
    let a = side * angle;
    let ly = side * e;
    (px + d * a.cos() - ly * a.sin(), py + d * a.sin() + ly * a.cos())
}

/// 爪 1 本の外形（腕と梁の 2 つの凸多角形）。
pub fn jaw_polygons(sim: &SimCfg, side: f64, angle: f64) -> [[(f64, f64); 4]; 2] {
    let g = &sim.gripper;
    let (hw, e, d0, l) = (g.jaw_width_m / 2.0, g.jaw_lateral_offset_m, g.jaw_beam_start_m, g.jaw_length_m);
    let beam = [(d0, e - hw), (l, e - hw), (l, e + hw), (d0, e + hw)];
    let arm = [(0.0, -hw), (d0, e - hw), (d0, e + hw), (0.0, hw)];
    let f = |poly: [(f64, f64); 4]| poly.map(|(d, y)| jaw_point(sim, side, angle, d, y));
    [f(arm), f(beam)]
}

pub fn gripper_center_x(sim: &SimCfg) -> f64 {
    let g = &sim.gripper;
    front_x(sim) + g.pivot_x_from_front_m + (g.jaw_beam_start_m + g.jaw_length_m) / 2.0
}

/// 前方位置 x で、左右の梁の内側の縁の間隔の半分（梁の範囲外なら -inf）。
pub fn jaw_inner_half_gap(sim: &SimCfg, angle: f64, x: f64) -> f64 {
    let g = &sim.gripper;
    let ei = g.jaw_lateral_offset_m - g.jaw_width_m / 2.0;
    let (x0, y0) = jaw_point(sim, 1.0, angle, g.jaw_beam_start_m, ei);
    let t = (x - x0) / angle.cos();
    if t < 0.0 || t > g.jaw_length_m - g.jaw_beam_start_m {
        return f64::NEG_INFINITY;
    }
    y0 + t * angle.sin()
}

/// 前方位置 x にある幅 2*half_extent の物体に、梁の内側の縁がちょうど触れる角度（閉〜開の範囲に収める）。
pub fn angle_touching(sim: &SimCfg, x: f64, half_extent: f64) -> f64 {
    let g = &sim.gripper;
    let (mut lo, mut hi) = (g.jaw_closed_angle_rad, g.jaw_open_angle_rad);
    if jaw_inner_half_gap(sim, lo, x) >= half_extent {
        return lo;
    }
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if jaw_inner_half_gap(sim, mid, x) < half_extent {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// 車軸中点から、車体と開いた爪の最も遠い点までの距離（配置の重なり判定用）。
pub fn robot_radius(sim: &SimCfg) -> f64 {
    let (bx, by) = (sim.robot.body_size_m[0], sim.robot.body_size_m[1]);
    let cx = body_center_x(sim);
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for sx in [1.0, -1.0] {
        for sy in [1.0, -1.0] {
            pts.push((cx + sx * bx / 2.0, sy * by / 2.0));
        }
    }
    for side in SIDES {
        for poly in jaw_polygons(sim, side, sim.gripper.jaw_open_angle_rad) {
            pts.extend_from_slice(&poly);
        }
    }
    // Python の max（最初の最大）
    let mut best = f64::NEG_INFINITY;
    for (x, y) in pts {
        let r = hypot(x, y);
        if r > best {
            best = r;
        }
    }
    best
}
