//! 平面幾何（toy_sweeper.geometry と同じ計算。浮動小数点の演算の順序まで Python と同じにする）。

pub const PI: f64 = std::f64::consts::PI;

/// Python の float の `%`（CPython の float_rem）: 結果の符号は割る数と同じ。
pub fn py_mod(x: f64, w: f64) -> f64 {
    let m = x % w;
    if m != 0.0 {
        if (w < 0.0) != (m < 0.0) {
            m + w
        } else {
            m
        }
    } else {
        0.0f64.copysign(w)
    }
}

/// [-pi, pi) に折り返す（geometry.wrap_angle）。
pub fn wrap_angle(a: f64) -> f64 {
    py_mod(a + PI, 2.0 * PI) - PI
}

extern "C" {
    fn remainder(x: f64, y: f64) -> f64;
}

/// math.remainder（IEEE の剰余。正確な値は 1 つに決まるので libm と CPython で同じ）。
pub fn ieee_remainder(x: f64, y: f64) -> f64 {
    unsafe { remainder(x, y) }
}

#[inline]
fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

#[inline]
fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let z = x - a;
    (x, b - z)
}

/// math.hypot（CPython 3.12 の vector_norm と同じ計算。libm の hypot とは最後の桁が違うことがある）。
pub fn hypot(a: f64, b: f64) -> f64 {
    let v = [a.abs(), b.abs()];
    if v[0].is_nan() || v[1].is_nan() {
        if v[0].is_infinite() || v[1].is_infinite() {
            return f64::INFINITY;
        }
        return f64::NAN;
    }
    let max = if v[1] > v[0] { v[1] } else { v[0] };
    vector_norm(&v, max)
}

fn vector_norm(vec: &[f64; 2], max: f64) -> f64 {
    if max.is_infinite() {
        return max;
    }
    if max == 0.0 {
        return max;
    }
    let (_, max_e) = frexp(max);
    if max_e < -1023 {
        let v2 = [vec[0] / f64::MIN_POSITIVE, vec[1] / f64::MIN_POSITIVE];
        return f64::MIN_POSITIVE * vector_norm(&v2, max / f64::MIN_POSITIVE);
    }
    let scale = ldexp(1.0, -max_e);
    let (mut csum, mut frac1, mut frac2) = (1.0f64, 0.0f64, 0.0f64);
    for &x0 in vec.iter() {
        let x = x0 * scale;
        let pr = dl_mul(x, x);
        let sm = dl_fast_sum(csum, pr.0);
        csum = sm.0;
        frac1 += pr.1;
        frac2 += sm.1;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let pr = dl_mul(-h, h);
    let sm = dl_fast_sum(csum, pr.0);
    csum = sm.0;
    frac1 += pr.1;
    frac2 += sm.1;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

extern "C" {
    #[link_name = "frexp"]
    fn c_frexp(x: f64, e: *mut i32) -> f64;
    #[link_name = "ldexp"]
    fn c_ldexp(x: f64, e: i32) -> f64;
}

fn frexp(x: f64) -> (f64, i32) {
    let mut e = 0i32;
    let m = unsafe { c_frexp(x, &mut e) };
    (m, e)
}

fn ldexp(x: f64, e: i32) -> f64 {
    unsafe { c_ldexp(x, e) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose2D {
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

impl Pose2D {
    pub fn new(x: f64, y: f64, yaw: f64) -> Pose2D {
        Pose2D { x, y, yaw }
    }
    pub fn xy(&self) -> (f64, f64) {
        (self.x, self.y)
    }
    /// ローカル座標の点を世界座標へ（Pose2D.transform_point）。
    pub fn transform_point(&self, local: (f64, f64)) -> (f64, f64) {
        let (c, s) = (self.yaw.cos(), self.yaw.sin());
        (self.x + c * local.0 - s * local.1, self.y + s * local.0 + c * local.1)
    }
}

/// 軸平行の長方形（geometry.Rect）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub center: (f64, f64),
    pub size: (f64, f64),
}

impl Rect {
    pub fn contains(&self, p: (f64, f64)) -> bool {
        (p.0 - self.center.0).abs() <= self.size.0 / 2.0 && (p.1 - self.center.1).abs() <= self.size.1 / 2.0
    }
    pub fn contains_all(&self, pts: &[(f64, f64)]) -> bool {
        pts.iter().all(|&p| self.contains(p))
    }
}

pub fn box_corners(pose: &Pose2D, size_xy: (f64, f64)) -> [(f64, f64); 4] {
    let (hx, hy) = (size_xy.0 / 2.0, size_xy.1 / 2.0);
    [
        pose.transform_point((hx, hy)),
        pose.transform_point((-hx, hy)),
        pose.transform_point((-hx, -hy)),
        pose.transform_point((hx, -hy)),
    ]
}

pub fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    hypot(a.0 - b.0, a.1 - b.1)
}

/// 点から長方形の縁までの符号付き距離（外は正、中は負）。
pub fn signed_distance_to_rect(p: (f64, f64), r: &Rect) -> f64 {
    let dx = (p.0 - r.center.0).abs() - r.size.0 / 2.0;
    let dy = (p.1 - r.center.1).abs() - r.size.1 / 2.0;
    if dx > 0.0 || dy > 0.0 {
        return hypot(py_max(dx, 0.0), py_max(dy, 0.0));
    }
    py_max(dx, dy)
}

/// Python の max(a, b)（等しければ前の方。NaN は考えない）
#[inline]
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python の min(a, b)
#[inline]
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_angle_matches_python_semantics() {
        assert_eq!(wrap_angle(PI), -PI);
        assert!((wrap_angle(3.0 * PI / 2.0) + PI / 2.0).abs() < 1e-15);
        assert_eq!(py_mod(-1.0, 3.0), 2.0);
        assert_eq!(py_mod(1.0, -3.0), -2.0);
    }

    #[test]
    fn hypot_basic() {
        assert_eq!(hypot(3.0, 4.0), 5.0);
        assert_eq!(hypot(0.0, 0.0), 0.0);
        assert_eq!(hypot(-2.0, 0.0), 2.0);
    }
}
