//! Python から使う型の登録。各モジュールの Python 向けの型（#[pyclass]）は、そのモジュールの py_register で登録する。
use pyo3::prelude::*;

/// 試験用: geometry.hypot（CPython の math.hypot と同じになることを確かめる）
#[pyfunction]
fn _hypot(a: f64, b: f64) -> f64 {
    crate::geometry::hypot(a, b)
}

/// 試験用: geometry.wrap_angle
#[pyfunction]
fn _wrap_angle(a: f64) -> f64 {
    crate::geometry::wrap_angle(a)
}

/// 試験用: geometry.ieee_remainder（math.remainder）
#[pyfunction]
fn _remainder(a: f64, b: f64) -> f64 {
    crate::geometry::ieee_remainder(a, b)
}

/// 試験用: robot_model の値（設定の JSON から）。Python の robot_model と同じになることを確かめる
#[pyfunction]
fn _robot_model(cfg_json: &str, angle: f64, x: f64, half_extent: f64) -> PyResult<Vec<f64>> {
    let c = crate::config::Cfg::from_json(cfg_json).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let s = &c.sim;
    use crate::robot_model::*;
    let mut v = vec![body_center_x(s), front_x(s), gripper_center_x(s), robot_radius(s),
                     jaw_inner_half_gap(s, angle, x), angle_touching(s, x, half_extent)];
    for side in SIDES {
        for poly in jaw_polygons(s, side, angle) {
            for (px, py) in poly {
                v.push(px);
                v.push(py);
            }
        }
    }
    Ok(v)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_hypot, m)?)?;
    m.add_function(wrap_pyfunction!(_wrap_angle, m)?)?;
    m.add_function(wrap_pyfunction!(_remainder, m)?)?;
    m.add_function(wrap_pyfunction!(_robot_model, m)?)?;
    crate::rng::py_register(m)?;
    crate::sim2d::py_register(m)?;
    crate::sampler::py_register(m)?;
    crate::env::py_register(m)?;
    crate::vecenv::py_register(m)?;
    Ok(())
}
