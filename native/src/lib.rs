//! おもちゃ片付けの学習環境の Rust 版。Python 版（toy_sweeper.env.ToySweeperEnv）と同じ盤面・物理・
//! 観測・報酬を、同じ乱数列で再現する。違いは test/test_native_*.py が Python 版と突き合わせて見つける。
//!
//! モジュール:
//! - rng: numpy の Generator(PCG64) と同じ乱数（random / uniform / integers / normal）
//! - cp: 同梱の Chipmunk2D の C の関数の宣言（FFI）
//! - geometry, robot_model: toy_sweeper.geometry / robot_model と同じ平面幾何
//! - config: Python の設定（AllCfg を JSON にしたもの）
//! - sim2d: toy_sweeper.sim2d.Sim2D
//! - sampler: toy_sweeper.task.TaskSampler
//! - env: toy_sweeper.env.ToySweeperEnv（観測は spaces.ObservationBuilder、snapshot.build_snapshot）
//! - vecenv: 多数の env を rayon で並列に進める（SB3 の VecEnv の中身）
//! - py: Python から使う型（pyo3）

pub mod config;
pub mod cp;
pub mod env;
pub mod geometry;
pub mod py;
pub mod rng;
pub mod robot_model;
pub mod sampler;
pub mod sim2d;
pub mod vecenv;
pub mod ziggurat_tables;

use pyo3::prelude::*;

#[pymodule]
fn toy_sweeper_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py::register(m)
}
