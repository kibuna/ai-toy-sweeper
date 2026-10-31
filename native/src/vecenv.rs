//! 多数の env を rayon で並列に進める（SB3 の VecEnv の中身。Python 側は toy_sweeper.native.NativeVecEnv）。
//!
//! SB3 の DummyVecEnv / SubprocVecEnv と Monitor と同じ動き:
//! - 終わった env はその場で reset する（Seed なし、env の乱数のまま）。返す観測は新しいエピソードの最初の観測、
//!   終わったときの観測は terminal_obs に分けて返す。
//! - Monitor と同じく、エピソードの報酬の和（Python の sum() と同じ総和）と長さを数える。
//! - どれかの env が失敗したら、その env の番号と状況を付けて例外にする（飛ばして続けない）。
//! - step / reset の途中の失敗は、ほかの env がもう進んでいて半端な状態になる。なので失敗した VecEnv は止まった
//!   ことにし（poisoned）、以後の step / reset / set_curriculum は最初の失敗を付けてすぐ例外にする。行動の値など、
//!   どの env も進める前に分かる失敗は、何も変えずに例外にする（止めない）。
use crate::config::Cfg;
use crate::env::{guard, rng_from, Env, EnvError, PySum, RngState, N_ACTIONS, N_TERMS, REWARD_TERM_NAMES};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

struct Slot {
    env: Env,
    /// Monitor の rewards の和と長さ
    ret: PySum,
    len: i64,
}

pub struct VecEnv {
    slots: Vec<Slot>,
    pool: rayon::ThreadPool,
    dim: usize,
    /// 半端な状態にした最初の失敗（あれば以後は進めない）
    poisoned: Option<EnvError>,
}

/// step の書き先（env i は i 行目だけに書く）
#[derive(Clone, Copy)]
struct Out {
    obs: *mut f32,
    terminal_obs: *mut f32,
    reward: *mut f64,
    terminated: *mut bool,
    truncated: *mut bool,
    is_success: *mut bool,
    failure: *mut i8,
    stage: *mut i64,
    new_stage: *mut i64,
    steps: *mut i64,
    n_delivered: *mut i64,
    n_targets: *mut i64,
    n_foreign_in_goal: *mut i64,
    n_home_out: *mut i64,
    n_chassis_contact: *mut i64,
    reward_terms: *mut f64,
    episode_return: *mut f64,
    episode_length: *mut i64,
}

// 各 env は自分の行だけに書くので、スレッドの間で重ならない
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl VecEnv {
    pub fn new(cfg: &Cfg, rngs: Vec<crate::rng::NpRng>, stage: i64, threads: usize) -> Result<VecEnv, EnvError> {
        if rngs.is_empty() {
            return Err(EnvError::Invalid("env の数は 1 以上".into()));
        }
        if threads == 0 {
            return Err(EnvError::Invalid("threads は 1 以上".into()));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("toy-sweeper-env-{i}"))
            .build()
            .map_err(|e| EnvError::State(format!("スレッドを作れない: {e}")))?;
        let mut slots = Vec::with_capacity(rngs.len());
        for rng in rngs {
            let mut env = Env::new(cfg.clone(), stage)?;
            env.set_rng(rng);
            slots.push(Slot { env, ret: PySum::default(), len: 0 });
        }
        let dim = slots[0].env.obs_dim();
        Ok(VecEnv { slots, pool, dim, poisoned: None })
    }

    pub fn n_envs(&self) -> usize {
        self.slots.len()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn env(&self, i: usize) -> &Env {
        &self.slots[i].env
    }

    /// 前の失敗で止まっていたら、それを付けた例外
    fn alive(&self) -> Result<(), EnvError> {
        match &self.poisoned {
            Some(e) => Err(EnvError::State(format!(
                "VecEnv は前の失敗で止まっている（一部の env だけ進んだ半端な状態なので続けない。作り直す）。最初の失敗: {e}"))),
            None => Ok(()),
        }
    }

    /// 失敗したら止まったことにする
    fn poison_on_error(&mut self, r: Result<(), EnvError>) -> Result<(), EnvError> {
        if let Err(e) = &r {
            self.poisoned = Some(e.clone());
        }
        r
    }

    pub fn set_curriculum(&mut self, stage: i64, previous: &[i64], p: f64, indices: &[usize]) -> Result<(), EnvError> {
        self.alive()?;
        for &i in indices {
            if i >= self.slots.len() {
                return Err(EnvError::Invalid(format!("env の番号 {i} が範囲外（{} 個）", self.slots.len())));
            }
        }
        for (k, &i) in indices.iter().enumerate() {
            if let Err(e) = self.slots[i].env.set_curriculum(stage, previous, p) {
                // Env::set_curriculum は確かめてから書き換えるので、最初の env で失敗したなら何も変わっていない
                return if k == 0 { Err(e) } else { self.poison_on_error(Err(e)) };
            }
        }
        Ok(())
    }

    /// 全 env を reset する。rngs[i] が Some なら env i の乱数をそれに入れ替える（Seed あり）、None なら今の乱数のまま。
    /// obs は (N, dim) の行優先
    pub fn reset(&mut self, rngs: Vec<Option<crate::rng::NpRng>>, obs: &mut [f32]) -> Result<(), EnvError> {
        self.alive()?;
        if rngs.len() != self.slots.len() || obs.len() != self.slots.len() * self.dim {
            return Err(EnvError::Invalid(format!("reset: 乱数 {} 個・観測の書き先 {} が env {} 個 × {} と合わない",
                                                 rngs.len(), obs.len(), self.slots.len(), self.dim)));
        }
        let dim = self.dim;
        let slots = &mut self.slots;
        let res: Vec<Result<(), EnvError>> = self.pool.install(|| {
            slots.par_iter_mut().zip(rngs.into_par_iter()).zip(obs.par_chunks_mut(dim)).enumerate()
                .map(|(i, ((s, rng), o))| {
                    let _cp = crate::cp::enter_env_index(i);
                    guard(|| {
                        if let Some(r) = rng {
                            s.env.set_rng(r);
                        }
                        s.ret = PySum::default();
                        s.len = 0;
                        s.env.reset_into(o)
                    })
                }).collect()
        });
        let r = first_error(res);
        self.poison_on_error(r)
    }

    /// 全 env を 1 歩進める（終わった env は reset する）
    fn step_all(&mut self, actions: &[i64], out: Out) -> Result<(), EnvError> {
        self.alive()?;
        // どの env も進める前に分かる失敗は、ここで（止めずに）返す
        if actions.len() != self.slots.len() {
            return Err(EnvError::Invalid(format!("actions の数 {} が env の数 {} と違う", actions.len(), self.slots.len())));
        }
        for (i, (&a, s)) in actions.iter().zip(&self.slots).enumerate() {
            if !(0..N_ACTIONS as i64).contains(&a) {
                return Err(EnvError::Invalid(format!("env {i}: {a} is not a valid Action")));
            }
            if s.env.episode_stage().is_none() {
                return Err(EnvError::State(format!("env {i}: reset() が先")));
            }
        }
        let dim = self.dim;
        let slots = &mut self.slots;
        let res: Vec<Result<(), EnvError>> = self.pool.install(|| {
            slots.par_iter_mut().enumerate().map(|(i, s)| {
                let _cp = crate::cp::enter_env_index(i);
                guard(|| step_one(s, i, actions[i], dim, out))
            }).collect()
        });
        let r = first_error(res);
        self.poison_on_error(r)
    }
}

/// 最初に失敗した env の番号を付ける
fn first_error(res: Vec<Result<(), EnvError>>) -> Result<(), EnvError> {
    match res.into_iter().enumerate().find_map(|(i, r)| r.err().map(|e| (i, e))) {
        Some((i, e)) => Err(e.prefixed(&format!("env {i}: "))),
        None => Ok(()),
    }
}

fn step_one(s: &mut Slot, i: usize, action: i64, dim: usize, o: Out) -> Result<(), EnvError> {
    // SAFETY: 書き先は (N, dim) / (N, N_TERMS) / (N,) の配列で、i 行目は env i だけが書く
    unsafe {
        let obs = std::slice::from_raw_parts_mut(o.obs.add(i * dim), dim);
        let (reward, terminated, truncated, info) = s.env.step_into(action, obs)?;
        s.ret.add(reward);
        s.len += 1;
        *o.reward.add(i) = reward;
        *o.terminated.add(i) = terminated;
        *o.truncated.add(i) = truncated;
        *o.is_success.add(i) = info.is_success;
        *o.failure.add(i) = info.failure.map_or(-1, |f| f as i8);
        *o.stage.add(i) = info.stage;
        *o.steps.add(i) = info.steps;
        *o.n_delivered.add(i) = info.n_delivered;
        *o.n_targets.add(i) = info.n_targets;
        *o.n_foreign_in_goal.add(i) = info.n_foreign_in_goal;
        *o.n_home_out.add(i) = info.n_home_out;
        *o.n_chassis_contact.add(i) = info.n_chassis_contact;
        std::slice::from_raw_parts_mut(o.reward_terms.add(i * N_TERMS), N_TERMS).copy_from_slice(&info.reward_terms);
        if terminated || truncated {
            *o.episode_return.add(i) = s.ret.value();
            *o.episode_length.add(i) = s.len;
            std::slice::from_raw_parts_mut(o.terminal_obs.add(i * dim), dim).copy_from_slice(obs);
            s.ret = PySum::default();
            s.len = 0;
            s.env.reset_into(obs)?;
        }
        *o.new_stage.add(i) = s.env.episode_stage().ok_or_else(|| EnvError::State("reset の後なのにエピソードがない".into()))?;
    }
    Ok(())
}

/// 多数の env（Python の NativeVecEnv が使う）
#[pyclass(name = "VecEnv", module = "toy_sweeper_native")]
pub struct PyVecEnv {
    inner: VecEnv,
}

#[pymethods]
impl PyVecEnv {
    /// rng_states[i]: env i の乱数の状態 (state, inc, 32 ビットの残りか None)
    #[new]
    fn py_new(cfg_json: &str, rng_states: Vec<RngState>, stage: i64, threads: usize) -> PyResult<Self> {
        let cfg = Cfg::from_json(cfg_json).map_err(PyValueError::new_err)?;
        let rngs = rng_states.into_iter().map(rng_from).collect();
        Ok(PyVecEnv { inner: guard(|| VecEnv::new(&cfg, rngs, stage, threads))? })
    }

    #[getter]
    fn n_envs(&self) -> usize {
        self.inner.n_envs()
    }

    #[getter]
    fn obs_dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn reward_term_names(&self) -> Vec<&'static str> {
        REWARD_TERM_NAMES.to_vec()
    }

    #[pyo3(signature = (stage, previous, rehearsal_probability, indices=None))]
    fn set_curriculum(&mut self, stage: i64, previous: Vec<i64>, rehearsal_probability: f64, indices: Option<Vec<usize>>)
        -> PyResult<()> {
        let idx = indices.unwrap_or_else(|| (0..self.inner.n_envs()).collect());
        Ok(self.inner.set_curriculum(stage, &previous, rehearsal_probability, &idx)?)
    }

    /// 全 env を reset する。rng_states[i] が None でなければ env i の乱数をそれにする（Seed あり）。
    /// (obs (N, dim) f32, 各 env の Stage (N,) i64) を返す
    #[pyo3(signature = (rng_states=None))]
    fn reset<'py>(&mut self, py: Python<'py>, rng_states: Option<Vec<Option<RngState>>>)
        -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<i64>>)> {
        let n = self.inner.n_envs();
        let rngs: Vec<_> = match rng_states {
            Some(v) => {
                if v.len() != n {
                    return Err(PyValueError::new_err(format!("rng_states の数 {} が env の数 {n} と違う", v.len())));
                }
                v.into_iter().map(|s| s.map(rng_from)).collect()
            }
            None => vec![None; n],
        };
        let obs = PyArray2::<f32>::zeros(py, [n, self.inner.dim()], false);
        // SAFETY: 作ったばかりの配列で、ほかから見えていない
        let o = unsafe { obs.as_slice_mut()? };
        let inner = &mut self.inner;
        py.allow_threads(|| inner.reset(rngs, o))?;
        let stages: Vec<i64> = (0..n).map(|i| self.inner.env(i).episode_stage().unwrap()).collect();
        Ok((obs, PyArray1::from_vec(py, stages)))
    }

    /// 全 env を 1 歩進める。actions: (N,) int64。結果は numpy の配列の dict（toy_sweeper.native.NativeVecEnv が読む）
    fn step<'py>(&mut self, py: Python<'py>, actions: PyReadonlyArray1<'py, i64>) -> PyResult<Bound<'py, PyDict>> {
        let n = self.inner.n_envs();
        let dim = self.inner.dim();
        if actions.len() != n {
            return Err(PyValueError::new_err(format!("actions の数 {} が env の数 {n} と違う", actions.len())));
        }
        let acts: Vec<i64> = actions.as_slice()?.to_vec();
        let obs = PyArray2::<f32>::zeros(py, [n, dim], false);
        let terminal_obs = PyArray2::<f32>::zeros(py, [n, dim], false);
        let reward = PyArray1::<f64>::zeros(py, n, false);
        let terminated = PyArray1::<bool>::zeros(py, n, false);
        let truncated = PyArray1::<bool>::zeros(py, n, false);
        let is_success = PyArray1::<bool>::zeros(py, n, false);
        let failure = PyArray1::<i8>::zeros(py, n, false);
        let ints: Vec<Bound<'py, PyArray1<i64>>> = (0..9).map(|_| PyArray1::<i64>::zeros(py, n, false)).collect();
        let reward_terms = PyArray2::<f64>::zeros(py, [n, N_TERMS], false);
        let episode_return = PyArray1::<f64>::zeros(py, n, false);
        // 作ったばかりの配列（連続、ほかから見えていない）。GIL を外している間は Rust だけが書く
        let out = {
            Out {
                obs: obs.data(),
                terminal_obs: terminal_obs.data(),
                reward: reward.data(),
                terminated: terminated.data(),
                truncated: truncated.data(),
                is_success: is_success.data(),
                failure: failure.data(),
                stage: ints[0].data(),
                new_stage: ints[1].data(),
                steps: ints[2].data(),
                n_delivered: ints[3].data(),
                n_targets: ints[4].data(),
                n_foreign_in_goal: ints[5].data(),
                n_home_out: ints[6].data(),
                n_chassis_contact: ints[7].data(),
                reward_terms: reward_terms.data(),
                episode_return: episode_return.data(),
                episode_length: ints[8].data(),
            }
        };
        let inner = &mut self.inner;
        py.allow_threads(|| inner.step_all(&acts, out))?;
        let d = PyDict::new(py);
        d.set_item("obs", obs)?;
        d.set_item("reward", reward)?;
        d.set_item("terminated", terminated)?;
        d.set_item("truncated", truncated)?;
        d.set_item("terminal_obs", terminal_obs)?;
        d.set_item("is_success", is_success)?;
        d.set_item("failure", failure)?;
        for (k, a) in ["stage", "new_stage", "steps", "n_delivered", "n_targets", "n_foreign_in_goal", "n_home_out",
                       "n_chassis_contact"].iter().zip(&ints[..8]) {
            d.set_item(k, a)?;
        }
        d.set_item("reward_terms", reward_terms)?;
        d.set_item("episode_return", episode_return)?;
        d.set_item("episode_length", &ints[8])?;
        Ok(d)
    }

    /// 各 env の乱数の状態（試験用）
    fn rng_states(&self) -> Vec<Option<RngState>> {
        (0..self.inner.n_envs()).map(|i| self.inner.env(i).rng_full_state()).collect()
    }

    fn stages(&self) -> Vec<Option<i64>> {
        (0..self.inner.n_envs()).map(|i| self.inner.env(i).episode_stage()).collect()
    }

    fn diagnostics(&self, i: usize) -> PyResult<String> {
        if i >= self.inner.n_envs() {
            return Err(PyRuntimeError::new_err(format!("env の番号 {i} が範囲外")));
        }
        Ok(self.inner.env(i).diagnostics())
    }
}

pub fn py_register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVecEnv>()?;
    Ok(())
}
