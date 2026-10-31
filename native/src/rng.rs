//! numpy の np.random.Generator(np.random.PCG64(...)) と同じ乱数（numpy 2.2.6 の C と Cython を写した）。
//!
//! 写した元（numpy 2.2.6 の sdist）:
//! - PCG64 の 1 歩と出力: numpy/random/src/pcg64/pcg64.h（pcg_setseq_128_xsl_rr_64_random_r。状態を進めてから出す）
//! - 32 ビットの取り出し: pcg64.h の pcg64_next32。64 ビットを 1 個引いて下位を返し、上位を残す。残りはビット生成器の
//!   状態（numpy の state の has_uint32 / uinteger）で、次の 32 ビットの要求で使う。random() / normal() は 64 ビットを
//!   直接引くので残りに触らない
//! - random(): _common.pxd の uint64_to_double（(x >> 11) * 2^-53）
//! - uniform(): _generator.pyx の uniform（範囲 high - low が有限で符号が負でないことを確かめる）と
//!   distributions.c の random_uniform（low + range * next_double）
//! - integers(): _bounded_integers.pyx.in の _rand_int64 → distributions.c の random_bounded_uint64_fill
//!   （Lemire 法、範囲が 32 ビットに入れば 32 ビットの Lemire 法）
//! - normal(): distributions.c の random_standard_normal（ziggurat、表は ziggurat_tables.rs）と random_normal
//! - choice(n, size=k, replace=False): _generator.pyx の choice（Floyd 法と _shuffle_int、大きい母集団では末尾の並べ替え）
use crate::ziggurat_tables::{FI_DOUBLE, KI_DOUBLE, WI_DOUBLE, ZIGGURAT_NOR_INV_R, ZIGGURAT_NOR_R};
use pyo3::prelude::*;

/// pcg64.h の PCG_DEFAULT_MULTIPLIER_128
const PCG_MULT: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

/// numpy の PCG64 と同じ状態（128 ビットの state と inc、32 ビットの残り）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NpRng {
    state: u128,
    inc: u128,
    /// pcg64_next32 が残した上位 32 ビット（numpy の has_uint32 == 1 のときの uinteger）
    buf: Option<u32>,
}

impl NpRng {
    /// numpy の bit_generator.state["state"]["state"] / ["inc"] から作る（32 ビットの残りなし。Seed から作った直後と同じ）。
    pub fn from_state(state: u128, inc: u128) -> Self {
        Self::from_full_state(state, inc, None)
    }

    /// 32 ビットの残りも含めて作る。buf は numpy の has_uint32 が 1 のときの uinteger、0 のときは None。
    pub fn from_full_state(state: u128, inc: u128, buf: Option<u32>) -> Self {
        assert!(inc & 1 == 1, "PCG64 の inc は奇数のはず（inc = {inc}）");
        Self { state, inc, buf }
    }

    pub fn state(&self) -> (u128, u128) {
        (self.state, self.inc)
    }

    pub fn uint32_buffer(&self) -> Option<u32> {
        self.buf
    }

    /// pcg64_next64: 状態を進めてから XSL-RR で 64 ビットを出す。
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
        let x = ((self.state >> 64) as u64) ^ (self.state as u64);
        x.rotate_right((self.state >> 122) as u32)
    }

    /// pcg64_next32: 残りがあればそれを、なければ 64 ビットを引いて下位を返し上位を残す。
    pub fn next_u32(&mut self) -> u32 {
        if let Some(u) = self.buf.take() {
            return u;
        }
        let n = self.next_u64();
        self.buf = Some((n >> 32) as u32);
        n as u32
    }

    /// Generator.random(): [0, 1) の一様乱数。
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Generator.uniform(low, high)（スカラー）。範囲が無限・NaN・負（-0.0 も）なら numpy と同じく止める。
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        let range = high - low;
        assert!(range.is_finite(), "uniform: high - low が有限でない（low = {low}, high = {high}）");
        assert!(!range.is_sign_negative(), "uniform: high - low < 0（low = {low}, high = {high}）");
        low + range * self.random()
    }

    /// Generator.integers(n)（スカラー、int64）: [0, n)。n <= 0 なら止める。
    pub fn integers(&mut self, n: i64) -> i64 {
        self.integers_range(0, n)
    }

    /// Generator.integers(low, high)（スカラー、int64）: [low, high)。low >= high なら止める。
    pub fn integers_range(&mut self, low: i64, high: i64) -> i64 {
        assert!(low < high, "integers: low >= high（low = {low}, high = {high}）");
        let rng = (high - 1).wrapping_sub(low) as u64; // 閉区間 [low, high - 1] の幅
        (low as u64).wrapping_add(self.bounded_u64(rng)) as i64
    }

    /// distributions.c の random_bounded_uint64（use_masked = false）: [0, rng] の一様な整数。
    /// rng == 0 なら乱数を使わずに 0 を返す。
    fn bounded_u64(&mut self, rng: u64) -> u64 {
        if rng == 0 {
            0
        } else if rng <= 0xFFFF_FFFF {
            if rng == 0xFFFF_FFFF {
                self.next_u32() as u64 // 32 ビットの Lemire 法は rng = 0xFFFFFFFF を扱えない
            } else {
                self.bounded_lemire_u32(rng as u32) as u64
            }
        } else if rng == u64::MAX {
            self.next_u64()
        } else {
            self.bounded_lemire_u64(rng)
        }
    }

    /// distributions.c の buffered_bounded_lemire_uint32
    fn bounded_lemire_u32(&mut self, rng: u32) -> u32 {
        let rng_excl = rng + 1;
        let mut m = self.next_u32() as u64 * rng_excl as u64;
        let mut leftover = m as u32;
        if leftover < rng_excl {
            let threshold = (u32::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = self.next_u32() as u64 * rng_excl as u64;
                leftover = m as u32;
            }
        }
        (m >> 32) as u32
    }

    /// distributions.c の bounded_lemire_uint64（__uint128_t の版）
    fn bounded_lemire_u64(&mut self, rng: u64) -> u64 {
        let rng_excl = rng + 1;
        let mut m = self.next_u64() as u128 * rng_excl as u128;
        let mut leftover = m as u64;
        if leftover < rng_excl {
            let threshold = (u64::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = self.next_u64() as u128 * rng_excl as u128;
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Generator.standard_normal()（distributions.c の random_standard_normal、ziggurat）。
    pub fn standard_normal(&mut self) -> f64 {
        loop {
            let mut r = self.next_u64();
            let idx = (r & 0xff) as usize;
            r >>= 8;
            let sign = r & 0x1;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = rabs as f64 * WI_DOUBLE[idx];
            if sign & 0x1 == 1 {
                x = -x;
            }
            if rabs < KI_DOUBLE[idx] {
                return x; // 99.3% はここで返る
            }
            if idx == 0 {
                // 裾。log(0) を避けて 1 - U を使う（numpy の GH 13361）
                loop {
                    let xx = -ZIGGURAT_NOR_INV_R * (-self.random()).ln_1p();
                    let yy = -(-self.random()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 0x1 == 1 { -(ZIGGURAT_NOR_R + xx) } else { ZIGGURAT_NOR_R + xx };
                    }
                }
            } else if (FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]) * self.random() + FI_DOUBLE[idx] < (-0.5 * x * x).exp() {
                return x;
            }
        }
    }

    /// Generator.normal()（loc = 0.0, scale = 1.0）。random_normal は loc + scale * z なので、z = -0.0 は 0.0 になる。
    pub fn normal(&mut self) -> f64 {
        0.0 + 1.0 * self.standard_normal()
    }

    /// Generator.choice(pop_size, size=size, replace=False)（shuffle=True、p なし）。母集団 [0, pop_size) から重複なしに選ぶ。
    pub fn choice_no_replace(&mut self, pop_size: i64, size: usize) -> Vec<i64> {
        assert!(pop_size > 0 || size == 0, "choice: pop_size = {pop_size} では選べない");
        assert!(size as i64 <= pop_size.max(0), "choice: size = {size} > pop_size = {pop_size}（replace=False）");
        let k = size as i64;
        if pop_size > 10000 && k > pop_size / 50 {
            // 末尾の並べ替え（_shuffle_int(pop, max(pop - size, 1)) の後ろ size 個）
            let mut idx: Vec<i64> = (0..pop_size).collect();
            self.shuffle_int(&mut idx, (pop_size - k).max(1) as usize);
            return idx[(pop_size - k) as usize..].to_vec();
        }
        // Floyd 法。numpy は開番地法のハッシュで「もう選んだか」を見るだけなので、結果は選んだ順の列で決まる
        let mut out: Vec<i64> = Vec::with_capacity(size);
        let mut seen = std::collections::HashSet::with_capacity(size);
        for j in (pop_size - k)..pop_size {
            let val = self.bounded_u64(j as u64) as i64;
            if seen.insert(val) {
                out.push(val);
            } else {
                seen.insert(j); // j はまだ選ばれていない（これまでの値は j 未満）
                out.push(j);
            }
        }
        self.shuffle_int(&mut out, 1);
        out
    }

    /// _generator.pyx の _shuffle_int: i = n-1, ..., first の順に data[i] と data[bounded(i)] を入れ替える。
    fn shuffle_int(&mut self, data: &mut [i64], first: usize) {
        for i in (first..data.len()).rev() {
            let j = self.bounded_u64(i as u64) as usize;
            data.swap(i, j);
        }
    }
}

/// 試験用（test/test_native_rng.py）に Python から使う NpRng。
#[pyclass(name = "NpRng", module = "toy_sweeper_native")]
pub struct PyNpRng {
    inner: NpRng,
}

#[pymethods]
impl PyNpRng {
    /// numpy の bit_generator.state の state / inc と、has_uint32 が 1 なら uinteger（0 なら None）。
    #[new]
    #[pyo3(signature = (state, inc, uint32_buffer=None))]
    fn new(state: u128, inc: u128, uint32_buffer: Option<u32>) -> Self {
        Self { inner: NpRng::from_full_state(state, inc, uint32_buffer) }
    }

    fn state(&self) -> (u128, u128) {
        self.inner.state()
    }

    fn uint32_buffer(&self) -> Option<u32> {
        self.inner.uint32_buffer()
    }

    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    fn random(&mut self) -> f64 {
        self.inner.random()
    }

    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        self.inner.uniform(low, high)
    }

    fn integers(&mut self, n: i64) -> i64 {
        self.inner.integers(n)
    }

    fn integers_range(&mut self, low: i64, high: i64) -> i64 {
        self.inner.integers_range(low, high)
    }

    fn normal(&mut self) -> f64 {
        self.inner.normal()
    }

    fn standard_normal(&mut self) -> f64 {
        self.inner.standard_normal()
    }

    fn choice_no_replace(&mut self, pop_size: i64, size: usize) -> Vec<i64> {
        self.inner.choice_no_replace(pop_size, size)
    }
}

pub fn py_register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNpRng>()
}

#[cfg(test)]
mod tests {
    //! 値は numpy 2.2.6 で np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed))) から引いたもの。
    use super::*;

    const SEED0_STATE: u128 = 35399562948360463058890781895381311971;
    const SEED0_INC: u128 = 87136372517582989555478159403783844777;

    fn seed0() -> NpRng {
        NpRng::from_state(SEED0_STATE, SEED0_INC)
    }

    #[test]
    fn next_u64_matches_numpy_random_raw() {
        let mut r = seed0();
        let v: Vec<u64> = (0..3).map(|_| r.next_u64()).collect();
        assert_eq!(v, [11749869230777074271, 4976686463289251617, 755828109848996024]);
    }

    #[test]
    fn random_and_uniform_match_numpy() {
        let mut r = seed0();
        // 0x1.461fd79fb3850p-1, 0x1.1442f7e20b674p-2
        assert_eq!(r.random().to_bits(), 0x3FE461FD79FB3850);
        assert_eq!(r.random().to_bits(), 0x3FD1442F7E20B674);
        let mut r = seed0();
        // 0x1.a3e77cc707704p+0, -0x1.7e265a615af08p-1
        assert_eq!(r.uniform(-2.5, 4.0).to_bits(), 0x3FFA3E77CC707704);
        assert_eq!(r.uniform(-2.5, 4.0).to_bits(), 0xBFE7E265A615AF08);
    }

    #[test]
    fn normal_matches_numpy() {
        let mut r = seed0();
        // 0x1.017ed89db8441p-3, -0x1.0e8cfe9bd45ccp-3, 0x1.47e57a468b06dp-1
        assert_eq!(r.normal().to_bits(), 0x3FC017ED89DB8441);
        assert_eq!(r.normal().to_bits(), 0xBFC0E8CFE9BD45CC);
        assert_eq!(r.normal().to_bits(), 0x3FE47E57A468B06D);
    }

    #[test]
    fn normal_tail_matches_numpy() {
        // seed 1 の 861 個目の normal() は裾（idx == 0）に入る
        let mut r = NpRng::from_state(45948035941562197576018527454357159123, 194290289479364712180083596243593368443);
        assert_eq!(r.normal().to_bits(), 0x400E035931978327); // 0x1.e035931978327p+1
        assert_eq!(r.state().0, 215387605741918675568394071316505153900);
    }

    #[test]
    fn integers_use_the_32bit_buffer() {
        let mut r = seed0();
        let v: Vec<i64> = (0..5).map(|_| r.integers(10)).collect();
        assert_eq!(v, [8, 6, 5, 2, 3]);
        assert_eq!(r.state().0, 18792671013009471063546316241193924174);
        assert_eq!(r.uint32_buffer(), Some(175979945));
        // 64 ビットの道は残りを作らない
        let mut r = seed0();
        let v: Vec<i64> = (0..3).map(|_| r.integers(1 << 40)).collect();
        assert_eq!(v, [700346781657, 296633628802, 45050865998]);
        assert_eq!(r.uint32_buffer(), None);
    }

    #[test]
    fn integers_one_draws_nothing() {
        let mut r = seed0();
        assert_eq!(r.integers(1), 0);
        assert_eq!(r, seed0());
    }

    #[test]
    fn choice_matches_numpy() {
        let mut r = seed0();
        assert_eq!(r.choice_no_replace(20, 5), [10, 9, 5, 6, 13]);
        assert_eq!(r.state().0, 133411349017971402732463711865589153492);
        assert_eq!(r.uint32_buffer(), Some(3492969080));
    }

    #[test]
    #[should_panic]
    fn integers_zero_panics() {
        seed0().integers(0);
    }

    #[test]
    #[should_panic]
    fn uniform_reversed_panics() {
        seed0().uniform(1.0, 0.0);
    }

    #[test]
    #[should_panic]
    fn even_inc_panics() {
        NpRng::from_state(1, 2);
    }
}
