// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The part of NumPy's `default_rng(seed)` the DeepSeek-V4.1 Engram layout draws
//! from: `SeedSequence` seeding a `PCG64` generator, and `Generator.integers(low, high)` for
//! `int64`. The publisher derives the Engram hash multipliers with
//! `np.random.default_rng(10007 * layer_id).integers(0, bound, size=max_ngram_size)`, so the
//! multipliers are only reproducible bit for bit by this exact stream.
//!
//! Owner: config (model parsers).
//! Invariants:
//! - Wrapping `u32` arithmetic in `SeedSequence`, wrapping `u128` in PCG64, as in NumPy.
//! - The 32-bit draw is buffered in the generator (low half first), so mixed 32/64-bit draws
//!   from one generator interleave as NumPy's do.

/// 2026-10-18: NumPy `SeedSequence` constants (`bit_generator.pyx`).
const POOL_SIZE: usize = 4;
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

/// 2026-10-18: PCG64's 128-bit LCG multiplier (`PCG_DEFAULT_MULTIPLIER_128`).
const PCG_MULT: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// 2026-10-18: A non-negative integer seed as NumPy coerces it: little-endian 32-bit words,
/// `[0]` for zero.
fn seed_words(seed: u128) -> Vec<u32> {
    if seed == 0 {
        return vec![0];
    }
    let mut words = Vec::new();
    let mut n = seed;
    while n > 0 {
        words.push((n & 0xffff_ffff) as u32);
        n >>= 32;
    }
    words
}

/// 2026-10-18: `SeedSequence(seed).pool` with no spawn key.
fn seed_pool(seed: u128) -> [u32; POOL_SIZE] {
    let entropy = seed_words(seed);
    let mut pool = [0u32; POOL_SIZE];
    let mut hash_const = INIT_A;
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hash_const);
    }
    for src in 0..POOL_SIZE {
        for dst in 0..POOL_SIZE {
            if src != dst {
                pool[dst] = mix(pool[dst], hashmix(pool[src], &mut hash_const));
            }
        }
    }
    for &word in entropy.iter().skip(POOL_SIZE) {
        for slot in pool.iter_mut() {
            *slot = mix(*slot, hashmix(word, &mut hash_const));
        }
    }
    pool
}

/// 2026-10-18: `SeedSequence(seed).generate_state(4, np.uint64)`: eight 32-bit words drawn
/// from the cycled pool, paired little-endian into four `u64`.
fn generate_state_u64x4(seed: u128) -> [u64; 4] {
    let pool = seed_pool(seed);
    let mut hash_const = INIT_B;
    let mut words = [0u32; 8];
    for (i, w) in words.iter_mut().enumerate() {
        let mut v = pool[i % POOL_SIZE] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        v = v.wrapping_mul(hash_const);
        *w = v ^ (v >> XSHIFT);
    }
    std::array::from_fn(|i| u64::from(words[2 * i]) | (u64::from(words[2 * i + 1]) << 32))
}

/// 2026-10-18: NumPy's `PCG64` (XSL-RR 128/64) as `default_rng(seed)` builds it.
pub(crate) struct NumpyPcg64 {
    state: u128,
    inc: u128,
    buffered_u32: Option<u32>,
}

impl NumpyPcg64 {
    /// 2026-10-18: `np.random.default_rng(seed)` for a non-negative integer seed.
    pub(crate) fn default_rng(seed: u128) -> Self {
        let s = generate_state_u64x4(seed);
        let initstate = (u128::from(s[0]) << 64) | u128::from(s[1]);
        let initseq = (u128::from(s[2]) << 64) | u128::from(s[3]);
        let mut g = Self {
            state: 0,
            inc: (initseq << 1) | 1,
            buffered_u32: None,
        };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }

    fn next_u32(&mut self) -> u32 {
        if let Some(v) = self.buffered_u32.take() {
            return v;
        }
        let next = self.next_u64();
        self.buffered_u32 = Some((next >> 32) as u32);
        next as u32
    }

    /// 2026-10-18: `Generator.random()`: 53 high bits scaled to [0, 1).
    /// Test-only: the published `random()` values pin the seeding and the stream.
    #[cfg(test)]
    pub(crate) fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// 2026-10-18: `Generator.integers(low, high, size=n, dtype=np.int64)` (high exclusive,
    /// `low < high`), Lemire's bounded draw on the 32- or 64-bit path NumPy picks for the
    /// range.
    pub(crate) fn integers_i64(&mut self, low: i64, high: i64, n: usize) -> Vec<i64> {
        assert!(low < high, "integers: low {low} >= high {high}");
        let rng = high.wrapping_sub(low).wrapping_sub(1) as u64;
        (0..n)
            .map(|_| {
                let off = if rng == 0 {
                    0
                } else if rng <= u64::from(u32::MAX) {
                    if rng == u64::from(u32::MAX) {
                        u64::from(self.next_u32())
                    } else {
                        u64::from(self.bounded_lemire_u32(rng as u32))
                    }
                } else if rng == u64::MAX {
                    self.next_u64()
                } else {
                    self.bounded_lemire_u64(rng)
                };
                low.wrapping_add(off as i64)
            })
            .collect()
    }

    fn bounded_lemire_u32(&mut self, rng: u32) -> u32 {
        let excl = rng + 1;
        let mut m = u64::from(self.next_u32()) * u64::from(excl);
        let mut leftover = m as u32;
        if leftover < excl {
            let threshold = (u32::MAX - rng) % excl;
            while leftover < threshold {
                m = u64::from(self.next_u32()) * u64::from(excl);
                leftover = m as u32;
            }
        }
        (m >> 32) as u32
    }

    fn bounded_lemire_u64(&mut self, rng: u64) -> u64 {
        let excl = rng + 1;
        let mut m = u128::from(self.next_u64()) * u128::from(excl);
        let mut leftover = m as u64;
        if leftover < excl {
            let threshold = (u64::MAX - rng) % excl;
            while leftover < threshold {
                m = u128::from(self.next_u64()) * u128::from(excl);
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::NumpyPcg64;

    /// 2026-10-18: Values printed by NumPy itself: `default_rng(42).random()`,
    /// `default_rng(0).random()`.
    #[test]
    fn random_matches_numpy() {
        assert_eq!(
            NumpyPcg64::default_rng(42).random(),
            0.773_956_048_555_963_3
        );
        assert_eq!(NumpyPcg64::default_rng(0).random(), 0.636_961_687_321_454_3);
    }

    /// 2026-10-18: `default_rng(42).integers(0, 10, 3)` is `[0, 7, 6]` in NumPy: the 32-bit
    /// Lemire path with the buffered half-words.
    #[test]
    fn small_range_integers_take_the_buffered_32_bit_path() {
        assert_eq!(
            NumpyPcg64::default_rng(42).integers_i64(0, 10, 3),
            vec![0, 7, 6]
        );
    }

    /// 2026-10-18: A multi-word seed (above `u32::MAX`) differs from its low word alone, so
    /// the high entropy word reaches the pool.
    #[test]
    fn high_seed_words_reach_the_pool() {
        let lo = NumpyPcg64::default_rng(5).next_u64();
        let hi = NumpyPcg64::default_rng(5 | (1u128 << 32)).next_u64();
        assert_ne!(lo, hi);
    }
}
