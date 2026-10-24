// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The DeepSeek-V4.1 Engram hash layout, derived from `config.json` the way the
//! publisher's `inference/engram.py` derives it (`EngramLayout.from_args`,
//! `compute_hash_multipliers`, the offsets in `NgramHashState`). None of it is stored in the
//! checkpoint, so it is computed here once at parse time.
//!
//! Owner: config (model parsers).
//! Invariants:
//! - Layouts are flat and ngram-major: `multipliers[layer * max_ngram + i]`,
//!   `primes[(layer * (max_ngram - 1) + ngram) * n_heads + head]`, `offsets` alike — the shape
//!   `EngramHashTables::from_flat` reads.
//! - Primes are never reused across layers, n-gram sizes or heads (one global seen set).
//! - Each layer's prime sum is that layer's table row count; a mismatch with
//!   `engram_num_embeddings` is an error, never a silent rehash.

use std::collections::BTreeSet;

use anyhow::{Result, ensure};

use super::numpy_rng::NumpyPcg64;

/// 2026-10-18: The per-layer RNG seed stride in `compute_hash_multipliers`.
const LAYER_SEED_STRIDE: u128 = 10007;

/// 2026-10-18: Deterministic Miller–Rabin for `u64` (these bases are exact below 2^64).
fn is_prime(n: u64) -> bool {
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 {
        return false;
    }
    for p in BASES {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let mulmod = |a: u64, b: u64| ((u128::from(a) * u128::from(b)) % u128::from(n)) as u64;
    let powmod = |mut b: u64, mut e: u64| {
        let mut r = 1u64;
        b %= n;
        while e > 0 {
            if e & 1 == 1 {
                r = mulmod(r, b);
            }
            b = mulmod(b, b);
            e >>= 1;
        }
        r
    };
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    'witness: for a in BASES {
        let mut x = powmod(a, d);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mulmod(x, x);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// 2026-10-18: `find_next_prime`: the smallest prime above `start` not yet handed out.
fn next_unseen_prime(start: u64, seen: &BTreeSet<u64>) -> u64 {
    let mut c = start + 1;
    while !is_prime(c) || seen.contains(&c) {
        c += 1;
    }
    c
}

/// 2026-10-18: The inputs `config.json` supplies.
pub(crate) struct EngramSpec<'a> {
    pub layer_ids: &'a [usize],
    pub max_ngram: usize,
    pub n_heads: usize,
    pub vocab_size: u64,
    pub compressed_vocab_size: u64,
}

/// 2026-10-18: The derived hash layout, flat (see the module invariants).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct EngramLayout {
    pub multipliers: Vec<u64>,
    pub primes: Vec<u64>,
    pub offsets: Vec<u64>,
    /// Sum of each layer's primes: its table row count.
    pub rows_per_layer: Vec<u64>,
}

pub(crate) fn engram_layout(spec: &EngramSpec<'_>) -> Result<EngramLayout> {
    ensure!(
        spec.max_ngram >= 2,
        "Engram max_ngram_size {} must be >= 2",
        spec.max_ngram
    );
    ensure!(spec.n_heads > 0, "Engram n_heads must be > 0");
    ensure!(spec.vocab_size > 0, "Engram vocab_size must be > 0");
    ensure!(
        spec.compressed_vocab_size > 0,
        "Engram compressed_vocab_size must be > 0"
    );

    let mut seen = BTreeSet::new();
    let mut primes = Vec::with_capacity(spec.layer_ids.len() * (spec.max_ngram - 1) * spec.n_heads);
    let mut offsets = Vec::with_capacity(primes.capacity());
    let mut rows_per_layer = Vec::with_capacity(spec.layer_ids.len());
    for _ in spec.layer_ids {
        let mut offset = 0u64;
        for _ in 0..spec.max_ngram - 1 {
            let mut current = spec.vocab_size - 1;
            for _ in 0..spec.n_heads {
                current = next_unseen_prime(current, &seen);
                seen.insert(current);
                primes.push(current);
                offsets.push(offset);
                offset += current;
            }
        }
        rows_per_layer.push(offset);
    }

    // 2026-10-18: Bounded so `token * multiplier` cannot overflow int64; drawn from the
    // compressed vocab size, not the bucket base.
    let bound = (i64::MAX / spec.compressed_vocab_size as i64 / 2).max(1);
    let mut multipliers = Vec::with_capacity(spec.layer_ids.len() * spec.max_ngram);
    for &layer in spec.layer_ids {
        let mut rng = NumpyPcg64::default_rng(LAYER_SEED_STRIDE * layer as u128);
        for v in rng.integers_i64(0, bound, spec.max_ngram) {
            multipliers.push(v as u64 * 2 + 1);
        }
    }
    Ok(EngramLayout {
        multipliers,
        primes,
        offsets,
        rows_per_layer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> EngramLayout {
        engram_layout(&EngramSpec {
            layer_ids: &[1, 4],
            max_ngram: 4,
            n_heads: 2,
            vocab_size: 64,
            compressed_vocab_size: 64,
        })
        .unwrap()
    }

    /// 2026-10-18: The tiny golden `ds41_tiny_golden.json` (model-arch) was generated by the
    /// publisher's `engram.py` under NumPy; these are its multipliers, primes and offsets.
    #[test]
    fn tiny_layout_matches_the_publisher_generated_golden() {
        let l = tiny();
        assert_eq!(
            l.multipliers,
            vec![
                118650432209603643,
                7493640653728489,
                55676810147952557,
                114555519428547269,
                127364524771245009,
                7800076361716407,
                114038564133282483,
                29348051253447937,
            ]
        );
        assert_eq!(
            l.primes,
            vec![67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
        );
        assert_eq!(
            l.offsets,
            vec![0, 67, 138, 211, 290, 373, 0, 97, 198, 301, 408, 517]
        );
        assert_eq!(l.rows_per_layer, vec![462, 630]);
    }

    /// 2026-10-18: The publisher's `config.json` lists each layer's table rows; the derived
    /// prime sums must equal them (the second layer differs only because primes are global).
    #[test]
    fn real_layout_prime_sums_equal_the_published_row_counts() {
        let l = engram_layout(&EngramSpec {
            layer_ids: &[1, 14],
            max_ngram: 4,
            n_heads: 8,
            vocab_size: 16_000_000,
            compressed_vocab_size: 99092,
        })
        .unwrap();
        assert_eq!(l.rows_per_layer, vec![384_006_168, 384_016_682]);
        assert_eq!(
            (l.multipliers.len(), l.primes.len(), l.offsets.len()),
            (8, 48, 48)
        );
        assert!(l.multipliers.iter().all(|m| m % 2 == 1));
        let bound = (i64::MAX / 99092 / 2) as u64;
        assert!(l.multipliers.iter().all(|&m| m < 2 * bound + 1));
    }

    /// 2026-10-18: Against trial division over a range spanning small primes, squares of
    /// primes and Carmichael numbers.
    #[test]
    fn miller_rabin_agrees_with_trial_division() {
        let trial = |n: u64| {
            n >= 2
                && (2..)
                    .take_while(|d| d * d <= n)
                    .all(|d| !n.is_multiple_of(d))
        };
        for n in 0..20_000u64 {
            assert_eq!(is_prime(n), trial(n), "{n}");
        }
        for n in [561u64, 1105, 1729, 2465, 2821, 6601, 8911, 3_215_031_751] {
            assert!(!is_prime(n), "{n}");
        }
        assert!(is_prime(18_446_744_073_709_551_557));
        assert!(!is_prime(4_294_967_291 * 4_294_967_291));
    }

    #[test]
    fn a_one_gram_layout_is_refused() {
        let e = engram_layout(&EngramSpec {
            layer_ids: &[1],
            max_ngram: 1,
            n_heads: 2,
            vocab_size: 64,
            compressed_vocab_size: 64,
        })
        .unwrap_err();
        assert!(e.to_string().contains("max_ngram_size"), "{e:#}");
    }
}
