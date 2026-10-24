// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The host twin's arithmetic, the round trip's error bound, and how much of the
//! indexer's top-2048 selection survives the FP8 cache.

use super::*;

/// 2026-10-16: A deterministic generator (splitmix64), so every run sees the same keys.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// 2026-10-16: A standard normal by Box-Muller.
    fn normal(&mut self) -> f32 {
        let u = self.uniform().max(1e-7);
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * self.uniform()).cos()
    }
}

/// 2026-10-16: Rotating twice multiplies by 128 exactly on small integers (exact in f32),
/// and the rotation of a unit vector is all +-1: it is the transform, not a permutation.
#[test]
fn fwht_is_the_scaled_involution() {
    let mut rng = Rng(7);
    let orig: [f32; D] = std::array::from_fn(|_| (rng.next_u64() % 41) as f32 - 20.0);
    let mut v = orig;
    fwht128(&mut v);
    assert_ne!(v, orig);
    fwht128(&mut v);
    assert_eq!(v, orig.map(|x| x * 128.0));
    let mut e = [0.0f32; D];
    e[5] = 1.0;
    fwht128(&mut e);
    assert!(e.iter().all(|x| x.abs() == 1.0));
}

/// 2026-10-16: The scale is the smallest power of two that keeps amax within 448, clamped to
/// the normal exponents (a finite amax tops out at 2^120; only infinity reaches 2^127).
#[test]
fn ue8m0_scale_values() {
    assert_eq!(ue8m0_scale(448.0), 1.0);
    assert_eq!(ue8m0_scale(449.0), 2.0);
    assert_eq!(ue8m0_scale(896.0), 2.0);
    assert_eq!(ue8m0_scale(224.0), 0.5);
    assert_eq!(ue8m0_scale(300.0), 1.0);
    assert_eq!(ue8m0_scale(0.0), f32::from_bits(1 << 23));
    assert_eq!(ue8m0_scale(1e-40), f32::from_bits(1 << 23));
    assert_eq!(ue8m0_scale(f32::MAX), f32::from_bits((127 + 120) << 23));
    assert_eq!(ue8m0_scale(f32::INFINITY), f32::from_bits(254 << 23));
    for a in [1e-3f32, 0.7, 3.0, 12.5, 1e4] {
        let s = ue8m0_scale(a);
        assert!(a / s <= E4M3_MAX && a / s > E4M3_MAX / 2.0, "{a}: {s}");
    }
}

/// 2026-10-16: Every finite code encodes back to itself; a midpoint goes to the even code;
/// past 448 saturates; the sign survives, including on zero.
#[test]
fn e4m3_encode_round_trips_and_rounds_to_even() {
    for c in (0u8..=0x7e).chain(0x80..=0xfe) {
        assert_eq!(e4m3_encode(e4m3_decode(c)), c, "code {c:#x}");
    }
    // 2026-10-16: 1.0 is 0x38, 1.125 is 0x39, 1.25 is 0x3a: 1.0625 ties to 0x38 and 1.1875
    // to 0x3a, the even codes.
    assert_eq!(e4m3_encode(1.0625), 0x38);
    assert_eq!(e4m3_encode(1.1875), 0x3a);
    assert_eq!(e4m3_encode(1.07), 0x39);
    assert_eq!(e4m3_encode(460.0), 0x7e);
    assert_eq!(e4m3_encode(1e9), 0x7e);
    assert_eq!(e4m3_encode(-1e9), 0xfe);
    assert_eq!(e4m3_encode(-0.0), 0x80);
    assert_eq!(e4m3_encode(0.0001), 0x00);
}

/// 2026-10-16: An indexer-like pool key: unit noise, a per-channel offset (the LayerNorm
/// bias), a few large outlier channels, and `signal` along a direction queries share.
fn pool_key(rng: &mut Rng, bias: &[f32; D], dir: &[f32; D], signal: f32) -> [f32; D] {
    std::array::from_fn(|d| {
        let outlier = if d % 41 == 3 { 8.0 } else { 1.0 };
        rng.normal() * outlier + bias[d] + signal * dir[d]
    })
}

/// 2026-10-16: The key the store would write back for `key`, through the twin.
fn round_trip(key: &[f32; D]) -> [f32; D] {
    let (q, s) = quantize_pool(key);
    dequantize_pool(&q, s)
}

/// 2026-10-16: Per element of the rotated key, the E4M3 error is at most half a step: 2^-4 of
/// the value for a normal code, 2^-10 of the scale for a subnormal one. The key's relative L2
/// error is then at most 2^-4 plus the subnormal term. The control encodes with a quarter of
/// the scale, which saturates the largest element, and must break the bound.
#[test]
fn round_trip_error_is_within_the_e4m3_bound() {
    let mut rng = Rng(11);
    let bias: [f32; D] = std::array::from_fn(|_| 0.5 * rng.normal());
    let dir: [f32; D] = std::array::from_fn(|_| rng.normal());
    let mut worst = 0.0f32;
    for i in 0..2000 {
        let key = pool_key(&mut rng, &bias, &dir, (i % 7) as f32);
        let (q, s) = quantize_pool(&key);
        let mut y = key;
        fwht128(&mut y);
        for (qi, yi) in q.iter().zip(y) {
            let err = (e4m3_decode(*qi) * s - yi).abs();
            assert!(
                err <= yi.abs() / 16.0 + s / 1024.0,
                "{yi} -> {}",
                e4m3_decode(*qi) * s
            );
        }
        let back = dequantize_pool(&q, s);
        let num: f32 = back.iter().zip(&key).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = key.iter().map(|b| b * b).sum();
        worst = worst.max((num / den).sqrt());
        let sq = s / 4.0;
        let broken = y.iter().any(|yi| {
            let err = (e4m3_decode(e4m3_encode(yi / sq)) * sq - yi).abs();
            err > yi.abs() / 16.0 + sq / 1024.0
        });
        assert!(broken, "the quarter-scale control met the bound");
    }
    assert!(worst <= 1.0 / 16.0, "worst relative L2 error {worst}");
}

/// 2026-10-16: The indexer's score of one pool: `sum_h w_h * relu(scale * q_h . k)`.
fn score(qs: &[[f32; D]], w: &[f32], scale: f32, k: &[f32; D]) -> f32 {
    qs.iter()
        .zip(w)
        .map(|(q, wh)| {
            let dot: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
            wh * (scale * dot).max(0.0)
        })
        .sum()
}

/// 2026-10-16: The top `n` pool ids by score (ties to the lower id).
fn top(scores: &[f32], n: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..scores.len()).collect();
    ids.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    ids.truncate(n);
    ids.sort_unstable();
    ids
}

fn overlap(a: &[usize], b: &[usize]) -> f32 {
    let hits = a.iter().filter(|x| b.binary_search(x).is_ok()).count();
    hits as f32 / a.len() as f32
}

/// 2026-10-16: Of the 512 pools (2048 tokens) the indexer selects from 8192 pools of exact
/// keys, at least 90% are still selected from the FP8 cache's keys, over 32 heads and several
/// queries. Two controls bound the measure: keys rounded far more coarsely (one mantissa bit)
/// lose more of the selection, and unrelated keys share only about chance (1/16).
#[test]
fn top_2048_selection_survives_the_fp8_cache() {
    const POOLS: usize = 8192;
    const SELECT: usize = 2048 / 4;
    let mut rng = Rng(23);
    let bias: [f32; D] = std::array::from_fn(|_| 0.5 * rng.normal());
    let dir: [f32; D] = std::array::from_fn(|_| rng.normal());
    let keys: Vec<[f32; D]> = (0..POOLS)
        .map(|_| {
            let signal = -2.0 * rng.uniform().max(1e-7).ln();
            pool_key(&mut rng, &bias, &dir, signal)
        })
        .collect();
    let fp8: Vec<[f32; D]> = keys.iter().map(round_trip).collect();
    let coarse: Vec<[f32; D]> = keys
        .iter()
        .map(|k| k.map(|x| f32::from_bits(x.to_bits() & 0xffc0_0000)))
        .collect();
    let unrelated: Vec<[f32; D]> = (0..POOLS)
        .map(|_| pool_key(&mut rng, &bias, &dir, 1.0))
        .collect();
    let scale = 1.0 / (D as f32).sqrt();
    let (mut worst, mut worst_coarse, mut best_unrelated) = (1.0f32, 0.0f32, 0.0f32);
    for _ in 0..4 {
        let qs: Vec<[f32; D]> = (0..32)
            .map(|_| std::array::from_fn(|d| rng.normal() + 0.3 * dir[d]))
            .collect();
        let w: Vec<f32> = (0..32).map(|_| rng.normal()).collect();
        let sel = |ks: &[[f32; D]]| {
            top(
                &ks.iter()
                    .map(|k| score(&qs, &w, scale, k))
                    .collect::<Vec<_>>(),
                SELECT,
            )
        };
        let exact = sel(&keys);
        let o = overlap(&exact, &sel(&fp8));
        let oc = overlap(&exact, &sel(&coarse));
        let ou = overlap(&exact, &sel(&unrelated));
        worst = worst.min(o);
        worst_coarse = worst_coarse.max(oc);
        best_unrelated = best_unrelated.max(ou);
        assert!(oc < o, "coarse control {oc} not below fp8 {o}");
    }
    assert!(worst >= 0.90, "fp8 top-2048 overlap {worst}");
    assert!(best_unrelated < 0.25, "unrelated control {best_unrelated}");
    eprintln!(
        "top-2048 overlap: fp8 worst {worst}, coarse best {worst_coarse}, unrelated best {best_unrelated}"
    );
}
