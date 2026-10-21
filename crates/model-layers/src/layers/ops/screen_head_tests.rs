// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Tests of the screened head's launch-shape rules and bound (`screen_head.rs`).
//! The bound tests run the claim on the host: the reference's FP32 sum in its own order, the
//! screen copy as the load-time build makes it, and the candidate rule; the argmax must always
//! be a candidate, and a bound without its quantization term must lose it on a row built to sit
//! on that term.

use super::*;

fn bf16_round(v: f32) -> f32 {
    let b = v.to_bits();
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(r)
}

/// 2026-10-21: `dense_gemv_bf16_batchm`'s FP32 sum for one row (64 lanes, 8 products per
/// 64-vector slab, shuffle-down tree per warp, then warp 0 + warp 1).
fn reference(w: &[f32], x: &[f32]) -> f32 {
    let kvec = w.len() / 8;
    let mut acc = [0f32; 64];
    for base in (0..kvec).step_by(64) {
        for (lane, a) in acc.iter_mut().enumerate() {
            let kv = base + lane;
            if kv < kvec {
                for e in 0..8 {
                    *a += x[kv * 8 + e] * w[kv * 8 + e];
                }
            }
        }
    }
    let fold = |v: &mut [f32]| {
        let mut off = 16;
        while off > 0 {
            for l in 0..32 {
                let o = if l + off < 32 { v[l + off] } else { v[l] };
                v[l] += o;
            }
            off >>= 1;
        }
        v[0]
    };
    let (a, b) = acc.split_at_mut(32);
    fold(a) + fold(b)
}

/// 2026-10-21: The load-time build of one row: `(q, s)` per 128-block.
fn quantize(w: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; w.len()];
    let mut s = vec![0f32; w.len() / 128];
    for (b, blk) in w.chunks(128).enumerate() {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        s[b] = amax / 127.0;
        for (j, v) in blk.iter().enumerate() {
            let r = if s[b] > 0.0 {
                (f64::from(*v) / f64::from(s[b])).round_ties_even() as f32
            } else {
                0.0
            };
            q[b * 128 + j] = r.clamp(-127.0, 127.0) as i8;
        }
    }
    (q, s)
}

/// 2026-10-21: `(lo, up)` of one row under `kappa`, the screen sum taken exactly (its FP32
/// error is inside kappa's rounding term).
fn bounds(w: &[f32], x: &[f32], kappa: f64) -> (f64, f64) {
    let (q, s) = quantize(w);
    let (mut sum, mut e) = (0f64, 0f64);
    for (b, sb) in s.iter().enumerate() {
        let r = b * 128..(b + 1) * 128;
        let p: f64 = r.clone().map(|k| f64::from(x[k]) * f64::from(q[k])).sum();
        let h: f64 = r.map(|k| f64::from(x[k]).abs()).sum();
        sum += f64::from(*sb) * p;
        e += f64::from(*sb) * h;
    }
    let eps = kappa * e + 2f64.powi(-100);
    (sum - eps, sum + eps)
}

/// 2026-10-21: Whether the BF16 argmax (lowest index on ties; any rule picks among equals) is a
/// candidate under `kappa`.
fn argmax_kept(rows: &[Vec<f32>], x: &[f32], kappa: f64) -> bool {
    let vals: Vec<f32> = rows.iter().map(|w| bf16_round(reference(w, x))).collect();
    let best = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let b: Vec<(f64, f64)> = rows.iter().map(|w| bounds(w, x, kappa)).collect();
    let l = b.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let pad = 2f64.powi(-7) * (1.0 + 2f64.powi(-7)) * l.abs() + 2f64.powi(-120);
    vals.iter()
        .zip(&b)
        .all(|(v, (_, up))| *v != best || *up >= l - pad)
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

#[test]
fn shapes_the_kernels_take() {
    assert!(screen_shape_ok(1, 4096) && screen_shape_ok(2, 4096) && screen_shape_ok(4, 4096));
    assert!(!screen_shape_ok(0, 4096), "no rows");
    assert!(!screen_shape_ok(5, 512), "past SCREEN_MAX_M");
    assert!(!screen_shape_ok(1, 4000), "k not a multiple of 512");
    assert!(
        !screen_shape_ok(4, 8192),
        "m * k past the shared-memory stage"
    );
    assert_eq!(scan_grid(51_648), 807);
    assert_eq!(rescore_grid(51_648), 202);
    assert_eq!((scan_grid(1), rescore_grid(1)), (1, 1));
}

#[test]
fn kappa_is_the_quantization_term_plus_small_rounding_terms() {
    let (q, r) = screen_kappa_parts(4096);
    assert!(q > 0.5 && q < 0.5 + 1e-6);
    assert!(
        r > 127.5 * 70.0 * 2f64.powi(-24),
        "reference term too small: {r}"
    );
    assert!(r < 1e-3);
    let k = screen_kappa(4096);
    assert!(f64::from(k) >= q + r, "kappa must round up");
    assert!(f64::from(k) < (q + r) * (1.0 + 1e-6));
    assert_eq!(f32_up(0.5), 0.5);
    assert!(f64::from(f32_up(0.1)) >= 0.1);
}

#[test]
fn the_argmax_is_always_a_candidate() {
    let k = 512usize;
    let kappa = f64::from(screen_kappa(k as u32));
    let mut g = Lcg(7);
    for case in 0..60 {
        let rows: Vec<Vec<f32>> = (0..40)
            .map(|_| (0..k).map(|_| bf16_round(g.f() * 0.05)).collect())
            .collect();
        let scale = [0.5f32, 4.0, 60.0][case % 3];
        let mut x: Vec<f32> = (0..k).map(|_| bf16_round(g.f() * scale)).collect();
        if case % 2 == 0 {
            // 2026-10-21: A near tie: x along rows 3 + 9, so the two lead together.
            for (j, v) in x.iter_mut().enumerate() {
                *v = bf16_round(*v * 0.1 + (rows[3][j] + rows[9][j]) * 40.0 * scale);
            }
        }
        assert!(argmax_kept(&rows, &x, kappa), "case {case}");
    }
}

#[test]
fn a_bound_without_the_quantization_term_loses_the_argmax() {
    // 2026-10-21: Scale 2^-10 in every block (an anchor weight of 127 / 1024). The leader sits
    // 0.375 of a step above its grid point along x on every weight; the other row sits 0.375 below
    // a point one step up on half the weights and 0.375 below its point on the rest. The leader is
    // ahead by 0.25 of a step per weight, the screen puts it behind by 0.5.
    let k = 512usize;
    let s = 2f32.powi(-10);
    let mut g = Lcg(11);
    let x: Vec<f32> = (0..k)
        .map(|j| {
            if j % 128 == 0 || g.f() >= 0.0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    let grid: Vec<f32> = (0..k).map(|_| (g.f() * 15.0).round()).collect();
    let row = |shift: &dyn Fn(usize) -> f32| -> Vec<f32> {
        (0..k)
            .map(|j| {
                if j % 128 == 0 {
                    127.0 * s
                } else {
                    (grid[j] + shift(j) * x[j]) * s
                }
            })
            .collect()
    };
    let leader = row(&|_| 0.375);
    let other = row(&|j| if j % 2 == 0 { 0.625 } else { -0.375 });
    for w in leader.iter().chain(&other) {
        assert_eq!(*w, bf16_round(*w), "the rows must be exact BF16");
    }
    let rows = vec![other, leader];
    let (_, rounding) = screen_kappa_parts(k as u32);
    assert!(argmax_kept(&rows, &x, f64::from(screen_kappa(k as u32))));
    assert!(
        !argmax_kept(&rows, &x, rounding),
        "the rounding terms alone must not cover a coherent quantization error"
    );
}
