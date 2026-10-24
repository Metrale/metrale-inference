// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `census [rows]`: the candidate sets a 4-bit screen (one scale per 32 weights,
//! about half the INT8 copy's bytes) would leave, beside the INT8 screen's on the same rows, before any 4-bit kernel exists. The MTP drafter's head is an FP8
//! copy of the same head rows (one byte per weight, as the INT8 screen), so only a narrower copy
//! can read fewer bytes than it; this measures what the narrower copy's looser bound costs.
//!
//! Host arithmetic in f64 on the real head under the serve's split (no GPU): per weight block
//! `s = max|w| / qmax`, `q = clamp(rint(w / s))`, `S = sum_b s_b sum x q`, `E = sum_b s_b |x_b|_1`,
//! `eps = kappa E` with `kappa = 0.5 (1 + 2^-20)` plus the FP32 rounding terms at `qmax`, and a
//! row is a candidate when `S + eps >= L - pad` (`L = max (S - eps)`, the module docs of
//! `ops/screen_head.rs`). The INT8 column is a check of this host model against the gate's
//! device counts.

use anyhow::Result;
use half::bf16;

use super::rows::{self, Family, REALISTIC, Rng};
use super::{K, SPLIT, pct};

/// 2026-10-21: One quantized copy: `q` per weight, `s` per block.
struct QCopy {
    q: Vec<i8>,
    s: Vec<f64>,
    blk: usize,
    kappa: f64,
}

fn gamma(n: f64) -> f64 {
    let nu = n * 2f64.powi(-24);
    nu / (1.0 - nu)
}

fn build(head: &[bf16], qmax: f64, blk: usize) -> QCopy {
    let nb = head.len() / blk;
    let mut q = vec![0i8; head.len()];
    let mut s = vec![0f64; nb];
    for b in 0..nb {
        let w = &head[b * blk..(b + 1) * blk];
        let amax = w
            .iter()
            .map(|v| f64::from(v.to_f32()).abs())
            .fold(0.0, f64::max);
        let sc = f64::from((amax as f32) / (qmax as f32));
        s[b] = sc;
        for (j, v) in w.iter().enumerate() {
            let r = if sc > 0.0 {
                (f64::from(v.to_f32()) / sc).round_ties_even()
            } else {
                0.0
            };
            q[b * blk + j] = r.clamp(-qmax, qmax) as i8;
        }
    }
    let t = 1.0 + 2f64.powi(-20);
    let d_ref = (8 * K.div_ceil(512) + 6) as f64;
    let d_scr = (16 + K.div_ceil(512) + 6) as f64;
    let kappa = 0.5 * t + (qmax + 0.5) * t * gamma(d_ref) + qmax * gamma(d_scr);
    QCopy { q, s, blk, kappa }
}

/// 2026-10-21: Candidates (summed over the three shards) of activation row `x` under `c`.
fn candidates(c: &QCopy, x: &[f32]) -> u32 {
    let blk = c.blk;
    let nb = K / blk;
    let h: Vec<f64> = (0..nb)
        .map(|b| {
            x[b * blk..(b + 1) * blk]
                .iter()
                .map(|v| f64::from(*v).abs())
                .sum()
        })
        .collect();
    let mut total = 0u32;
    for &(begin, len) in &SPLIT {
        let threads = 16usize;
        let per = len.div_ceil(threads);
        let bounds: Vec<(f64, f64)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    let h = &h;
                    sc.spawn(move || {
                        let lo = begin + t * per;
                        let hi = (lo + per).min(begin + len);
                        (lo..hi)
                            .map(|i| {
                                let (mut sum, mut e) = (0f64, 0f64);
                                for b in 0..nb {
                                    let off = i * K + b * blk;
                                    let d: f64 = c.q[off..off + blk]
                                        .iter()
                                        .zip(&x[b * blk..(b + 1) * blk])
                                        .map(|(q, v)| f64::from(*q) * f64::from(*v))
                                        .sum();
                                    let sb = c.s[i * nb + b];
                                    sum += sb * d;
                                    e += sb * h[b];
                                }
                                (sum, c.kappa * e)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let l = bounds
            .iter()
            .map(|(s, e)| s - e)
            .fold(f64::NEG_INFINITY, f64::max);
        let pad = 2f64.powi(-7) * (1.0 + 2f64.powi(-7)) * l.abs() + 2f64.powi(-120);
        total += bounds.iter().filter(|(s, e)| s + e >= l - pad).count() as u32;
    }
    total
}

pub(crate) fn census(head: &[bf16], gamma_w: &[f32], rows: usize) -> Result<()> {
    let int8 = build(head, 127.0, 128);
    let int4 = build(head, 7.0, 32);
    eprintln!(
        "census: kappa int8 g128 {:.7} int4 g32 {:.7}",
        int8.kappa, int4.kappa
    );
    let mut g = Rng(0xce45_2026_1021);
    let mut stats: Vec<(Family, Vec<u32>, Vec<u32>)> = REALISTIC
        .iter()
        .map(|&f| (f, Vec::new(), Vec::new()))
        .collect();
    for r in 0..rows {
        let fi = r % stats.len();
        let x = rows::realistic(stats[fi].0, &mut g, head, K, gamma_w);
        stats[fi].1.push(candidates(&int8, &x));
        stats[fi].2.push(candidates(&int4, &x));
    }
    println!("census: {rows} rows, candidates summed over the three shards");
    println!(
        "{:<10} {:>6} | {:>6} {:>6} {:>6} | {:>7} {:>7} {:>7} {:>9}",
        "family", "rows", "i8 p50", "p99", "max", "i4 p50", "p99", "max", "i4 mean"
    );
    for (f, mut a, mut b) in stats {
        let mean = b.iter().map(|&v| f64::from(v)).sum::<f64>() / b.len().max(1) as f64;
        println!(
            "{:<10} {:>6} | {:>6} {:>6} {:>6} | {:>7} {:>7} {:>7} {:>9.1}",
            format!("{f:?}"),
            a.len(),
            pct(&mut a, 0.5),
            pct(&mut a, 0.99),
            pct(&mut a, 1.0),
            pct(&mut b, 0.5),
            pct(&mut b, 0.99),
            pct(&mut b, 1.0),
            mean
        );
    }
    Ok(())
}
