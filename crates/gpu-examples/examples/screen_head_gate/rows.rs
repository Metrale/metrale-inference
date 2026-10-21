// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The activation rows of the screened-head gate: realistic families (post-RMSNorm
//! rows under the checkpoint's final-norm weight) and an adversarial family built on the
//! screen's own quantization residuals.

use half::bf16;

/// 2026-10-21: A seeded LCG with a Box-Muller normal.
pub struct Rng(pub u64);

impl Rng {
    pub fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn n(&mut self) -> f64 {
        let (a, b) = (self.u(), self.u());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.u() * n as f64) as usize % n
    }
}

/// 2026-10-21: The families. `Iso` .. `BigNorm` are post-RMSNorm rows; `Adversarial` sits on the
/// screen's residuals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// 2026-10-21: An isotropic normal before the norm: flat logits, the widest candidate sets.
    Iso,
    /// 2026-10-21: Toward one token's head row, at a random strength (a confident step).
    Peaked,
    /// 2026-10-21: Toward two tokens' rows equally (a near tie between them).
    Tie2,
    /// 2026-10-21: A few channels 50x the rest (massive activations).
    Outlier,
    /// 2026-10-21: `Peaked` scaled by 64 (a large-norm row).
    BigNorm,
    /// 2026-10-21: See [`adversarial`].
    Adversarial,
}

pub const REALISTIC: [Family; 5] = [
    Family::Iso,
    Family::Peaked,
    Family::Tie2,
    Family::Outlier,
    Family::BigNorm,
];

/// 2026-10-21: `gamma * z / rms(z)` rounded to BF16, as the final RMSNorm writes it.
fn normed(z: &[f64], gamma: &[f32]) -> Vec<f32> {
    let rms = (z.iter().map(|v| v * v).sum::<f64>() / z.len() as f64)
        .sqrt()
        .max(1e-30);
    z.iter()
        .zip(gamma)
        .map(|(v, g)| bf16::from_f64(v / rms * f64::from(*g)).to_f32())
        .collect()
}

/// 2026-10-21: One row of a realistic family over the head `w` (`[v, k]`).
pub fn realistic(fam: Family, g: &mut Rng, w: &[bf16], k: usize, gamma: &[f32]) -> Vec<f32> {
    let v = w.len() / k;
    let row = |t: usize| -> Vec<f64> {
        (0..k)
            .map(|j| f64::from(w[t * k + j].to_f32()) / f64::from(gamma[j]).abs().max(1e-3))
            .collect()
    };
    let unit = |r: Vec<f64>| -> Vec<f64> {
        let n = (r.iter().map(|x| x * x).sum::<f64>() / r.len() as f64)
            .sqrt()
            .max(1e-30);
        r.into_iter().map(|x| x / n).collect()
    };
    let noise: Vec<f64> = (0..k).map(|_| g.n()).collect();
    match fam {
        Family::Iso => normed(&noise, gamma),
        Family::Peaked | Family::BigNorm => {
            let beta = 0.2 + 3.0 * g.u();
            let d = unit(row(g.below(v)));
            let z: Vec<f64> = noise.iter().zip(&d).map(|(n, d)| n + beta * d).collect();
            let x = normed(&z, gamma);
            if fam == Family::BigNorm {
                x.iter()
                    .map(|v| bf16::from_f32(v * 64.0).to_f32())
                    .collect()
            } else {
                x
            }
        }
        Family::Tie2 => {
            let beta = 1.0 + 2.0 * g.u();
            let (a, b) = (unit(row(g.below(v))), unit(row(g.below(v))));
            let z: Vec<f64> = (0..k).map(|j| noise[j] + beta * (a[j] + b[j])).collect();
            normed(&z, gamma)
        }
        Family::Outlier => {
            let mut z = noise;
            for _ in 0..6 {
                let j = g.below(k);
                z[j] *= 50.0;
            }
            normed(&z, gamma)
        }
        Family::Adversarial => unreachable!("adversarial rows come from `adversarial`"),
    }
}

/// 2026-10-21: The screen copy of one head row as the load-time build makes it: `(q, s)` per
/// 128-weight block.
pub fn quantize_row(w: &[bf16]) -> (Vec<f32>, Vec<f32>) {
    let mut q = vec![0f32; w.len()];
    let mut s = vec![0f32; w.len() / 128];
    for (b, blk) in w.chunks(128).enumerate() {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
        s[b] = amax / 127.0;
        for (j, v) in blk.iter().enumerate() {
            q[b * 128 + j] = if s[b] > 0.0 {
                (f64::from(v.to_f32()) / f64::from(s[b]))
                    .round_ties_even()
                    .clamp(-127.0, 127.0) as f32
            } else {
                0.0
            };
        }
    }
    (q, s)
}

/// 2026-10-21: A row that puts two head rows `a`, `b` on top of the vocabulary with the true
/// leader `a` just ahead, supported only on the coordinates where both rows' quantization
/// residuals are large and of opposite sign, with signs that push the screen down on `a` and up
/// on `b`. The screen errors of both rows are then near their bounds and coherent, so a bound
/// at half its size drops `a`, while the true bound must keep it.
pub fn adversarial(g: &mut Rng, w: &[bf16], k: usize, v: usize) -> Vec<f32> {
    loop {
        let (a, b) = (g.below(v), g.below(v));
        if a == b {
            continue;
        }
        let (wa, wb) = (&w[a * k..(a + 1) * k], &w[b * k..(b + 1) * k]);
        let (qa, sa) = quantize_row(wa);
        let (qb, sb) = quantize_row(wb);
        let res = |w: &[bf16], q: &[f32], s: &[f32], j: usize| {
            f64::from(w[j].to_f32()) - f64::from(s[j / 128]) * f64::from(q[j])
        };
        let mut x = vec![0f64; k];
        let mut support = Vec::new();
        for j in 0..k {
            let (ra, rb) = (res(wa, &qa, &sa, j), res(wb, &qb, &sb, j));
            let (ta, tb) = (0.3 * f64::from(sa[j / 128]), 0.3 * f64::from(sb[j / 128]));
            if ra.abs() >= ta && rb.abs() >= tb && ra.signum() != rb.signum() {
                support.push(j);
            }
        }
        if support.len() < 64 {
            continue;
        }
        // 2026-10-21: d pushes S_a = T_a - x.res_a down and S_b up; the lift puts a and b on top;
        // lambda sets T_a - T_b to a small positive margin.
        let c = 1.0;
        let wf = |w: &[bf16], j: usize| f64::from(w[j].to_f32());
        let mean_w = support
            .iter()
            .map(|&j| wf(wa, j).abs() + wf(wb, j).abs())
            .sum::<f64>()
            / (2 * support.len()) as f64;
        let mu = 0.35 * c / mean_w.max(1e-12);
        for &j in &support {
            let d = res(wa, &qa, &sa, j).signum();
            x[j] = c * d + mu * (wf(wa, j) + wf(wb, j));
        }
        let diff = |x: &[f64]| {
            support
                .iter()
                .map(|&j| x[j] * (wf(wa, j) - wf(wb, j)))
                .sum::<f64>()
        };
        let dd: f64 = support
            .iter()
            .map(|&j| (wf(wa, j) - wf(wb, j)).powi(2))
            .sum();
        let e: f64 = support
            .iter()
            .map(|&j| f64::from(sa[j / 128]) * x[j].abs())
            .sum();
        let lambda = (0.02 * e - diff(&x)) / dd.max(1e-30);
        for &j in &support {
            x[j] += lambda * (wf(wa, j) - wf(wb, j));
        }
        // 2026-10-21: Scale so the leader's logit is order 20, as a real top logit is.
        let ta: f64 = support.iter().map(|&j| x[j] * wf(wa, j)).sum();
        let scale = if ta.abs() > 1e-30 {
            20.0 / ta.abs()
        } else {
            1.0
        };
        return x
            .iter()
            .map(|v| bf16::from_f64(v * scale).to_f32())
            .collect();
    }
}
