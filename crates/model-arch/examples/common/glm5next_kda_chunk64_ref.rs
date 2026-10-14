// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Host f64 reference of one (chunk, head)'s WY factors for `glm5next_kda_chunk64_bench`:
//! A, P, U, W, Qg, Kd and the decay exactly as `kda_chunk64_prepare.cu` defines them, so the
//! prepare kernel is gated on its own and not only through the scan. Inputs are the bench's
//! host arrays (q/k/v already rounded to BF16).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types; `C` = 64 tokens, `D` = 128 channels.

use metrale_model_arch::glm5next_kda::chunk64::{CHUNK64 as C, CHUNK64_D as D};

/// 2026-10-12: The factors of chunk `c`, head `h` over `t` live tokens; q/k/v rows `cd` apart (q
/// at 0, k at `qkv`, v at `2 qkv`), gate rows `qkv` apart, beta rows `heads` apart.
pub struct Factors {
    pub a: Vec<f64>,
    pub p: Vec<f64>,
    pub u: Vec<f64>,
    pub w: Vec<f64>,
    pub qg: Vec<f64>,
    pub kd: Vec<f64>,
    pub decay: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn factors(
    qkv: &[f32],
    gate: &[f32],
    beta: &[f32],
    t: usize,
    heads: usize,
    c: usize,
    h: usize,
    scale: f64,
) -> Factors {
    let qkv_w = heads * D;
    let cd = 3 * qkv_w;
    let at = |i: usize, part: usize, d: usize| -> f64 {
        let tok = c * C + i;
        if tok < t {
            f64::from(qkv[tok * cd + part * qkv_w + h * D + d])
        } else {
            0.0
        }
    };
    let mut g = vec![0f64; C * D];
    let mut b = vec![0f64; C];
    for i in 0..C {
        let tok = c * C + i;
        if tok < t {
            for d in 0..D {
                g[i * D + d] = f64::from(gate[tok * qkv_w + h * D + d]);
            }
            b[i] = f64::from(beta[tok * heads + h]);
        }
    }
    for i in 1..C {
        for d in 0..D {
            g[i * D + d] += g[(i - 1) * D + d];
        }
    }
    let (mut a, mut p) = (vec![0f64; C * C], vec![0f64; C * C]);
    for i in 0..C {
        for j in 0..=i {
            let (mut sa, mut sp) = (0f64, 0f64);
            for d in 0..D {
                let kj = at(j, 1, d) * (g[i * D + d] - g[j * D + d]).exp();
                sa += at(i, 1, d) * kj;
                sp += at(i, 0, d) * kj;
            }
            if j < i {
                a[i * C + j] = sa;
            }
            p[i * C + j] = scale * sp;
        }
    }
    let (mut u, mut w) = (vec![0f64; C * D], vec![0f64; C * D]);
    for i in 0..C {
        for d in 0..D {
            let (mut su, mut sw) = (at(i, 2, d), g[i * D + d].exp() * at(i, 1, d));
            for j in 0..i {
                su -= a[i * C + j] * u[j * D + d];
                sw -= a[i * C + j] * w[j * D + d];
            }
            u[i * D + d] = b[i] * su;
            w[i * D + d] = b[i] * sw;
        }
    }
    let (mut qg, mut kd, mut decay) = (vec![0f64; C * D], vec![0f64; C * D], vec![0f64; D]);
    for d in 0..D {
        let gl = g[(C - 1) * D + d];
        decay[d] = gl.exp();
        for i in 0..C {
            qg[i * D + d] = scale * g[i * D + d].exp() * at(i, 0, d);
            kd[i * D + d] = (gl - g[i * D + d]).exp() * at(i, 1, d);
        }
    }
    Factors {
        a,
        p,
        u,
        w,
        qg,
        kd,
        decay,
    }
}

/// 2026-10-12: Relative RMS of `dev - reference`.
pub fn rel_rms(dev: &[f32], reference: &[f64]) -> f64 {
    assert_eq!(dev.len(), reference.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in dev.iter().zip(reference) {
        let d = f64::from(x) - y;
        num += d * d;
        den += y * y;
    }
    (num / den.max(1e-30)).sqrt()
}
