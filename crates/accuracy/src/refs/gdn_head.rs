// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: One (sequence, value head) of the gated delta rule step ([`super::gdn`]), in
//! bounded arithmetic and as a conforming emulation, and the flat output index map. Every
//! rounding point below is a line of gated_delta_rule.cu (gated_delta_rule_decode_f32_strided,
//! :696-813; gated_delta_rule_decode_f32, :233-346, is the same text); the kernels compile with
//! `--fmad=false` (kernels/gb10/common/KERNEL.toml), so each product and each sum rounds.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - A head is computed whole (the clamp reads every column's state), once per call.
//! - The clamp decision is taken on the bound: certain either way, or both outcomes covered.

use std::collections::BTreeMap;

use crate::bounded::{Bounded, sum};
use crate::case::Case;
use crate::elem::Elem;
use crate::plan::Plan;
use crate::refs::gdn::{Formats, Geom, STATE_MAX_NORM, decay_clamp, formats};

/// 2026-10-09: The depths the step's reductions are declared with: the q/k dot products
/// (`k`), and the clamp's sum of squares (`state_k` within a thread, then `state_v` across).
#[derive(Debug, Clone, Copy)]
struct Depths {
    k: u64,
    norm: u64,
}

/// 2026-10-09: The declared depths; a missing reduction is an error naming it.
fn depths(plan: &Plan) -> Result<Depths, String> {
    Ok(Depths {
        k: plan.depth_of("k")?,
        norm: plan.depth_of("state_k")? + plan.depth_of("state_v")?,
    })
}

/// 2026-10-09: The operands of one (row, value head), exactly.
struct Head {
    g: f64,
    beta: f64,
    k: Vec<f64>,
    q: Vec<f64>,
    v: Vec<f64>,
    s: Vec<f64>,
}

/// 2026-10-09: Read one (row, value head)'s operands, the decay clamped as the kernel does.
fn head(case: &Case, geo: &Geom, r: usize, vh: usize) -> Result<Head, String> {
    let (kd, vd) = (geo.k_dim, geo.v_dim);
    let kh = vh / (geo.v_heads / geo.k_heads);
    let at = |name: &str, from: usize, n: usize| -> Result<Vec<f64>, String> {
        let t = case.tensor(name)?;
        Ok((from..from + n).map(|i| t.get(i)).collect())
    };
    let qk = (r * geo.k_heads + kh) * kd;
    let (lo, hi) = decay_clamp();
    Ok(Head {
        g: case.tensor("gate")?.get(r * geo.v_heads + vh).clamp(lo, hi),
        beta: case.tensor("beta")?.get(r * geo.v_heads + vh),
        k: at("k", qk, kd)?,
        q: at("q", qk, kd)?,
        v: at("v", (r * geo.v_heads + vh) * vd, vd)?,
        s: at("state", (r * geo.v_heads + vh) * kd * vd, kd * vd)?,
    })
}

/// 2026-10-09: A head's outputs: `o` per value column, then the new state `[k_dim, v_dim]`.
struct Out<T> {
    o: Vec<T>,
    s: Vec<T>,
}

/// 2026-10-09: The bounded step of one head, before the kernel's final roundings (the stores
/// of `o` and of the state).
fn bounded_head(h: &Head, f: &Formats, d: Depths, rsqrt: f64) -> Result<Out<Bounded>, String> {
    let (kd, vd) = (h.k.len(), h.v.len());
    let x = Bounded::exact;
    let cp = f.compute;
    let mut o = Vec::with_capacity(vd);
    let mut pre = vec![x(0.0); kd * vd];
    let mut stored = vec![x(0.0); kd * vd];
    // 2026-10-09: :811 `rsqrtf((float)k_dim)`.
    let inv = x(kd as f64).rsqrt(rsqrt);
    for c in 0..vd {
        // 2026-10-09: :743-751, hk_dot: each product rounds, four per step, then into the sum.
        let terms: Vec<Bounded> = (0..kd)
            .map(|j| x(h.s[j * vd + c]).mul(x(h.k[j])).round(cp))
            .collect();
        let hk = sum(&terms, cp, d.k);
        // 2026-10-09: :753 `(v_i - g * hk_dot) * bt`.
        let u = x(h.v[c])
            .sub(x(h.g).mul(hk).round(cp))
            .round(cp)
            .mul(x(h.beta))
            .round(cp);
        let mut qt = Vec::with_capacity(kd);
        for j in 0..kd {
            // 2026-10-09: :765-772 `g * h + k * v_new`, stored; :773 reads the stored value.
            let p = x(h.g)
                .mul(x(h.s[j * vd + c]))
                .round(cp)
                .add(x(h.k[j]).mul(u).round(cp));
            let st = p.round(f.state);
            pre[j * vd + c] = p;
            stored[j * vd + c] = st;
            qt.push(st.mul(x(h.q[j])).round(cp));
        }
        // 2026-10-09: :773 q_dot, as hk_dot; :812 `q_dot * inv_sqrt_d` is the store's rounding.
        o.push(sum(&qt, cp, d.k).mul(inv));
    }
    // 2026-10-09: :780-797, the sum of squares of the stored state over the head.
    let squares: Vec<Bounded> = stored.iter().map(|s| s.mul(*s).round(cp)).collect();
    let norm = sum(&squares, cp, d.norm);
    if norm.mag() >= f.compute.max_finite {
        return Err(format!(
            "the state norm^2 reaches {:e}: f32 overflows and the clamp zeroes the state",
            norm.mag()
        ));
    }
    let limit = STATE_MAX_NORM * STATE_MAX_NORM;
    // 2026-10-09: :803 `SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq)`; :805 `H *= scale` is the
    // store's rounding.
    let clamped = || -> Vec<Bounded> {
        let scale = x(STATE_MAX_NORM).mul(norm.rsqrt(rsqrt)).round(cp);
        stored.iter().map(|s| s.mul(scale)).collect()
    };
    let s = if norm.v - norm.e > limit {
        clamped()
    } else if norm.v + norm.e <= limit {
        pre
    } else {
        // 2026-10-09: `head_norm_sq > 1e6` may go either way: cover both outcomes.
        pre.iter()
            .zip(clamped())
            .map(|(a, b)| Bounded {
                v: a.v,
                e: a.e.max((b.v - a.v).abs() + b.e),
            })
            .collect()
    };
    Ok(Out { o, s })
}

/// 2026-10-09: RNE into `e`, overflowing to a signed infinity as IEEE arithmetic does.
fn rnd(e: Elem, v: f64) -> f64 {
    e.round(v).unwrap_or(if v.is_nan() {
        v
    } else {
        f64::INFINITY.copysign(v)
    })
}

/// 2026-10-09: A conforming emulation of one head: the kernel's roundings, its reductions in
/// `acc` bracketed per [`crate::emulate::reduce`] `variant`, `rsqrtf` correctly rounded.
fn emulated_head(h: &Head, f: &Formats, d: Depths, acc: Elem, variant: u32) -> Out<f64> {
    let (kd, vd) = (h.k.len(), h.v.len());
    let cp = f.compute;
    let red = |t: &[f64], depth: u64| crate::emulate::reduce(t, acc, depth, variant);
    let inv = rnd(cp, 1.0 / (kd as f64).sqrt());
    let mut o = Vec::with_capacity(vd);
    let mut st = vec![0.0; kd * vd];
    for c in 0..vd {
        let terms: Vec<f64> = (0..kd).map(|j| rnd(cp, h.s[j * vd + c] * h.k[j])).collect();
        let hk = red(&terms, d.k);
        let u = rnd(cp, rnd(cp, h.v[c] - rnd(cp, h.g * hk)) * h.beta);
        let mut qt = Vec::with_capacity(kd);
        for j in 0..kd {
            let s = rnd(
                f.state,
                rnd(cp, h.g * h.s[j * vd + c]) + rnd(cp, h.k[j] * u),
            );
            st[j * vd + c] = s;
            qt.push(rnd(cp, s * h.q[j]));
        }
        o.push(rnd(f.out, red(&qt, d.k) * inv));
    }
    let squares: Vec<f64> = st.iter().map(|s| rnd(cp, s * s)).collect();
    let norm = red(&squares, d.norm);
    let s = if norm > STATE_MAX_NORM * STATE_MAX_NORM {
        let scale = rnd(cp, STATE_MAX_NORM * rnd(cp, 1.0 / norm.sqrt()));
        st.iter().map(|s| rnd(f.state, s * scale)).collect()
    } else {
        st
    };
    Out { o, s }
}

/// 2026-10-09: Where flat output index `i` lands: (row, value head, element of that head's
/// `o` (`< v_dim`) or of its state (`v_dim + j * v_dim + c`)).
fn locate(geo: &Geom, i: usize) -> Result<(usize, usize, usize), String> {
    let cols = geo.cols();
    let (r, c) = (i / cols, i % cols);
    if r >= geo.rows {
        return Err(format!("index {i} beyond {} rows", geo.rows));
    }
    let per = geo.k_dim * geo.v_dim;
    Ok(if c < geo.o_len() {
        (r, c / geo.v_dim, c % geo.v_dim)
    } else {
        let s = c - geo.o_len();
        (r, s / per, geo.v_dim + s % per)
    })
}

/// 2026-10-09: `f(head)` per distinct (row, head) of `idx`, then each index's element.
fn gather<T: Copy>(
    case: &Case,
    idx: &[usize],
    f: impl Fn(&Head) -> Result<Out<T>, String>,
) -> Result<Vec<T>, String> {
    let geo = Geom::of_case(case)?;
    let mut heads: BTreeMap<(usize, usize), Out<T>> = BTreeMap::new();
    let mut out = Vec::with_capacity(idx.len());
    for &i in idx {
        let (r, vh, e) = locate(&geo, i)?;
        let h = match heads.entry((r, vh)) {
            std::collections::btree_map::Entry::Occupied(o) => o.into_mut(),
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(f(&head(case, &geo, r, vh)?)?)
            }
        };
        out.push(if e < geo.v_dim {
            h.o[e]
        } else {
            h.s[e - geo.v_dim]
        });
    }
    Ok(out)
}

/// 2026-10-09: The bounded reference at flat output indices `idx`.
pub fn reference(case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
    let (f, d) = (formats(plan)?, depths(plan)?);
    let rsqrt = plan.approx_of("rsqrt")?;
    gather(case, idx, |h| bounded_head(h, &f, d, rsqrt))
}

/// 2026-10-09: The conforming emulation at `idx`, the reductions in `acc` when given.
pub fn emulate(
    case: &Case,
    plan: &Plan,
    acc: Option<Elem>,
    variant: u32,
    idx: &[usize],
) -> Result<Vec<f64>, String> {
    let (f, d) = (formats(plan)?, depths(plan)?);
    let acc = acc.unwrap_or(f.compute);
    gather(case, idx, |h| Ok(emulated_head(h, &f, d, acc, variant)))
}
