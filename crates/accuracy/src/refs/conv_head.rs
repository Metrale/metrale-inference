// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: One (sequence, head of channels) of the conv step ([`super::conv`]) in bounded
//! arithmetic and as a conforming emulation, and the flat output index map. Rounding points are
//! lines of causal_conv1d.cu (causal_conv1d_update_l2norm_f32_strided, :496-557;
//! causal_conv1d_update_l2norm_f32, :413-472, is the same text); the kernels compile with
//! `--fmad=false` (kernels/gb10/common/KERNEL.toml), and `__expf(-a)` is
//! `ex2.approx.f32(a * -log2(e) as f32)`, `1/x` is `rcp.rn.f32`, `rsqrtf` `rsqrt.approx.f32`
//! (nvcc -O3 --fmad=false PTX).
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - A group of `head_dim` channels is computed whole (the L2 norm reads every channel), once
//!   per call.
//! - An exponent at the edge of f32's range is covered both ways (overflow to a zero sigmoid,
//!   or a finite one), never assumed.

use std::collections::BTreeMap;

use crate::bounded::{Bounded, sum};
use crate::case::Case;
use crate::elem::{self, Elem};
use crate::plan::Plan;
use crate::refs::conv::{Formats, Geom, formats};

/// 2026-10-09: `-log2(e)` as the f32 constant `__expf` multiplies by (`0fBFB8AA3B`).
fn neg_log2e() -> f64 {
    f64::from(f32::from_bits(0xBFB8_AA3B))
}

/// 2026-10-09: The declared depths and approximate-function errors.
#[derive(Debug, Clone, Copy)]
struct Decl {
    conv: u64,
    head: u64,
    ex2: f64,
    rsqrt: f64,
}

/// 2026-10-09: Read the declarations; a missing one is an error naming it.
fn decl(plan: &Plan) -> Result<Decl, String> {
    Ok(Decl {
        conv: plan.depth_of("conv")?,
        head: plan.depth_of("head")?,
        ex2: plan.approx_of("ex2")?,
        rsqrt: plan.approx_of("rsqrt")?,
    })
}

/// 2026-10-09: The operands of one channel: its updated window and its weights.
struct Chan {
    window: Vec<f64>,
    w: Vec<f64>,
}

/// 2026-10-09: Channel `ch` of row `r`, the window shifted and the input appended (:522-523).
fn chan(case: &Case, g: &Geom, r: usize, ch: usize) -> Result<Chan, String> {
    let (win, x, w) = (case.tensor("window")?, case.tensor("x")?, case.tensor("w")?);
    let base = (r * g.dim + ch) * g.d_conv;
    let mut window: Vec<f64> = (1..g.d_conv).map(|t| win.get(base + t)).collect();
    window.push(x.get(r * g.in_dim + ch));
    Ok(Chan {
        window,
        w: (0..g.d_conv).map(|t| w.get(ch * g.d_conv + t)).collect(),
    })
}

/// 2026-10-09: The sigmoid of `a` as :528 computes it: `rcp(1 + ex2(round(a * c)))`.
fn sigmoid(a: Bounded, cp: Elem, d: &Decl) -> Bounded {
    let y = a.mul(Bounded::exact(neg_log2e())).round(cp);
    // 2026-10-09: ex2.approx.f32 returns +inf from 128 on; 1 + inf is inf and rcp(inf) is 0.
    if y.v - y.e >= 128.0 {
        return Bounded::exact(0.0);
    }
    if y.v + y.e < 127.0 {
        let ln2 = Bounded {
            v: std::f64::consts::LN_2,
            e: std::f64::consts::LN_2 * elem::F64.unit_roundoff(),
        };
        let ey = y.mul(ln2).exp(d.ex2);
        // 2026-10-09: A subnormal 2^y carries the floor of the subnormal range besides `rel`.
        let ey = Bounded {
            v: ey.v,
            e: ey.e + elem::pow2(cp.emin),
        };
        let den = Bounded::exact(1.0).add(ey).round(cp);
        return Bounded::exact(1.0).div(den).round(cp);
    }
    // 2026-10-09: 2^y is near f32's largest value: the sigmoid lies in [0, 1 / (1 + 2^y_lo)].
    let lo = (y.v - y.e).exp2() * (1.0 - d.ex2);
    let hi = (1.0 / (1.0 + lo)) * (1.0 + 4.0 * cp.unit_roundoff()) + cp.min_subnormal();
    Bounded {
        v: hi / 2.0,
        e: hi / 2.0,
    }
}

/// 2026-10-09: A group's outputs: the conv output per channel, then each channel's window.
struct Out<T> {
    y: Vec<T>,
    window: Vec<Vec<T>>,
}

/// 2026-10-09: The bounded step of a group of channels, before the final stores.
fn bounded_group(cs: &[Chan], l2: bool, f: &Formats, d: &Decl, eps: f64) -> Out<Bounded> {
    let cp = f.compute;
    let x = Bounded::exact;
    let silu: Vec<Bounded> = cs
        .iter()
        .map(|c| {
            // 2026-10-09: :526-527, `acc += state[k] * w[k]` from 0 (no bias), in tap order.
            let terms: Vec<Bounded> = c
                .window
                .iter()
                .zip(&c.w)
                .map(|(s, w)| x(*s).mul(x(*w)).round(cp))
                .collect();
            let a = sum(&terms, cp, d.conv);
            // 2026-10-09: :529 `acc * sigmoid_acc`.
            a.mul(sigmoid(a, cp, d))
        })
        .collect();
    let y = if l2 {
        let stored: Vec<Bounded> = silu.iter().map(|s| s.round(cp)).collect();
        // 2026-10-09: :533-544, squares, a 32-lane tree, the head's four warps in sequence.
        let squares: Vec<Bounded> = stored.iter().map(|s| s.mul(*s).round(cp)).collect();
        let total = sum(&squares, cp, d.head);
        // 2026-10-09: :546 `rsqrtf(total + l2_eps)`; :550 `silu *=` is the store's rounding.
        let r = total.add(x(eps)).round(cp).rsqrt(d.rsqrt);
        stored.iter().map(|s| s.mul(r)).collect()
    } else {
        silu
    };
    Out {
        y,
        window: cs
            .iter()
            .map(|c| c.window.iter().map(|v| x(*v)).collect())
            .collect(),
    }
}

/// 2026-10-09: RNE into `e`, overflowing to a signed infinity as IEEE arithmetic does.
fn rnd(e: Elem, v: f64) -> f64 {
    e.round(v).unwrap_or(if v.is_nan() {
        v
    } else {
        f64::INFINITY.copysign(v)
    })
}

/// 2026-10-09: A conforming emulation of a group: the kernel's roundings, sums in `acc`
/// bracketed per [`crate::emulate::reduce`] `variant`, `ex2` and `rsqrt` correctly rounded.
fn emulated_group(
    cs: &[Chan],
    l2: bool,
    f: &Formats,
    d: &Decl,
    eps: f64,
    acc: Elem,
    v: u32,
) -> Out<f64> {
    let cp = f.compute;
    let silu: Vec<f64> = cs
        .iter()
        .map(|c| {
            let terms: Vec<f64> = c
                .window
                .iter()
                .zip(&c.w)
                .map(|(s, w)| rnd(cp, s * w))
                .collect();
            let a = crate::emulate::reduce(&terms, acc, d.conv, v);
            let ey = rnd(cp, rnd(cp, a * neg_log2e()).exp2());
            let den = rnd(cp, 1.0 + ey);
            let sig = if den.is_infinite() {
                0.0
            } else {
                rnd(cp, 1.0 / den)
            };
            rnd(cp, a * sig)
        })
        .collect();
    let y = if l2 {
        let squares: Vec<f64> = silu.iter().map(|s| rnd(cp, s * s)).collect();
        let total = crate::emulate::reduce(&squares, acc, d.head, v);
        let r = rnd(cp, 1.0 / rnd(cp, total + eps).sqrt());
        silu.iter().map(|s| rnd(f.out, s * r)).collect()
    } else {
        silu.iter().map(|s| rnd(f.out, *s)).collect()
    };
    Out {
        y,
        window: cs.iter().map(|c| c.window.clone()).collect(),
    }
}

/// 2026-10-09: Where flat output index `i` lands: (row, channel group, element): element
/// `< head_dim` is that channel's conv output, else `head_dim + c * d_conv + t` its window.
fn locate(g: &Geom, i: usize) -> Result<(usize, usize, usize), String> {
    let cols = g.cols();
    let (r, c) = (i / cols, i % cols);
    if r >= g.rows {
        return Err(format!("index {i} beyond {} rows", g.rows));
    }
    let hd = g.head_dim;
    Ok(if c < g.dim {
        (r, c / hd, c % hd)
    } else {
        let (ch, t) = ((c - g.dim) / g.d_conv, (c - g.dim) % g.d_conv);
        (r, ch / hd, hd + (ch % hd) * g.d_conv + t)
    })
}

/// 2026-10-09: `f(channels, normalized)` per distinct (row, group) of `idx`, then each index's
/// element. A group is `head_dim` channels; the last may be shorter.
fn gather<T: Copy>(
    case: &Case,
    idx: &[usize],
    f: impl Fn(&[Chan], bool) -> Out<T>,
) -> Result<Vec<T>, String> {
    let g = Geom::of_case(case)?;
    let hd = g.head_dim;
    let mut groups: BTreeMap<(usize, usize), Out<T>> = BTreeMap::new();
    let mut out = Vec::with_capacity(idx.len());
    for &i in idx {
        let (r, grp, e) = locate(&g, i)?;
        let o = match groups.entry((r, grp)) {
            std::collections::btree_map::Entry::Occupied(o) => o.into_mut(),
            std::collections::btree_map::Entry::Vacant(v) => {
                let chans = (grp * hd..((grp + 1) * hd).min(g.dim))
                    .map(|ch| chan(case, &g, r, ch))
                    .collect::<Result<Vec<_>, _>>()?;
                v.insert(f(&chans, grp * hd < g.qk_channels))
            }
        };
        out.push(if e < hd {
            o.y[e]
        } else {
            let k = e - hd;
            o.window[k / g.d_conv][k % g.d_conv]
        });
    }
    Ok(out)
}

/// 2026-10-09: The bounded reference at flat output indices `idx`.
pub fn reference(case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
    let (f, d) = (formats(plan)?, decl(plan)?);
    let eps = case.scalar("eps")?;
    gather(case, idx, |cs, l2| bounded_group(cs, l2, &f, &d, eps))
}

/// 2026-10-09: The conforming emulation at `idx`, the sums in `acc` when given.
pub fn emulate(
    case: &Case,
    plan: &Plan,
    acc: Option<Elem>,
    variant: u32,
    idx: &[usize],
) -> Result<Vec<f64>, String> {
    let (f, d) = (formats(plan)?, decl(plan)?);
    let eps = case.scalar("eps")?;
    let acc = acc.unwrap_or(f.compute);
    gather(case, idx, |cs, l2| {
        emulated_group(cs, l2, &f, &d, eps, acc, variant)
    })
}
