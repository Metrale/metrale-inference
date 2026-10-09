// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The RoPE reference (rope.cu `rope_forward`): in every Q and K head, the channel
//! pairs `(i, i + rotary_dim/2)`, `i < rotary_dim/2`, rotate by `angle_i = pos * freq_i`,
//! `freq_i = theta^(-2i / rotary_dim)`; channels at and past `rotary_dim` are untouched. In
//! bounded arithmetic following the source's steps:
//! - `freq_i = f32(1 / pow(theta, 2i / rotary_dim))`, the quotient, the power and its
//!   reciprocal in f64 (rope.cu:69-72; `pow` is the CUDA double-precision library function, its
//!   error the contract's `approx.pow`);
//! - `angle = f32(f32(pos) * freq_i)` (:84-85; positions below 2^24 convert exactly);
//! - `cosf`, `sinf` (:86-87), the accurate library functions (no fast-math flag), their error
//!   the contract's `approx.cos` / `approx.sin`, relative to the rounded angle;
//! - `y0 = x0 cos - x1 sin`, `y1 = x1 cos + x0 sin`, each product and the sum rounded in the
//!   compute precision (:109-110, `--fmad=false`).
//!
//! Tensors of a rope case: `q` `[rows, q_heads * head_dim]`, `k` `[rows, kv_heads * head_dim]`
//! (bf16), `pos` `[rows]` (i32). Scalars: `theta` (an f32 value), `head_dim`, `rotary_dim`.
//! Output `[rows, (q_heads + kv_heads) * head_dim]`: each row's rotated Q heads, then its K
//! heads.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - The angle's own rounding is part of the declared computation: the bound grows with the
//!   position, so the case draws positions from a serving context ([`MAX_POSITION`]).

use metrale_circuit::format::Format;
use metrale_circuit::pipeline::{StepKind, Value};

use crate::bounded::Bounded;
use crate::case::{Case, Enc, Tensor};
use crate::elem::{self, Elem};
use crate::inputs::{InputClass, SplitMix64, tensor};
use crate::plan::Plan;
use crate::points::Shape;
use crate::refs::linear::num;
use crate::refs::norm::HEAD_DIM;

/// 2026-10-09: The rotary width a case runs at: the swept checkpoints' (Qwen3.5, Qwen3.6,
/// Qwen3.8) `partial_rotary_factor` 0.25 of [`HEAD_DIM`], a config parameter the sweep's shape
/// does not carry.
pub const ROTARY_DIM: usize = 64;

/// 2026-10-09: The rope base a case passes: the swept checkpoints' `rope_theta`.
pub const THETA: f64 = 1.0e7;

/// 2026-10-09: The K heads a case rotates beside its Q heads (the sweep's shape carries the Q
/// width only): Qwen3.6-35B-A3B's two. Each head is a block of its own, so the count sizes the
/// grid, not the arithmetic.
pub const KV_HEADS: usize = 2;

/// 2026-10-09: Positions are drawn below this: a 32k serving context.
pub const MAX_POSITION: u64 = 1 << 15;

/// 2026-10-09: The sizes of a rope case.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    rows: usize,
    nq: usize,
    hd: usize,
    rot: usize,
    cols: usize,
}

/// 2026-10-09: The layout of `case`.
pub fn layout(case: &Case) -> Result<Layout, String> {
    let (q, k) = (case.tensor("q")?, case.tensor("k")?);
    let hd = case.scalar("head_dim")? as usize;
    let rot = case.scalar("rotary_dim")? as usize;
    if hd == 0 || rot == 0 || !rot.is_multiple_of(2) || rot > hd {
        return Err(format!("rotary_dim {rot} of head_dim {hd}"));
    }
    let (rows, qw, kw) = (q.dims[0], q.dims[1], k.dims[1]);
    if k.dims[0] != rows || !qw.is_multiple_of(hd) || !kw.is_multiple_of(hd) {
        return Err(format!(
            "q {:?} and k {:?} with head_dim {hd}",
            q.dims, k.dims
        ));
    }
    Ok(Layout {
        rows,
        nq: qw / hd,
        hd,
        rot,
        cols: qw + kw,
    })
}

impl Layout {
    /// 2026-10-09: Column strides whose edges a sample covers: the heads (the Q/K seam is one)
    /// and the rotary half (the pair seam).
    pub fn strides(&self) -> Vec<usize> {
        vec![self.hd, self.rot / 2]
    }
}

/// 2026-10-09: The compute precision and output format of a rope plan.
fn formats(plan: &Plan) -> Result<(Elem, Elem), String> {
    let p = &plan.pipeline;
    if p.inputs
        .iter()
        .chain(&p.outputs)
        .any(|f| *f != Format::Bf16)
        || p.outputs.len() != 2
    {
        return Err("the rope reference reads and writes bf16 Q and K".into());
    }
    match plan.step(StepKind::Compute) {
        Some(Value::Num(n)) => Ok((num(*n, plan.ftz), elem::BF16)),
        _ => Err("the pipeline has no `compute` step".into()),
    }
}

/// 2026-10-09: Fill `case` with a rope launch at `shape` (Q width `in_dim`), Q and K drawn for
/// `class`, positions uniform below [`MAX_POSITION`].
pub fn fill(
    case: &mut Case,
    plan: &Plan,
    shape: &Shape,
    class: InputClass,
    stream: &dyn Fn(&str) -> SplitMix64,
) -> Result<(), String> {
    formats(plan)?;
    let (rows, qw) = (shape.rows as usize, shape.in_dim as usize);
    if rows == 0 || qw == 0 || !qw.is_multiple_of(HEAD_DIM) {
        return Err(format!(
            "a rope over {rows} rows of {qw}: not whole heads of {HEAD_DIM}"
        ));
    }
    let kw = KV_HEADS * HEAD_DIM;
    let q = tensor(&mut stream("q"), class, rows, qw, 1.0, elem::BF16);
    let k = tensor(&mut stream("k"), class, rows, kw, 1.0, elem::BF16);
    let mut rp = stream("pos");
    let pos: Vec<f64> = (0..rows).map(|_| rp.below(MAX_POSITION) as f64).collect();
    case.tensors
        .insert("q".into(), Tensor::encode(Enc::Bf16, vec![rows, qw], &q)?);
    case.tensors
        .insert("k".into(), Tensor::encode(Enc::Bf16, vec![rows, kw], &k)?);
    case.tensors
        .insert("pos".into(), Tensor::encode(Enc::I32, vec![rows], &pos)?);
    let theta = elem::F32.round(THETA).ok_or("theta is not an f32 value")?;
    case.scalars.insert("theta".into(), theta);
    case.scalars.insert("head_dim".into(), HEAD_DIM as f64);
    case.scalars.insert("rotary_dim".into(), ROTARY_DIM as f64);
    case.out = (vec![rows, qw + kw], Enc::Bf16);
    Ok(())
}

/// 2026-10-09: What output element `(r, c)` is: an untouched channel, or the first (`y0`) or
/// second (`y1`) channel of pair `p` with its operands `(x0, x1)`.
enum Elt {
    /// 2026-10-09: A channel at or past `rotary_dim`, returned as read.
    Pass(f64),
    /// 2026-10-09: A rotated channel.
    Rot {
        p: usize,
        x0: f64,
        x1: f64,
        second: bool,
    },
}

/// 2026-10-09: Output element `(r, c)` of `case`.
fn elt(case: &Case, l: &Layout, r: usize, c: usize) -> Result<Elt, String> {
    if r >= l.rows || c >= l.cols {
        return Err(format!("({r}, {c}) beyond [{}, {}]", l.rows, l.cols));
    }
    let qw = l.nq * l.hd;
    let (t, width, col) = if c < qw {
        (case.tensor("q")?, qw, c)
    } else {
        (case.tensor("k")?, l.cols - qw, c - qw)
    };
    let (head, d) = (col / l.hd, col % l.hd);
    let at = |dd: usize| t.get(r * width + head * l.hd + dd);
    let half = l.rot / 2;
    Ok(if d >= l.rot {
        Elt::Pass(at(d))
    } else if d < half {
        Elt::Rot {
            p: d,
            x0: at(d),
            x1: at(d + half),
            second: false,
        }
    } else {
        Elt::Rot {
            p: d - half,
            x0: at(d - half),
            x1: at(d),
            second: true,
        }
    })
}

/// 2026-10-09: Row `r`'s position, refused unless the kernel converts it to f32 exactly.
fn position(case: &Case, r: usize) -> Result<f64, String> {
    let p = case.tensor("pos")?.get(r);
    if !(0.0..(1u64 << 24) as f64).contains(&p) {
        return Err(format!("position {p} does not convert to f32 exactly"));
    }
    Ok(p)
}

/// 2026-10-09: The bounded reference at flat output indices `idx` (`r * cols + c`), before the
/// kernel's final rounding into bf16.
pub fn reference(case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
    let (c32, _) = formats(plan)?;
    let l = layout(case)?;
    let theta = case.scalar("theta")?;
    let (rel_pow, rel_cos, rel_sin) = (
        plan.approx_of("pow")?,
        plan.approx_of("cos")?,
        plan.approx_of("sin")?,
    );
    // 2026-10-09: ln(theta) in f64 is itself within an ulp of the real value.
    let ln_t = Bounded {
        v: theta.ln(),
        e: theta.ln().abs() * elem::F64.unit_roundoff() * 2.0,
    };
    let freqs: Vec<Bounded> = (0..l.rot / 2)
        .map(|p| {
            let fe = Bounded::exact(2.0 * p as f64)
                .div(Bounded::exact(l.rot as f64))
                .round(elem::F64);
            let pow = fe.mul(ln_t).exp(rel_pow);
            Bounded::exact(1.0).div(pow).round(elem::F64).round(c32)
        })
        .collect();
    // 2026-10-09: A library sine or cosine within 2 ulp of a subnormal result is off by up to
    // twice the subnormal spacing: four of the format's rounding floors.
    let trig = |b: Bounded, f: fn(f64) -> f64, rel: f64| {
        let t = b.lipschitz(f, 1.0, rel);
        Bounded {
            v: t.v,
            e: t.e + 4.0 * c32.underflow_floor(),
        }
    };
    let mut out = Vec::with_capacity(idx.len());
    for &i in idx {
        let (r, c) = (i / l.cols, i % l.cols);
        match elt(case, &l, r, c)? {
            Elt::Pass(x) => out.push(Bounded::exact(x)),
            Elt::Rot { p, x0, x1, second } => {
                let angle = Bounded::exact(position(case, r)?).mul(freqs[p]).round(c32);
                let (cs, sn) = (
                    trig(angle, f64::cos, rel_cos),
                    trig(angle, f64::sin, rel_sin),
                );
                let prod = |x: f64, t: Bounded| Bounded::exact(x).mul(t).round(c32);
                let y = if second {
                    prod(x1, cs).add(prod(x0, sn))
                } else {
                    prod(x0, cs).sub(prod(x1, sn))
                };
                out.push(y.round(c32));
            }
        }
    }
    Ok(out)
}

/// 2026-10-09: A conforming emulation at `idx`, in bf16: the source's steps with `pow`, `cos`
/// and `sin` correctly rounded. The rotation's two products and their sum are its only
/// accumulation: with `acc` set they run in `acc` (the `accumulate:<fmt>` arm).
pub fn emulate(
    case: &Case,
    plan: &Plan,
    acc: Option<Elem>,
    idx: &[usize],
) -> Result<Vec<f64>, String> {
    let (c32, out) = formats(plan)?;
    let a = acc.unwrap_or(c32);
    let l = layout(case)?;
    let theta = case.scalar("theta")?;
    let rnd = |e: Elem, v: f64| e.round_saturating(v).unwrap_or(f64::NAN);
    let freqs: Vec<f64> = (0..l.rot / 2)
        .map(|p| rnd(c32, 1.0 / theta.powf(2.0 * p as f64 / l.rot as f64)))
        .collect();
    let mut ys = Vec::with_capacity(idx.len());
    for &i in idx {
        let (r, c) = (i / l.cols, i % l.cols);
        let y = match elt(case, &l, r, c)? {
            Elt::Pass(x) => x,
            Elt::Rot { p, x0, x1, second } => {
                let angle = rnd(c32, position(case, r)? * freqs[p]);
                let (cs, sn) = (rnd(c32, angle.cos()), rnd(c32, angle.sin()));
                let prod = |x: f64, t: f64| rnd(a, x * t);
                if second {
                    rnd(a, prod(x1, cs) + prod(x0, sn))
                } else {
                    rnd(a, prod(x0, cs) - prod(x1, sn))
                }
            }
        };
        ys.push(rnd(out, y));
    }
    Ok(ys)
}
