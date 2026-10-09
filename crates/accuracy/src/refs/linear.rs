// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The projection reference (linear, lm_head, router, expert projections run as one
//! weight): `y[r, c] = sum_k x[r, k] * w[c, k]` in bounded arithmetic, following the family's
//! declared pipeline: the stored formats of both operands, the operand precision of the
//! weight, the product precision, the accumulator, the scale step and the output format.
//!
//! Tensors of a linear case: `x` `[rows, k]` (+ `x_block`, `x_row`, scalar `x_global`), `w`
//! `[n, k]` (+ `w_block`, `w_row`, scalar `w_global`). Output `[rows, n]`.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - Operands are decoded exactly from the stored bytes.
//! - Scale placement the pipeline does not state (a per-tensor or per-row scale applied inside
//!   each group's scale, or after the reduction) is covered both ways: the bound applies the
//!   rounding at both places.

use metrale_circuit::format::Format;
use metrale_circuit::pipeline::{Num, StepKind, Value};

use crate::bounded::{Bounded, sum};
use crate::case::{Case, Enc, Tensor};
use crate::contract::ScaleFold;
use crate::elem::{self, Elem, F32_FTZ};
use crate::inputs::{InputClass, SplitMix64, tensor};
use crate::plan::Plan;
use crate::refs::quant::{Layout, Stored, quantize};

/// 2026-10-09: Distinct weight rows drawn before tiling: prime, so no shard or tile edge (a
/// multiple of 64) maps a row onto its neighbour's values.
pub const ROW_PERIOD: usize = 127;

/// 2026-10-09: The sizes of one projection launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dims {
    /// 2026-10-09: Rows (tokens).
    pub rows: usize,
    /// 2026-10-09: Reduced width.
    pub k: usize,
    /// 2026-10-09: Output width.
    pub n: usize,
}

/// 2026-10-09: The formats of a linear plan: (activation as stored, weight as stored, weight
/// operand precision, product precisions, accumulator, scale precision, output).
struct Formats {
    act: Layout,
    weight: Layout,
    operand: Elem,
    mma: (Elem, Elem),
    acc: Elem,
    scale: Option<Elem>,
    out: Elem,
}

fn num(n: Num, ftz: bool) -> Elem {
    let e = elem::of_num(n);
    if ftz && n == Num::F32 { F32_FTZ } else { e }
}

fn formats(plan: &Plan) -> Result<Formats, String> {
    let p = &plan.pipeline;
    let input = *p.inputs.first().ok_or("the pipeline names no input")?;
    let act_step = match plan.step(StepKind::Act) {
        Some(Value::Format(f)) => *f,
        _ => return Err("the pipeline has no `act` step".into()),
    };
    if act_step != input {
        return Err(format!(
            "in-kernel activation quantization ({} read, {} multiplied) needs the quantizing reference",
            input.name(),
            act_step.name()
        ));
    }
    let (stored, operand) = match plan.step(StepKind::Weight) {
        Some(Value::Weight { stored, operand }) => (*stored, *operand),
        _ => return Err("the pipeline has no `weight` step".into()),
    };
    let (a, b) = match plan.step(StepKind::Mma) {
        Some(Value::Mma { a, b }) => (*a, *b),
        _ => return Err("the pipeline has no `mma` step".into()),
    };
    let acc = match plan.step(StepKind::Accumulate) {
        Some(Value::Num(n)) => num(*n, plan.ftz),
        _ => return Err("the pipeline has no `accumulate` step".into()),
    };
    let scale = match plan.step(StepKind::Scale) {
        Some(Value::OptNum(o)) => o.map(|n| num(n, plan.ftz)),
        _ => return Err("the pipeline has no `scale` step".into()),
    };
    let out = match p.outputs.first() {
        Some(Format::Bf16) => elem::BF16,
        Some(Format::F32) => {
            if plan.ftz {
                F32_FTZ
            } else {
                elem::F32
            }
        }
        Some(o) => return Err(format!("a linear output in {}", o.name())),
        None => return Err("the pipeline names no output".into()),
    };
    Ok(Formats {
        act: Layout::of(input).ok_or("activation format")?,
        weight: Layout::of(stored).ok_or("weight format")?,
        operand: num(operand, plan.ftz),
        mma: (num(a, plan.ftz), num(b, plan.ftz)),
        acc,
        scale,
        out,
    })
}

fn put(case: &mut Case, prefix: &str, s: Stored) {
    case.tensors.insert(prefix.to_string(), s.values);
    if let Some(b) = s.block {
        case.tensors.insert(format!("{prefix}_block"), b);
        case.scalars
            .insert(format!("{prefix}_block_rows"), s.block_rows as f64);
    }
    if let Some(r) = s.row {
        case.tensors.insert(format!("{prefix}_row"), r);
    }
    case.scalars.insert(format!("{prefix}_global"), s.global);
}

/// 2026-10-09: Fill `case` with the operands of a linear launch of `dims`, drawn for `class`.
pub fn fill(
    case: &mut Case,
    plan: &Plan,
    dims: Dims,
    class: InputClass,
    rx: &mut SplitMix64,
    rw: &mut SplitMix64,
) -> Result<(), String> {
    let f = formats(plan)?;
    let real_fmt = elem::F32;
    let x = tensor(rx, class, dims.rows, dims.k, 1.0, real_fmt);
    let round_into = |l: Layout, v: Vec<f64>| -> Vec<f64> {
        match l {
            Layout::Bf16 => v
                .iter()
                .map(|x| elem::BF16.round_saturating(*x).unwrap_or(0.0))
                .collect(),
            _ => v,
        }
    };
    // 2026-10-09: The class shapes the activations; weights are drawn as a checkpoint holds
    // them (gaussian, then quantized into the stored layout), as a period of distinct rows
    // tiled to N (a lm_head's 248k rows are not drawn one by one).
    let period = match f.weight {
        Layout::Fp8Block(r, _) => (2 * r).min(dims.n.next_multiple_of(r)),
        _ => ROW_PERIOD.min(dims.n),
    };
    let w = tensor(
        rw,
        InputClass::Gaussian,
        period,
        dims.k,
        1.0 / (dims.k as f64).sqrt(),
        real_fmt,
    );
    put(
        case,
        "x",
        quantize(f.act, &round_into(f.act, x), dims.rows, dims.k, true)?,
    );
    let stored = quantize(f.weight, &round_into(f.weight, w), period, dims.k, false)?;
    put(case, "w", stored.tile(dims.n)?);
    case.out = (
        vec![dims.rows, dims.n],
        if f.out == elem::BF16 {
            Enc::Bf16
        } else {
            Enc::F32
        },
    );
    Ok(())
}

struct Operand<'a> {
    values: &'a Tensor,
    block: Option<&'a Tensor>,
    block_rows: usize,
    row: Option<&'a Tensor>,
    global: f64,
}

fn operand<'a>(case: &'a Case, p: &str) -> Result<Operand<'a>, String> {
    Ok(Operand {
        values: case.tensor(p)?,
        block: case.tensors.get(&format!("{p}_block")),
        block_rows: case
            .scalars
            .get(&format!("{p}_block_rows"))
            .map_or(1, |v| *v as usize),
        row: case.tensors.get(&format!("{p}_row")),
        global: case.scalar(&format!("{p}_global"))?,
    })
}

impl Operand<'_> {
    fn block_scale(&self, r: usize, c: usize, group: usize) -> f64 {
        self.block.map_or(1.0, |t| {
            t.get((r / self.block_rows) * t.dims[1] + c / group)
        })
    }
    fn row_scale(&self, r: usize) -> f64 {
        self.row.map_or(1.0, |t| t.get(r))
    }
}

/// 2026-10-09: The bounded reference at flat output indices `idx` (`r * n + c`), before the
/// kernel's final rounding into the output format (the comparison accounts for that rounding).
pub fn reference(case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
    let f = formats(plan)?;
    let (x, w) = (operand(case, "x")?, operand(case, "w")?);
    let (rows, k) = (x.values.dims[0], x.values.dims[1]);
    let n = w.values.dims[0];
    if w.values.dims[1] != k {
        return Err(format!("x is [{rows}, {k}] but w is {:?}", w.values.dims));
    }
    let group = match (f.act.k_group(), f.weight.k_group()) {
        (Some(a), Some(b)) if a != b => {
            return Err(format!("activation group {a} and weight group {b}"));
        }
        (a, b) => a.or(b),
    };
    let product_rounds = f.mma.0.precision + f.mma.1.precision > f.acc.precision;
    let depth_k = plan.depth_of("k")?;
    let scale_fmt = |what: &str| {
        f.scale
            .ok_or_else(|| format!("{what} scales with a `scale = none` pipeline"))
    };
    let mut out = Vec::with_capacity(idx.len());
    for &i in idx {
        let (r, c) = (i / n, i % n);
        if r >= rows {
            return Err(format!("index {i} beyond [{rows}, {n}]"));
        }
        let xv = |j: usize| x.values.get(r * k + j);
        let wq = |j: usize| w.values.get(c * k + j);
        let prod = |a: f64, b: Bounded| {
            let p = Bounded::exact(a).mul(b);
            if product_rounds { p.round(f.acc) } else { p }
        };
        let y = match (plan.scale_fold, group) {
            (ScaleFold::Group, Some(g)) => {
                let sf = scale_fmt("block")?;
                let depth_g = plan.depth_of("group")?;
                let globals = x.global * w.global * x.row_scale(r) * w.row_scale(c);
                let mut groups = Vec::with_capacity(k / g);
                for b in 0..k / g {
                    let terms: Vec<Bounded> = (b * g..(b + 1) * g)
                        .map(|j| prod(xv(j), Bounded::exact(wq(j))))
                        .collect();
                    let part = sum(&terms, f.acc, depth_g);
                    let s = Bounded::exact(w.block_scale(c, b * g, g))
                        .mul(Bounded::exact(x.block_scale(r, b * g, g)))
                        .mul(Bounded::exact(globals))
                        .round(sf);
                    groups.push(part.mul(s).round(sf));
                }
                // 2026-10-09: The after-reduction placement of the per-row and per-tensor scales.
                sum(&groups, f.acc, depth_k).round(sf)
            }
            (ScaleFold::Group, None) => {
                return Err("scale_fold = group without a block-scaled operand".into());
            }
            (fold, _) => {
                if f.act.k_group().is_some() {
                    return Err("an activation with block scales needs scale_fold = group".into());
                }
                let terms: Vec<Bounded> = (0..k)
                    .map(|j| {
                        let wop = match (fold, group) {
                            (ScaleFold::Element, Some(g)) => {
                                Bounded::exact(wq(j) * w.block_scale(c, j, g)).round(f.operand)
                            }
                            (ScaleFold::None, Some(_)) => {
                                return Err(
                                    "a block-scaled weight needs scale_fold element or group"
                                        .to_string(),
                                );
                            }
                            _ => Bounded::exact(wq(j)),
                        };
                        Ok(prod(xv(j), wop))
                    })
                    .collect::<Result<_, String>>()?;
                let mut y = sum(&terms, f.acc, depth_k);
                for s in [w.row_scale(c), x.row_scale(r), w.global, x.global] {
                    if s != 1.0 {
                        y = y.mul(Bounded::exact(s)).round(scale_fmt("row or tensor")?);
                    }
                }
                y
            }
        };
        out.push(y);
    }
    Ok(out)
}

/// 2026-10-09: A conforming f32 emulation of the declared linear pipeline at `idx` (the noise
/// floor), or, with `acc` set, the same emulation with every accumulation rounded into `acc`
/// (the `accumulate:<fmt>` mutation arm). Bracketing per [`crate::emulate::reduce`].
pub fn emulate(
    case: &Case,
    plan: &Plan,
    acc: Option<Elem>,
    variant: u32,
    idx: &[usize],
) -> Result<Vec<f64>, String> {
    let f = formats(plan)?;
    let (x, w) = (operand(case, "x")?, operand(case, "w")?);
    let k = x.values.dims[1];
    let n = w.values.dims[0];
    let acc = acc.unwrap_or(f.acc);
    let group = f.act.k_group().or(f.weight.k_group());
    let depth_k = plan.depth_of("k")?;
    let rnd = |e: Elem, v: f64| e.round_saturating(v).unwrap_or(f64::NAN);
    let product_rounds = f.mma.0.precision + f.mma.1.precision > f.acc.precision;
    let prod = |a: f64, b: f64| {
        if product_rounds {
            rnd(acc, a * b)
        } else {
            a * b
        }
    };
    let sf = f.scale.unwrap_or(elem::F32);
    let mut out = Vec::with_capacity(idx.len());
    for &i in idx {
        let (r, c) = (i / n, i % n);
        let xv = |j: usize| x.values.get(r * k + j);
        let wq = |j: usize| w.values.get(c * k + j);
        let y = match (plan.scale_fold, group) {
            (ScaleFold::Group, Some(g)) => {
                let depth_g = plan.depth_of("group")?;
                let globals = rnd(sf, x.global * w.global * x.row_scale(r) * w.row_scale(c));
                let parts: Vec<f64> = (0..k / g)
                    .map(|b| {
                        let terms: Vec<f64> =
                            (b * g..(b + 1) * g).map(|j| prod(xv(j), wq(j))).collect();
                        let part = crate::emulate::reduce(&terms, acc, depth_g, variant);
                        let s = rnd(
                            sf,
                            rnd(sf, w.block_scale(c, b * g, g) * x.block_scale(r, b * g, g))
                                * globals,
                        );
                        rnd(sf, part * s)
                    })
                    .collect();
                crate::emulate::reduce(&parts, acc, depth_k, variant)
            }
            _ => {
                let terms: Vec<f64> = (0..k)
                    .map(|j| {
                        let wop = match group {
                            Some(g) => rnd(f.operand, wq(j) * w.block_scale(c, j, g)),
                            None => wq(j),
                        };
                        prod(xv(j), wop)
                    })
                    .collect();
                let mut y = crate::emulate::reduce(&terms, acc, depth_k, variant);
                for s in [w.row_scale(c), x.row_scale(r), w.global, x.global] {
                    if s != 1.0 {
                        y = rnd(sf, y * s);
                    }
                }
                y
            }
        };
        out.push(rnd(f.out, y));
    }
    Ok(out)
}
