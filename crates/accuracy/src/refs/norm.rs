// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The RMSNorm reference: `y = x * rsqrt(sum(x^2) / n + eps) * (1 + w)` (weight form
//! `one_plus`) or `* w` (`plain`) over rows of `n`, in bounded arithmetic following the family's
//! declared pipeline: the format the norm reads its input in, the compute precision of the
//! squares, their reduction and every scalar step, and the output format.
//!
//! Tensors of a norm case: `x` `[rows, n]` in the input format; or, when the declared input is
//! f32 (the residual-add norms, which read the f32 sum of the stream `h` and the branch `s` they
//! add themselves), `h` and `s` `[rows, n]` bf16. `w` `[n]` bf16. Scalars: `eps` (an f32 value)
//! and `one_plus` (1 or 0, the point's `weight_form`). Output `[rows, n]`.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - The scalar steps are rms_norm.cu's (rms_norm, :96 and :106): a correctly rounded division
//!   by `n`, an f32 add of `eps`, `rsqrtf` (the approximate instruction: its error is the
//!   contract's `approx.rsqrt`), then `x * rms` and `* (1 + w)`, each rounded in the compute
//!   precision.
//! - A residual-add norm squares the sum in its declared input format and scales the sum as it
//!   stores it, in the stream's own format (rms_norm.cu:409-412 and :452-456).

use metrale_circuit::format::Format;
use metrale_circuit::pipeline::{StepKind, Value};

use crate::bounded::{Bounded, sum};
use crate::case::{Case, Enc, Tensor};
use crate::elem::{self, Elem};
use crate::inputs::{InputClass, SplitMix64, tensor};
use crate::plan::Plan;
use crate::points::Shape;
use crate::refs::linear::num;

/// 2026-10-09: The head width a per-head norm (`qk_norm`) or RoPE case runs at. The sweep's
/// shape carries the node edge's heads x head width, not the head width (the rms_norm and rope
/// families declare no head_dim parameter); every swept instance that runs these kernels
/// (Qwen3.5, Qwen3.6, Qwen3.8) has head_dim 256. A width it does not divide is refused.
pub const HEAD_DIM: usize = 256;

/// 2026-10-09: The eps a case passes: the swept checkpoints' `rms_norm_eps`, a config parameter
/// the sweep's shape does not carry.
pub const EPS: f64 = 1e-6;

/// 2026-10-09: Standard deviation of a drawn norm weight around its form's identity (0 for
/// `one_plus`, 1 for `plain`): checkpoint norm weights stay within a few tenths of it.
const WEIGHT_SPREAD: f64 = 0.25;

/// 2026-10-09: Rows and row length of a norm launch at `shape`: a per-head norm runs one row per
/// head of [`HEAD_DIM`].
pub fn dims(shape: &Shape) -> Result<(usize, usize), String> {
    let (width, rows) = (shape.in_dim as usize, shape.rows as usize);
    let (rows, n) = if shape.op.split(':').next() == Some("qk_norm") {
        if width == 0 || !width.is_multiple_of(HEAD_DIM) {
            return Err(format!(
                "a qk_norm over {width} is not whole heads of {HEAD_DIM}"
            ));
        }
        (rows * (width / HEAD_DIM), HEAD_DIM)
    } else {
        (rows, width)
    };
    // 2026-10-09: rms_norm.cu reads rows as BF16 pairs from their start (its invariant).
    if n == 0 || !n.is_multiple_of(2) || rows == 0 {
        return Err(format!(
            "a norm of {rows} rows of {n}: rows must be even and non-empty"
        ));
    }
    Ok((rows, n))
}

/// 2026-10-09: The formats of a norm plan: the input as the reduction reads it, the compute
/// precision, the output.
struct Formats {
    input: Elem,
    compute: Elem,
    out: Elem,
}

/// 2026-10-09: The rounding model of a norm tensor format (bf16 or f32).
fn elem_of(f: Format, ftz: bool) -> Result<Elem, String> {
    match f {
        Format::Bf16 => Ok(elem::BF16),
        Format::F32 if ftz => Ok(elem::F32_FTZ),
        Format::F32 => Ok(elem::F32),
        o => Err(format!("a norm over {}", o.name())),
    }
}

/// 2026-10-09: The formats `plan` declares for a norm.
fn formats(plan: &Plan) -> Result<Formats, String> {
    let p = &plan.pipeline;
    let input = match p.inputs.as_slice() {
        [f] => elem_of(*f, plan.ftz)?,
        other => {
            return Err(format!(
                "a norm reads one input, the pipeline names {}",
                other.len()
            ));
        }
    };
    let compute = match plan.step(StepKind::Compute) {
        Some(Value::Num(n)) => num(*n, plan.ftz),
        _ => return Err("the pipeline has no `compute` step".into()),
    };
    let out = elem_of(
        *p.outputs.first().ok_or("the pipeline names no output")?,
        plan.ftz,
    )?;
    Ok(Formats {
        input,
        compute,
        out,
    })
}

/// 2026-10-09: The point's weight form: `true` for `1 + w`.
fn weight_form(plan: &Plan) -> Result<bool, String> {
    match plan.value_of("weight_form")? {
        "one_plus" => Ok(true),
        "plain" => Ok(false),
        o => Err(format!("weight_form `{o}` (one_plus | plain)")),
    }
}

/// 2026-10-09: Fill `case` with a norm launch at `shape`, the row values drawn for `class` and the
/// weight as a checkpoint of the point's weight form holds it.
pub fn fill(
    case: &mut Case,
    plan: &Plan,
    shape: &Shape,
    class: InputClass,
    stream: &dyn Fn(&str) -> SplitMix64,
) -> Result<(), String> {
    let f = formats(plan)?;
    let one_plus = weight_form(plan)?;
    let (rows, n) = dims(shape)?;
    let draw = |name: &str| {
        let v = tensor(&mut stream(name), class, rows, n, 1.0, elem::BF16);
        Tensor::encode(Enc::Bf16, vec![rows, n], &v)
    };
    if f.input == elem::BF16 {
        case.tensors.insert("x".into(), draw("x")?);
    } else if f.input.name == elem::F32.name {
        case.tensors.insert("h".into(), draw("h")?);
        case.tensors.insert("s".into(), draw("s")?);
    } else {
        return Err(format!("a norm input in {}", f.input.name));
    }
    let centre = if one_plus { 0.0 } else { 1.0 };
    let w: Vec<f64> = tensor(
        &mut stream("w"),
        InputClass::Gaussian,
        1,
        n,
        WEIGHT_SPREAD,
        elem::F64,
    )
    .iter()
    .map(|v| elem::BF16.round_saturating(centre + v).unwrap_or(0.0))
    .collect();
    case.tensors
        .insert("w".into(), Tensor::encode(Enc::Bf16, vec![n], &w)?);
    let eps = elem::F32.round(EPS).ok_or("eps is not an f32 value")?;
    case.scalars.insert("eps".into(), eps);
    case.scalars
        .insert("one_plus".into(), f64::from(u8::from(one_plus)));
    let enc = if f.out == elem::BF16 {
        Enc::Bf16
    } else {
        Enc::F32
    };
    case.out = (vec![rows, n], enc);
    Ok(())
}

/// 2026-10-09: One row's values: what the squares read and what the scaling multiplies.
struct Row {
    squared: Vec<f64>,
    scaled: Vec<f64>,
}

/// 2026-10-09: Row `r`'s values as the squares and the scaling read them.
fn row(case: &Case, f: &Formats, r: usize, n: usize) -> Result<Row, String> {
    let slice = |t: &Tensor| (r * n..(r + 1) * n).map(|i| t.get(i)).collect::<Vec<f64>>();
    if let Some(x) = case.tensors.get("x") {
        let v = slice(x);
        return Ok(Row {
            squared: v.clone(),
            scaled: v,
        });
    }
    let (h, s) = (case.tensor("h")?, case.tensor("s")?);
    let stored = h.enc.elem().ok_or("the stream has no rounding model")?;
    let (mut squared, mut scaled) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for (a, b) in slice(h).into_iter().zip(slice(s)) {
        // 2026-10-09: The add is one rounding into the compute precision; the reduction reads
        // that sum in the declared input format and the stream keeps it in its own.
        let total = f.compute.round(a + b).ok_or("the residual sum overflows")?;
        squared.push(
            f.input
                .round(total)
                .ok_or("the sum overflows the input format")?,
        );
        scaled.push(
            stored
                .round(total)
                .ok_or("the sum overflows the stream format")?,
        );
    }
    Ok(Row { squared, scaled })
}

/// 2026-10-09: The operands every element of a case shares.
struct Shared<'a> {
    f: Formats,
    w: &'a Tensor,
    eps: f64,
    one_plus: bool,
    rows: usize,
    n: usize,
}

/// 2026-10-09: The operands of `case` every element shares, checked against its output.
fn shared<'a>(case: &'a Case, plan: &Plan) -> Result<Shared<'a>, String> {
    let (rows, n) = (case.out.0[0], case.out.0[1]);
    let w = case.tensor("w")?;
    if w.len() != n {
        return Err(format!("w has {} values for rows of {n}", w.len()));
    }
    Ok(Shared {
        f: formats(plan)?,
        w,
        eps: case.scalar("eps")?,
        one_plus: case.scalar("one_plus")? == 1.0,
        rows,
        n,
    })
}

/// 2026-10-09: Each index's row and column, rows grouped so each row's statistic is computed
/// once.
fn by_row(idx: &[usize], rows: usize, n: usize) -> Result<Vec<(usize, usize)>, String> {
    idx.iter()
        .map(|&i| {
            let (r, c) = (i / n, i % n);
            if r >= rows {
                Err(format!("index {i} beyond [{rows}, {n}]"))
            } else {
                Ok((r, c))
            }
        })
        .collect()
}

/// 2026-10-09: The bounded reference at flat output indices `idx` (`r * n + c`), before the
/// kernel's final rounding into the output format.
pub fn reference(case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
    let sh = shared(case, plan)?;
    let (depth, rel) = (plan.depth_of("k")?, plan.approx_of("rsqrt")?);
    let c = sh.f.compute;
    let mut cache: Option<(usize, Row, Bounded)> = None;
    let mut out = Vec::with_capacity(idx.len());
    for (r, col) in by_row(idx, sh.rows, sh.n)? {
        if cache.as_ref().is_none_or(|(cr, _, _)| *cr != r) {
            let rw = row(case, &sh.f, r, sh.n)?;
            let terms: Vec<Bounded> = rw
                .squared
                .iter()
                .map(|&x| Bounded::exact(x).mul(Bounded::exact(x)).round(c))
                .collect();
            let rms = sum(&terms, c, depth)
                .div(Bounded::exact(sh.n as f64))
                .round(c)
                .add(Bounded::exact(sh.eps))
                .round(c)
                .rsqrt(rel);
            cache = Some((r, rw, rms));
        }
        let (_, rw, rms) = cache.as_ref().ok_or("no row")?;
        let w = sh.w.get(col);
        let wf = if sh.one_plus {
            Bounded::exact(1.0).add(Bounded::exact(w)).round(c)
        } else {
            Bounded::exact(w)
        };
        out.push(
            Bounded::exact(rw.scaled[col])
                .mul(*rms)
                .round(c)
                .mul(wf)
                .round(c),
        );
    }
    Ok(out)
}

/// 2026-10-09: A conforming emulation of the declared norm at `idx`, in the output format: the
/// squares and their reduction in the compute precision (or, with `acc` set, in `acc`: the
/// `accumulate:<fmt>` arm), bracketed per [`crate::emulate::reduce`], `rsqrt` correctly
/// rounded.
pub fn emulate(
    case: &Case,
    plan: &Plan,
    acc: Option<Elem>,
    variant: u32,
    idx: &[usize],
) -> Result<Vec<f64>, String> {
    let sh = shared(case, plan)?;
    let depth = plan.depth_of("k")?;
    let c = sh.f.compute;
    let acc = acc.unwrap_or(c);
    let rnd = |e: Elem, v: f64| e.round_saturating(v).unwrap_or(f64::NAN);
    let mut cache: Option<(usize, Row, f64)> = None;
    let mut out = Vec::with_capacity(idx.len());
    for (r, col) in by_row(idx, sh.rows, sh.n)? {
        if cache.as_ref().is_none_or(|(cr, _, _)| *cr != r) {
            let rw = row(case, &sh.f, r, sh.n)?;
            let terms: Vec<f64> = rw.squared.iter().map(|&x| rnd(acc, x * x)).collect();
            let total = crate::emulate::reduce(&terms, acc, depth, variant);
            let arg = rnd(c, rnd(c, total / sh.n as f64) + sh.eps);
            cache = Some((r, rw, rnd(c, 1.0 / arg.sqrt())));
        }
        let (_, rw, rms) = cache.as_ref().ok_or("no row")?;
        let w = sh.w.get(col);
        let wf = if sh.one_plus { rnd(c, 1.0 + w) } else { w };
        out.push(rnd(sh.f.out, rnd(c, rnd(c, rw.scaled[col] * rms) * wf)));
    }
    Ok(out)
}
