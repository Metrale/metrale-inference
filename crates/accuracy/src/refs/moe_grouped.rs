// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The grouped routed-expert reference's case: the routing (an input, never
//! computed), each routed expert's stored weights, the activations, and the per-expert
//! projection views the reference evaluates with [`super::linear`]. Also the pure row
//! permutations between a kernel's sorted positions and the case's slot order, which the GPU
//! adapters apply to what the engine's own sort produced.
//!
//! Tensors of a case: `ids` `[tokens, top_k]` i32 (the routing); `x` the projection input; for
//! each routed expert `e` and projection `p` (`gate` and `up` for `expert_gate_up`, `down` for
//! `expert_down`) the stored weight `w_<p>.<e>` `[n, k]` with its `_block` scales and the
//! scalars `w_<p>.<e>_global` and `w_<p>.<e>_block_rows` ([`super::linear::put`]'s names);
//! scalar `experts` (the model's expert count, which sizes the pointer tables). Output rows are
//! slots `token * top_k + j` in routing order.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - Only routed experts hold weights, so every expert a mutation picks is routed.
//! - `expert_gate_up`: `x` `[tokens, k]` as the pipeline's input; output `[slots, n]` f32, the
//!   SiLU product the down projection reads (a stored BF16 hi + lo pair is decoded to f32 by
//!   [`pair_rows_by_slot`]).
//! - `expert_down`: a BF16 input is the hi + lo pair a grouped gate+up stores, `x`
//!   `[slots, 2k]` (hi then lo, each multiplied by the same weight); an f32 input is the plain
//!   product, `x` `[slots, k]`. Output `[slots, n]` in the pipeline's output format.

use std::collections::BTreeMap;

use metrale_circuit::format::Format;
use metrale_circuit::pipeline::NodePipeline;

use super::linear::{self, put};
use crate::case::{Case, Enc, Tensor};
use crate::elem::{self, BF16};
use crate::inputs::{InputClass, SplitMix64, tensor};
use crate::plan::Plan;
use crate::points::Shape;

/// 2026-10-09: Zipf exponent of the routing: expert popularity `(rank + 1)^-0.9`. At 256
/// experts and top-8 it gives about the distinct-expert counts the Qwen3.6-35B-A3B verify step
/// routes to (121 at 32 rows, 153 at 64, measured with the engine's expert-id dump on the
/// concurrency ladder), so experts carry one row and many rows (several passes) alike.
pub const ZIPF_ALPHA: f64 = 0.9;

/// 2026-10-09: The two grouped expert ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// 2026-10-09: Gate and up projections, then the SiLU product.
    GateUp,
    /// 2026-10-09: The down projection of the SiLU product.
    Down,
}

impl Op {
    /// 2026-10-09: The op of a contract key (`expert_gate_up`, `expert_down`).
    pub fn of(op: &str) -> Result<Op, String> {
        match op.split(':').next().unwrap_or(op) {
            "expert_gate_up" => Ok(Op::GateUp),
            "expert_down" => Ok(Op::Down),
            o => Err(format!("`{o}` is not a grouped expert op")),
        }
    }

    /// 2026-10-09: The projections it runs per expert.
    pub fn projections(self) -> &'static [&'static str] {
        match self {
            Op::GateUp => &["gate", "up"],
            Op::Down => &["down"],
        }
    }
}

/// 2026-10-09: The down input is the BF16 hi + lo pair (two MMA operands per K value).
pub fn pair_input(pipeline: &NodePipeline) -> Result<bool, String> {
    match pipeline.inputs.first() {
        Some(Format::Bf16) => Ok(true),
        Some(Format::F32) => Ok(false),
        Some(f) => Err(format!("a grouped down projection reading {}", f.name())),
        None => Err("the pipeline names no input".into()),
    }
}

/// 2026-10-09: A runtime value of the node; the sweep reads it from the family's runtime
/// parameters, and a missing one is an error, never a default.
fn runtime(shape: &Shape, name: &str) -> Result<usize, String> {
    shape
        .runtime
        .get(name)
        .ok_or_else(|| {
            format!(
                "the point carries no runtime `{name}` (the family must declare it as a runtime \
                 parameter)"
            )
        })?
        .parse()
        .map_err(|e| format!("runtime `{name}`: {e}"))
}

/// 2026-10-09: `tokens` rows of `top_k` distinct experts out of `experts`, drawn with Zipf
/// popularity [`ZIPF_ALPHA`] over a seeded ranking of the experts (so popular experts sit at any
/// id, low and high). Row-major `[tokens, top_k]`.
pub fn zipf_routing(rng: &mut SplitMix64, experts: usize, top_k: usize, tokens: usize) -> Vec<u32> {
    let mut rank: Vec<u32> = (0..experts as u32).collect();
    for i in (1..experts).rev() {
        rank.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let mut cdf = Vec::with_capacity(experts);
    let mut acc = 0.0;
    for r in 0..experts {
        acc += ((r + 1) as f64).powf(-ZIPF_ALPHA);
        cdf.push(acc);
    }
    let mut ids = Vec::with_capacity(tokens * top_k);
    for _ in 0..tokens {
        let mut row: Vec<u32> = Vec::with_capacity(top_k);
        while row.len() < top_k {
            let u = rng.unit() * acc;
            let e = rank[cdf.partition_point(|&c| c <= u).min(experts - 1)];
            if !row.contains(&e) {
                row.push(e);
            }
        }
        ids.extend(row);
    }
    ids
}

/// 2026-10-09: The routing of a case: `(ids by slot, top_k)`.
pub fn routing(case: &Case) -> Result<(Vec<usize>, usize), String> {
    let t = case.tensor("ids")?;
    Ok((t.values().iter().map(|&v| v as usize).collect(), t.dims[1]))
}

/// 2026-10-09: The routed experts, ascending, each with its slots in order.
pub fn slots_by_expert(ids: &[usize]) -> BTreeMap<usize, Vec<usize>> {
    let mut m: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (s, &e) in ids.iter().enumerate() {
        m.entry(e).or_default().push(s);
    }
    m
}

/// 2026-10-09: The name of expert `e`'s projection `p` weight.
pub fn weight_name(p: &str, e: usize) -> String {
    format!("w_{p}.{e}")
}

/// 2026-10-09: Fill `case` with a grouped launch of `shape` for `class`.
pub fn fill(
    case: &mut Case,
    plan: &Plan,
    shape: &Shape,
    class: InputClass,
    stream: &dyn Fn(&str) -> SplitMix64,
) -> Result<(), String> {
    let op = Op::of(&case.op)?;
    let f = linear::formats(plan)?;
    let (experts, top_k) = (runtime(shape, "experts")?, runtime(shape, "top_k")?);
    if top_k == 0 || top_k > experts {
        return Err(format!("top_k {top_k} of {experts} experts"));
    }
    let (tokens, k) = (shape.rows as usize, shape.in_dim as usize);
    let n = match op {
        Op::GateUp if shape.out_dim.is_multiple_of(2) => shape.out_dim as usize / 2,
        Op::GateUp => return Err(format!("gate+up width {} is odd", shape.out_dim)),
        Op::Down => shape.out_dim as usize,
    };
    let ids = zipf_routing(&mut stream("ids"), experts, top_k, tokens);
    let slots = tokens * top_k;
    let id_vals: Vec<f64> = ids.iter().map(|&e| f64::from(e)).collect();
    case.tensors.insert(
        "ids".into(),
        Tensor::encode(Enc::I32, vec![tokens, top_k], &id_vals)?,
    );
    case.scalars.insert("experts".into(), experts as f64);
    match op {
        Op::GateUp => put(
            case,
            "x",
            linear::activation(&mut stream("x"), f.act, class, tokens, k)?,
        ),
        Op::Down if pair_input(&plan.pipeline)? => {
            let a = tensor(&mut stream("x"), class, slots, k, 1.0, elem::F32);
            case.tensors.insert("x".into(), hi_lo_rows(&a, slots, k)?);
            case.scalars.insert("x_global".into(), 1.0);
        }
        Op::Down => put(
            case,
            "x",
            linear::activation(&mut stream("x"), f.act, class, slots, k)?,
        ),
    }
    let routed: Vec<usize> = slots_by_expert(&ids.iter().map(|&e| e as usize).collect::<Vec<_>>())
        .into_keys()
        .collect();
    let names: Vec<String> = routed
        .iter()
        .flat_map(|&e| op.projections().iter().map(move |p| weight_name(p, e)))
        .collect();
    for (name, w) in names
        .iter()
        .zip(draw_weights(&names, f.weight, n, k, stream)?)
    {
        put(case, name, w);
    }
    case.out = (
        vec![slots, n],
        match op {
            Op::GateUp => Enc::F32,
            Op::Down if f.out == BF16 => Enc::Bf16,
            Op::Down => Enc::F32,
        },
    );
    Ok(())
}

/// 2026-10-09: Every named weight from its own stream, drawn on scoped threads (each weight
/// depends on its name alone, so the thread count changes the speed and never a byte).
fn draw_weights(
    names: &[String],
    layout: super::quant::Layout,
    n: usize,
    k: usize,
    stream: &dyn Fn(&str) -> SplitMix64,
) -> Result<Vec<super::quant::Stored>, String> {
    let mut rngs: Vec<SplitMix64> = names.iter().map(|nm| stream(nm)).collect();
    let workers = std::thread::available_parallelism()
        .map_or(1, |v| v.get())
        .min(names.len().max(1));
    let per = names.len().div_ceil(workers).max(1);
    let parts: Vec<Result<Vec<super::quant::Stored>, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = rngs
            .chunks_mut(per)
            .map(|c| {
                s.spawn(move || {
                    c.iter_mut()
                        .map(|r| linear::weight(r, layout, n, k))
                        .collect::<Result<Vec<_>, String>>()
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("a weight worker panicked".into()))
            })
            .collect()
    });
    let mut out = Vec::with_capacity(names.len());
    for p in parts {
        out.extend(p?);
    }
    Ok(out)
}

/// 2026-10-09: f32 values `a` `[rows, k]` as the BF16 hi + lo rows a grouped gate+up stores:
/// `hi = bf16(a)`, `lo = bf16(a - hi)`, row r holding k hi values then k lo values.
pub fn hi_lo_rows(a: &[f64], rows: usize, k: usize) -> Result<Tensor, String> {
    let mut v = Vec::with_capacity(2 * rows * k);
    for r in a.chunks(k) {
        let hi: Vec<f64> = r
            .iter()
            .map(|&x| BF16.round_saturating(x).ok_or("a non-finite product"))
            .collect::<Result<_, _>>()?;
        let lo: Vec<f64> = r
            .iter()
            .zip(&hi)
            .map(|(&x, &h)| BF16.round_saturating(x - h).ok_or("a non-finite product"))
            .collect::<Result<_, _>>()?;
        v.extend(hi);
        v.extend(lo);
    }
    Tensor::encode(Enc::Bf16, vec![rows, 2 * k], &v)
}

/// 2026-10-09: The linear view of expert `e`'s projection `p`: the case's input and that weight
/// under [`super::linear`]'s names. For a paired down input the weight is repeated along K
/// (`[w | w]`), so the hi and lo halves of a row meet the same weights in one reduction.
pub fn view(case: &Case, p: &str, e: usize, pair: bool) -> Result<Case, String> {
    let name = weight_name(p, e);
    let mut sub = Case {
        family: case.family.clone(),
        kernel: case.kernel.clone(),
        launcher: case.launcher.clone(),
        op: case.op.clone(),
        tensors: BTreeMap::new(),
        scalars: BTreeMap::new(),
        out: case.out.clone(),
        split: Vec::new(),
    };
    sub.tensors.insert("x".into(), case.tensor("x")?.clone());
    sub.scalars
        .insert("x_global".into(), case.scalar("x_global")?);
    for (from, to) in [(name.clone(), "w"), (format!("{name}_block"), "w_block")] {
        if let Some(t) = case.tensors.get(&from) {
            sub.tensors
                .insert(to.into(), if pair { repeat_k(t) } else { t.clone() });
        }
    }
    for (from, to) in [
        (format!("{name}_global"), "w_global"),
        (format!("{name}_block_rows"), "w_block_rows"),
    ] {
        if let Some(&v) = case.scalars.get(&from) {
            sub.scalars.insert(to.into(), v);
        }
    }
    if !sub.tensors.contains_key("w") {
        return Err(format!("case has no weight `{name}`"));
    }
    Ok(sub)
}

/// 2026-10-09: `[rows, c]` as `[rows, 2c]`, each row followed by itself.
fn repeat_k(t: &Tensor) -> Tensor {
    let row = t.bytes.len() / t.dims[0];
    let mut bytes = Vec::with_capacity(2 * t.bytes.len());
    for r in t.bytes.chunks(row) {
        bytes.extend_from_slice(r);
        bytes.extend_from_slice(r);
    }
    Tensor {
        enc: t.enc,
        dims: vec![t.dims[0], 2 * t.dims[1]],
        bytes: std::sync::Arc::new(bytes),
    }
}

/// 2026-10-09: Rows `[slots, row_bytes]` in slot order placed at their sorted positions:
/// position `perm[s]` holds slot `s` (what the down kernel reads by position).
pub fn rows_by_position(by_slot: &[u8], perm: &[i32], row_bytes: usize) -> Result<Vec<u8>, String> {
    let mut out = vec![0u8; by_slot.len()];
    for (s, &p) in perm.iter().enumerate() {
        let p = position(p, perm.len())?;
        out[p * row_bytes..(p + 1) * row_bytes]
            .copy_from_slice(&by_slot[s * row_bytes..(s + 1) * row_bytes]);
    }
    Ok(out)
}

/// 2026-10-09: Rows written by sorted position back in slot order: slot `s` is position
/// `perm[s]` (the blend's mapping).
pub fn rows_by_slot(by_pos: &[u8], perm: &[i32], row_bytes: usize) -> Result<Vec<u8>, String> {
    if by_pos.len() != perm.len() * row_bytes {
        return Err(format!(
            "{} bytes for {} rows of {row_bytes}",
            by_pos.len(),
            perm.len()
        ));
    }
    let mut out = Vec::with_capacity(by_pos.len());
    for &p in perm {
        let p = position(p, perm.len())?;
        out.extend_from_slice(&by_pos[p * row_bytes..(p + 1) * row_bytes]);
    }
    Ok(out)
}

/// 2026-10-09: BF16 hi | lo rows `[positions, 2n]` (the act buffer of a tensor-core gate+up) as
/// f32 rows `[slots, n]` in slot order: each element `hi + lo` (exact in f64) rounded once to
/// f32. A sentinel left in either half decodes to a non-finite or huge value, never a
/// plausible one.
pub fn pair_rows_by_slot(by_pos: &[u8], perm: &[i32], n: usize) -> Result<Vec<u8>, String> {
    let rows = rows_by_slot(by_pos, perm, 4 * n)?;
    let half = |b: &[u8], i: usize| elem::bf16_to_f64(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]));
    let mut out = Vec::with_capacity(rows.len());
    for r in rows.chunks(4 * n) {
        for c in 0..n {
            out.extend_from_slice(&((half(r, c) + half(r, n + c)) as f32).to_le_bytes());
        }
    }
    Ok(out)
}

fn position(p: i32, slots: usize) -> Result<usize, String> {
    usize::try_from(p)
        .ok()
        .filter(|&p| p < slots)
        .ok_or_else(|| format!("sorted position {p} outside {slots} slots"))
}
