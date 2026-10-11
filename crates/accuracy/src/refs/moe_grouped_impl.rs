// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `moe_grouped`: routed-expert projections run grouped by expert. For each routed
//! (token, slot) the expert's projection is the projection reference ([`super::linear`]) on the
//! expert's weight, so the formats, the scale fold and the reduction are the linear ones read
//! from the same plan; `expert_gate_up` adds the fused SiLU epilogue on the BF16-rounded gate
//! and up (`gate / (1 + exp(-gate)) * up` in f32, the product stored as the down projection
//! reads it).
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - The routing is an input of the case; a mutation edits only routed experts' weights.
//! - The epilogue's approximate exponential and the storage of its product are contract
//!   declarations (`approx.ex2`, `approx.silu_store`), never assumed exact.

use std::collections::BTreeMap;

use metrale_circuit::pipeline::NodePipeline;

use super::moe_grouped::{self as mg, Op};
use super::{RefImpl, linear, linear_impl, linear_mutate};
use crate::bounded::Bounded;
use crate::case::Case;
use crate::elem::{self, BF16, Elem, F32, F32_FTZ};
use crate::inputs::{InputClass, SplitMix64};
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::points::Shape;

/// 2026-10-09: The grouped routed-expert reference.
pub struct MoeGrouped;

/// 2026-10-09: A bound on |silu'(x)| = |s(x)(1 + x(1 - s(x)))| over the reals (its extremes are
/// about 1.0998 and -0.0998).
const SILU_LIP: f64 = 1.1;

/// 2026-10-09: Below this the kernel's `exp(-gate)` can overflow f32 (it does past -88.72):
/// the quotient is then 0 and the whole of |silu| is error, which the bound admits.
const EXP_OVERFLOW_GATE: f64 = -80.0;

fn f32_of(plan: &Plan) -> Elem {
    if plan.ftz { F32_FTZ } else { F32 }
}

/// 2026-10-09: `silu(g)` as the epilogue computes it, `g / (1 + __expf(-g))`, in f32. `__expf(x)`
/// is `ex2.approx(RN(x * L))` with `L` = log2(e) rounded to f32: the argument carries a relative
/// error of at most `2u + u^2`, which scales `e^-g` by at most `exp(|g| (2u + u^2))`, and ex2
/// adds the declared `approx.ex2`. With `E' = E(1 + eps)`, `(1 + E') / (1 + E)` is within
/// `eps`, so the quotient (two more roundings) is within `r` of `silu` at the kernel's `g`; a
/// result below the normal range moves `1 + E` by at most `2^(emin + 1)`.
fn silu(g: Bounded, plan: &Plan) -> Result<Bounded, String> {
    let f32e = f32_of(plan);
    let u = f32e.unit_roundoff();
    let arg = (g.mag() * (2.0 * u + u * u)).exp_m1();
    let eps = (1.0 + arg) * (1.0 + plan.approx_of("ex2")?) - 1.0 + elem::pow2(f32e.emin + 1);
    if eps >= 1.0 {
        return Ok(Bounded {
            v: g.v,
            e: f64::INFINITY,
        });
    }
    let mut r = (1.0 + u) / ((1.0 - u) * (1.0 - eps)) - 1.0;
    if g.v - g.e < EXP_OVERFLOW_GATE {
        r = r.max(1.0);
    }
    let s = g.lipschitz(|x| x / (1.0 + (-x).exp()), SILU_LIP, r);
    Ok(s.add(Bounded {
        v: 0.0,
        e: f32e.underflow_floor(),
    }))
}

/// 2026-10-09: The kernel's RNE of a value it holds anywhere in `[v - e, v + e]` into `fmt`:
/// RNE is monotone, so the result lies between the roundings of the two ends (widened by the
/// f64 evaluation of the ends). When the interval holds no rounding boundary the result is
/// known exactly (the gate and up of a conforming kernel round to the same BF16 as the exact
/// projection unless it sits within the bound of a tie); otherwise both outcomes are admitted.
fn round_monotone(b: Bounded, fmt: Elem) -> Bounded {
    let w = b.e * (1.0 + elem::pow2(-50)) + b.v.abs() * elem::pow2(-50);
    match (fmt.round(b.v - w), fmt.round(b.v + w)) {
        (Some(lo), Some(hi)) => Bounded {
            v: 0.5 * (lo + hi),
            e: 0.5 * (hi - lo),
        },
        _ => Bounded {
            v: b.v,
            e: f64::INFINITY,
        },
    }
}

/// 2026-10-09: The stored SiLU product of BF16 gate `g` and up `u`: `silu(g) * u` rounded to
/// f32, then the declared storage error `approx.silu_store` (a BF16 hi + lo pair keeps it to
/// `u_bf16^2` plus two BF16 floors; an f32 store is exact).
fn product(g: Bounded, u: Bounded, plan: &Plan) -> Result<Bounded, String> {
    let p = silu(g, plan)?.mul(u).round(f32_of(plan));
    let rho = plan.approx_of("silu_store")?;
    Ok(if rho > 0.0 {
        p.add(Bounded {
            v: 0.0,
            e: rho * p.mag() + 2.0 * BF16.underflow_floor(),
        })
    } else {
        p
    })
}

/// 2026-10-09: A conforming f32 epilogue on BF16 `g`, `u`: the exponential correctly rounded
/// (inside any declared ex2 error), an overflow kept infinite as the hardware keeps it.
fn emulate_product(g: f64, u: f64, plan: &Plan) -> f64 {
    let f = f32_of(plan);
    let r = |v: f64| f.round(v).unwrap_or(f64::INFINITY.copysign(v));
    let q = r(1.0 + r((-g).exp()));
    r(r(g / q) * u)
}

/// 2026-10-09: `f(op, expert, linear indices)` for each routed expert over the outputs of `idx`
/// it owns, reassembled in `idx` order. A gate+up output `(slot, c)` is row `token` of the
/// expert's projection, a down output row `slot` (its input rows are slots).
fn per_expert<T: Clone>(
    case: &Case,
    idx: &[usize],
    f: impl Fn(Op, usize, &[usize]) -> Result<Vec<T>, String>,
) -> Result<Vec<T>, String> {
    let op = Op::of(&case.op)?;
    let (ids, top_k) = mg::routing(case)?;
    let n = case.out.0[1];
    let mut groups: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (i, &flat) in idx.iter().enumerate() {
        let (s, c) = (flat / n, flat % n);
        let e = *ids
            .get(s)
            .ok_or_else(|| format!("index {flat} beyond [{}, {n}]", ids.len()))?;
        let row = match op {
            Op::GateUp => s / top_k,
            Op::Down => s,
        };
        groups.entry(e).or_default().push((i, row * n + c));
    }
    let mut out: Vec<Option<T>> = vec![None; idx.len()];
    for (e, items) in groups {
        let lin: Vec<usize> = items.iter().map(|x| x.1).collect();
        let vals = f(op, e, &lin)?;
        if vals.len() != items.len() {
            return Err(format!(
                "expert {e}: {} values for {}",
                vals.len(),
                items.len()
            ));
        }
        for ((i, _), v) in items.iter().zip(vals) {
            out[*i] = Some(v);
        }
    }
    out.into_iter()
        .map(|v| v.ok_or_else(|| "an output no expert owns".to_string()))
        .collect()
}

fn pair(op: Op, plan: &Plan) -> Result<bool, String> {
    Ok(op == Op::Down && mg::pair_input(&plan.pipeline)?)
}

impl RefImpl for MoeGrouped {
    fn name(&self) -> &'static str {
        "moe_grouped"
    }

    fn serves(&self, op: &str) -> bool {
        Op::of(op).is_ok()
    }

    fn lens(&self, shape: &Shape, pipeline: &NodePipeline) -> BTreeMap<String, u64> {
        let mut m = linear_impl::Linear.lens(shape, pipeline);
        if Op::of(&shape.op) == Ok(Op::Down) && mg::pair_input(pipeline) == Ok(true) {
            m.insert("k".into(), 2 * shape.in_dim);
        }
        m
    }

    fn fill(
        &self,
        case: &mut Case,
        plan: &Plan,
        shape: &Shape,
        class: InputClass,
        stream: &dyn Fn(&str) -> SplitMix64,
    ) -> Result<(), String> {
        mg::fill(case, plan, shape, class, stream)
    }

    fn reference(&self, case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
        let out = linear::formats(plan)?.out;
        let paired = pair(Op::of(&case.op)?, plan)?;
        per_expert(case, idx, |op, e, lin| match op {
            Op::GateUp => {
                let g = linear::reference(&mg::view(case, "gate", e, false)?, plan, lin)?;
                let u = linear::reference(&mg::view(case, "up", e, false)?, plan, lin)?;
                g.into_iter()
                    .zip(u)
                    .map(|(g, u)| product(round_monotone(g, out), round_monotone(u, out), plan))
                    .collect()
            }
            Op::Down => linear::reference(&mg::view(case, "down", e, paired)?, plan, lin),
        })
    }

    fn emulate(
        &self,
        case: &Case,
        plan: &Plan,
        acc: Option<Elem>,
        variant: u32,
        idx: &[usize],
    ) -> Result<Vec<f64>, String> {
        let paired = pair(Op::of(&case.op)?, plan)?;
        per_expert(case, idx, |op, e, lin| match op {
            Op::GateUp => {
                let em = |p: &str| {
                    linear::emulate(&mg::view(case, p, e, false)?, plan, acc, variant, lin)
                };
                let (g, u) = (em("gate")?, em("up")?);
                Ok(g.into_iter()
                    .zip(u)
                    .map(|(g, u)| emulate_product(g, u, plan))
                    .collect())
            }
            Op::Down => {
                linear::emulate(&mg::view(case, "down", e, paired)?, plan, acc, variant, lin)
            }
        })
    }

    /// 2026-10-09: `zero_expert` zeroes every projection weight of the most-routed expert
    /// (ties: the lowest id) and reaches all of its slots' outputs; `corrupt_block_scale`
    /// flips one block scale of one of its projections (the linear mutation on that weight) and
    /// reaches its slots at the columns the scale row covers.
    fn mutate(
        &self,
        case: &mut Case,
        m: &Mutation,
        rng: &mut SplitMix64,
    ) -> Result<Vec<usize>, String> {
        let op = Op::of(&case.op)?;
        let (ids, _) = mg::routing(case)?;
        let n = case.out.0[1];
        let by = mg::slots_by_expert(&ids);
        let (e, slots) = by
            .iter()
            .max_by_key(|(e, s)| (s.len(), std::cmp::Reverse(**e)))
            .ok_or("the case routes no slot")?;
        let cols: Vec<usize> = match m {
            Mutation::ZeroExpert => {
                for p in op.projections() {
                    let t = case
                        .tensors
                        .get_mut(&mg::weight_name(p, *e))
                        .ok_or_else(|| format!("expert {e} has no `{p}` weight"))?;
                    std::sync::Arc::make_mut(&mut t.bytes).fill(0);
                }
                (0..n).collect()
            }
            Mutation::CorruptBlockScale => {
                let ps = op.projections();
                let p = ps[rng.below(ps.len() as u64) as usize];
                let mut sub = mg::view(case, p, *e, false)?;
                let cols = linear_mutate::mutate(&mut sub, m, rng)?;
                let block = sub.tensors.remove("w_block").ok_or("no block scales")?;
                case.tensors
                    .insert(format!("{}_block", mg::weight_name(p, *e)), block);
                cols
            }
            other => {
                return Err(format!(
                    "`{}` does not apply to grouped experts",
                    other.name()
                ));
            }
        };
        Ok(slots
            .iter()
            .flat_map(|s| cols.iter().map(move |c| s * n + c))
            .collect())
    }

    /// 2026-10-09: The weights' block-row edges (FP8 128-row scale blocks) besides 64.
    fn strides(&self, case: &Case) -> Vec<usize> {
        let mut v: Vec<usize> = case
            .scalars
            .iter()
            .filter(|(k, r)| k.ends_with("_block_rows") && **r > 1.0)
            .map(|(_, &r)| r as usize)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}
