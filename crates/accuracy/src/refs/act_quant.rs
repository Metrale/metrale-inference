// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The activation quantizer reference (`act_quant` to an FP8 format): per row (or per
//! 128-value group) `s = max(f32(amax / 448), floor)`, `q = E4M3(clamp(f32(v / s), ±448))`, every
//! division correctly rounded. That rule is exact, so the reference is exact (`e = 0`): the
//! kernel's codes and scales must be the host model's, byte for byte, on every input class.
//!
//! The kernel writes codes and scales; the adapter decodes them with the layout the pipeline
//! declares (`fp8/token` or `fp8/g128`) into one f32 output `[rows, k]` of `q * s`, so a kernel
//! that writes the other granularity's scales decodes to wrong values and fails.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - The floor is the contract's constant `scale_floor`, never assumed.

use std::collections::BTreeMap;

use metrale_circuit::format::{Format, Scale};
use metrale_circuit::pipeline::NodePipeline;

use super::RefImpl;
use crate::bounded::Bounded;
use crate::case::{Case, Enc, Tensor};
use crate::elem::{self, E4M3, Elem};
use crate::inputs::{InputClass, SplitMix64, tensor};
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::points::Shape;

/// 2026-10-09: `act_quant`.
pub struct ActQuant;

/// 2026-10-09: K values sharing one scale (`k` for a per-token scale).
fn group_of(plan: &Plan, k: usize) -> Result<usize, String> {
    match plan.pipeline.outputs.first() {
        Some(Format::Fp8E4m3 {
            scale: Scale::PerToken,
        }) => Ok(k),
        Some(Format::Fp8E4m3 {
            scale: Scale::Group(g),
        }) => Ok(*g as usize),
        other => Err(format!("an FP8 activation quantizer output, not {other:?}")),
    }
}

fn f32r(x: f64) -> f64 {
    elem::F32.round(x).unwrap_or(f64::NAN)
}

/// 2026-10-09: The host model of one group: its scale and the decoded values.
pub fn quantize_group(v: &[f64], floor: f64) -> (f64, Vec<f64>) {
    let amax = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let s = f32r(amax / 448.0).max(f32r(floor));
    let q = v
        .iter()
        .map(|x| {
            E4M3.round(f32r(x / s).clamp(-448.0, 448.0))
                .unwrap_or(f64::NAN)
                * s
        })
        .collect();
    (s, q)
}

impl ActQuant {
    fn model(case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<f64>, String> {
        let x = case.tensor("x")?;
        let k = x.dims[1];
        let g = group_of(plan, k)?;
        let floor = plan.constant_of("scale_floor")?;
        let mut cache: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        idx.iter()
            .map(|&i| {
                let (r, c) = (i / k, i % k);
                let start = r * k + (c / g) * g;
                let vals = cache.entry(start).or_insert_with(|| {
                    let v: Vec<f64> = (start..start + g).map(|j| x.get(j)).collect();
                    quantize_group(&v, floor).1
                });
                Ok(vals[c % g])
            })
            .collect()
    }
}

impl RefImpl for ActQuant {
    fn name(&self) -> &'static str {
        "act_quant"
    }

    fn serves(&self, op: &str) -> bool {
        op == "act_quant" || op.starts_with("act_quant:fp8")
    }

    fn lens(&self, shape: &Shape, _pipeline: &NodePipeline) -> BTreeMap<String, u64> {
        BTreeMap::from([("k".to_string(), shape.in_dim)])
    }

    fn fill(
        &self,
        case: &mut Case,
        plan: &Plan,
        shape: &Shape,
        class: InputClass,
        stream: &dyn Fn(&str) -> SplitMix64,
    ) -> Result<(), String> {
        let (rows, k) = (shape.rows as usize, shape.in_dim as usize);
        group_of(plan, k)?;
        let x = tensor(&mut stream("x"), class, rows, k, 1.0, elem::BF16);
        case.tensors
            .insert("x".into(), Tensor::encode(Enc::Bf16, vec![rows, k], &x)?);
        case.out = (vec![rows, k], Enc::F32);
        Ok(())
    }

    fn reference(&self, case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
        Ok(Self::model(case, plan, idx)?
            .into_iter()
            .map(Bounded::exact)
            .collect())
    }

    fn emulate(
        &self,
        case: &Case,
        plan: &Plan,
        _acc: Option<Elem>,
        _variant: u32,
        idx: &[usize],
    ) -> Result<Vec<f64>, String> {
        Self::model(case, plan, idx)
    }

    fn mutate(
        &self,
        _case: &mut Case,
        m: &Mutation,
        _rng: &mut SplitMix64,
    ) -> Result<Vec<usize>, String> {
        Err(format!(
            "`{}` does not apply to an activation quantizer (use a symbol mutation)",
            m.name()
        ))
    }

    fn strides(&self, case: &Case) -> Vec<usize> {
        let k = case.out.0.get(1).copied().unwrap_or(0);
        if k > 128 { vec![128] } else { Vec::new() }
    }
}
