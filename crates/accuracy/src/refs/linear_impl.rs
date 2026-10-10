// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The projection reference as a [`RefImpl`]: [`super::linear`] (case, bound,
//! emulation) and [`super::linear_mutate`] (mutations) behind the common interface.
//!
//! Owner: metrale-accuracy.
//! Invariants: none beyond the trait's.

use std::collections::BTreeMap;

use metrale_circuit::pipeline::{NodePipeline, StepKind, Value};

use super::{RefImpl, linear, linear_mutate, quant};
use crate::bounded::Bounded;
use crate::case::Case;
use crate::elem::Elem;
use crate::inputs::{InputClass, SplitMix64};
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::points::Shape;

/// 2026-10-09: `linear`: linear, lm_head, router and the expert projections run as one weight.
pub struct Linear;

impl RefImpl for Linear {
    fn name(&self) -> &'static str {
        "linear"
    }

    fn serves(&self, op: &str) -> bool {
        let base = op.split(':').next().unwrap_or(op);
        matches!(
            base,
            "linear" | "lm_head" | "router" | "expert_gate_up" | "expert_down"
        )
    }

    fn lens(&self, shape: &Shape, pipeline: &NodePipeline) -> BTreeMap<String, u64> {
        let mut m = BTreeMap::from([("k".to_string(), shape.in_dim)]);
        let weight = pipeline
            .steps
            .iter()
            .find(|s| s.kind == StepKind::Weight)
            .and_then(|s| match &s.value {
                Value::Weight { stored, .. } => {
                    quant::Layout::of(*stored).and_then(|l| l.k_group())
                }
                _ => None,
            });
        let act = pipeline
            .inputs
            .first()
            .and_then(|f| quant::Layout::of(*f))
            .and_then(|l| l.k_group());
        if let Some(g) = act.or(weight) {
            m.insert("group".to_string(), g as u64);
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
        let dims = linear::Dims {
            rows: shape.rows as usize,
            k: shape.in_dim as usize,
            n: shape.out_dim as usize,
        };
        let (mut rx, mut rw) = (stream("x"), stream("w"));
        linear::fill(case, plan, dims, class, &mut rx, &mut rw)
    }

    fn reference(&self, case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
        linear::reference(case, plan, idx)
    }

    fn emulate(
        &self,
        case: &Case,
        plan: &Plan,
        acc: Option<Elem>,
        variant: u32,
        idx: &[usize],
    ) -> Result<Vec<f64>, String> {
        linear::emulate(case, plan, acc, variant, idx)
    }

    fn mutate(
        &self,
        case: &mut Case,
        m: &Mutation,
        rng: &mut SplitMix64,
    ) -> Result<Vec<usize>, String> {
        let (rows, cols) = (case.out.0[0], case.out.0[1]);
        let reach = linear_mutate::mutate(case, m, rng)?;
        Ok((0..rows)
            .flat_map(|r| reach.iter().map(move |c| r * cols + c))
            .collect())
    }

    fn strides(&self, case: &Case) -> Vec<usize> {
        case.tensors
            .get("w_block")
            .and_then(|b| case.tensor("w").ok().map(|w| w.dims[1] / b.dims[1].max(1)))
            .into_iter()
            .collect()
    }
}
