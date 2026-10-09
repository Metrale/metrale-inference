// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The RMSNorm reference as a [`RefImpl`]: [`super::norm`] (case, bound, emulation)
//! behind the common interface. A norm has no data mutation in the catalogue: its contracts
//! detect with `accumulate:<fmt>` and with a wrong-weight-form symbol.
//!
//! Owner: metrale-accuracy.
//! Invariants: none beyond the trait's.

use std::collections::BTreeMap;

use metrale_circuit::pipeline::NodePipeline;

use super::{RefImpl, norm};
use crate::bounded::Bounded;
use crate::case::Case;
use crate::elem::Elem;
use crate::inputs::{InputClass, SplitMix64};
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::points::Shape;

/// 2026-10-09: Columns one pass of a norm block covers: rms_norm.cu's threads read BF16 pairs,
/// at most 1024 threads a row (ops::rms_norm launches min(n, 1024)), so a thread's second pair
/// starts at column 2048.
const PASS: usize = 2 * 1024;

/// 2026-10-09: `rms_norm`: the input, final and per-head q/k norms, plain or after a fused
/// residual add.
pub struct RmsNorm;

impl RefImpl for RmsNorm {
    fn name(&self) -> &'static str {
        "rms_norm"
    }

    fn serves(&self, op: &str) -> bool {
        let base = op.split(':').next().unwrap_or(op);
        matches!(base, "rms_norm" | "final_norm" | "qk_norm")
    }

    fn lens(&self, shape: &Shape, _pipeline: &NodePipeline) -> BTreeMap<String, u64> {
        // 2026-10-09: A shape `dims` refuses gives no length; the plan then names the missing
        // reduction and `fill` the reason.
        norm::dims(shape)
            .map(|(_, n)| BTreeMap::from([("k".to_string(), n as u64)]))
            .unwrap_or_default()
    }

    fn fill(
        &self,
        case: &mut Case,
        plan: &Plan,
        shape: &Shape,
        class: InputClass,
        stream: &dyn Fn(&str) -> SplitMix64,
    ) -> Result<(), String> {
        norm::fill(case, plan, shape, class, stream)
    }

    fn reference(&self, case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
        norm::reference(case, plan, idx)
    }

    fn emulate(
        &self,
        case: &Case,
        plan: &Plan,
        acc: Option<Elem>,
        variant: u32,
        idx: &[usize],
    ) -> Result<Vec<f64>, String> {
        norm::emulate(case, plan, acc, variant, idx)
    }

    fn mutate(
        &self,
        _case: &mut Case,
        m: &Mutation,
        _rng: &mut SplitMix64,
    ) -> Result<Vec<usize>, String> {
        Err(format!("`{}` does not apply to a norm", m.name()))
    }

    fn strides(&self, _case: &Case) -> Vec<usize> {
        vec![PASS]
    }
}
