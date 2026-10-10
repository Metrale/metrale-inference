// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The RoPE reference as a [`RefImpl`]: [`super::rope`] (case, bound, emulation)
//! and its one data mutation, the wrong rope base.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - A mutation other than `wrong_rope_base` is refused, never a silent no-op.

use std::collections::BTreeMap;

use metrale_circuit::pipeline::NodePipeline;

use super::{RefImpl, rope};
use crate::bounded::Bounded;
use crate::case::Case;
use crate::elem::{self, Elem};
use crate::inputs::{InputClass, SplitMix64};
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::points::Shape;

/// 2026-10-09: The factor `wrong_rope_base` multiplies the rope base by.
const WRONG_BASE_FACTOR: f64 = 10.0;

/// 2026-10-09: `rope`: rotary position embedding of Q and K, rotate-half pairs.
pub struct Rope;

impl RefImpl for Rope {
    fn name(&self) -> &'static str {
        "rope"
    }

    fn serves(&self, op: &str) -> bool {
        op.split(':').next() == Some("rope")
    }

    fn lens(&self, _shape: &Shape, _pipeline: &NodePipeline) -> BTreeMap<String, u64> {
        BTreeMap::new()
    }

    fn fill(
        &self,
        case: &mut Case,
        plan: &Plan,
        shape: &Shape,
        class: InputClass,
        stream: &dyn Fn(&str) -> SplitMix64,
    ) -> Result<(), String> {
        rope::fill(case, plan, shape, class, stream)
    }

    fn reference(&self, case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
        rope::reference(case, plan, idx)
    }

    fn emulate(
        &self,
        case: &Case,
        plan: &Plan,
        acc: Option<Elem>,
        _variant: u32,
        idx: &[usize],
    ) -> Result<Vec<f64>, String> {
        rope::emulate(case, plan, acc, idx)
    }

    fn mutate(
        &self,
        case: &mut Case,
        m: &Mutation,
        _rng: &mut SplitMix64,
    ) -> Result<Vec<usize>, String> {
        match m {
            Mutation::WrongRopeBase => {
                let theta = case.scalar("theta")?;
                let wrong = elem::F32
                    .round(theta * WRONG_BASE_FACTOR)
                    .ok_or("ten times the rope base overflows f32")?;
                case.scalars.insert("theta".into(), wrong);
                Ok(Vec::new())
            }
            other => Err(format!("`{}` does not apply to a rope", other.name())),
        }
    }

    fn strides(&self, case: &Case) -> Vec<usize> {
        rope::layout(case).map(|l| l.strides()).unwrap_or_default()
    }
}
