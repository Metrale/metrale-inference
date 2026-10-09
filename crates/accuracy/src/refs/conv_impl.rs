// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The GDN conv step reference as a [`RefImpl`]: [`super::conv`] (geometry,
//! formats, case, mutation) and [`super::conv_head`] (bound, emulation) behind the common
//! interface.
//!
//! Owner: metrale-accuracy.
//! Invariants: none beyond the trait's.

use std::collections::BTreeMap;

use metrale_circuit::pipeline::NodePipeline;

use super::{RefImpl, conv, conv_head};
use crate::bounded::Bounded;
use crate::case::Case;
use crate::elem::Elem;
use crate::inputs::{InputClass, SplitMix64};
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::points::Shape;

/// 2026-10-09: `conv1d_l2norm`: the conv1d update with SiLU and the q/k heads' L2 norm.
pub struct Conv1dL2norm;

impl RefImpl for Conv1dL2norm {
    fn name(&self) -> &'static str {
        "conv1d_l2norm"
    }

    fn serves(&self, op: &str) -> bool {
        op == "conv1d_update"
    }

    /// 2026-10-09: `conv` (the taps of one channel) and `head` (the channels of one
    /// normalized head).
    fn lens(&self, shape: &Shape, _pipeline: &NodePipeline) -> BTreeMap<String, u64> {
        let get = |n: &str| shape.runtime.get(n).and_then(|v| v.parse::<u64>().ok());
        let mut m = BTreeMap::new();
        if let Some(d) = get("d_conv") {
            m.insert("conv".to_string(), d);
        }
        if let Some(k) = get("k_dim") {
            m.insert("head".to_string(), k);
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
        conv::fill(case, plan, conv::Geom::of_shape(shape)?, class, stream)
    }

    fn reference(&self, case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
        conv_head::reference(case, plan, idx)
    }

    fn emulate(
        &self,
        case: &Case,
        plan: &Plan,
        acc: Option<Elem>,
        variant: u32,
        idx: &[usize],
    ) -> Result<Vec<f64>, String> {
        conv_head::emulate(case, plan, acc, variant, idx)
    }

    fn mutate(
        &self,
        case: &mut Case,
        m: &Mutation,
        _rng: &mut SplitMix64,
    ) -> Result<Vec<usize>, String> {
        conv::mutate(case, m)
    }

    /// 2026-10-09: The head and block edges of the output, the q/k-to-v edge, the output to
    /// window edge, and the window's channel edges (multiples of `d_conv` from `dim`, which a
    /// multiple of the block is).
    fn strides(&self, case: &Case) -> Vec<usize> {
        conv::Geom::of_case(case).map_or_else(
            |_| Vec::new(),
            |g| vec![g.head_dim, conv::BLOCK, g.qk_channels, g.dim, g.d_conv],
        )
    }
}
