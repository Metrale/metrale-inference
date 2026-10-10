// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The gated delta rule reference as a [`RefImpl`]: [`super::gdn`] (geometry,
//! formats, case, mutation) and [`super::gdn_head`] (bound, emulation) behind the common
//! interface.
//!
//! Owner: metrale-accuracy.
//! Invariants: none beyond the trait's.

use std::collections::BTreeMap;

use metrale_circuit::pipeline::NodePipeline;

use super::{RefImpl, gdn, gdn_head, grid_sample};
use crate::bounded::Bounded;
use crate::case::Case;
use crate::elem::Elem;
use crate::inputs::{InputClass, SplitMix64};
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::points::Shape;

/// 2026-10-09: `gdn_recurrence`: one decode step of the gated delta rule.
pub struct GdnRecurrence;

impl RefImpl for GdnRecurrence {
    fn name(&self) -> &'static str {
        "gdn_recurrence"
    }

    fn serves(&self, op: &str) -> bool {
        op == "gdn_recurrence"
    }

    /// 2026-10-09: `k` (each q/k dot product), `state_k` (a thread's share of the clamp's sum
    /// of squares: one value column, k_dim rows) and `state_v` (the columns across threads).
    fn lens(&self, shape: &Shape, _pipeline: &NodePipeline) -> BTreeMap<String, u64> {
        let get = |n: &str| shape.runtime.get(n).and_then(|v| v.parse::<u64>().ok());
        let mut m = BTreeMap::new();
        if let Some(k) = get("k_dim") {
            m.insert("k".to_string(), k);
            m.insert("state_k".to_string(), k);
        }
        if let Some(v) = get("v_dim") {
            m.insert("state_v".to_string(), v);
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
        gdn::fill(case, plan, gdn::Geom::of_shape(shape)?, class, stream)
    }

    fn reference(&self, case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String> {
        gdn_head::reference(case, plan, idx)
    }

    fn emulate(
        &self,
        case: &Case,
        plan: &Plan,
        acc: Option<Elem>,
        variant: u32,
        idx: &[usize],
    ) -> Result<Vec<f64>, String> {
        gdn_head::emulate(case, plan, acc, variant, idx)
    }

    fn mutate(
        &self,
        case: &mut Case,
        m: &Mutation,
        _rng: &mut SplitMix64,
    ) -> Result<Vec<usize>, String> {
        gdn::mutate(case, m)
    }

    /// 2026-10-09: The default sample plus, in every sampled row, both sides of every state
    /// head's first element (the state starts at `o_len`, so a stride from column 0 does not
    /// fall on them).
    fn sample(&self, case: &Case, rng: &mut SplitMix64) -> Vec<usize> {
        let mut idx = grid_sample(case, &self.strides(case), rng);
        if let Ok(g) = gdn::Geom::of_case(case) {
            let cols = g.cols();
            let mut rows: Vec<usize> = idx.iter().map(|i| i / cols).collect();
            rows.dedup();
            for r in rows {
                for h in 0..g.v_heads {
                    let first = g.o_len() + h * g.k_dim * g.v_dim;
                    idx.extend([r * cols + first - 1, r * cols + first]);
                }
            }
            idx.sort_unstable();
            idx.dedup();
        }
        idx
    }

    /// 2026-10-09: The value-head edges of `o` and the value-column edges of the state (both
    /// multiples of `v_dim` from column 0), and the `o`/state boundary.
    fn strides(&self, case: &Case) -> Vec<usize> {
        gdn::Geom::of_case(case).map_or_else(|_| Vec::new(), |g| vec![g.v_dim, g.o_len()])
    }
}
