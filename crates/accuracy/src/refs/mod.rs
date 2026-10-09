// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: References, one per op class, in bounded arithmetic, each with the case builder
//! that draws its operands, the sample of outputs it compares, a conforming emulation and its
//! data mutations. [`Reference`] is the closed set a contract's `reference` names.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - A reference reads its formats from the plan only ([`crate::plan`]).
//! - Every sample contains the structural indices of the output (tile, group, alignment and
//!   shard edges) besides seeded random ones.

use std::collections::BTreeMap;

use metrale_circuit::pipeline::{NodePipeline, StepKind, Value};

use crate::bounded::Bounded;
use crate::case::Case;
use crate::elem::Elem;
use crate::inputs::{InputClass, SplitMix64, structural_indices};
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::points::Shape;

pub mod linear;
pub mod linear_mutate;
pub mod quant;

/// 2026-10-09: The references a contract can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reference {
    /// 2026-10-09: A projection ([`linear`]).
    Linear,
}

const REFS: [(Reference, &str); 1] = [(Reference::Linear, "linear")];

/// 2026-10-09: Random output columns a sample adds to the structural ones.
const EXTRA_COLUMNS: usize = 48;

impl Reference {
    /// 2026-10-09: The contract spelling.
    pub fn name(self) -> &'static str {
        REFS.iter()
            .find(|(r, _)| *r == self)
            .map_or("?", |(_, n)| n)
    }

    /// 2026-10-09: Parse the contract spelling.
    pub fn parse(s: &str) -> Option<Self> {
        REFS.iter().find(|(_, n)| *n == s).map(|(r, _)| *r)
    }

    /// 2026-10-09: The ops a reference can stand for (the contract's `op` must be one).
    pub fn serves(self, op: &str) -> bool {
        let base = op.split(':').next().unwrap_or(op);
        match self {
            Reference::Linear => matches!(
                base,
                "linear" | "lm_head" | "router" | "expert_gate_up" | "expert_down"
            ),
        }
    }

    /// 2026-10-09: Lengths of the reduced dimensions a contract's reduction may name.
    pub fn lens(self, shape: &Shape, pipeline: &NodePipeline) -> BTreeMap<String, u64> {
        let mut m = BTreeMap::from([("k".to_string(), shape.in_dim)]);
        match self {
            Reference::Linear => {
                let group = pipeline
                    .inputs
                    .first()
                    .and_then(|f| quant::Layout::of(*f))
                    .and_then(|l| l.k_group())
                    .or_else(|| {
                        pipeline
                            .steps
                            .iter()
                            .find(|s| s.kind == StepKind::Weight)
                            .and_then(|s| match &s.value {
                                Value::Weight { stored, .. } => {
                                    quant::Layout::of(*stored).and_then(|l| l.k_group())
                                }
                                _ => None,
                            })
                    });
                if let Some(g) = group {
                    m.insert("group".to_string(), g as u64);
                }
            }
        }
        m
    }

    /// 2026-10-09: Draw the operands of `shape` for `class` into `case`.
    pub fn fill(
        self,
        case: &mut Case,
        plan: &Plan,
        shape: &Shape,
        class: InputClass,
        seed: u64,
        key: &str,
    ) -> Result<(), String> {
        let stream =
            |t: &str| SplitMix64::keyed(seed, &[&case.family, &case.kernel, key, class.name(), t]);
        match self {
            Reference::Linear => {
                let dims = linear::Dims {
                    rows: shape.rows as usize,
                    k: shape.in_dim as usize,
                    n: shape.out_dim as usize,
                };
                let (mut rx, mut rw) = (stream("x"), stream("w"));
                linear::fill(case, plan, dims, class, &mut rx, &mut rw)
            }
        }
    }

    /// 2026-10-09: The flat output indices compared: every row, the structural columns
    /// (both ends, both sides of every multiple of 64 and of the scale group, every shard edge)
    /// and seeded random ones.
    pub fn sample(self, case: &Case, rng: &mut SplitMix64) -> Vec<usize> {
        let (rows, cols) = (case.out.0[0], case.out.0[1]);
        let mut strides = vec![64usize];
        if let Some(g) = case
            .tensors
            .get("w_block")
            .map(|b| case.tensor("w").map_or(0, |w| w.dims[1] / b.dims[1]))
        {
            strides.push(g);
        }
        let mut colset = structural_indices(cols, &strides, EXTRA_COLUMNS, rng);
        for s in &case.split {
            for c in [s.lo.saturating_sub(1), s.lo, s.hi.saturating_sub(1), s.hi] {
                if c < cols {
                    colset.push(c);
                }
            }
        }
        colset.sort_unstable();
        colset.dedup();
        (0..rows)
            .flat_map(|r| colset.iter().map(move |c| r * cols + c))
            .collect()
    }

    /// 2026-10-09: The bounded reference at `idx`.
    pub fn reference(
        self,
        case: &Case,
        plan: &Plan,
        idx: &[usize],
    ) -> Result<Vec<Bounded>, String> {
        match self {
            Reference::Linear => linear::reference(case, plan, idx),
        }
    }

    /// 2026-10-09: The conforming emulation at `idx` in bracketing `variant`
    /// ([`crate::emulate::VARIANTS`]), with the accumulator replaced by `acc`.
    pub fn emulate(
        self,
        case: &Case,
        plan: &Plan,
        acc: Option<Elem>,
        variant: u32,
        idx: &[usize],
    ) -> Result<Vec<f64>, String> {
        match self {
            Reference::Linear => linear::emulate(case, plan, acc, variant, idx),
        }
    }

    /// 2026-10-09: Apply a data mutation; the flat output indices it can reach.
    pub fn mutate(
        self,
        case: &mut Case,
        m: &Mutation,
        rng: &mut SplitMix64,
    ) -> Result<Vec<usize>, String> {
        match self {
            Reference::Linear => {
                let (rows, cols) = (case.out.0[0], case.out.0[1]);
                let reach = linear_mutate::mutate(case, m, rng)?;
                Ok((0..rows)
                    .flat_map(|r| reach.iter().map(move |c| r * cols + c))
                    .collect())
            }
        }
    }
}
