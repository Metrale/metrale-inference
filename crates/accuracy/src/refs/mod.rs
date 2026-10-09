// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: References, one per op class, in bounded arithmetic. Each implements [`RefImpl`]:
//! the case builder that draws its operands, the sample of outputs it compares, the bounded
//! reference, a conforming emulation and its data mutations. [`REFS`] is the closed set a
//! contract's `reference` names; adding a reference is a module and one row there.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - A reference reads its formats from the plan only ([`crate::plan`]).
//! - Every sample contains the structural indices of the output (tile, group, alignment and
//!   shard edges) besides seeded random ones.
//! - A reference returns values BEFORE the kernel's final rounding into the output format; the
//!   comparison accounts for that rounding ([`crate::compare::bounded`]).

use std::collections::BTreeMap;

use metrale_circuit::pipeline::NodePipeline;

use crate::bounded::Bounded;
use crate::case::Case;
use crate::elem::Elem;
use crate::inputs::{InputClass, SplitMix64, structural_indices};
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::points::Shape;

pub mod act_quant;
pub mod attention;
pub mod attention_impl;
pub mod conv;
pub mod conv_head;
pub mod conv_impl;
pub mod gdn;
pub mod gdn_head;
pub mod gdn_impl;
pub mod linear;
pub mod linear_impl;
pub mod linear_mutate;
pub mod moe_grouped;
pub mod moe_grouped_impl;
pub mod norm;
pub mod norm_impl;
pub mod quant;
pub mod rope;
pub mod rope_impl;

/// 2026-10-09: What a reference provides. Every method but [`RefImpl::sample`] is required.
pub trait RefImpl: Sync {
    /// 2026-10-09: The contract spelling.
    fn name(&self) -> &'static str;
    /// 2026-10-09: The op keys it can stand for (a contract's `op` must be one).
    fn serves(&self, op: &str) -> bool;
    /// 2026-10-09: Lengths of the reduced dimensions a contract's reduction may name.
    fn lens(&self, shape: &Shape, pipeline: &NodePipeline) -> BTreeMap<String, u64>;
    /// 2026-10-09: Draw the operands of `shape` for `class` into `case` (`stream(name)` gives
    /// each tensor its own keyed stream).
    fn fill(
        &self,
        case: &mut Case,
        plan: &Plan,
        shape: &Shape,
        class: InputClass,
        stream: &dyn Fn(&str) -> SplitMix64,
    ) -> Result<(), String>;
    /// 2026-10-09: The bounded reference at flat output indices `idx`.
    fn reference(&self, case: &Case, plan: &Plan, idx: &[usize]) -> Result<Vec<Bounded>, String>;
    /// 2026-10-09: The conforming emulation at `idx` in bracketing `variant`, with the
    /// accumulator replaced by `acc`.
    fn emulate(
        &self,
        case: &Case,
        plan: &Plan,
        acc: Option<Elem>,
        variant: u32,
        idx: &[usize],
    ) -> Result<Vec<f64>, String>;
    /// 2026-10-09: Apply a data mutation; the flat output indices it reaches that a sample may
    /// miss (empty when it reaches every compared output).
    fn mutate(
        &self,
        case: &mut Case,
        m: &Mutation,
        rng: &mut SplitMix64,
    ) -> Result<Vec<usize>, String>;
    /// 2026-10-09: The flat output indices compared, for a `[rows, cols]` output: every row
    /// (or the row-tile edges and seeded rows above [`ALL_ROWS`]), the structural columns (both
    /// ends, the edges of 64 and of the reference's own strides at both ends, every shard edge)
    /// and seeded random ones.
    fn sample(&self, case: &Case, rng: &mut SplitMix64) -> Vec<usize> {
        grid_sample(case, &self.strides(case), rng)
    }
    /// 2026-10-09: Column strides whose edges the sample covers besides 64.
    fn strides(&self, case: &Case) -> Vec<usize>;
}

/// 2026-10-09: The references, by contract spelling.
pub const REFS: &[&dyn RefImpl] = &[
    &linear_impl::Linear,
    &act_quant::ActQuant,
    &norm_impl::RmsNorm,
    &rope_impl::Rope,
    &attention_impl::PagedAttention,
    &gdn_impl::GdnRecurrence,
    &conv_impl::Conv1dL2norm,
    &moe_grouped_impl::MoeGrouped,
];

/// 2026-10-09: Random output columns a sample adds to the structural ones.
const EXTRA_COLUMNS: usize = 48;

/// 2026-10-09: Up to this many rows every row is compared; above, the row-tile edges (this
/// stride) and [`EXTRA_ROWS`] random rows.
pub const ALL_ROWS: usize = 16;

/// 2026-10-09: Random rows a sample adds above [`ALL_ROWS`] rows.
const EXTRA_ROWS: usize = 8;

/// 2026-10-09: A named reference.
#[derive(Clone, Copy)]
pub struct Reference(&'static dyn RefImpl);

impl std::fmt::Debug for Reference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.name())
    }
}

impl Reference {
    /// 2026-10-09: The reference a contract names.
    pub fn parse(s: &str) -> Option<Self> {
        REFS.iter().find(|r| r.name() == s).map(|r| Reference(*r))
    }

    /// 2026-10-09: Its implementation.
    pub fn imp(self) -> &'static dyn RefImpl {
        self.0
    }

    /// 2026-10-09: See [`RefImpl::serves`].
    pub fn serves(self, op: &str) -> bool {
        self.0.serves(op)
    }

    /// 2026-10-09: See [`RefImpl::lens`].
    pub fn lens(self, shape: &Shape, pipeline: &NodePipeline) -> BTreeMap<String, u64> {
        self.0.lens(shape, pipeline)
    }

    /// 2026-10-09: Draw the case's operands from streams keyed by (seed, family, kernel, key,
    /// class, tensor).
    pub fn fill(
        self,
        case: &mut Case,
        plan: &Plan,
        shape: &Shape,
        class: InputClass,
        seed: u64,
        key: &str,
    ) -> Result<(), String> {
        let (family, kernel) = (case.family.clone(), case.kernel.clone());
        let stream =
            move |t: &str| SplitMix64::keyed(seed, &[&family, &kernel, key, class.name(), t]);
        self.0.fill(case, plan, shape, class, &stream)
    }

    /// 2026-10-09: See [`RefImpl::sample`].
    pub fn sample(self, case: &Case, rng: &mut SplitMix64) -> Vec<usize> {
        self.0.sample(case, rng)
    }

    /// 2026-10-09: The bounded reference at `idx`, computed in parallel chunks.
    pub fn reference(
        self,
        case: &Case,
        plan: &Plan,
        idx: &[usize],
    ) -> Result<Vec<Bounded>, String> {
        par(idx, |part| self.0.reference(case, plan, part))
    }

    /// 2026-10-09: The conforming emulation at `idx`, computed in parallel chunks.
    pub fn emulate(
        self,
        case: &Case,
        plan: &Plan,
        acc: Option<Elem>,
        variant: u32,
        idx: &[usize],
    ) -> Result<Vec<f64>, String> {
        par(idx, |part| self.0.emulate(case, plan, acc, variant, part))
    }

    /// 2026-10-09: See [`RefImpl::mutate`].
    pub fn mutate(
        self,
        case: &mut Case,
        m: &Mutation,
        rng: &mut SplitMix64,
    ) -> Result<Vec<usize>, String> {
        self.0.mutate(case, m, rng)
    }
}

/// 2026-10-09: The default sample of a `[rows, cols]` output (see [`RefImpl::sample`]).
pub fn grid_sample(case: &Case, extra_strides: &[usize], rng: &mut SplitMix64) -> Vec<usize> {
    let (rows, cols) = (case.out.0[0], case.out.0[1]);
    let mut strides = vec![64usize];
    strides.extend_from_slice(extra_strides);
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
    let rowset: Vec<usize> = if rows <= ALL_ROWS {
        (0..rows).collect()
    } else {
        structural_indices(rows, &[ALL_ROWS], EXTRA_ROWS, rng)
    };
    rowset
        .iter()
        .flat_map(|r| colset.iter().map(move |c| r * cols + c))
        .collect()
}

/// 2026-10-09: `f` over `idx` in contiguous chunks on scoped threads, concatenated in order.
/// Each output element is computed from the case alone, so the thread count changes the speed
/// and never a value.
fn par<T: Send>(
    idx: &[usize],
    f: impl Fn(&[usize]) -> Result<Vec<T>, String> + Sync,
) -> Result<Vec<T>, String> {
    const CHUNK: usize = 256;
    if idx.len() <= CHUNK {
        return f(idx);
    }
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(idx.len().div_ceil(CHUNK));
    let per = idx.len().div_ceil(workers);
    let parts: Vec<Result<Vec<T>, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = idx.chunks(per).map(|c| s.spawn(|| f(c))).collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("a reference worker panicked".into()))
            })
            .collect()
    });
    let mut out = Vec::with_capacity(idx.len());
    for p in parts {
        out.extend(p?);
    }
    Ok(out)
}
