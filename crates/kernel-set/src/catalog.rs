// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Whole-registry lookups over the compiled targets: enumerate
//! them, or find one by a substring of its model name. [`super::resolve`]
//! picks the target for a checkpoint.
//!
//! 2026-10-24: Over a [`KernelCatalog`], the registry one binary compiled
//! (`metrale_kernels::CATALOG`), so a library that is handed it reads the registry without
//! depending on the crate that compiles it.
//!
//! Owner: kernels crate.
//! Invariants: none beyond the types.

use super::TargetPtxSet;
use super::resolve::{ResolveCandidate, TargetResolveError, resolve_pinned, resolve_target};

/// 2026-10-24: One build's compiled kernel registry: its targets and the two attestations the
/// build script computed for them.
pub struct KernelCatalog {
    /// 2026-10-24: Every compiled target, one entry per target (the generated
    /// `all_ptx_sets()`); empty under the `METRALE_SKIP_BUILD` stub.
    pub all_ptx_sets: fn() -> Vec<TargetPtxSet>,
    /// 2026-10-24: `metrale_kernels::KERNEL_SET_HASH`.
    pub kernel_set_hash: &'static str,
    /// 2026-10-24: `metrale_kernels::TARGET_CLOSURES`.
    pub target_closures: &'static str,
}

impl KernelCatalog {
    /// 2026-09-25: Every compiled kernel target.
    pub fn available_targets(&self) -> Vec<TargetPtxSet> {
        (self.all_ptx_sets)()
    }

    /// 2026-09-25: The first compiled target whose model name contains
    /// `needle` (case-sensitive), or `None`.
    pub fn ptx_for_model(&self, needle: &str) -> Option<TargetPtxSet> {
        (self.all_ptx_sets)()
            .into_iter()
            .find(|t| t.target.model.contains(needle))
    }

    /// 2026-09-25: The compiled target that serves a checkpoint, by the resolution
    /// rules: [`resolve_pinned`] when `pinned_target` (`--kernel-target`) is set,
    /// else [`resolve_target`] over every compiled target. `Ok(None)` when no compiled
    /// target declares the pair.
    pub fn ptx_for_config(
        &self,
        model_type: &str,
        hidden_size: usize,
        model_refs: &[&str],
        pinned_target: Option<&str>,
    ) -> Result<Option<TargetPtxSet>, TargetResolveError> {
        let targets = (self.all_ptx_sets)();
        let candidates: Vec<ResolveCandidate<'_>> = targets
            .iter()
            .map(|t| ResolveCandidate {
                name: t.target.model,
                type_matches: &t.model_type_matches,
                match_names: t.match_names,
            })
            .collect();
        let idx = match pinned_target {
            Some(pin) => Some(resolve_pinned(&candidates, pin, model_type, hidden_size)?),
            None => resolve_target(&candidates, model_type, hidden_size, model_refs)?,
        };
        drop(candidates);
        Ok(idx.and_then(|i| targets.into_iter().nth(i)))
    }

    /// 2026-09-25: The compiled target with exactly this `(model, quant)`
    /// identity, or `None`. For callers that already know the resolved target,
    /// such as the dashboard's kernel table, which looks up the target `serve`
    /// published.
    pub fn ptx_for_exact_target(&self, model: &str, quant: &str) -> Option<TargetPtxSet> {
        (self.all_ptx_sets)()
            .into_iter()
            .find(|t| t.target.model == model && t.target.quant == quant)
    }
}
