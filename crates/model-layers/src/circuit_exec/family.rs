// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Plan-group emitters for a model family whose layers keep their own buffers: the
//! family registers one [`GroupEmitter`] per FUSIONS emitter id, and compiling a block of a plan
//! asks each group's emitter to turn the group's typed fields (its kernels, its members' ops and
//! the rank circuit's dims) into the family's own typed launch description `S`. The family runs
//! that description through its existing launch helpers, so a plan-executed block launches what
//! its hand-written path launches, with every choice taken from the plan.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - No family is named here: a family hands its emitters to [`EmitterRegistry::new`] (FACE).
//! - An emitter id is registered once; a group whose emitter is not registered is an error
//!   naming the emitter and the rule, never a skip.
//! - A plan kernel resolves through the device port by its `(module, func)`; an absent one is an
//!   error naming it.

use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail, ensure};
use metrale_circuit::ir::Node;
use metrale_circuit::{Circuit, FusionPlan, Group, KernelId, Mode};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-10-23: One group of a plan, as an emitter reads it.
#[derive(Clone, Copy)]
pub struct PlanGroup<'a> {
    pub circuit: &'a Circuit,
    pub group: &'a Group,
    /// 2026-10-23: Its index in the plan.
    pub index: usize,
    /// 2026-10-23: The plan's mode and padded rows.
    pub mode: Mode,
    pub rows: u64,
}

impl<'a> PlanGroup<'a> {
    /// 2026-10-23: Every group of `plan` over `circuit`, in execution order.
    pub fn all(circuit: &'a Circuit, plan: &'a FusionPlan) -> impl Iterator<Item = Self> + 'a {
        plan.groups
            .iter()
            .enumerate()
            .map(move |(index, group)| Self {
                circuit,
                group,
                index,
                mode: plan.mode,
                rows: plan.rows,
            })
    }

    /// 2026-10-23: Member `i`, in pattern order.
    pub fn node(&self, i: usize) -> &'a Node {
        &self.circuit.nodes[self.group.nodes[i]]
    }

    /// 2026-10-23: The members' template-local ids, in pattern order.
    pub fn locals(&self) -> Vec<&'a str> {
        (0..self.group.nodes.len())
            .map(|i| self.node(i).local.as_str())
            .collect()
    }

    /// 2026-10-23: The block template of the first member.
    pub fn block(&self) -> &'a str {
        self.node(0).block.as_str()
    }

    /// 2026-10-23: The layer of the first member; `None` in the prologue, epilogue and draft.
    pub fn layer(&self) -> Option<usize> {
        self.node(0).layer
    }

    /// 2026-10-23: `block [locals]`, as the plan reports name a site.
    pub fn site(&self) -> String {
        format!("{} [{}]", self.block(), self.locals().join(","))
    }

    /// 2026-10-23: Refuse a group whose members are not exactly `locals`, in order.
    pub fn expect_locals(&self, emitter: &str, locals: &[&str]) -> Result<()> {
        let got = self.locals();
        ensure!(
            got == locals,
            "emitter `{emitter}` cannot launch group {} ({}, rule `{}`); it takes [{}]",
            self.index,
            self.site(),
            self.group.rule,
            locals.join(",")
        );
        Ok(())
    }

    /// 2026-10-23: Kernel `i` of the group.
    pub fn kernel(&self, i: usize) -> Result<&'a KernelId> {
        self.group.kernels.get(i).with_context(|| {
            format!(
                "group {} (rule `{}`) names {} kernels; the emitter reads kernel {i}",
                self.index,
                self.group.rule,
                self.group.kernels.len()
            )
        })
    }

    /// 2026-10-23: Refuse a group that does not name exactly `n` kernels.
    pub fn expect_kernels(&self, emitter: &str, n: usize) -> Result<()> {
        ensure!(
            self.group.kernels.len() == n,
            "emitter `{emitter}`: group {} (rule `{}`) names {} kernels, it launches {n}",
            self.index,
            self.group.rule,
            self.group.kernels.len()
        );
        Ok(())
    }

    /// 2026-10-23: A dim of the planned (rank) circuit.
    pub fn dim(&self, name: &str) -> Result<u64> {
        self.circuit
            .dims
            .get(name)
            .copied()
            .with_context(|| format!("the planned circuit has no dim `{name}`"))
    }
}

/// 2026-10-23: The emitter of one FUSIONS emitter id for a family whose compile target is `S`.
pub trait GroupEmitter<S: ?Sized>: Sync {
    /// 2026-10-23: The FUSIONS `emitter` id it serves.
    fn id(&self) -> &'static str;
    /// 2026-10-23: Record the group's launches in `site`; refuse a group it was not written for.
    fn emit(&self, g: &PlanGroup<'_>, site: &mut S) -> Result<()>;
}

/// 2026-10-23: A family's emitters by id.
pub struct EmitterRegistry<S: ?Sized + 'static> {
    by_id: BTreeMap<&'static str, &'static dyn GroupEmitter<S>>,
}

impl<S: ?Sized + 'static> EmitterRegistry<S> {
    /// 2026-10-23: The registry of `emitters`; an error when two serve one id.
    pub fn new(emitters: &[&'static dyn GroupEmitter<S>]) -> Result<Self> {
        let mut by_id = BTreeMap::new();
        for &e in emitters {
            if by_id.insert(e.id(), e).is_some() {
                bail!("two emitters are registered for `{}`", e.id());
            }
        }
        Ok(Self { by_id })
    }

    /// 2026-10-23: The registered ids, sorted.
    pub fn ids(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.by_id.keys().copied()
    }

    /// 2026-10-23: Whether `id` has an emitter.
    pub fn serves(&self, id: &str) -> bool {
        self.by_id.contains_key(id)
    }

    /// 2026-10-23: Emit `g` through the emitter its plan names.
    pub fn emit(&self, g: &PlanGroup<'_>, site: &mut S) -> Result<()> {
        let e = self.by_id.get(g.group.emitter.as_str()).ok_or_else(|| {
            anyhow!(
                "no emitter `{}` is registered (group {} {}, rule `{}`)",
                g.group.emitter,
                g.index,
                g.site(),
                g.group.rule
            )
        })?;
        e.emit(g, site).with_context(|| {
            format!(
                "emitting group {} {} (rule `{}`)",
                g.index,
                g.site(),
                g.group.rule
            )
        })
    }
}

/// 2026-10-23: The handle of a plan kernel on `gpu`.
pub fn plan_kernel(gpu: &dyn GpuBackend, k: &KernelId) -> Result<KernelHandle> {
    gpu.kernel(&k.module, &k.func)
        .with_context(|| format!("the plan names `{k}`, which this target cannot launch"))
}

/// 2026-10-23: One rank's fused plans over its circuit, as a layer installs plan execution
/// from them: one plan per `(mode, rows)` the engine fused, in that order. The executed
/// selection is `--forward circuit` (the engine's `ForwardSelect`); this carries only its data.
#[derive(Clone, Copy)]
pub struct RankPlans<'a> {
    pub circuit: &'a Circuit,
    pub plans: &'a [FusionPlan],
}

#[cfg(test)]
#[path = "family_tests.rs"]
mod tests;
