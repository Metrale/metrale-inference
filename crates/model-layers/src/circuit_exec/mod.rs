// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The circuit executor: run a model's decode from its fused circuit plan instead of
//! the hand-written layer loops. At boot it instantiates the model's circuit, fuses it under the
//! live policy with the kernels the loaded modules contain, lays its buffers out in one
//! workspace and compiles a straight-line [`program::Program`] for single-sequence decode and
//! one per multi-sequence batch width; a decode step then runs the program for its width
//! between the legacy prologue (embedding, metadata upload) and the logits.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Every emitter launches through an existing `ops::*` function; the executor adds no launch
//!   code of its own.
//! - Anything the circuit does not model (a layer feature, a switch, a head feature, a plan
//!   kernel no emitter launches) refuses the build; a decode never runs a plan that
//!   misdescribes the model.
//! - The workspace is allocated once, at build, and never moves, so captured graphs stay valid.
//!   Every program lays its buffers out from the workspace base: programs run one at a time on
//!   the model's stream, so they share it, and it is sized for the largest.

pub mod bindings;
pub mod compile;
mod emitters;
pub mod kernels;
pub mod policy;
pub mod program;
pub mod sources;

use anyhow::{Context, Result, bail};
use metrale_circuit::planner::plan_buffers_with;
use metrale_circuit::{FusionPlan, Instance, Mode, Numerics, Policy, fuse};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

pub use bindings::{
    AttnFacts, BoundWeight, CircuitBindings, CircuitLayer, GdnFacts, HeadBinding, MixerFacts,
    RopeFacts, WeightSlot,
};
pub use compile::Fixed;
pub use program::{GdnState, Program, StepEnv};

/// 2026-09-28: Which rules a build may select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fusions {
    /// 2026-09-28: Every rule whose kernel is present.
    All,
    /// 2026-09-28: Only `reference` rules: the kernels of `bit_identical` rules are treated as
    /// absent, so the plan is today's routing exactly. The parity harness compares the two.
    ReferenceOnly,
}

/// 2026-09-28: The compiled kernel modules of the served target, `(module, PTX)`: the fuser's
/// view of which kernels exist ([`kernels::available_in`]).
#[derive(Clone)]
pub struct TargetModules(pub Vec<(&'static str, &'static [u8])>);

impl std::fmt::Debug for TargetModules {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TargetModules({} modules)", self.0.len())
    }
}

/// 2026-09-28: What a build reads.
pub struct Boot<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub config: &'a ModelConfig,
    /// 2026-09-28: The model's levers, as its legacy layers read them (`ForwardContext::levers`).
    pub levers: &'a crate::layers::ops::ModelLevers,
    pub instance: &'a Instance,
    pub policy: Policy,
    pub layers: Vec<Option<CircuitLayer>>,
    pub head: HeadBinding,
    pub fixed: Fixed,
    pub fusions: Fusions,
    pub modules: &'a TargetModules,
    /// 2026-09-28: The padded multi-sequence batch widths to compile (the decode graph ladder
    /// up to the serve's widest batch).
    pub multi_seq_rows: Vec<u64>,
}

/// 2026-09-28: A built executor: the decode program and the workspace it runs in.
pub struct CircuitExec {
    /// 2026-09-28: Decode at one row.
    pub decode: Program,
    /// 2026-09-28: The plan `decode` was compiled from.
    pub decode_plan: FusionPlan,
    /// 2026-09-28: Multi-sequence decode, one program per padded width, ascending; each with
    /// the plan it was compiled from.
    pub multi_seq: Vec<(Program, FusionPlan)>,
    /// 2026-09-28: SHA-256 of the FUSIONS.toml the plan was chosen from.
    pub rules_digest: String,
    /// 2026-09-28: Which rules the build allowed.
    pub fusions: Fusions,
    workspace: DevicePtr,
    workspace_bytes: u64,
}

impl CircuitExec {
    /// 2026-09-28: Build the executor.
    pub fn build(b: Boot<'_>) -> Result<Self> {
        let mut served = b.instance.clone();
        served.shape = sources::served_shape(&b.instance.shape, &sources::arch_shape(b.config)?)?;
        let src = sources::sources(&served)?;
        let loaded = match &served.precision {
            // 2026-09-28: The served model's own policy: its parsed quantization_config under
            // the published tier and kernel capabilities, with the instance's engine formats.
            metrale_circuit::PrecisionSpec::Policy { engine, .. } => {
                let policy = metrale_config::WeightQuantPolicy::for_checkpoint(
                    crate::layers::weight_quantization(),
                    b.config.quantization_config.as_ref(),
                    crate::layers::kernel_caps(),
                );
                metrale_circuit::load_with(
                    &served,
                    src,
                    &metrale_circuit::precision_policy::PolicyPrecision::new(policy, engine),
                )?
            }
            metrale_circuit::PrecisionSpec::Table(_) => metrale_circuit::load(&served, src)?,
        };
        let unmodelled = policy::unmodelled_switches(b.levers);
        if !unmodelled.is_empty() {
            bail!(
                "the circuit does not model these switches: {}",
                unmodelled.join(", ")
            );
        }
        let layers = compile::check_bindings(&loaded.circuit, &b.layers, &b.head)?;
        let present = kernels::available_in(&loaded.rules, &b.modules.0)?;
        let mut available = present.clone();
        if b.fusions == Fusions::ReferenceOnly {
            for r in &loaded.rules {
                if matches!(r.numerics, Numerics::BitIdentical { .. }) {
                    for k in &r.kernels {
                        available.kernels.remove(k);
                    }
                }
            }
        }
        let table = kernels::KernelTable::resolve(b.gpu, &present);
        let inputs = compile::Inputs {
            gpu: b.gpu,
            config: b.config,
            kernels: &table,
            fixed: &b.fixed,
            layers: &layers,
            head: &b.head,
        };
        let shapes = std::iter::once((Mode::Decode, 1))
            .chain(b.multi_seq_rows.iter().map(|&r| (Mode::MultiSeq, r)));
        let mut laid = Vec::new();
        for (mode, rows) in shapes {
            let plan = fuse(
                &loaded.circuit,
                &loaded.rules,
                &available,
                &b.policy,
                mode,
                rows,
            )
            .with_context(|| format!("fusing {mode:?} at {rows} rows"))?;
            let layout = compile::layout(&loaded.circuit, &plan)?;
            let buffers = plan_buffers_with(&loaded.circuit, &plan, rows, &layout)?;
            laid.push((plan, layout, buffers));
        }
        let workspace_bytes = laid
            .iter()
            .map(|(_, _, buf)| buf.arena_bytes)
            .max()
            .unwrap_or(0)
            .max(1);
        let workspace = b
            .gpu
            .alloc(workspace_bytes as usize)
            .context("allocating the circuit workspace")?;
        let mut programs = Vec::with_capacity(laid.len());
        for (plan, layout, buffers) in laid {
            match compile::compile(
                &loaded.circuit,
                &plan,
                &layout,
                &buffers,
                workspace,
                &inputs,
            ) {
                Ok(p) => programs.push((p, plan)),
                Err(e) => {
                    b.gpu.free(workspace).ok();
                    return Err(
                        e.context(format!("compiling {:?} at {} rows", plan.mode, plan.rows))
                    );
                }
            }
        }
        let mut programs = programs.into_iter();
        let (decode, plan) = programs.next().context("no decode program")?;
        let multi_seq: Vec<_> = programs.collect();
        tracing::info!(
            "circuit decode: {} launches/step, plan {} ({} rules, {:?}), {} multi-seq widths, \
             workspace {} KiB",
            decode.launches.len(),
            &plan.digest[..12],
            loaded.rules.len(),
            b.fusions,
            multi_seq.len(),
            workspace_bytes / 1024
        );
        Ok(CircuitExec {
            decode,
            decode_plan: plan,
            multi_seq,
            rules_digest: loaded.rules_digest,
            fusions: b.fusions,
            workspace,
            workspace_bytes,
        })
    }

    /// 2026-09-28: The multi-sequence program for `rows` padded rows, if one was compiled.
    pub fn multi_seq_program(&self, rows: u64) -> Option<&Program> {
        self.multi_seq
            .iter()
            .find(|(p, _)| p.rows == rows)
            .map(|(p, _)| p)
    }

    /// 2026-09-28: One digest over every compiled plan, in order (decode, then the
    /// multi-sequence widths ascending): what a record of this forward discloses.
    pub fn plans_digest(&self) -> String {
        metrale_circuit::digest::plans_digest(
            std::iter::once(self.decode_plan.digest.as_str())
                .chain(self.multi_seq.iter().map(|(_, p)| p.digest.as_str())),
        )
    }

    /// 2026-09-28: Bytes of the workspace.
    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    /// 2026-09-28: Free the workspace. The caller must first destroy every graph that captured
    /// a program of this executor.
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.workspace)
    }
}

#[cfg(test)]
#[path = "exec_fixture.rs"]
mod exec_fixture;
#[cfg(test)]
#[path = "exec_multi_tests.rs"]
mod exec_multi_tests;
#[cfg(test)]
#[path = "exec_tests.rs"]
mod exec_tests;
