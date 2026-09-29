// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The circuit executor: run a model's decode from its fused circuit plan instead of
//! the hand-written layer loops. At boot it instantiates the model's circuit, fuses it under the
//! live policy with the kernels the loaded modules contain, lays its buffers out in one
//! workspace and compiles a straight-line [`program::Program`]; a decode step then runs that
//! program between the legacy prologue (embedding, metadata upload) and the logits.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Every emitter launches through an existing `ops::*` function; the executor adds no launch
//!   code of its own.
//! - Anything the circuit does not model (a layer feature, a switch, a head feature, a plan
//!   kernel no emitter launches) refuses the build; a decode never runs a plan that
//!   misdescribes the model.
//! - The workspace is allocated once, at build, and never moves, so captured graphs stay valid.

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
}

/// 2026-09-28: A built executor: the decode program and the workspace it runs in.
pub struct CircuitExec {
    /// 2026-09-28: Decode at one row.
    pub decode: Program,
    /// 2026-09-28: The plan `decode` was compiled from.
    pub decode_plan: FusionPlan,
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
        let plan = fuse(
            &loaded.circuit,
            &loaded.rules,
            &available,
            &b.policy,
            Mode::Decode,
            1,
        )?;
        let layout = compile::layout(&loaded.circuit, &plan)?;
        let buffers = plan_buffers_with(&loaded.circuit, &plan, 1, &layout)?;
        let table = kernels::KernelTable::resolve(b.gpu, &present);
        let workspace_bytes = buffers.arena_bytes.max(1);
        let workspace = b
            .gpu
            .alloc(workspace_bytes as usize)
            .context("allocating the circuit workspace")?;
        let inputs = compile::Inputs {
            config: b.config,
            kernels: &table,
            fixed: &b.fixed,
            layers: &layers,
            head: &b.head,
        };
        let decode = match compile::compile(
            &loaded.circuit,
            &plan,
            &layout,
            &buffers,
            workspace,
            &inputs,
        ) {
            Ok(p) => p,
            Err(e) => {
                b.gpu.free(workspace).ok();
                return Err(e);
            }
        };
        tracing::info!(
            "circuit decode: {} launches/step, plan {} ({} rules, {:?}), workspace {} KiB",
            decode.launches.len(),
            &plan.digest[..12],
            loaded.rules.len(),
            b.fusions,
            workspace_bytes / 1024
        );
        Ok(CircuitExec {
            decode,
            decode_plan: plan,
            rules_digest: loaded.rules_digest,
            fusions: b.fusions,
            workspace,
            workspace_bytes,
        })
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
#[path = "exec_tests.rs"]
mod exec_tests;
