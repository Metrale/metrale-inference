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
mod draft_rows;
mod emitters;
pub mod fixed;
pub mod kernels;
pub mod policy;
pub mod prefill;
pub mod program;
pub mod routes;
pub mod sources;
pub mod state_bind;
pub mod verify_batch;

use anyhow::{Context, Result, bail};
use metrale_circuit::planner::plan_buffers_with;
use metrale_circuit::{FusionPlan, Instance, Mode, Numerics, Policy, fuse};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

pub use bindings::{
    AttnFacts, BoundWeight, CircuitBindings, CircuitLayer, DraftBinding, DraftRows, GdnFacts,
    HeadBinding, MixerFacts, RopeFacts, WeightSlot,
};
pub use compile::{DraftFixed, Fixed};
pub use program::{DraftPrograms, DraftRunner, GdnState, Program, StepEnv};

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
    /// 2026-09-30: The checkpoint's `config.json`, which the circuit's config map turns into the
    /// served shape ([`sources::checkpoint_shape`]).
    pub config_json: &'a str,
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
    /// 2026-09-29: The MTP verify widths `K` to compile (none without speculative decode).
    pub verify_rows: Vec<u64>,
    /// 2026-09-29: The MTP draft head, when one is loaded; its buffers are `fixed.draft`.
    pub draft: Option<CircuitLayer>,
    /// 2026-09-30: The widest batched MTP verify, in rows (`Σ k`), when the serve verifies
    /// several sequences in one forward; `None` otherwise.
    pub verify_batch_rows: Option<u64>,
    /// 2026-09-30: The widest batched propose (`propose_batch_max`), whose widths from 2 get
    /// n-row draft programs; `None` when the head proposes one sequence at a time.
    pub draft_rows: Option<u64>,
    /// 2026-10-03: The SSM pool the state programs bind to.
    pub state_pool: state_bind::StatePool,
    /// 2026-10-03: The legacy forward's buffer arena, where prefill plans place their edges.
    pub arena: &'a metrale_gpu_runtime::buffers::BufferArena,
    /// 2026-10-03: The widest prefill pass to compile programs for; `None` builds none (the
    /// serve's prefill then runs the legacy layers, disclosed).
    pub prefill_max_tokens: Option<u64>,
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
    /// 2026-09-29: The single-sequence MTP verify, one program per `K`, ascending.
    pub verify: Vec<(Program, FusionPlan)>,
    /// 2026-09-29: The MTP draft head's single-row step and (2026-09-30) its n-row ones, which
    /// the head runs itself (`DraftRunner`).
    pub draft: Option<std::sync::Arc<DraftPrograms>>,
    /// 2026-09-30: The runtime routes' arms, beside the primary program of each mode and row
    /// count they apply to ([`routes`]).
    pub routes: Vec<routes::RoutedProgram>,
    /// 2026-09-30: Per layer, the `(h, conv)` slot pitch of its GDN state; `None` elsewhere.
    pub gdn_pitch: Vec<Option<(usize, usize)>>,
    /// 2026-09-30: The batched MTP verify's programs, compiled per row table.
    pub verify_batch: Option<verify_batch::VerifyBatch>,
    /// 2026-10-03: The state programs, bound to the SSM pool (`state_bind`).
    pub state: state_bind::StatePrograms,
    /// 2026-10-03: The prefill programs; `None` when the build compiled none.
    pub prefill: Option<prefill::PrefillPrograms>,
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
        let from_checkpoint = sources::checkpoint_shape(b.config_json, b.config.vocab_size as u64)?;
        served.shape = sources::served_shape(&b.instance.shape, &from_checkpoint)?;
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
        let unmodelled = policy::unmodelled_switches(
            b.levers,
            !b.verify_rows.is_empty() || b.verify_batch_rows.is_some(),
        );
        if !unmodelled.is_empty() {
            bail!(
                "the circuit does not model these switches: {}",
                unmodelled.join(", ")
            );
        }
        let layers = bindings::check_bindings(&loaded.circuit, &b.layers, &b.head)?;
        let state = b.state_pool.bind(&loaded.circuit, b.config)?;
        if let Some(d) = &b.draft {
            if !d.unmodelled.is_empty() {
                bail!(
                    "the circuit does not model this draft head: {}",
                    d.unmodelled.join(", ")
                );
            }
            anyhow::ensure!(b.fixed.draft.is_some(), "a draft head without its buffers");
        }
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
            draft: b.draft.as_ref(),
            arena: Some(b.arena),
        };
        let shapes = std::iter::once((Mode::Decode, 1))
            .chain(b.multi_seq_rows.iter().map(|&r| (Mode::MultiSeq, r)))
            .chain(b.verify_rows.iter().map(|&r| (Mode::Verify, r)))
            .chain(b.draft.iter().map(|_| (Mode::Draft, 1)));
        routes::check_known(&loaded.runtime)?;
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
            let set = (loaded.rules.as_slice(), loaded.runtime.as_slice());
            let arms = metrale_circuit::runtime::route_arms(
                &loaded.circuit,
                set,
                &available,
                &b.policy,
                &plan,
            )
            .with_context(|| format!("fusing the runtime routes of {mode:?} at {rows} rows"))?;
            for (route, plan) in
                std::iter::once((None, plan)).chain(arms.into_iter().map(|(r, p)| (Some(r), p)))
            {
                let layout = compile::layout(&loaded.circuit, &plan)?;
                let buffers = plan_buffers_with(&loaded.circuit, &plan, rows, &layout)?;
                laid.push((route, plan, layout, buffers));
            }
        }
        let mut workspace_bytes = laid
            .iter()
            .map(|(_, _, _, buf)| buf.arena_bytes)
            .max()
            .unwrap_or(0)
            .max(1);
        // 2026-09-30: The batched verify compiles later, per table, in this same workspace; only
        // for an instance whose batched-verify plans are checked in (`Instance::verify_batch`).
        let verify_batch_rows = b
            .verify_batch_rows
            .filter(|_| !b.instance.verify_batch.is_empty());
        if b.verify_batch_rows.is_some() && verify_batch_rows.is_none() {
            tracing::info!(
                "circuit: `{}` states no batched-verify plans; each sequence verifies alone",
                b.instance.recipe
            );
        }
        if let Some(max) = verify_batch_rows {
            for t in verify_batch::sizing_tables(max) {
                let a = verify_batch::arena_bytes(
                    &loaded.circuit,
                    &loaded.rules,
                    &available,
                    &b.policy,
                    &t,
                )?;
                workspace_bytes = workspace_bytes.max(a);
            }
        }
        let draft_laid = match (&b.fixed.draft, b.draft_rows) {
            (Some(d), Some(max)) if b.draft.is_some() && d.rows.is_some() => {
                draft_rows::lay_out(&loaded, &available, &b.policy, max)
            }
            _ => Vec::new(),
        };
        for l in &draft_laid {
            workspace_bytes = workspace_bytes.max(l.buffers.arena_bytes);
        }
        let workspace = b
            .gpu
            .alloc(workspace_bytes as usize)
            .context("allocating the circuit workspace")?;
        let mut programs = Vec::with_capacity(laid.len());
        let mut routed = Vec::new();
        for (route, plan, layout, buffers) in laid {
            match compile::compile(
                &loaded.circuit,
                &plan,
                &layout,
                &buffers,
                workspace,
                &inputs,
            ) {
                Ok(program) => match route {
                    None => programs.push((program, plan)),
                    Some(route) => routed.push(routes::RoutedProgram {
                        route,
                        program,
                        plan,
                    }),
                },
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
        let (draft, rest): (Vec<_>, Vec<_>) = programs.partition(|(p, _)| p.mode == Mode::Draft);
        let (multi_seq, verify): (Vec<_>, Vec<_>) = rest
            .into_iter()
            .partition(|(p, _)| p.mode == Mode::MultiSeq);
        let draft = draft.into_iter().next().map(|one| {
            let mut programs = vec![one];
            programs.extend(draft_rows::compile_all(
                &loaded.circuit,
                draft_laid,
                workspace,
                &inputs,
            ));
            std::sync::Arc::new(DraftPrograms { programs })
        });
        tracing::info!(
            "circuit decode: {} launches/step, plan {} ({} rules, {:?}), {} multi-seq widths, \
             {} verify widths, {} draft widths, {} runtime-route programs, workspace {} KiB",
            decode.launches.len(),
            &plan.digest[..12],
            loaded.rules.len(),
            b.fusions,
            multi_seq.len(),
            verify.len(),
            draft.as_ref().map_or(0, |d| d.programs.len()),
            routed.len(),
            workspace_bytes / 1024
        );
        let gdn_pitch = layers
            .iter()
            .map(|l| match l.mixer {
                MixerFacts::Gdn(g) => Some((g.h_slot_bytes as usize, g.conv_state_bytes as usize)),
                MixerFacts::Attention(_) => None,
            })
            .collect();
        let prefill = match b.prefill_max_tokens.map(|max_tokens| {
            let pb = prefill::PrefillBoot {
                circuit: &loaded.circuit,
                rules: &loaded.rules,
                runtime: &loaded.runtime,
                available: &available,
                policy: &b.policy,
                arena: b.arena,
                max_tokens,
            };
            prefill::PrefillPrograms::build(&pb, &inputs)
        }) {
            Some(Err(e)) => {
                b.gpu.free(workspace).ok();
                return Err(e.context("building the prefill programs"));
            }
            built => built.transpose()?,
        };
        let verify_batch = verify_batch_rows.map(|max| {
            verify_batch::VerifyBatch::new(
                verify_batch::Parts {
                    circuit: loaded.circuit.clone(),
                    rules: loaded.rules.clone(),
                    available: available.clone(),
                    policy: b.policy.clone(),
                    kernels: table.clone(),
                    fixed: b.fixed.clone(),
                    layers: layers.clone(),
                    head: b.head.clone(),
                    config: b.config.clone(),
                },
                workspace,
                workspace_bytes,
                max,
            )
        });
        Ok(CircuitExec {
            decode,
            decode_plan: plan,
            multi_seq,
            verify,
            draft,
            routes: routed,
            gdn_pitch,
            verify_batch,
            state,
            prefill,
            rules_digest: loaded.rules_digest,
            fusions: b.fusions,
            workspace,
            workspace_bytes,
        })
    }
}

#[path = "exec_access.rs"]
mod exec_access;

#[cfg(test)]
#[path = "exec_declared_tests.rs"]
mod exec_declared_tests;
#[cfg(test)]
#[path = "exec_draft_rows_tests.rs"]
mod exec_draft_rows_tests;
#[cfg(test)]
#[path = "exec_draft_tests.rs"]
mod exec_draft_tests;
#[cfg(test)]
#[path = "exec_fixture.rs"]
mod exec_fixture;
#[cfg(test)]
#[path = "exec_fixture_run.rs"]
mod exec_fixture_run;
#[cfg(test)]
#[path = "exec_multi_tests.rs"]
mod exec_multi_tests;
#[cfg(test)]
#[path = "exec_prefill_tests.rs"]
mod exec_prefill_tests;
#[cfg(test)]
#[path = "exec_tests.rs"]
mod exec_tests;
#[cfg(test)]
#[path = "exec_verify_batch_tests.rs"]
mod exec_verify_batch_tests;
#[cfg(test)]
#[path = "exec_verify_tests.rs"]
mod exec_verify_tests;
