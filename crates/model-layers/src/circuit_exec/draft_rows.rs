// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The MTP draft head's n-row programs (the batched propose,
//! `forward_batch_position`): one per width from 2 rows to the widest batch the head proposes.
//! A width the rules do not cover, or whose plan the head would not run (an LM-head or
//! projection kernel other than the plan's, the confidences off), gets no program, and the
//! head drafts those sequences one at a time on its single-row program.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Each width is fused, laid out and compiled exactly as the primary programs are, in the
//!   one workspace; only the refusal is softer (a skipped width, logged with its reason).

use metrale_circuit::planner::{BufferPlan, Layout, plan_buffers_with};
use metrale_circuit::{AvailableKernels, Circuit, FusionPlan, Loaded, Mode, Policy, fuse};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::compile::{self, Inputs};
use super::program::Program;

/// 2026-09-30: A fused and laid-out n-row draft plan, before the workspace exists.
pub(super) struct LaidDraft {
    plan: FusionPlan,
    layout: Layout,
    pub(super) buffers: BufferPlan,
}

/// 2026-09-30: Fuse and lay out the draft at each width in `2..=max`, skipping (with a log) a
/// width the rules or the layout refuse.
pub(super) fn lay_out(
    loaded: &Loaded,
    available: &AvailableKernels,
    policy: &Policy,
    max: u64,
) -> Vec<LaidDraft> {
    (2..=max)
        .filter_map(|n| {
            let laid = (|| -> anyhow::Result<LaidDraft> {
                let plan = fuse(
                    &loaded.circuit,
                    &loaded.rules,
                    available,
                    policy,
                    Mode::Draft,
                    n,
                )?;
                let layout = compile::layout(&loaded.circuit, &plan)?;
                let buffers = plan_buffers_with(&loaded.circuit, &plan, n, &layout)?;
                Ok(LaidDraft {
                    plan,
                    layout,
                    buffers,
                })
            })();
            laid.map_err(|e| skipped(n, &e)).ok()
        })
        .collect()
}

/// 2026-09-30: Compile each laid-out width at `workspace`, skipping (with a log) one an
/// emitter refuses.
pub(super) fn compile_all(
    circuit: &Circuit,
    laid: Vec<LaidDraft>,
    workspace: DevicePtr,
    inputs: &Inputs<'_>,
) -> Vec<(Program, FusionPlan)> {
    laid.into_iter()
        .filter_map(|l| {
            match compile::compile(circuit, &l.plan, &l.layout, &l.buffers, workspace, inputs) {
                Ok(p) => Some((p, l.plan)),
                Err(e) => {
                    skipped(l.plan.rows, &e);
                    None
                }
            }
        })
        .collect()
}

fn skipped(n: u64, e: &anyhow::Error) {
    tracing::info!(
        "circuit: no {n}-row draft program ({e:#}); those batches draft each sequence alone"
    );
}
