// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: A compiled circuit program: the straight-line launches of one plan, with every
//! host decision (kernel, grid, sizes, pointers into the model's fixed buffers and the
//! workspace) made at compile time.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - A step reads only [`StepEnv`]: the stream, each row's GDN state pointers and the
//!   block-table width uploaded for the step. Everything else a launch uses was fixed when it
//!   was compiled, so a CUDA-graph capture of [`Program::run`] replays correctly for any step
//!   with the same state pointers (the graph caches key on the sequence's slot).
//! - `run` issues no synchronize and no device-to-host copy, so it may be captured.
//! - A program holds exactly `FusionPlan::launches` launches.

use anyhow::{Context, Result};
use metrale_circuit::Mode;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-09-28: One sequence's recurrent state in one GatedDeltaNet layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnState {
    /// 2026-09-28: The h state (FP32).
    pub h: DevicePtr,
    /// 2026-09-28: The conv1d window.
    pub conv: DevicePtr,
}

/// 2026-09-28: What varies between two runs of one program.
pub struct StepEnv<'a> {
    /// 2026-09-28: The backend.
    pub gpu: &'a dyn GpuBackend,
    /// 2026-09-28: The stream the step runs (and is captured) on.
    pub stream: u64,
    /// 2026-09-28: Per layer, each row's GDN state (row `i` = sequence `i`, padding rows
    /// included); empty for a layer without one.
    pub gdn: &'a [Vec<GdnState>],
    /// 2026-09-28: `AttnMetadataDev::max_blocks_per_seq` of this step.
    pub max_blocks_per_seq: u32,
}

impl StepEnv<'_> {
    /// 2026-09-28: Row `row`'s GDN state in layer `layer`; an error when there is none.
    pub fn gdn_state(&self, layer: usize, row: usize) -> Result<GdnState> {
        self.gdn
            .get(layer)
            .and_then(|l| l.get(row))
            .copied()
            .with_context(|| format!("layer {layer} has no GDN state for row {row}"))
    }
}

/// 2026-09-28: The body of one launch.
pub(crate) type RunFn = Box<dyn Fn(&StepEnv<'_>) -> Result<()> + Send + Sync>;

/// 2026-09-28: One kernel launch.
pub struct Launch {
    /// 2026-09-28: Index of the plan group it belongs to.
    pub group: usize,
    /// 2026-09-28: `module::func` it launches.
    pub kernel: String,
    pub(crate) run: RunFn,
}

/// 2026-09-28: The compiled launches of one plan.
pub struct Program {
    /// 2026-09-28: The plan's mode.
    pub mode: Mode,
    /// 2026-09-28: The plan's padded rows.
    pub rows: u64,
    /// 2026-09-28: The plan's digest (`metrale_circuit::digest::plan_digest`).
    pub plan_digest: String,
    /// 2026-09-28: The launches, in order.
    pub launches: Vec<Launch>,
}

impl Program {
    /// 2026-09-28: Issue every launch on `env.stream`.
    pub fn run(&self, env: &StepEnv<'_>) -> Result<()> {
        for l in &self.launches {
            (l.run)(env).with_context(|| format!("circuit group {} ({})", l.group, l.kernel))?;
        }
        Ok(())
    }
}
