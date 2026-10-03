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
//! - A program holds exactly `FusionPlan::launches` kernel launches and `FusionPlan::copies`
//!   copies.

use anyhow::{Context, Result};
use metrale_circuit::Mode;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layer::AttnMetadataDev;

/// 2026-09-29: The most rollback points an MTP verify writes: one per row but the last, at
/// K = 4.
pub const MAX_VERIFY_STEPS: usize = 3;

/// 2026-09-28: One sequence's recurrent state in one GatedDeltaNet layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnState {
    /// 2026-09-28: The h state (FP32).
    pub h: DevicePtr,
    /// 2026-09-28: The conv1d window.
    pub conv: DevicePtr,
    /// 2026-09-29: The h state after verify row `t` (`SsmLayerState::h_state_intermediates`);
    /// NULL where the sequence has none. Only a verify reads them.
    pub h_steps: [DevicePtr; MAX_VERIFY_STEPS],
    /// 2026-09-29: The conv window after verify row `t`, indexed like `h_steps`.
    pub conv_steps: [DevicePtr; MAX_VERIFY_STEPS],
}

/// 2026-10-03: What a prefill pass supplies at run time (LIFECYCLE-DESIGN.md 15.4): its row
/// count and position, and the metadata it uploaded. A prefill program is eager and compiled per
/// row bucket, so its emitters size grids from these, as the legacy layers do.
#[derive(Clone, Copy)]
pub struct PrefillStep {
    /// 2026-10-03: Rows of this pass (`T`, the legacy `proc_count`).
    pub tokens: u32,
    /// 2026-10-03: The absolute position of row 0 (`effective_seq_len_start`).
    pub start: u32,
    /// 2026-10-03: Rows below this are not written to the KV cache (the shared prefix-cache
    /// blocks a restore recomputes over).
    pub kv_write_floor: u32,
    /// 2026-10-03: The pass's attention metadata: positions and slots, and on a paged pass the
    /// block table and sequence length. `None` for the head's steps, which read none.
    pub meta: Option<AttnMetadataDev>,
}

impl PrefillStep {
    /// 2026-10-03: The pass's attention metadata; an error for a step that carries none.
    pub fn meta(&self) -> Result<AttnMetadataDev> {
        self.meta
            .context("a prefill launch read attention metadata the step does not carry")
    }
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
    /// 2026-10-03: A prefill pass's run-time facts; `None` for every other mode.
    pub prefill: Option<PrefillStep>,
}

impl StepEnv<'_> {
    /// 2026-10-03: The prefill pass this step runs; an error outside a prefill.
    pub fn prefill(&self) -> Result<PrefillStep> {
        self.prefill
            .context("a prefill launch ran without the pass's rows and metadata")
    }

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

/// 2026-09-29: What a launch puts on the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchKind {
    /// 2026-09-29: A kernel.
    Kernel,
    /// 2026-09-29: A copy-engine transfer.
    Copy,
}

/// 2026-09-28: One kernel launch or copy.
pub struct Launch {
    /// 2026-09-28: Index of the plan group it belongs to.
    pub group: usize,
    /// 2026-09-28: `module::func` it launches, or `copy` for a transfer.
    pub kernel: String,
    /// 2026-09-29: Kernel or copy.
    pub kind: LaunchKind,
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
    /// 2026-10-03: The launches of each layer and of the head, in order (LIFECYCLE-DESIGN.md
    /// 15.4): what a host driver composes when it runs part of a program (a prefill pass that is
    /// not the last runs no head).
    pub segments: Vec<Segment>,
}

/// 2026-10-03: Where a run of launches belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentOf {
    /// 2026-10-03: The embedding, before the first layer.
    Embed,
    /// 2026-10-03: Layer `i`.
    Layer(usize),
    /// 2026-10-03: A step of the head, by its first node's op (`final_norm`, `lm_head`,
    /// `argmax`): a driver runs host work between them (the prefix cache's exact restore puts
    /// the snapshot's hidden row between the final norm and the LM head).
    Head(metrale_circuit::OpKind),
}

/// 2026-10-03: A contiguous run of launches of one part of the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub of: SegmentOf,
    pub launches: std::ops::Range<usize>,
}

impl Program {
    /// 2026-09-28: Issue every launch on `env.stream`.
    pub fn run(&self, env: &StepEnv<'_>) -> Result<()> {
        self.run_launches(0..self.launches.len(), env)
    }

    /// 2026-10-03: Issue the launches of every segment `keep` selects, in order.
    pub fn run_segments(
        &self,
        keep: impl Fn(SegmentOf) -> bool,
        env: &StepEnv<'_>,
    ) -> Result<()> {
        for s in self.segments.iter().filter(|s| keep(s.of)) {
            self.run_launches(s.launches.clone(), env)?;
        }
        Ok(())
    }

    fn run_launches(&self, range: std::ops::Range<usize>, env: &StepEnv<'_>) -> Result<()> {
        for l in &self.launches[range] {
            (l.run)(env).with_context(|| format!("circuit group {} ({})", l.group, l.kernel))?;
        }
        Ok(())
    }
}

/// 2026-09-29: A program another component runs in place of its own forward (the MTP draft
/// head's step), without GDN state. 2026-09-30: One program per row count: one row for a
/// single sequence's draft, n rows for the batched propose's.
pub trait DraftRunner: Send + Sync {
    /// 2026-09-30: Whether a program for `rows` draft rows was compiled.
    fn serves(&self, rows: u64) -> bool;
    /// 2026-09-29: Issue the `rows`-row program on `stream` for a step whose block table is
    /// `max_blocks_per_seq` wide; an error when none was compiled for `rows`.
    fn run_draft(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        rows: u64,
        max_blocks_per_seq: u32,
    ) -> Result<()>;
}

/// 2026-09-30: The draft head's compiled programs, one per row count, ascending (one row
/// first).
pub struct DraftPrograms {
    pub programs: Vec<(Program, metrale_circuit::FusionPlan)>,
}

impl DraftRunner for DraftPrograms {
    fn serves(&self, rows: u64) -> bool {
        self.programs.iter().any(|(p, _)| p.rows == rows)
    }

    fn run_draft(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        rows: u64,
        max_blocks_per_seq: u32,
    ) -> Result<()> {
        let (p, _) = self
            .programs
            .iter()
            .find(|(p, _)| p.rows == rows)
            .with_context(|| format!("no draft program was compiled for {rows} rows"))?;
        p.run(&StepEnv {
            gpu,
            stream,
            gdn: &[],
            max_blocks_per_seq,
            prefill: None,
        })
    }
}
