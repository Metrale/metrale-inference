// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: A built executor's accessors: the program a step runs, the disclosed digest, and
//! freeing the workspace. Split from `mod.rs`.
//!
//! Owner: model-layers circuit executor.
//! Invariants: see `mod.rs`.

use anyhow::{Context, Result};
use metrale_circuit::Mode;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::program::{DraftRunner, GdnState, Program};
use super::{CircuitExec, routes};

impl CircuitExec {
    /// 2026-09-28: The multi-sequence program for `rows` padded rows, if one was compiled.
    pub fn multi_seq_program(&self, rows: u64) -> Option<&Program> {
        self.multi_seq
            .iter()
            .find(|(p, _)| p.rows == rows)
            .map(|(p, _)| p)
    }

    /// 2026-09-30: The program a multi-sequence step of `rows` padded rows runs, given its GDN
    /// states: the arm of the first runtime route whose condition holds, else the primary one.
    pub fn multi_seq_step(&self, rows: u64, gdn: &[Vec<GdnState>]) -> Result<&Program> {
        for r in self
            .routes
            .iter()
            .filter(|r| r.program.mode == Mode::MultiSeq && r.program.rows == rows)
        {
            if routes::holds(&r.route, &self.gdn_pitch, gdn)? {
                return Ok(&r.program);
            }
        }
        self.multi_seq_program(rows)
            .with_context(|| format!("no circuit program was compiled for {rows} rows"))
    }

    /// 2026-09-29: The draft head's program, if one was compiled.
    pub fn draft_runner(&self) -> Option<std::sync::Arc<dyn DraftRunner>> {
        self.draft
            .as_ref()
            .map(|d| d.clone() as std::sync::Arc<dyn DraftRunner>)
    }

    /// 2026-09-29: The verify program for `k` rows, if one was compiled.
    pub fn verify_program(&self, k: u64) -> Option<&Program> {
        self.verify
            .iter()
            .find(|(p, _)| p.rows == k)
            .map(|(p, _)| p)
    }

    /// 2026-09-28: One digest over every compiled plan, in order (decode, then the
    /// multi-sequence widths ascending, then the verify widths, the draft step and, 2026-09-30,
    /// the runtime routes' arms, and 2026-10-03 the prefill programs): what a record of this
    /// forward discloses.
    pub fn plans_digest(&self) -> String {
        let prefill = self
            .prefill
            .iter()
            .flat_map(|p| p.programs.iter().map(|p| p.plan.digest.as_str()));
        metrale_circuit::digest::plans_digest(
            std::iter::once(self.decode_plan.digest.as_str()).chain(
                self.multi_seq
                    .iter()
                    .chain(&self.verify)
                    .map(|(_, p)| p.digest.as_str())
                    .chain(
                        self.draft
                            .iter()
                            .flat_map(|d| d.programs.iter().map(|(_, p)| p.digest.as_str())),
                    )
                    .chain(self.routes.iter().map(|r| r.plan.digest.as_str()))
                    .chain(prefill),
            ),
        )
    }

    /// 2026-09-28: Bytes of the workspace.
    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    /// 2026-09-28: Free the workspace. The caller must first destroy every graph that captured
    /// a program of this executor.
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(p) = &self.draft {
            anyhow::ensure!(
                std::sync::Arc::strong_count(p) == 1,
                "the draft program is still installed; remove it before freeing the workspace"
            );
        }
        if let Some(lane) = &self.lane {
            lane.free(gpu)?;
        }
        gpu.free(self.workspace)
    }
}
