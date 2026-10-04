// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Running the circuit's prefill programs in place of the legacy layer loop of a
//! prefill pass and of its head (M6, LIFECYCLE-DESIGN.md 15.4). The host driver stays the
//! legacy one (chunking, prefix lookup, embedding, metadata upload, snapshots); a pass hands
//! the program its rows, position, KV floor and metadata, and the sequence's GDN state.
//!
//! Owner: model-engine (prefill).
//! Invariants:
//! - A pass at offset 0 runs the `prefill` program, any other the `prefill_chunk` one; the GDN
//!   arm is the pass's exact-replay flag, as the legacy `gdn_exact_replay` is.
//! - A pass the circuit does not run (no prefill programs, a shape they do not cover) runs the
//!   legacy layers and is counted (`circuit_fallbacks`).

use anyhow::Result;
use metrale_circuit::Mode;
use metrale_model_layers::circuit_exec::program::{PrefillStep, SegmentOf, StepEnv};

use super::super::types::TransformerModel;
use super::impl_circuit_run::gdn_states;
use crate::traits::SequenceState;

/// 2026-10-03: The prefill mode of a pass starting at `start`.
pub(super) fn prefill_mode(start: u32) -> Mode {
    if start == 0 {
        Mode::Prefill
    } else {
        Mode::PrefillChunk
    }
}

impl TransformerModel {
    /// 2026-10-03: Run the layers of a prefill pass of `seq` on the circuit; `false` when the
    /// circuit runs no prefill (the caller runs the legacy layers). 2026-10-04: The pass's
    /// embedding first (the program's embed segment from the staged ids, then `splice`'s vision
    /// rows when the pass covers a whole chunk), over the rows the driver embedded before it
    /// decided the pass runs here; at the flip the driver's embedding goes, this stays.
    pub(super) fn circuit_prefill_layers(
        &self,
        seq: &SequenceState,
        step: PrefillStep,
        exact_replay: bool,
        splice: Option<&[u32]>,
        stream: u64,
    ) -> Result<bool> {
        let guard = self.circuit.read();
        let Some(prefill) = guard.as_ref().and_then(|e| e.prefill.as_ref()) else {
            return Ok(false);
        };
        let p = prefill.select(prefill_mode(step.start), step.tokens as u64, exact_replay)?;
        let gdn = gdn_states(self.layers.len(), &[&seq.layer_states])?;
        let env = StepEnv {
            gpu: self.gpu.as_ref(),
            stream,
            gdn: &gdn,
            max_blocks_per_seq: step.meta()?.max_blocks_per_seq,
            prefill: Some(step),
        };
        p.program.run_segments(|s| s == SegmentOf::Embed, &env)?;
        if let Some(chunk) = splice {
            self.prefill_vision_splice(chunk, self.buffers.hidden_states(), stream)?;
        }
        p.program
            .run_segments(|s| matches!(s, SegmentOf::Layer(_)), &env)?;
        Ok(true)
    }

    /// 2026-10-03: Run the head step `op` (`FinalNorm` or `LmHead`) of the last prefill pass on
    /// the circuit, on the pass's last row; `false` when the circuit runs no prefill.
    pub(super) fn circuit_prefill_head(
        &self,
        op: metrale_circuit::OpKind,
        tokens: u32,
        stream: u64,
    ) -> Result<bool> {
        let guard = self.circuit.read();
        let Some(prefill) = guard.as_ref().and_then(|e| e.prefill.as_ref()) else {
            return Ok(false);
        };
        let p = prefill.head(tokens as u64)?;
        let step = PrefillStep {
            tokens,
            start: 0,
            kv_write_floor: 0,
            ids_row0: 0,
            meta: None,
        };
        p.program.run_segments(
            |s| s == SegmentOf::Head(op),
            &StepEnv {
                gpu: self.gpu.as_ref(),
                stream,
                gdn: &[],
                max_blocks_per_seq: 0,
                prefill: Some(step),
            },
        )?;
        Ok(true)
    }
}
