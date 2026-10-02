// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: `forward_batch_position_circuit`: one n-row draft position on the circuit's
//! draft program, the counterpart of `forward_batch_position`. The host steps are the legacy
//! forward's (`stage.rs`); the program runs everything from the pre-fc norms to the argmax.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - The program reads row i's target hidden from the stream buffer (`hidden_states`) at
//!   `i * hidden`; the first position stacks the caller's rows there, and a chained position's
//!   rows already are there (`chain_hidden`, the program's stream output).
//! - The program writes the ids and the confidences (`argmax_bf16_batch_lp`), so they are
//!   read back as the legacy forward reads them with D-Cut on.

use super::*;

impl MtpHead {
    /// 2026-09-30: One draft position for the n rows on `runner`'s n-row program; the
    /// arguments are `forward_batch_position`'s, with the confidences required.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_batch_position_circuit(
        &self,
        runner: &dyn crate::circuit_exec::DraftRunner,
        tokens: &[u32],
        hiddens: &[DevicePtr],
        positions: &[usize],
        states: &mut [&mut MtpProposerState],
        ctx: &ForwardContext,
        stream: u64,
        out_ids: &mut [u32],
        out_lp: &mut [f32],
    ) -> Result<()> {
        let n = tokens.len();
        ensure!(
            hiddens.len() == n && out_ids.len() == n && out_lp.len() == n,
            "propose_batch: length mismatch"
        );
        self.stage_batch_embeds(tokens, ctx, stream)?;
        let row = ctx.config.hidden_size * 2;
        let stacked = ctx.buffers.hidden_states();
        let span = stacked.0..stacked.0 + (n * row) as u64;
        ensure!(
            hiddens
                .iter()
                .enumerate()
                .all(|(i, &hp)| hp == stacked.offset(i * row)
                    || hp.0 + row as u64 <= span.start
                    || hp.0 >= span.end),
            "propose_batch: a target hidden overlaps another row of the stream buffer"
        );
        for (i, &hp) in hiddens.iter().enumerate() {
            let at = stacked.offset(i * row);
            if hp != at {
                ctx.gpu.copy_d2d_async(hp, at, row, stream)?;
            }
        }
        // 2026-09-30: The host buffer is held until the readback below synchronizes.
        let (_meta_buf, max_blocks) = self.stage_batch_meta(states, positions, ctx, stream)?;
        runner.run_draft(ctx.gpu, stream, n as u64, max_blocks)?;
        Self::read_batch_ids(ctx, true, out_ids, Some(out_lp))?;
        Self::finish_batch_rows(states, positions);
        Ok(())
    }
}
