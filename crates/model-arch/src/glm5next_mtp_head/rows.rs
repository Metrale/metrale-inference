// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The drafter's context rows (whole-prompt prefill and catch-up), split out of
//! `glm5next_mtp_head.rs` on 2026-10-11 unchanged.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants: the row space is dense (a row's KV slot is its place in the state's block table,
//! its indexer row and RoPE position its pair key).

use super::*;

impl Glm5NextMtpHead {
    /// 2026-09-25: Append `tokens.len() - 1` drafter context rows: row `r` is pair key
    /// `row_base + r` = `(embed(tokens[r + 1]), hiddens row r)`. Used for both the whole-prompt
    /// prefill and the catch-up feed; the only difference between them is `row_base`.
    ///
    /// The row space is dense: a row's KV slot, indexer row and RoPE position are all
    /// `seq_len` (see `Glm5NextDsaLayer::write_kv_row`), so every pair key from 0 up is
    /// written and slot == key.
    ///
    /// Unless `METRALE_GLM_MTP_PREFILL_FULL=1`, a context row runs only `input_norm` and the
    /// DSA cache writes (`drafter_write_kv_row`): no MoE, no attention, no `lm_head`, because
    /// its block output is discarded.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rows_impl(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let st = match state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
        {
            Some(s) => s,
            None => return Ok(0),
        };
        // 2026-09-25: Rows must append exactly at the drafter's current length, or the dense
        // row space grows a hole and every later RoPE position is wrong. 2026-10-11: A state
        // without a KV region writes nothing.
        if st.seq_len != row_base || tokens.len() < 2 || st.kv_region.is_none() {
            return Ok(0);
        }
        let h = self.hidden;
        let rows = tokens.len() - 1;
        if row_base + rows > self.max_seq_len {
            return Ok(0);
        }
        let gpu = ctx.gpu;
        let dbg = metrale_model_layers::speculative::mtp_refeed_debug();
        let prefill_full = self.levers.prefill_full;
        // 2026-10-11: Two rows or more run in passes (`ctx_rows.rs`); the per-row loop below
        // stays for one row and for the diagnostic arms (full block, per-row fingerprints).
        if rows >= 2 && !prefill_full && !dbg && self.ctx_rows_batched() {
            self.ctx_rows_impl(tokens, hiddens, row_base, rows, st, ctx, stream)?;
            return Ok(rows);
        }
        let mut kv = self.kv_cache.lock();
        let Glm5NextMtpProposerState {
            dsa,
            seq_len,
            block_table,
            concat,
            x,
            ..
        } = st;
        for r in 0..rows {
            let embed_row = self
                .embed_tokens
                .weight
                .offset(tokens[r + 1] as usize * h * 2);
            // 2026-10-11: Row r is pair key `row_base + r`.
            let embed =
                (!super::hidden_half::zeroes_embedding(ctx.levers.mtp_zero_pos0, row_base + r))
                    .then_some(embed_row);
            self.embed_half(gpu, embed, *concat, stream)?;
            // 2026-10-11: A context row is always a target row; `x` stages its target norm
            // before the `eh_proj` below writes it.
            self.hidden_half(ctx, hiddens.offset(r * h * 2), true, *x, *concat, stream)?;
            ops::dense_gemv(
                gpu,
                self.gemv_k,
                *concat,
                &self.module.eh_proj,
                *x,
                h as u32,
                (2 * h) as u32,
                stream,
            )?;
            let dsa_state: &mut dyn LayerState = dsa;
            // 2026-09-25: Diagnostic arm `METRALE_GLM_MTP_PREFILL_FULL=1`: build the row through
            // the full block path a propose uses (`decode_one_for_drafter`) instead of the
            // cache-only write.
            if prefill_full {
                self.module.layer.decode_one_for_drafter(
                    *x,
                    dsa_state,
                    &mut kv,
                    *seq_len,
                    block_table,
                    ctx,
                    stream,
                )?;
            } else {
                // 2026-10-11: `concat` is free once `eh_proj` has read it: the normed row's
                // scratch, so no shared forward buffer is written (`prefill_uses_shared_buffers`).
                self.module.layer.drafter_write_kv_row(
                    *x,
                    *concat,
                    dsa_state,
                    &mut kv,
                    *seq_len,
                    block_table,
                    ctx,
                    stream,
                )?;
            }
            if dbg {
                let fp = metrale_model_layers::speculative::hidden_fingerprint(
                    gpu,
                    hiddens.offset(r * h * 2),
                    h,
                );
                tracing::info!(
                    "GLM_MTP_DBG ctx row slot={} key={} tok={} fp_hidden={fp:016x}",
                    *seq_len,
                    row_base + r,
                    tokens[r + 1],
                );
            }
            *seq_len += 1;
        }
        Ok(rows)
    }
}
