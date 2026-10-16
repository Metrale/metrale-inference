// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The drafter's context rows (whole-prompt prefill and catch-up) in passes of up to
//! [`CTX_ROWS_CHUNK`] rows, instead of one row at a time (`rows.rs`): one embedding gather, the
//! input norms over all rows, the two halves interleaved into the `eh_proj` input with two 2-D
//! copies, one `eh_proj`, then the block's KV and indexer rows in one pass
//! (`Glm5NextLayer::drafter_write_kv_rows`). One row at a time cost ~0.35 ms per prompt row,
//! which kept the eager drafter prefill (`METRALE_GLM_MTP_EAGER_PREFILL`) opt-in.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - Row `r` of a call is pair key `row_base + r` = `(embed(tokens[r + 1]), hiddens row r)`, at
//!   the drafter's next KV slot and indexer row, as the one-row path writes it.
//! - Draft-only: the projections take the batched or tile path where the one-row path takes the
//!   GEMV, so a row's bits can differ; the target verifies every draft.
//! - Writes only the head's own scratch, the state's KV region and indexer cache: no shared
//!   forward buffer (`prefill_uses_shared_buffers`).

use super::*;

/// 2026-10-11: Rows per pass; the MTP block's DSA workspace is built for at least this many
/// (`drafter_block_rows`).
pub const CTX_ROWS_CHUNK: usize = 64;

/// 2026-10-11: The passes' scratch, `CTX_ROWS_CHUNK` rows each.
pub(super) struct CtxRowsScratch {
    ids: DevicePtr,
    emb: DevicePtr,
    hid: DevicePtr,
    concat: DevicePtr,
    x: DevicePtr,
    slots: DevicePtr,
    gather_k: KernelHandle,
}

impl CtxRowsScratch {
    /// 2026-10-11: `None` without the embedding gather kernel (the rows then run one at a
    /// time).
    pub(super) fn new(gpu: &dyn GpuBackend, hidden: usize) -> Result<Option<Self>> {
        let gather_k =
            metrale_model_layers::layers::try_kernel(gpu, "embed_from_argmax", "batched_embed");
        if gather_k.0 == 0 {
            return Ok(None);
        }
        let n = CTX_ROWS_CHUNK;
        Ok(Some(Self {
            ids: gpu.alloc(n * 4)?,
            emb: gpu.alloc(n * hidden * 2)?,
            hid: gpu.alloc(n * hidden * 2)?,
            concat: gpu.alloc(n * 2 * hidden * 2)?,
            x: gpu.alloc(n * hidden * 2)?,
            slots: gpu.alloc(n * 8)?,
            gather_k,
        }))
    }
}

/// 2026-10-11: The KV slots of `m` drafter rows from `row0`, through the state's block table,
/// as the one-row write computes each (`staged_words`).
pub(super) fn ctx_row_slots(row0: usize, m: usize, table: &[u32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(m * 8);
    for r in 0..m {
        out.extend_from_slice(
            &super::graph::staged_words(row0 + r, table)?
                .slot
                .to_le_bytes(),
        );
    }
    Ok(out)
}

impl Glm5NextMtpHead {
    /// 2026-10-11: Whether the context rows run in passes: the scratch exists and the block's
    /// DSA workspace takes at least two rows.
    pub(super) fn ctx_rows_batched(&self) -> bool {
        self.ctx_rows.is_some() && self.module.layer.drafter_context_rows_max() >= 2
    }

    /// 2026-10-11: `rows` context rows for `tokens` / `hiddens` from drafter row `row_base`, in
    /// passes (module doc). The caller checked the row space (`rows_impl`). Advances `seq_len`
    /// by `rows`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ctx_rows_impl(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        rows: usize,
        st: &mut Glm5NextMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(sc) = self.ctx_rows.as_ref() else {
            bail!("GLM MTP context rows: no batched scratch");
        };
        let sc = sc.lock();
        let crate::glm5next_layer::Glm5NextMixer::Dsa(dsa) = &self.module.layer.mixer else {
            bail!("GLM MTP block is not a DSA layer");
        };
        let gpu = ctx.gpu;
        let h = self.hidden;
        let chunk = CTX_ROWS_CHUNK.min(self.module.layer.drafter_context_rows_max());
        let mut kv = self.kv_cache.lock();
        let mut r0 = 0usize;
        while r0 < rows {
            let m = chunk.min(rows - r0);
            // 2026-10-11: Row r of this pass is pair key `row_base + r0 + r`: the embedding of
            // `tokens[r0 + r + 1]`, normed, zeroed at key 0 under `METRALE_MTP_ZERO_POS0`.
            let ids: Vec<u8> = tokens[r0 + 1..r0 + 1 + m]
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect();
            gpu.copy_h2d(&ids, sc.ids)?;
            ops::batched_embed(
                gpu,
                sc.gather_k,
                sc.ids,
                self.embed_tokens.weight,
                sc.emb,
                m as u32,
                h as u32,
                stream,
            )?;
            self.norm_rows(gpu, sc.emb, self.module.enorm, sc.emb, h, m, stream)?;
            if row_base + r0 == 0
                && super::hidden_half::zeroes_embedding(ctx.levers.mtp_zero_pos0, 0)
            {
                super::hidden_half::note_applied("ZERO_POS0");
                gpu.memset_async(sc.emb, 0, h * 2, stream)?;
            }
            // 2026-10-11: The hidden half: a context row is always a target row, so the target's
            // final norm first under `METRALE_MTP_TARGET_POSTNORM` (`hidden_half`), then `hnorm`.
            let src = hiddens.offset(r0 * h * 2);
            if super::hidden_half::takes_target_norm(ctx.levers.mtp_target_postnorm, true) {
                super::hidden_half::note_applied("TARGET_POSTNORM");
                self.norm_rows(gpu, src, self.target_final_norm, sc.hid, h, m, stream)?;
                self.norm_rows(gpu, sc.hid, self.module.hnorm, sc.hid, h, m, stream)?;
            } else {
                self.norm_rows(gpu, src, self.module.hnorm, sc.hid, h, m, stream)?;
            }
            // 2026-10-11: `concat[r] = [emb[r] | hid[r]]`: two pitched copies, rows `h` and
            // `2h` elements apart.
            gpu.copy_d2d_2d_async(sc.emb, h * 2, sc.concat, 4 * h, h * 2, m, stream)?;
            gpu.copy_d2d_2d_async(
                sc.hid,
                h * 2,
                sc.concat.offset(h * 2),
                4 * h,
                h * 2,
                m,
                stream,
            )?;
            dsa.proj_rows(
                gpu,
                sc.concat,
                self.module.eh_proj.weight,
                sc.x,
                m,
                h,
                2 * h,
                metrale_model_layers::layers::ops::DenseCallSite::General,
                stream,
            )?;
            gpu.copy_h2d(&ctx_row_slots(st.seq_len, m, &st.block_table)?, sc.slots)?;
            // 2026-10-11: `emb` is free once `concat` holds it: the input norm's scratch.
            let dsa_state: &mut dyn LayerState = &mut st.dsa;
            self.module.layer.drafter_write_kv_rows(
                sc.x, sc.emb, m, dsa_state, &mut kv, st.seq_len, sc.slots, ctx, stream,
            )?;
            st.seq_len += m;
            r0 += m;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: The pass's slots are the one-row write's, row by row, across a block
    /// boundary and into a later region; a row past the table is refused.
    #[test]
    fn a_pass_writes_each_row_to_its_own_slot() {
        let table = super::super::kv_pool::region_table(2, 3);
        let bytes = ctx_row_slots(14, 4, &table).unwrap();
        let slots: Vec<i64> = bytes
            .chunks(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(slots, [6 * 16 + 14, 6 * 16 + 15, 7 * 16, 7 * 16 + 1]);
        for (r, s) in slots.iter().enumerate() {
            assert_eq!(
                *s,
                super::super::graph::staged_words(14 + r, &table)
                    .unwrap()
                    .slot
            );
        }
        assert!(ctx_row_slots(46, 4, &table).is_err());
    }
}
