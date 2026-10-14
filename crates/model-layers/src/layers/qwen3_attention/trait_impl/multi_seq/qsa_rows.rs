// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: QSA on the multi-sequence decode step: every live row ingests
//! its indexer key, and a row past the inert bound re-runs its attention over
//! its selected tokens into its output row, so a batch that holds a
//! long-context sequence no longer falls back to the per-sequence loop.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - Rows run one at a time through the indexer's single selection scratch:
//!   row `i`'s selected attention is enqueued before row `i + 1`'s selection
//!   overwrites the scratch, on one stream.
//! - Padding rows (`seq_len` 0) are skipped: they carry a dummy state that is
//!   dropped without `release_state`, so a carry created for one would leak.
//! - The batched paged decode ran dense attention for every row first; a QSA
//!   row's dense result is overwritten by its selected attention.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;

use super::attn::MsAttn;
use super::ctx::MultiSeqCtx;
use crate::layer::{AttnMetadataDev, LayerState};
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use metrale_cache::kv_cache::KvCacheDtype;

impl Qwen3AttentionLayer {
    /// 2026-10-13: Ingest and, where selection is active, attend over the
    /// selection for the live rows of the batch. A no-op on a layer without an
    /// indexer.
    pub(super) fn ms_qsa_rows(
        &self,
        c: &MultiSeqCtx<'_>,
        states: &mut [&mut (dyn LayerState + 'static)],
        seq_lens: &[usize],
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
        attn: &MsAttn,
    ) -> Result<()> {
        let Some(qsa) = self.qsa.as_ref() else {
            return Ok(());
        };
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            bs,
            bf16,
            normed,
            ..
        } = *c;
        anyhow::ensure!(
            matches!(self.kv_dtype.kv_pair().0, KvCacheDtype::Bf16)
                && matches!(self.kv_dtype.kv_pair().1, KvCacheDtype::Bf16),
            "QSA selection requires a plain BF16 KV cache (the gather copies raw \
             NHD rows); serve with --kv-cache-dtype bf16"
        );
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let out_row = nq as usize * hd as usize * bf16;
        for (i, state) in states.iter_mut().enumerate().take(n) {
            if seq_lens.get(i).copied() == Some(0) {
                continue;
            }
            let st = crate::layers::qwen3_attention::helpers::qsa_seq_state(qsa, *state, fwd.gpu)?;
            let sel = qsa.decode_select(
                st,
                normed.offset(i * h * bf16),
                seq_lens[i],
                // 2026-10-13: Row `i`'s u32 position, which is `seq_lens[i]`
                // (`upload_batch_metadata_fixed`).
                meta.positions.offset(i * 4),
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                meta.block_table
                    .offset(i * meta.max_blocks_per_seq as usize * 4),
                bs,
                fwd.gpu,
                stream,
            )?;
            let Some(sel) = sel else {
                continue;
            };
            // 2026-10-13: As `decode/attention_forward.rs` does for one sequence:
            // the BF16 route over the gathered scratch through its identity table,
            // from this row's Q into this row's output (2026-10-14: the same route
            // and split count, so a row's bits do not depend on its batch).
            self.run_paged_decode_bf16(
                fwd.gpu,
                attn.q.offset(i * attn.q_row_bytes),
                (sel.k_scratch, sel.v_scratch),
                0,
                attn.attn_out.offset(i * out_row),
                sel.table_dev,
                sel.seq_len_dev,
                sel.max_blocks,
                1,
                nq,
                nkv,
                hd,
                bs,
                inv_sqrt_d,
                nq * hd,
                fwd.buffers.splitk_workspace(),
                fwd.levers.max_decode_seqs,
                stream,
            )?;
        }
        Ok(())
    }
}
