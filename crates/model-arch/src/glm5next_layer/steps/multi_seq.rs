// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: `forward_multi`, the batched multi-sequence decode body: one token for each of
//! `n` sequences through the layer, the rows in groups of at most `multi_seq_chunk_rows()`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Row `r` uses highway slot `ctx.hc_row_offset + r`, sequence `r`'s state, `seq_lens[r]`
//!   and metadata row `r`; no row reads another row's state.
//! - A group is at most `DENSE_GEMV_BATCHM_MAX_M` rows (`multi_seq_chunk_rows`), the width up
//!   to which every projection gives each row the M = 1 GEMV's bits.
//!
//! # Why a row's output equals its sequence decoded alone (`forward_one`)
//!
//! `forward_one` and this path launch the same kernels per row; the only differences are the
//! row counts of the launches that span rows:
//! - mHC (`hc_expand`, `hc_mix`, `hc_finish`, `hc_post`, `hc_head`) and `rms_norm_vanilla` run
//!   one block (or one block row) per token and share nothing across tokens.
//! - The KDA and DSA projections, the shared expert and the dense MLP run the batched GEMV,
//!   whose rows are bit-identical to the M = 1 GEMV up to `DENSE_GEMV_BATCHM_MAX_M`.
//! - The router logits stay one M = 1 GEMV per row; the top-k and the combine run one block
//!   per row.
//! - The routed experts take the row union (`w4a16_gemv_sw_moe_batchm_m<R>`), whose per
//!   (row, slot) arithmetic is `w4a16_gemv_sw_moe`'s, as
//!   `examples/glm5next_moe_row_batch_microtest.rs` checks byte for byte.
//! - The KDA conv and recurrence and the DSA latent write, indexer and selection are the
//!   single-row launches, on the row's own state.
//! - The all-reduce sums `rows * hidden` elements across ranks. Single-GPU there is none; over
//!   two ranks a sum of two is order-free; over three, NCCL's reduction order for an element
//!   may depend on the message size and the element's offset, so identity across batch
//!   compositions is not promised there (the per-sequence path has the same exposure).
//!
//! One lever breaks the identity: `METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS` set at or below the
//! group width sends a group's routed experts to the grouped tensor-core GEMM (the default,
//! 128, is above it). It acts on `forward_k` the same way.

use super::*;

impl Glm5NextLayer {
    /// 2026-10-08: One decode token for each of `num_seqs` sequences, in groups of
    /// `multi_seq_chunk_rows()` rows, each group through `forward_rows_with` with a per-row
    /// mixer. A DSA layer reads each row's position, KV slot, `seq_len` and block table from
    /// `ctx.attn_metadata`, which must hold at least `num_seqs` rows; a KDA layer reads each
    /// row's pool state. Errors on a layer without a hyper-connection.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_multi<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(mhc) = self.mhc.as_ref() else {
            bail!(
                "GLM layer {}: the batched decode needs the hyper-connection; this is the MTP \
                 block",
                self.layer_idx
            );
        };
        if states.len() < num_seqs || seq_lens.len() < num_seqs {
            bail!(
                "GLM layer {}: a {num_seqs}-row batched decode got {} states and {} seq_lens",
                self.layer_idx,
                states.len(),
                seq_lens.len()
            );
        }
        let cap = ctx.buffers.max_batch_tokens();
        if ctx.hc_row_offset + num_seqs > cap {
            bail!(
                "GLM layer {}: rows {}..{} pass the {cap}-slot mHC highway",
                self.layer_idx,
                ctx.hc_row_offset,
                ctx.hc_row_offset + num_seqs
            );
        }
        let meta = match &self.mixer {
            Glm5NextMixer::Dsa(_) => Some(ctx.attn_metadata.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a batched DSA decode reads each row's position, KV slot and \
                     block table from attn_metadata, and none was uploaded",
                    self.layer_idx
                )
            })?),
            Glm5NextMixer::Kda { .. } => None,
        };
        let row_bytes = self.hidden * 2;
        for (base, m) in multi_seq_chunks(num_seqs, multi_seq_chunk_rows()) {
            let rows = &mut states[base..base + m];
            let lens = &seq_lens[base..base + m];
            let x = hidden.offset(base * row_bytes);
            let slot_base = ctx.hc_row_offset + base;
            match (&self.mixer, meta) {
                (Glm5NextMixer::Kda { layer, ws, .. }, _) => {
                    let kda = rows
                        .iter_mut()
                        .map(|s| {
                            let st = self.kda_state(&mut **s)?;
                            Ok(KdaSeqState {
                                conv: st.conv_state,
                                recurrent: st.h_state,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    self.forward_rows_with(mhc, x, m, slot_base, ctx, stream, |normed| {
                        layer.decode_rows(ctx.gpu, normed, &kda, ws, stream)?;
                        Ok(ws.final_out)
                    })?;
                }
                (Glm5NextMixer::Dsa(layer), Some(meta)) => {
                    self.forward_rows_with(mhc, x, m, slot_base, ctx, stream, |normed| {
                        // 2026-10-08: Writes its output projection over its input buffer, as
                        // `decode_k` does.
                        layer.decode_rows(normed, rows, lens, kv_cache, meta, base, ctx, stream)?;
                        Ok(normed)
                    })?;
                }
                (Glm5NextMixer::Dsa(_), None) => {
                    bail!("GLM layer {}: DSA mixer without metadata", self.layer_idx)
                }
            }
        }
        Ok(())
    }
}
