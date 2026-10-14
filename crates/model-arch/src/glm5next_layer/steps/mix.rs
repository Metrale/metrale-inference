// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The attention sublayer of `k` rows of one sequence: what `forward_k`'s mixer
//! closure ran inline before, moved here unchanged so the SP pipeline
//! (`steps/forward_sp_pipe.rs`) runs each sub-chunk's mixer by the same code.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The launches, their order and their arguments are `forward_k`'s before the move.

use super::*;

impl Glm5NextLayer {
    /// 2026-10-14: The attention sublayer on the `k` normed rows at `normed` (positions from
    /// `seq_len`), returning its output: KDA's `final_out`, or DSA's over its input. With
    /// `take_snapshots` a KDA layer writes its state after row `t` to intermediate `t`;
    /// `is_prefill` as `forward_k`'s.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn mix(
        &self,
        normed: DevicePtr,
        k: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        take_snapshots: bool,
        is_prefill: bool,
    ) -> Result<DevicePtr> {
        let gpu = ctx.gpu;
        let kda_ctx = match &self.mixer {
            Glm5NextMixer::Kda { ws, .. } => {
                if k > ws.max_tokens() {
                    bail!(
                        "GLM layer {}: a {k}-token verify exceeds the KDA workspace built for {}",
                        self.layer_idx,
                        ws.max_tokens()
                    );
                }
                let st = self.kda_state(state)?;
                // 2026-09-25: Intermediates only when `take_snapshots` (a verify); prefill passes
                // none. `decode_k` looks each row up with `get`, so an empty list takes none.
                let snaps: Vec<(DevicePtr, DevicePtr)> = if take_snapshots {
                    (0..k.saturating_sub(1))
                        .map(|t| (st.h_state_intermediates[t], st.conv_state_intermediates[t]))
                        .collect()
                } else {
                    Vec::new()
                };
                Some((
                    KdaSeqState {
                        conv: st.conv_state,
                        recurrent: st.h_state,
                    },
                    snaps,
                ))
            }
            Glm5NextMixer::Dsa(_) => None,
        };

        match (&self.mixer, &kda_ctx) {
            (Glm5NextMixer::Kda { layer, ws, .. }, Some((kda, snaps))) => {
                // 2026-09-25: The chunked scan runs only for a prefill sub-chunk of more
                // than one row that asked for no intermediates: it never materialises the
                // state after an interior row, and its order differs from `decode_k`'s,
                // which a verify must match.
                // 2026-10-12: `METRALE_GLM_KDA_CHUNK64` (`decode_k` picks it) is ahead of
                // `METRALE_GLM_KDA_CHUNK_PREFILL` where its floor is reached.
                if is_prefill
                    && k > 1
                    && snaps.is_empty()
                    && crate::glm5next_layer::levers::kda_chunk_prefill()
                    && !layer
                        .chunk64_applies(k, crate::glm5next_kda::chunk64::kda_chunk64_min_rows()?)
                {
                    layer.prefill(gpu, normed, k, kda, ws, stream)?;
                } else {
                    layer.decode_k(gpu, normed, k, kda, ws, snaps, stream)?;
                }
                Ok(ws.final_out)
            }
            (Glm5NextMixer::Dsa(layer), _) => {
                // 2026-09-25: DSA's `decode_k` writes its output projection over its input
                // buffer.
                layer.decode_k(
                    normed,
                    k,
                    state,
                    kv_cache,
                    seq_len,
                    block_table,
                    ctx,
                    stream,
                    is_prefill,
                )?;
                Ok(normed)
            }
            (Glm5NextMixer::Kda { .. }, None) => {
                bail!("GLM layer {}: KDA mixer without KDA state", self.layer_idx)
            }
        }
    }
}
