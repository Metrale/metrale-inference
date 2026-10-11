// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The mHC layer bodies: `forward_one` (one token) and `forward_k` (K rows of one
//! sequence in one call), both of whose multi-row work (2026-10-08) is `forward_rows_with`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `forward_one` uses highway slot `slot`; `forward_k` uses slots `slot_base..slot_base + k`.
//! - `forward_rows_with` launches what `forward_k` launched inline before 2026-10-08, in the
//!   same order; only the mixer call moved into the caller's closure.
//! - 2026-10-10: With `METRALE_GLM_AR_FUSED_POST=1` an all-reduce that takes the one-shot path
//!   in one piece is the exchange alone, and its sum runs inside the post-mix after it
//!   (`hc_post_reduced`); off, every launch is as before.
//! - 2026-10-10: With `METRALE_GLM_AR_L2_PREFETCH` the first layer drops a stale warm fork and
//!   the last joins the pending one (`l2_warm`).

use super::*;

impl Glm5NextLayer {
    /// 2026-09-25: One token through the whole layer, using highway slot `slot`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_one(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        slot: usize,
        st: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_offloaded: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let Some(mhc) = self.mhc.as_ref() else {
            return self.forward_one_plain(
                hidden,
                residual,
                st,
                kv_cache,
                seq_len,
                block_table,
                disk_block_ids,
                disk_offloaded,
                ctx,
                stream,
            );
        };
        let hc = mhc.hc_mult;
        let streams = ctx.buffers.hc_streams().offset(slot * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot * hc * hc * 4);
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();

        let t_mhc = profile::start();
        if self.is_first
            && let Some(l2) = &self.l2_warm
        {
            l2.reset();
        }
        if self.is_first {
            glm_hc_expand(
                gpu,
                mhc.kernels.hc_expand,
                hidden,
                streams,
                1,
                h as u32,
                hc as u32,
                stream,
            )?;
        }

        self.mhc_pre_norm(
            gpu,
            mhc,
            &mhc.attn,
            self.input_norm,
            streams,
            hidden,
            post,
            comb,
            normed,
            1,
            t_mhc,
            stream,
        )?;
        let attn_out = self.mixer_forward(
            normed,
            residual,
            st,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_offloaded,
            ctx,
            stream,
        )?;
        let fused = self.fused_post(mhc)?;
        // 2026-10-10: With the fold the attention site's sum and post-mix run inside the FFN
        // site's pre-norm, writing the post-mixed streams to `fold_streams`; the FFN site's
        // post-mix then reads them and writes the highway, so the slot ends the layer as before.
        let fold = self.post_fold(mhc)?;
        let mut partials = None;
        if self.mixer_all_reduce {
            self.reduce_probe(profile::REDUCE_ATTN_BAR, "attn", ctx, stream);
            let t = profile::start_hot();
            partials = self.reduce_or_exchange(attn_out, &[1], fused || fold, ctx, stream)?;
            profile::end_nosync(profile::REDUCE_ATTN_ENQ, t);
            let t = profile::start_hot();
            profile::end(profile::REDUCE_ATTN, t, ctx.gpu, stream);
        }
        let ffn_residual = if fold {
            let t_mhc = profile::start();
            crate::glm5next_mhc_post_fold::glm_hc_post_pre_norm(
                gpu,
                &mhc.kernels,
                &mhc.ffn,
                crate::glm5next_mhc_post_fold::PostFoldIo {
                    own: attn_out,
                    residual: streams,
                    streams_out: mhc.fold_streams,
                    post,
                    comb,
                    norm_w: self.post_attn_norm,
                    y_out: hidden,
                    normed_out: normed,
                },
                partials.as_ref(),
                [h as u32, hc as u32, mhc.sinkhorn_iters as u32],
                [self.rms_eps, mhc.hc_eps, self.rms_eps],
                stream,
            )?;
            profile::end(profile::MHC, t_mhc, gpu, stream);
            mhc.fold_streams
        } else {
            let t_mhc_post = profile::start();
            self.hc_post_reduced(
                gpu, mhc, attn_out, partials, streams, streams, post, comb, 1, stream,
            )?;
            profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);

            let t_mhc = profile::start();
            self.mhc_pre_norm(
                gpu,
                mhc,
                &mhc.ffn,
                self.post_attn_norm,
                streams,
                hidden,
                post,
                comb,
                normed,
                1,
                t_mhc,
                stream,
            )?;
            streams
        };
        let partials = self.mlp_forward(normed, ffn_out, 1, &[1], fused, ctx, stream)?;
        let t_mhc_post = profile::start();
        self.hc_post_reduced(
            gpu,
            mhc,
            ffn_out,
            partials,
            ffn_residual,
            streams,
            post,
            comb,
            1,
            stream,
        )?;

        if self.is_last {
            hc_head_mean(
                gpu,
                mhc.kernels.hc_head,
                streams,
                hidden,
                1,
                h as u32,
                hc as u32,
                stream,
            )?;
        }
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        if self.is_last {
            if let Some(l2) = &self.l2_warm {
                l2.join(gpu, stream)?;
            }
            profile::step();
        }
        Ok(())
    }

    /// 2026-09-25: `k` rows of one sequence through the layer in one call. Row `t` uses highway
    /// slot `slot_base + t` and position `seq_len + t`. Each mHC and norm launch covers all `k`
    /// rows and computes every row on its own, so a row's result equals a single-row launch.
    /// KDA's `decode_k` runs its recurrence row by row. Errors on a layer without a
    /// hyper-connection.
    ///
    /// With `take_snapshots`, a KDA layer writes its state after row `t` to intermediate `t`
    /// for `t < k - 1`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_k(
        &self,
        hidden: DevicePtr,
        k: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        take_snapshots: bool,
        slot_base: usize,
        // 2026-09-25: True only for a prefill sub-chunk (`Glm5NextLayer::prefill`). It selects
        // the DSA batched selector (`batch_select_enabled`) and, with
        // `METRALE_GLM_KDA_CHUNK_PREFILL=1`, the KDA chunked scan, both prefill-only.
        is_prefill: bool,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };

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

        self.forward_rows_with(mhc, hidden, k, slot_base, ctx, stream, &[k], |normed| {
            match (&self.mixer, &kda_ctx) {
                (Glm5NextMixer::Kda { layer, ws, .. }, Some((kda, snaps))) => {
                    // 2026-09-25: The chunked scan runs only for a prefill sub-chunk of more
                    // than one row that asked for no intermediates: it never materialises the
                    // state after an interior row, and its order differs from `decode_k`'s,
                    // which a verify must match.
                    if is_prefill
                        && k > 1
                        && snaps.is_empty()
                        && crate::glm5next_layer::levers::kda_chunk_prefill()
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
        })
    }

    /// 2026-10-08: The mHC layer body over `k` rows at highway slots `slot_base..slot_base + k`,
    /// with the mixer supplied by the caller: `mixer(normed)` runs the attention sublayer on
    /// the `k` normed rows and returns the pointer holding its output. `forward_k` (rows of one
    /// sequence) and `forward_multi` (one row per sequence) differ only there.
    ///
    /// Launch order: `hc_expand` (first layer), `hc_pre`, the input norm, the mixer, its
    /// all-reduce, `hc_post`; then `hc_pre`, the post-attention norm, the MLP with its
    /// all-reduce, `hc_post`, and `hc_head_mean` on the last layer. 2026-10-09: The MLP keeps
    /// separate launches per piece of `row_pieces` where they depend on the row count
    /// (`mlp_forward`); `[k]` launches as before.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_rows_with(
        &self,
        mhc: &Glm5NextMhc,
        hidden: DevicePtr,
        k: usize,
        slot_base: usize,
        ctx: &ForwardContext,
        stream: u64,
        row_pieces: &[usize],
        mixer: impl FnOnce(DevicePtr) -> Result<DevicePtr>,
    ) -> Result<()> {
        crate::glm5next_mlp::forward::check_pieces(k, row_pieces)?;
        let gpu = ctx.gpu;
        let h = self.hidden;
        let hc = mhc.hc_mult;
        // 2026-09-25: `slot_base` is the first row's token index within the whole forward, so
        // the sub-chunks of one prefill use disjoint highway slots.
        let streams = ctx.buffers.hc_streams().offset(slot_base * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot_base * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot_base * hc * hc * 4);
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();
        let (kt, ht, hct) = (k as u32, h as u32, hc as u32);

        let t_mhc = profile::start();
        if self.is_first
            && let Some(l2) = &self.l2_warm
        {
            l2.reset();
        }
        if self.is_first {
            glm_hc_expand(
                gpu,
                mhc.kernels.hc_expand,
                hidden,
                streams,
                kt,
                ht,
                hct,
                stream,
            )?;
        }

        self.mhc_pre_norm(
            gpu,
            mhc,
            &mhc.attn,
            self.input_norm,
            streams,
            hidden,
            post,
            comb,
            normed,
            k,
            t_mhc,
            stream,
        )?;
        let t = profile::start();
        let attn_out = mixer(normed)?;
        profile::end(profile::KDA, t, gpu, stream);
        let fused = self.fused_post(mhc)?;
        let mut partials = None;
        if self.mixer_all_reduce {
            self.reduce_probe(profile::REDUCE_ATTN_BAR, "attn", ctx, stream);
            let t = profile::start_hot();
            partials = self.reduce_or_exchange(attn_out, row_pieces, fused, ctx, stream)?;
            profile::end_nosync(profile::REDUCE_ATTN_ENQ, t);
            let t = profile::start_hot();
            profile::end(profile::REDUCE_ATTN, t, ctx.gpu, stream);
        }
        let t_mhc_post = profile::start();
        self.hc_post_reduced(
            gpu, mhc, attn_out, partials, streams, streams, post, comb, k, stream,
        )?;
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);

        let t_mhc = profile::start();
        self.mhc_pre_norm(
            gpu,
            mhc,
            &mhc.ffn,
            self.post_attn_norm,
            streams,
            hidden,
            post,
            comb,
            normed,
            k,
            t_mhc,
            stream,
        )?;
        let partials = self.mlp_forward(normed, ffn_out, k, row_pieces, fused, ctx, stream)?;
        let t_mhc_post = profile::start();
        self.hc_post_reduced(
            gpu, mhc, ffn_out, partials, streams, streams, post, comb, k, stream,
        )?;
        if self.is_last {
            hc_head_mean(
                gpu,
                mhc.kernels.hc_head,
                streams,
                hidden,
                kt,
                ht,
                hct,
                stream,
            )?;
        }
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        if self.is_last {
            if let Some(l2) = &self.l2_warm {
                l2.join(gpu, stream)?;
            }
            profile::step();
        }
        Ok(())
    }
}
