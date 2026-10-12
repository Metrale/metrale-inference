// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_SP_MHC=1`: the sequence-parallel mHC layer body of a TP prefill
//! sub-chunk (`glm5next_sp_mhc`): [`Glm5NextLayer::forward_rows_sp`] beside `forward_rows_with`,
//! same launch order, the per-row mHC launches on this rank's shard of rows only, the mixer
//! and the MLP on all rows, each all-reduce a reduce-scatter of rows summed in
//! `glm5next_hc_post_rank_sum`, and the normed rows all-gathered in place before the mixer and
//! the MLP; the last layer all-gathers `hidden` after `hc_head_mean`, so every rank leaves the
//! sub-chunk with every row as before.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Taken only from `forward_k`'s prefill branch under [`Glm5NextLayer::sp_takes`]: every rank
//!   decides from `k`, the world size and the levers alone, so all ranks take the same body.
//! - Needs `glm5next_hc_post_rank_sum` on the target (an error otherwise, never a silent
//!   unfused reduce) and the workspace `glm5next_sp_mhc::reserve` built at load.

use super::*;

impl Glm5NextLayer {
    /// 2026-10-12: Whether a `k`-row launch takes [`Self::forward_rows_sp`]: the lever on, a
    /// prefill sub-chunk outside a graph capture, a communicator of two or more ranks whose
    /// mixer output is all-reduced, and `k >= SP_MIN_ROWS`.
    pub(in crate::glm5next_layer) fn sp_takes(
        &self,
        ctx: &ForwardContext,
        k: usize,
        is_prefill: bool,
    ) -> Result<bool> {
        if !is_prefill || ctx.graph_capture || !self.mixer_all_reduce {
            return Ok(false);
        }
        let world = ctx.comm.map_or(1, |c| c.world_size());
        Ok(crate::glm5next_sp_mhc::applies(
            crate::glm5next_sp_mhc::enabled()?,
            world,
            k,
        ))
    }

    /// 2026-10-12: `forward_rows_with` with the mHC work on this rank's rows (see the module
    /// header). `mixer(normed)` runs the attention sublayer on all `k` normed rows.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_rows_sp(
        &self,
        mhc: &Glm5NextMhc,
        hidden: DevicePtr,
        k: usize,
        slot_base: usize,
        ctx: &ForwardContext,
        stream: u64,
        mixer: impl FnOnce(DevicePtr) -> Result<DevicePtr>,
    ) -> Result<()> {
        use crate::glm5next_sp_mhc as sp;
        let Some(comm) = ctx.comm else {
            bail!(
                "GLM layer {}: SP-mHC without a communicator",
                self.layer_idx
            );
        };
        anyhow::ensure!(
            mhc.kernels.hc_post_rank_sum.0 != 0,
            "GLM layer {}: METRALE_GLM_SP_MHC=1 needs glm5next_hc_post_rank_sum on this target",
            self.layer_idx
        );
        let ws = sp::workspace_get()?;
        let gpu = ctx.gpu;
        let h = self.hidden;
        let hc = mhc.hc_mult;
        let row_bytes = h * 2;
        let shards = sp::shards(k, comm.world_size());
        let (lo, hi) = shards[comm.rank()];
        let own = hi - lo;
        let (ot, ht, hct) = (own as u32, h as u32, hc as u32);

        // 2026-10-12: Full-chunk views at the sub-chunk's slots, then this rank's rows of each.
        let streams = ctx.buffers.hc_streams().offset(slot_base * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot_base * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot_base * hc * hc * 4);
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();
        let streams_o = streams.offset(lo * hc * h * 4);
        let post_o = post.offset(lo * hc * 4);
        let comb_o = comb.offset(lo * hc * hc * 4);
        let hidden_o = hidden.offset(lo * row_bytes);
        let normed_o = normed.offset(lo * row_bytes);

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
                hidden_o,
                streams_o,
                ot,
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
            streams_o,
            hidden_o,
            post_o,
            comb_o,
            normed_o,
            own,
            t_mhc,
            stream,
        )?;
        let t = profile::start_hot();
        sp::all_gather_rows(comm, normed, &shards, row_bytes, stream)?;
        profile::end(profile::REDUCE_ATTN_ENQ, t, gpu, stream);

        let t = profile::start();
        let attn_out = mixer(normed)?;
        profile::end(profile::KDA, t, gpu, stream);

        let t = profile::start_hot();
        let partials = sp::reduce_scatter_rows(comm, &ws, attn_out, &shards, row_bytes, stream)?;
        profile::end(profile::REDUCE_ATTN, t, gpu, stream);
        let t_mhc_post = profile::start();
        glm_hc_post_rank_sum(
            gpu,
            mhc.kernels.hc_post_rank_sum,
            attn_out.offset(lo * row_bytes),
            &partials,
            streams_o,
            post_o,
            comb_o,
            streams_o,
            ot,
            ht,
            hct,
            stream,
        )?;
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);

        let t_mhc = profile::start();
        self.mhc_pre_norm(
            gpu,
            mhc,
            &mhc.ffn,
            self.post_attn_norm,
            streams_o,
            hidden_o,
            post_o,
            comb_o,
            normed_o,
            own,
            t_mhc,
            stream,
        )?;
        let t = profile::start_hot();
        sp::all_gather_rows(comm, normed, &shards, row_bytes, stream)?;
        profile::end(profile::REDUCE_MLP_ENQ, t, gpu, stream);

        self.mlp_compute(normed, ffn_out, k, &[k], ctx, stream)?;
        let t_mhc_post = profile::start();
        if self.mlp_cfg.needs_all_reduce() {
            let t = profile::start_hot();
            let partials = sp::reduce_scatter_rows(comm, &ws, ffn_out, &shards, row_bytes, stream)?;
            profile::end(profile::REDUCE_MLP, t, gpu, stream);
            glm_hc_post_rank_sum(
                gpu,
                mhc.kernels.hc_post_rank_sum,
                ffn_out.offset(lo * row_bytes),
                &partials,
                streams_o,
                post_o,
                comb_o,
                streams_o,
                ot,
                ht,
                hct,
                stream,
            )?;
        } else {
            glm_hc_post(
                gpu,
                mhc.kernels.hc_post,
                ffn_out.offset(lo * row_bytes),
                streams_o,
                post_o,
                comb_o,
                streams_o,
                ot,
                ht,
                hct,
                stream,
            )?;
        }
        if self.is_last {
            hc_head_mean(
                gpu,
                mhc.kernels.hc_head,
                streams_o,
                hidden_o,
                ot,
                ht,
                hct,
                stream,
            )?;
            sp::all_gather_rows(comm, hidden, &shards, row_bytes, stream)?;
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
