// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_SP_MHC=1`: the sequence-parallel mHC layer body of a TP prefill
//! sub-chunk (`glm5next_sp_mhc`): [`Glm5NextLayer::forward_rows_sp`] beside `forward_rows_with`,
//! same launch order, the per-row mHC launches on this rank's shard of rows only, the mixer
//! and the MLP on all rows, each all-reduce a reduce-scatter of rows summed in
//! `glm5next_hc_post_rank_sum`, and the normed rows all-gathered in place before the mixer and
//! the MLP; the last layer all-gathers `hidden` after `hc_head_mean`, so every rank leaves the
//! sub-chunk with every row as before.
//!
//! 2026-10-14: The body is three mHC sections around the mixer and the MLP, each run by
//! `glm5next_sp_overlap::run_section`: serially on the caller's stream, or under
//! `METRALE_GLM_SP_OVERLAP=1` per row piece with the exchanges on a comm stream.
//! Under `METRALE_GLM_MHC_POST_PRE_FUSED=1` the second section's post-mix and FFN pre-norm of
//! a piece are one launch (`glm5next_mhc_post_pre_tok`), the pair's bytes.
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
            self.levers.mhc.sp_mhc,
            world,
            k,
        ))
    }

    /// 2026-10-12: `forward_rows_with` with the mHC work on this rank's rows (see the module
    /// header). `mixer(normed)` runs the attention sublayer on all `k` normed rows.
    /// 2026-10-13: `tap` dumps each point after the section that completes it; the summed mixer
    /// output inside its section, before the pre-norm overwrites a DSA output.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_rows_sp(
        &self,
        mhc: &Glm5NextMhc,
        hidden: DevicePtr,
        k: usize,
        slot_base: usize,
        ctx: &ForwardContext,
        stream: u64,
        // 2026-10-14: The MLP's per-piece launches, as `forward_rows_with`'s (`[k]` for one).
        row_pieces: &[usize],
        tap: Option<&dump::Tap<'_>>,
        mixer: impl FnOnce(DevicePtr) -> Result<DevicePtr>,
    ) -> Result<()> {
        use crate::glm5next_sp_mhc as sp;
        use crate::glm5next_sp_overlap as ov;
        crate::glm5next_mlp::forward::check_pieces(k, row_pieces)?;
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
        let world = comm.world_size();
        let shards = sp::shards(k, world);
        // 2026-10-14: `METRALE_GLM_SP_OVERLAP=1`: the shards in row pieces whose exchanges
        // overlap the mHC launches (`glm5next_sp_overlap`); one piece is the serial body.
        let m = ov::piece_count(self.levers.mhc.sp_overlap, k, world);
        let pieces = ov::pieces(&shards, m);
        let overlap = if m >= 2 {
            Some(ov::streams_get()?)
        } else {
            None
        };
        // 2026-10-14: `METRALE_GLM_MHC_POST_PRE_FUSED=1`: section 2's post-mix and FFN pre-norm
        // as one launch per wide piece (`glm5next_mhc_post_pre_tok`, same bytes).
        let post_pre = self.levers.mhc.post_pre_fused;

        // 2026-10-12: Full-chunk views at the sub-chunk's slots ([`SpRows`]).
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();
        let rows = SpRows::new(ctx, hc, h, slot_base, hidden, normed);
        let streams = rows.streams;
        let section = |rs, ag| ov::Section {
            pieces: &pieces,
            rs,
            ag,
            row_bytes,
            received: None,
            defer: None,
            prof: self.levers.profile,
        };

        if self.is_first
            && let Some(l2) = &self.l2_warm
        {
            l2.reset();
        }
        if self.is_first {
            dump::put(tap, dump::Point::HiddenIn, hidden)?;
        }
        ov::run_section(
            gpu,
            comm,
            &ws,
            overlap.as_ref(),
            stream,
            &section(None, Some((normed, Some(profile::REDUCE_ATTN_ENQ)))),
            &mut |a, b, _| self.sp_head_piece(gpu, mhc, &rows, a, b, stream),
        )?;
        dump::site(tap, false, [streams, rows.post, rows.comb, normed])?;

        let t = self.levers.profile.start();
        let attn_out = mixer(normed)?;
        profile::end(profile::KDA, t, gpu, stream);
        dump::put(tap, dump::Point::MixerOut, attn_out)?;
        dump::topk(tap)?;

        ov::run_section(
            gpu,
            comm,
            &ws,
            overlap.as_ref(),
            stream,
            &section(
                Some((attn_out, Some(profile::REDUCE_ATTN))),
                Some((normed, Some(profile::REDUCE_MLP_ENQ))),
            ),
            &mut |a, b, partials| {
                self.sp_mid_piece(
                    gpu, mhc, &rows, a, b, attn_out, partials, post_pre, tap, stream,
                )
            },
        )?;

        dump::site(tap, true, [streams, rows.post, rows.comb, normed])?;
        self.mlp_compute(normed, ffn_out, k, row_pieces, ctx, stream)?;
        dump::put(tap, dump::Point::MlpOut, ffn_out)?;
        dump::moe_ids(tap)?;
        let reduce = self.mlp_cfg.needs_all_reduce();
        ov::run_section(
            gpu,
            comm,
            &ws,
            overlap.as_ref(),
            stream,
            &section(
                reduce.then_some((ffn_out, Some(profile::REDUCE_MLP))),
                self.is_last.then_some((hidden, None)),
            ),
            &mut |a, b, partials| {
                self.sp_tail_piece(
                    gpu,
                    mhc,
                    &rows,
                    a,
                    b,
                    ffn_out,
                    partials,
                    reduce,
                    tap.is_some(),
                    stream,
                )
            },
        )?;
        self.dump_layer_out(tap, ffn_out, streams)?;
        if self.is_last {
            dump::put(tap, dump::Point::FinalHidden, hidden)?;
        }
        if self.is_last {
            if let Some(l2) = &self.l2_warm {
                l2.join(gpu, stream)?;
            }
            profile::step(self.levers.profile);
        }
        Ok(())
    }

    /// 2026-10-12: Section 1 of this rank's rows `[a, b)`: `hc_expand` on the first layer and the
    /// attention pre-norm. 2026-10-14: moved out of `forward_rows_sp` unchanged, shared with the
    /// pipeline (`steps/forward_sp_pipe.rs`).
    pub(in crate::glm5next_layer) fn sp_head_piece(
        &self,
        gpu: &dyn GpuBackend,
        mhc: &Glm5NextMhc,
        rows: &SpRows,
        a: usize,
        b: usize,
        stream: u64,
    ) -> Result<()> {
        let (ht, hct) = (self.hidden as u32, mhc.hc_mult as u32);
        let t_mhc = self.levers.profile.start();
        let (st, po, co, hi, no) = rows.at(a);
        if self.is_first {
            glm_hc_expand(
                gpu,
                mhc.kernels.hc_expand,
                hi,
                st,
                (b - a) as u32,
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
            st,
            hi,
            po,
            co,
            no,
            b - a,
            t_mhc,
            stream,
        )
    }

    /// 2026-10-12: Section 2 of rows `[a, b)`: the mixer output's post-mix summed with the
    /// peers' `partials` and the FFN pre-norm (one launch under `post_pre`). 2026-10-14: moved
    /// out of `forward_rows_sp` unchanged.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn sp_mid_piece(
        &self,
        gpu: &dyn GpuBackend,
        mhc: &Glm5NextMhc,
        rows: &SpRows,
        a: usize,
        b: usize,
        attn_out: DevicePtr,
        partials: Option<&metrale_comm::PeerPartials>,
        post_pre: bool,
        tap: Option<&dump::Tap<'_>>,
        stream: u64,
    ) -> Result<()> {
        let (ht, hct) = (self.hidden as u32, mhc.hc_mult as u32);
        let row_bytes = self.hidden * 2;
        let partials = partials.context("SP-mHC: the mixer's partials")?;
        let (st, po, co, hi, no) = rows.at(a);
        let n = (b - a) as u32;
        if let Some(kernel) =
            crate::glm5next_mhc_post_pre_tok::kernel_for(&mhc.kernels, post_pre, n)?
        {
            let t_mhc = self.levers.profile.start();
            crate::glm5next_mhc_post_pre_tok::launch(
                gpu,
                kernel,
                &crate::glm5next_mhc_post_pre_tok::PostPreTok {
                    block_out: attn_out.offset(a * row_bytes),
                    partials,
                    residual: st,
                    post: po,
                    comb: co,
                    streams_out: st,
                    site: &mhc.ffn,
                    norm_w: self.post_attn_norm,
                    y: hi,
                    post_out: po,
                    comb_out: co,
                    normed: no,
                    num_tokens: n,
                    dims: [ht, hct, mhc.sinkhorn_iters as u32],
                    eps: [self.rms_eps, mhc.hc_eps, self.rms_eps],
                },
                stream,
            )?;
            profile::end(profile::MHC, t_mhc, gpu, stream);
            return Ok(());
        }
        let t_mhc_post = self.levers.profile.start();
        crate::glm5next_hc_post_vec::post_rank_sum(
            gpu,
            &mhc.kernels,
            attn_out.offset(a * row_bytes),
            partials,
            [st, po, co, st],
            [n, ht, hct],
            tap.is_some(),
            stream,
        )?;
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        if let Some(t) = tap {
            t.put_rows(dump::Point::MixerRed, attn_out, a..b)?;
        }
        let t_mhc = self.levers.profile.start();
        self.mhc_pre_norm(
            gpu,
            mhc,
            &mhc.ffn,
            self.post_attn_norm,
            st,
            hi,
            po,
            co,
            no,
            b - a,
            t_mhc,
            stream,
        )
    }

    /// 2026-10-12: Section 3 of rows `[a, b)`: the MLP output's post-mix (summed with the peers'
    /// `partials` when `reduce`) and `hc_head_mean` on the last layer. 2026-10-14: moved out of
    /// `forward_rows_sp` unchanged. 2026-10-15: `write_x` asks the post-mix to leave the summed
    /// row in `ffn_out` (only a dump tap reads it; `glm5next_hc_post_vec`).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn sp_tail_piece(
        &self,
        gpu: &dyn GpuBackend,
        mhc: &Glm5NextMhc,
        rows: &SpRows,
        a: usize,
        b: usize,
        ffn_out: DevicePtr,
        partials: Option<&metrale_comm::PeerPartials>,
        reduce: bool,
        write_x: bool,
        stream: u64,
    ) -> Result<()> {
        let (ht, hct) = (self.hidden as u32, mhc.hc_mult as u32);
        let row_bytes = self.hidden * 2;
        let (st, po, co, hi, _) = rows.at(a);
        let n = (b - a) as u32;
        let t_mhc_post = self.levers.profile.start();
        if reduce {
            crate::glm5next_hc_post_vec::post_rank_sum(
                gpu,
                &mhc.kernels,
                ffn_out.offset(a * row_bytes),
                partials.context("SP-mHC: the MLP's partials")?,
                [st, po, co, st],
                [n, ht, hct],
                write_x,
                stream,
            )?;
        } else {
            glm_hc_post(
                gpu,
                mhc.kernels.hc_post,
                ffn_out.offset(a * row_bytes),
                st,
                po,
                co,
                st,
                n,
                ht,
                hct,
                stream,
            )?;
        }
        if self.is_last {
            hc_head_mean(gpu, mhc.kernels.hc_head, st, hi, n, ht, hct, stream)?;
        }
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        Ok(())
    }
}

/// 2026-10-12: A sub-chunk's full-chunk views at its highway slots (`hc_streams`, `hc_post`,
/// `hc_comb` from `slot_base`), its `hidden` rows and its normed rows; [`SpRows::at`] is row
/// `a` of each. 2026-10-14: a type, so the pipeline builds one per sub-chunk.
#[derive(Clone, Copy, Debug)]
pub(in crate::glm5next_layer) struct SpRows {
    streams: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    hidden: DevicePtr,
    normed: DevicePtr,
    hc: usize,
    h: usize,
}

impl SpRows {
    /// 2026-10-14: The views of a sub-chunk at `slot_base` of `hc` streams of width `h`.
    pub(in crate::glm5next_layer) fn new(
        ctx: &ForwardContext,
        hc: usize,
        h: usize,
        slot_base: usize,
        hidden: DevicePtr,
        normed: DevicePtr,
    ) -> Self {
        Self {
            streams: ctx.buffers.hc_streams().offset(slot_base * hc * h * 4),
            post: ctx.buffers.hc_post().offset(slot_base * hc * 4),
            comb: ctx.buffers.hc_comb().offset(slot_base * hc * hc * 4),
            hidden,
            normed,
            hc,
            h,
        }
    }

    /// 2026-10-14: Row `a` of the streams, post, comb, hidden and normed views.
    fn at(&self, a: usize) -> (DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr) {
        let (hc, h) = (self.hc, self.h);
        (
            self.streams.offset(a * hc * h * 4),
            self.post.offset(a * hc * 4),
            self.comb.offset(a * hc * hc * 4),
            self.hidden.offset(a * h * 2),
            self.normed.offset(a * h * 2),
        )
    }
}
