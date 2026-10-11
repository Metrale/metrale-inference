// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The building blocks of `Glm5NextLayer`: norm, mixer, MLP, all-reduce, the MTP
//! block's plain residual path, and the state accessors.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;

mod drafter;
mod forward;
mod mhc_slice;
mod multi_seq;
mod prefill_capture;
pub use multi_seq::{GroupSpan, group_spans};
mod prefill_spans;
pub use prefill_spans::{SpanPiece, prefill_span_groups};
mod replay;
mod verify_multi;

impl Glm5NextLayer {
    /// 2026-09-25: `rms_norm_vanilla` over `rows` contiguous `[hidden]` rows in one launch. The
    /// kernel runs one block per row (`token = blockIdx.x`) and the blocks share nothing, so the
    /// result equals `rows` single-row launches.
    fn norm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.rms_norm_k)
            .grid([rows as u32, 1, 1])
            .block([(self.hidden.min(1024)) as u32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(self.hidden as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        Ok(())
    }

    /// 2026-10-09: One mHC site's pre-mix of `k` rows into `y` (with `post` and `comb`), then
    /// the RMS norm of `y` with `norm_w` into `normed`: `glm_hc_pre` and `norm`, or with
    /// `mhc_fused()` the single `glm_hc_pre_norm` launch that writes the same bytes. Ends the
    /// caller's `profile::MHC` span `t_mhc`; the unfused norm keeps its `profile::NORM` span.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn mhc_pre_norm_one(
        &self,
        gpu: &dyn GpuBackend,
        mhc: &Glm5NextMhc,
        site: &Glm5NextMhcSiteWeights,
        norm_w: DevicePtr,
        streams: DevicePtr,
        y: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        normed: DevicePtr,
        k: usize,
        t_mhc: Option<std::time::Instant>,
        stream: u64,
    ) -> Result<()> {
        let (kt, ht, hct) = (k as u32, self.hidden as u32, mhc.hc_mult as u32);
        let iters = mhc.sinkhorn_iters as u32;
        if mhc_fused() {
            crate::glm5next_mhc_fused::glm_hc_pre_norm(
                gpu,
                &mhc.kernels,
                streams,
                site,
                norm_w,
                y,
                normed,
                post,
                comb,
                kt,
                ht,
                hct,
                iters,
                self.rms_eps,
                mhc.hc_eps,
                self.rms_eps,
                stream,
            )?;
            profile::end(profile::MHC, t_mhc, gpu, stream);
            return Ok(());
        }
        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            site,
            y,
            post,
            comb,
            kt,
            ht,
            hct,
            iters,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, y, norm_w, normed, k, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        Ok(())
    }

    /// 2026-09-25: Run the mixer on `normed`; returns the pointer that holds its output.
    #[allow(clippy::too_many_arguments)]
    fn mixer_forward(
        &self,
        normed: DevicePtr,
        residual: DevicePtr,
        st: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_offloaded: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match &self.mixer {
            Glm5NextMixer::Kda { layer, ws, .. } => {
                let ssm = self.kda_state(st)?;
                let kda = KdaSeqState {
                    conv: ssm.conv_state,
                    recurrent: ssm.h_state,
                };
                let t = profile::start();
                layer.decode(ctx.gpu, normed, &kda, ws, stream)?;
                profile::end(profile::KDA, t, ctx.gpu, stream);
                Ok(ws.final_out)
            }
            Glm5NextMixer::Dsa(layer) => {
                let dsa: &mut Glm5NextDsaState = self.dsa_state(st)?;
                layer.decode(
                    normed,
                    residual,
                    dsa,
                    kv_cache,
                    seq_len,
                    block_table,
                    disk_block_ids,
                    disk_offloaded,
                    ctx,
                    stream,
                )?;
                // 2026-09-25: DSA's `decode` writes its output projection over its input buffer.
                Ok(normed)
            }
        }
    }

    /// 2026-09-25: While profiling, a 2-byte all-reduce timed into bucket `bar`, then the
    /// route-trace line for `site`. No-op when profiling is off or there is no communicator.
    fn reduce_probe(&self, bar: usize, site: &str, ctx: &ForwardContext, stream: u64) {
        if !profile::on() {
            return;
        }
        let Some(comm) = ctx.comm else { return };
        let t = profile::start_hot();
        let p = profile::probe_buf(ctx.gpu);
        if p != 0 {
            let _ = comm.all_reduce_async(p, 2, stream);
        }
        let us = profile::end_us(bar, t, ctx.gpu, stream);
        profile::trace_bar(
            site,
            self.layer_idx,
            matches!(self.mlp, Glm5NextMlpSite::Moe(_)),
            us,
        );
    }

    /// 2026-09-25: All-reduce `rows` contiguous `[hidden]` BF16 rows in one collective when
    /// `ctx.comm` is set: blocking under graph capture, asynchronous otherwise. No-op without a
    /// communicator.
    fn reduce_partial(
        &self,
        p: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(comm) = ctx.comm {
            let bytes = rows * self.hidden * 2;
            if ctx.graph_capture {
                comm.all_reduce(p.0, bytes)?;
            } else {
                comm.all_reduce_async(p.0, bytes, stream)?;
            }
        }
        Ok(())
    }

    /// 2026-10-09: [`Self::reduce_partial`] once per piece of `row_pieces`, so each piece's
    /// collective has the size its own pass gives it (NCCL may order an element's sum by the
    /// message size); `[rows]` is one collective, as before.
    fn reduce_pieces(
        &self,
        p: DevicePtr,
        row_pieces: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut r0 = 0usize;
        for &n in row_pieces {
            self.reduce_partial(p.offset(r0 * self.hidden * 2), n, ctx, stream)?;
            r0 += n;
        }
        Ok(())
    }

    /// 2026-10-10: Whether this mHC site sums its all-reduce inside the post-mix
    /// (`METRALE_GLM_AR_FUSED_POST=1`). Errors when the lever is on and the target lacks
    /// `glm5next_hc_post_rank_sum`, rather than silently running the unfused pair.
    /// 2026-10-10: Whether the one-token decode folds the attention site's post-mix into the FFN
    /// site's pre-norm (`METRALE_GLM_MHC_POST_FOLD=1`); an error when the lever is on without the
    /// fused pre-norm, the kernel or the layer's `fold_streams`.
    pub(in crate::glm5next_layer) fn post_fold(&self, mhc: &Glm5NextMhc) -> Result<bool> {
        if !mhc_post_fold() {
            return Ok(false);
        }
        if !mhc_fused() || mhc.kernels.hc_post_pre_norm.0 == 0 || mhc.fold_streams.0 == 0 {
            bail!(
                "GLM layer {}: METRALE_GLM_MHC_POST_FOLD=1 needs METRALE_GLM_MHC_FUSED=1, \
                 glm5next_hc_post_pre_norm and the layer's fold streams",
                self.layer_idx
            );
        }
        Ok(true)
    }

    pub(in crate::glm5next_layer) fn fused_post(&self, mhc: &Glm5NextMhc) -> Result<bool> {
        if !ar_fused_post() {
            return Ok(false);
        }
        if mhc.kernels.hc_post_rank_sum.0 == 0 {
            bail!(
                "GLM layer {}: METRALE_GLM_AR_FUSED_POST=1 but glm5next_hc_post_rank_sum is not \
                 loaded on this target",
                self.layer_idx
            );
        }
        Ok(true)
    }

    /// 2026-10-10: The all-reduce before an mHC post-mix. With `fused` and one piece, asks the
    /// communicator for the exchange alone and returns where the partials landed, for
    /// [`Self::hc_post_reduced`] to sum; otherwise (or when the payload does not take the one-shot
    /// path) runs [`Self::reduce_pieces`] and returns `None`.
    pub(in crate::glm5next_layer) fn reduce_or_exchange(
        &self,
        p: DevicePtr,
        row_pieces: &[usize],
        fused: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<metrale_comm::PeerPartials>> {
        if let (true, [rows], Some(comm)) = (fused, row_pieces, ctx.comm) {
            let bytes = rows * self.hidden * 2;
            let order = (!ctx.graph_capture).then_some(stream);
            if let Some(partials) = comm.exchange_partials(p.0, bytes, order)? {
                return Ok(Some(partials));
            }
        }
        self.reduce_pieces(p, row_pieces, ctx, stream)?;
        Ok(None)
    }

    /// 2026-10-10: `glm_hc_post` of `k` rows of `x` and the streams at `residual` into the
    /// highway at `streams` (the same buffer except after a folded post, `forward_one`), or, when
    /// `partials` is set, `glm_hc_post_rank_sum`, which first sums `x` with the peers' partials
    /// (writing the sum back to `x`), with the same bits as the all-reduce sum then `glm_hc_post`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn hc_post_reduced(
        &self,
        gpu: &dyn GpuBackend,
        mhc: &Glm5NextMhc,
        x: DevicePtr,
        partials: Option<metrale_comm::PeerPartials>,
        residual: DevicePtr,
        streams: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        let (kt, ht, hct) = (k as u32, self.hidden as u32, mhc.hc_mult as u32);
        match partials {
            Some(p) => glm_hc_post_rank_sum(
                gpu,
                mhc.kernels.hc_post_rank_sum,
                x,
                &p,
                residual,
                post,
                comb,
                streams,
                kt,
                ht,
                hct,
                stream,
            ),
            None => glm_hc_post(
                gpu,
                mhc.kernels.hc_post,
                x,
                residual,
                post,
                comb,
                streams,
                kt,
                ht,
                hct,
                stream,
            ),
        }
    }

    /// 2026-09-25: The MLP over `rows` rows of `normed` into `out`, then one all-reduce of all
    /// rows when `Glm5NextMlpConfig::needs_all_reduce`. 2026-10-09: `row_pieces` splits the rows
    /// into consecutive pieces (`[rows]`: one) whose launches must not depend on each other: a
    /// dense site runs once per piece, a routed site launches its router and shared expert once
    /// per piece (`forward_moe_pieces`) and shares the rest, and the all-reduce runs per piece
    /// (`reduce_pieces`).
    /// 2026-10-10: `fused` hands the all-reduce's sum to the caller's post-mix
    /// ([`Self::reduce_or_exchange`]); the returned partials are then `Some`.
    #[allow(clippy::too_many_arguments)]
    fn mlp_forward(
        &self,
        normed: DevicePtr,
        out: DevicePtr,
        rows: usize,
        row_pieces: &[usize],
        fused: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<metrale_comm::PeerPartials>> {
        crate::glm5next_mlp::forward::check_pieces(rows, row_pieces)?;
        let t_dense = matches!(self.mlp, Glm5NextMlpSite::Dense(_))
            .then(profile::start)
            .flatten();
        match &self.mlp {
            Glm5NextMlpSite::Dense(w) => {
                let mut r0 = 0usize;
                for &n in row_pieces {
                    forward_dense_site(
                        ctx.gpu,
                        &self.mlp_kernels,
                        &self.mlp_cfg,
                        w,
                        self.mlp_cfg.local_dense_intermediate,
                        normed.offset(r0 * self.hidden * 2),
                        out.offset(r0 * self.hidden * 2),
                        n,
                        &self.mlp_ws,
                        stream,
                    )?;
                    r0 += n;
                }
            }
            Glm5NextMlpSite::Moe(w) => forward_moe_pieces(
                ctx.gpu,
                &self.mlp_kernels,
                &self.mlp_cfg,
                w,
                normed,
                out,
                rows,
                row_pieces,
                &self.mlp_ws,
                // 2026-10-09: A capture cannot read the expert histogram back to the host.
                ctx.graph_capture,
                stream,
            )?,
        }
        profile::end(profile::MLP_DENSE, t_dense, ctx.gpu, stream);
        // 2026-09-25: One all-reduce covers both partial sums, the EP-split routed experts and
        // the TP-split shared expert: `forward_moe` adds them together before returning.
        let mut partials = None;
        if self.mlp_cfg.needs_all_reduce() {
            // 2026-10-10: Forked before the reduce is enqueued, so the warm runs beside it.
            if let Some(l2) = &self.l2_warm {
                l2.fork(ctx.gpu, self.layer_idx, rows, stream)?;
            }
            self.reduce_probe(profile::REDUCE_MLP_BAR, "mlp", ctx, stream);
            let t = profile::start_hot();
            partials = self.reduce_or_exchange(out, row_pieces, fused, ctx, stream)?;
            profile::end_nosync(profile::REDUCE_MLP_ENQ, t);
            // 2026-09-25: This span times only the `synchronize` in `profile::end`.
            let t = profile::start_hot();
            profile::end(profile::REDUCE_MLP, t, ctx.gpu, stream);
        }
        Ok(partials)
    }

    /// 2026-09-25: `dst += src` over `n` BF16 elements (`bf16_add_inplace`); errors when the
    /// target lacks the kernel.
    fn add_inplace(
        &self,
        gpu: &dyn GpuBackend,
        dst: DevicePtr,
        src: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        if self.add_k.0 == 0 {
            bail!(
                "GLM layer {}: bf16_add_inplace is not loaded on this target; the MTP layer's \
                 plain residual path needs it",
                self.layer_idx
            );
        }
        KernelLaunch::new(gpu, self.add_k)
            .grid([(n as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_i32(n as i32)
            .launch(stream)
    }

    /// 2026-09-25: One token through a block with no hyper-connection (`mhc: None`, the MTP
    /// block): `x += mixer(norm(x))`, then `x += mlp(norm(x))`, each all-reduce before its add.
    #[allow(clippy::too_many_arguments)]
    fn forward_one_plain(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
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
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();

        self.norm(gpu, hidden, self.input_norm, normed, 1, stream)?;
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
        if self.mixer_all_reduce {
            self.reduce_partial(attn_out, 1, ctx, stream)?;
        }
        self.add_inplace(gpu, hidden, attn_out, h, stream)?;

        self.norm(gpu, hidden, self.post_attn_norm, normed, 1, stream)?;
        // 2026-10-10: No post-mix here to take the sum, so the all-reduce stays whole.
        self.mlp_forward(normed, ffn_out, 1, &[1], false, ctx, stream)?;
        self.add_inplace(gpu, hidden, ffn_out, h, stream)
    }

    /// 2026-09-25: This KDA layer's state as the SSM pool's `SsmLayerState`. Errors when the
    /// state is another type, or when its recurrent state is FP16: `h_is_f16` is set, or an FP32
    /// staging blob is attached, which only an FP16-sized pool has.
    pub(super) fn kda_state<'a>(
        &self,
        state: &'a mut dyn LayerState,
    ) -> Result<&'a mut SsmLayerState> {
        let st = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a KDA mixer was handed state that is not an SsmLayerState",
                    self.layer_idx
                )
            })?;
        if st.h_is_f16 || st.h_prefill_stage.is_some() {
            bail!(
                "GLM layer {}: KDA recurrent state is FP32-only; --ssm-h-dtype f16 narrowed it",
                self.layer_idx
            );
        }
        Ok(st)
    }

    /// 2026-09-25: This DSA layer's `Glm5NextDsaState`; errors on any other state type.
    pub(super) fn dsa_state<'a>(
        &self,
        state: &'a mut dyn LayerState,
    ) -> Result<&'a mut Glm5NextDsaState> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a DSA mixer was handed state that is not a Glm5NextDsaState",
                    self.layer_idx
                )
            })
    }
}
