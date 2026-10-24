// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: One mHC site's pre-mix + norm, all-reduce or exchange, and post-mix, as the GLM
//! layer runs them, over the inputs they read (the layer's levers, mHC weights, norm kernel and
//! communicator) rather than the whole layer: moved from `steps.rs` and `steps/reduce_post.rs`
//! unchanged except that `mhc` and the forward context's communicator are fields, so a launch
//! gate runs the hand-written site without building a layer. `Glm5NextLayer`'s methods of the
//! same names delegate here.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: as `steps.rs`'s.

use super::*;
use crate::glm5next_mhc_levers::fused_runs;

/// 2026-10-23: What a layer's norm and all-reduce launches read besides their buffers.
#[derive(Clone, Copy)]
pub(crate) struct LayerIo<'a> {
    pub levers: &'a GlmLevers,
    pub rms_norm_k: KernelHandle,
    pub hidden: usize,
    pub rms_eps: f32,
    pub layer_idx: usize,
    pub comm: Option<&'a dyn metrale_comm::CommBackend>,
    pub graph_capture: bool,
    /// 2026-10-23: The layer's installed mHC plans (`--forward circuit`): a launch at a row
    /// count with a plan runs the plan's step.
    pub plans: Option<&'a crate::glm5next_mhc_plan::InstalledMhc>,
}

/// 2026-10-23: An mHC site: the layer's [`LayerIo`] and its mHC weights and kernels.
#[derive(Clone, Copy)]
pub(crate) struct MhcSite<'a> {
    pub io: LayerIo<'a>,
    pub mhc: &'a Glm5NextMhc,
}

impl LayerIo<'_> {
    /// 2026-10-23: The installed mHC plan for a site launched as `row_pieces` (one piece only).
    pub(crate) fn plan(
        &self,
        row_pieces: &[usize],
    ) -> Option<crate::glm5next_mhc_plan::MhcSteps<KernelHandle>> {
        match row_pieces {
            [rows] => self.plans.and_then(|p| p.get(*rows)),
            _ => None,
        }
    }

    /// 2026-09-25: `rms_norm_vanilla` over `rows` contiguous `[hidden]` rows in one launch. The
    /// kernel runs one block per row (`token = blockIdx.x`) and the blocks share nothing, so the
    /// result equals `rows` single-row launches.
    pub(crate) fn norm(
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

    /// 2026-09-25: All-reduce `rows` contiguous `[hidden]` BF16 rows in one collective when
    /// `ctx.comm` is set: blocking under graph capture, asynchronous otherwise. No-op without a
    /// communicator.
    pub(crate) fn reduce_partial(&self, p: DevicePtr, rows: usize, stream: u64) -> Result<()> {
        // 2026-10-11: Sliced under `METRALE_GLM_BATCH_INVARIANT` (`glm5next_batch_inv`).
        let Some(comm) = self.comm else { return Ok(()) };
        let (h, cap) = (self.hidden * 2, self.graph_capture);
        crate::glm5next_batch_inv::all_reduce_rows(
            comm,
            self.levers.dense.batch_invariant,
            p,
            rows,
            h,
            cap,
            stream,
        )
    }

    /// 2026-10-09: [`Self::reduce_partial`] once per piece of `row_pieces`, so each piece's
    /// collective has the size its own pass gives it (NCCL may order an element's sum by the
    /// message size); `[rows]` is one collective, as before.
    pub(crate) fn reduce_pieces(
        &self,
        p: DevicePtr,
        row_pieces: &[usize],
        stream: u64,
    ) -> Result<()> {
        let mut r0 = 0usize;
        for &n in row_pieces {
            self.reduce_partial(p.offset(r0 * self.hidden * 2), n, stream)?;
            r0 += n;
        }
        Ok(())
    }

    /// 2026-10-10: The all-reduce before an mHC post-mix. With `fused` and one piece, asks the
    /// communicator for the exchange alone and returns where the partials landed, for
    /// [`Self::hc_post_reduced`] to sum; otherwise (or when the payload does not take the one-shot
    /// path) runs [`Self::reduce_pieces`] and returns `None`.
    pub(crate) fn reduce_or_exchange(
        &self,
        p: DevicePtr,
        row_pieces: &[usize],
        fused: bool,
        stream: u64,
    ) -> Result<Option<metrale_comm::PeerPartials>> {
        // 2026-10-23: A plan exchanges and never falls back to the all-reduce.
        if let ([rows], Some(_)) = (row_pieces, self.plan(row_pieces)) {
            let comm = self
                .comm
                .context("GLM mHC plan: the exchange needs a communicator")?;
            let bytes = rows * self.hidden * 2;
            let order = (!self.graph_capture).then_some(stream);
            let partials = comm.exchange_partials(p.0, bytes, order)?;
            return partials.map(Some).with_context(|| {
                format!("GLM mHC plan: the communicator has no one-shot exchange for {bytes} bytes")
            });
        }
        if let (true, [rows], Some(comm)) = (fused, row_pieces, self.comm) {
            let bytes = rows * self.hidden * 2;
            let order = (!self.graph_capture).then_some(stream);
            if let Some(partials) = comm.exchange_partials(p.0, bytes, order)? {
                return Ok(Some(partials));
            }
        }
        self.reduce_pieces(p, row_pieces, stream)?;
        Ok(None)
    }
}

impl MhcSite<'_> {
    /// 2026-10-09: One mHC site's pre-mix of `k` rows into `y` (with `post` and `comb`), then
    /// the RMS norm of `y` with `norm_w` into `normed`: `glm_hc_pre` and `norm`, or with
    /// `mhc_fused()` the single `glm_hc_pre_norm` launch that writes the same bytes (2026-10-22:
    /// where `fused_runs` holds, else the two launches). Ends the caller's `profile::MHC` span
    /// `t_mhc`; the unfused norm keeps its `profile::NORM` span.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mhc_pre_norm_one(
        &self,
        gpu: &dyn GpuBackend,
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
        if let Some(plan) = self.io.plan(&[k]) {
            let eps = [self.io.rms_eps, self.mhc.hc_eps, self.io.rms_eps];
            let outs = [y, post, comb, normed];
            let iters = self.mhc.sinkhorn_iters;
            plan.run_pre(gpu, streams, site, norm_w, outs, k, iters, eps, stream)?;
            profile::end(profile::MHC, t_mhc, gpu, stream);
            return Ok(());
        }
        let (kt, ht, hct) = (k as u32, self.io.hidden as u32, self.mhc.hc_mult as u32);
        let iters = self.mhc.sinkhorn_iters as u32;
        if self.io.levers.mhc_fused
            && fused_runs(&self.mhc.kernels, site, self.io.hidden, self.mhc.hc_mult)
        {
            crate::glm5next_mhc_fused::glm_hc_pre_norm(
                gpu,
                &self.mhc.kernels,
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
                self.io.rms_eps,
                self.mhc.hc_eps,
                self.io.rms_eps,
                stream,
            )?;
            profile::end(profile::MHC, t_mhc, gpu, stream);
            return Ok(());
        }
        glm_hc_pre(
            gpu,
            &self.mhc.kernels,
            streams,
            site,
            y,
            post,
            comb,
            kt,
            ht,
            hct,
            iters,
            self.io.rms_eps,
            self.mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = self.io.levers.profile.start();
        self.io.norm(gpu, y, norm_w, normed, k, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        Ok(())
    }

    /// 2026-10-10: Whether the one-token decode folds the attention site's post-mix into the FFN
    /// site's pre-norm (`METRALE_GLM_MHC_POST_FOLD=1`); an error when the lever is on without the
    /// fused pre-norm, the kernel or the layer's `fold_streams`.
    pub(crate) fn post_fold(&self) -> Result<bool> {
        if !self.io.levers.mhc_post_fold {
            return Ok(false);
        }
        if !self.io.levers.mhc_fused
            || self.mhc.kernels.hc_post_pre_norm.0 == 0
            || self.mhc.fold_streams.0 == 0
        {
            bail!(
                "GLM layer {}: METRALE_GLM_MHC_POST_FOLD=1 needs METRALE_GLM_MHC_FUSED=1, \
                 glm5next_hc_post_pre_norm and the layer's fold streams",
                self.io.layer_idx
            );
        }
        Ok(true)
    }

    pub(crate) fn fused_post(&self) -> Result<bool> {
        if !self.io.levers.ar_fused_post {
            return Ok(false);
        }
        if self.mhc.kernels.hc_post_rank_sum.0 == 0 {
            bail!(
                "GLM layer {}: METRALE_GLM_AR_FUSED_POST=1 but glm5next_hc_post_rank_sum is not \
                 loaded on this target",
                self.io.layer_idx
            );
        }
        Ok(true)
    }

    /// 2026-10-10: `glm_hc_post` of `k` rows of `x` and the streams at `residual` into the
    /// highway at `streams` (the same buffer except after a folded post, `forward_one`), or, when
    /// `partials` is set, `glm_hc_post_rank_sum`, which first sums `x` with the peers' partials
    /// (writing the sum back to `x`), with the same bits as the all-reduce sum then `glm_hc_post`.
    /// 2026-10-16: One exception at two ranks, where the all-reduce is `__hadd` and the rank sum
    /// starts its FP32 accumulator at +0: an element whose two partials are both -0 sums to +0
    /// here and -0 in the all-reduce (every other pair is the same bits, `world2_sum_tests`). At
    /// three or more ranks the all-reduce is the same rank sum, so there is no exception.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_post_reduced(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        partials: Option<metrale_comm::PeerPartials>,
        residual: DevicePtr,
        streams: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        if let (Some(plan), Some(p)) = (self.io.plan(&[k]), partials.as_ref()) {
            return plan.run_post(gpu, x, p, [residual, post, comb, streams], k, stream);
        }
        let (kt, ht, hct) = (k as u32, self.io.hidden as u32, self.mhc.hc_mult as u32);
        match partials {
            Some(p) => glm_hc_post_rank_sum(
                gpu,
                &self.mhc.kernels,
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
                self.mhc.kernels.hc_post,
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
}
