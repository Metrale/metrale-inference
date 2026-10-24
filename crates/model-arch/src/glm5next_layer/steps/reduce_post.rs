// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The all-reduce and post-mix steps of an mHC site (moved unchanged from `steps.rs`,
//! with the decode's post fold): which post path a site takes, the exchange that hands the
//! partials to a fused post, and the post-mix itself. 2026-10-23: The bodies moved to
//! `steps/mhc_site.rs` ([`MhcSite`], [`LayerIo`]); these methods build the site from the layer
//! and the forward context and delegate.
//!
//! 2026-10-24: And `--glm-mhc-post-pre` (R3): the attention site's post-sum and the FFN site's
//! pre-norm as one `glm5next_hc_post_pre_norm_c1` launch where the pair would run exactly the
//! post-sum then c1 (`crate::glm5next_mhc_post_pre_c1::kernel_for`; [`MhcSite::post_pre_c1`]).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: as `steps.rs`'s. The fused launch never runs under an installed mHC plan
//! (installation refuses the lever, `glm5next_plan::mhc`) or with a layer dump armed (the dump
//! reads the summed row between the two launches); it returns `false` without launching
//! whenever the pair must run, and the caller then runs it.

use super::*;
use crate::glm5next_mhc_levers::fused_runs;
use crate::glm5next_mhc_post_pre_tok::PostPreTok;

impl Glm5NextLayer {
    /// 2026-10-23: The layer's norm and all-reduce inputs under `comm` / `graph_capture`.
    pub(in crate::glm5next_layer) fn layer_io<'a>(
        &'a self,
        comm: Option<&'a dyn metrale_comm::CommBackend>,
        graph_capture: bool,
    ) -> LayerIo<'a> {
        LayerIo {
            levers: &self.levers,
            rms_norm_k: self.rms_norm_k,
            hidden: self.hidden,
            rms_eps: self.rms_eps,
            layer_idx: self.layer_idx,
            comm,
            graph_capture,
            plans: Some(&self.plans.mhc),
        }
    }

    /// 2026-10-23: The mHC site of `mhc` with the layer's inputs.
    pub(in crate::glm5next_layer) fn mhc_site<'a>(
        &'a self,
        mhc: &'a Glm5NextMhc,
        comm: Option<&'a dyn metrale_comm::CommBackend>,
        graph_capture: bool,
    ) -> MhcSite<'a> {
        MhcSite {
            io: self.layer_io(comm, graph_capture),
            mhc,
        }
    }

    /// 2026-10-23: [`LayerIo::norm`].
    pub(in crate::glm5next_layer) fn norm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        self.layer_io(None, false)
            .norm(gpu, x, w, out, rows, stream)
    }

    /// 2026-10-23: [`MhcSite::mhc_pre_norm_one`].
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
        self.mhc_site(mhc, None, false).mhc_pre_norm_one(
            gpu, site, norm_w, streams, y, post, comb, normed, k, t_mhc, stream,
        )
    }

    /// 2026-10-23: [`LayerIo::reduce_partial`].
    pub(in crate::glm5next_layer) fn reduce_partial(
        &self,
        p: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.layer_io(ctx.comm, ctx.graph_capture)
            .reduce_partial(p, rows, stream)
    }

    /// 2026-10-23: [`MhcSite::post_fold`].
    pub(in crate::glm5next_layer) fn post_fold(&self, mhc: &Glm5NextMhc) -> Result<bool> {
        self.mhc_site(mhc, None, false).post_fold()
    }

    /// 2026-10-23: [`MhcSite::fused_post`].
    pub(in crate::glm5next_layer) fn fused_post(&self, mhc: &Glm5NextMhc) -> Result<bool> {
        self.mhc_site(mhc, None, false).fused_post()
    }

    /// 2026-10-23: [`LayerIo::reduce_or_exchange`]: no mHC weights are read.
    pub(in crate::glm5next_layer) fn reduce_or_exchange(
        &self,
        p: DevicePtr,
        row_pieces: &[usize],
        fused: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<metrale_comm::PeerPartials>> {
        self.layer_io(ctx.comm, ctx.graph_capture)
            .reduce_or_exchange(p, row_pieces, fused, stream)
    }

    /// 2026-10-23: [`MhcSite::hc_post_reduced`].
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
        self.mhc_site(mhc, None, false)
            .hc_post_reduced(gpu, x, partials, residual, streams, post, comb, k, stream)
    }
}

impl MhcSite<'_> {
    /// 2026-10-24: The fused launch over `k` rows when it applies: `x` this rank's partial (the
    /// mixer output) with `partials` the peers', the highway `streams` (updated in place), the
    /// post-mix weights `post`/`comb` (overwritten with the FFN site's), and the FFN site's
    /// `norm_w`, `y` and `normed`. `dumping`: a layer dump tap is armed. `on`: the lever for this
    /// pass (`--glm-mhc-post-pre`, or the paired A/B's `mhc_post_pre` arm).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn post_pre_c1(
        &self,
        gpu: &dyn GpuBackend,
        on: bool,
        x: DevicePtr,
        partials: Option<&metrale_comm::PeerPartials>,
        norm_w: DevicePtr,
        [streams, y, post, comb, normed]: [DevicePtr; 5],
        k: usize,
        dumping: bool,
        stream: u64,
    ) -> Result<bool> {
        let (Some(p), false, None) = (partials, dumping, self.io.plan(&[k])) else {
            return Ok(false);
        };
        let (h, hc) = (self.io.hidden, self.mhc.hc_mult);
        let site = &self.mhc.ffn;
        let pre_fused = self.io.levers.mhc_fused && fused_runs(&self.mhc.kernels, site, h, hc);
        let flags = [on, true, pre_fused];
        let dims = [k as u32, h as u32, hc as u32];
        let Some(kernel) =
            crate::glm5next_mhc_post_pre_c1::kernel_for(&self.mhc.kernels, flags, dims)?
        else {
            return Ok(false);
        };
        let a = PostPreTok {
            block_out: x,
            partials: p,
            residual: streams,
            post,
            comb,
            streams_out: streams,
            site,
            norm_w,
            y,
            post_out: post,
            comb_out: comb,
            normed,
            num_tokens: k as u32,
            dims: [h as u32, hc as u32, self.mhc.sinkhorn_iters as u32],
            eps: [self.io.rms_eps, self.mhc.hc_eps, self.io.rms_eps],
        };
        crate::glm5next_mhc_post_pre_c1::launch(gpu, kernel, &a, stream)?;
        // 2026-10-24: Once per process: the serve log's proof that R3 reached the site.
        static TAKEN: std::sync::Once = std::sync::Once::new();
        TAKEN.call_once(|| tracing::info!("GLM mHC: fused post-sum + pre-norm taken ({k} rows)"));
        Ok(true)
    }
}

impl Glm5NextLayer {
    /// 2026-10-24: [`MhcSite::post_pre_c1`] under this pass's lever (`ab_levers::mhc_post_pre`).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn post_pre_c1(
        &self,
        ctx: &ForwardContext,
        mhc: &Glm5NextMhc,
        x: DevicePtr,
        partials: Option<&metrale_comm::PeerPartials>,
        norm_w: DevicePtr,
        bufs: [DevicePtr; 5],
        k: usize,
        dumping: bool,
        stream: u64,
    ) -> Result<bool> {
        let on = super::ab_levers::mhc_post_pre(self.levers.decode.mhc_post_pre, ctx)?;
        self.mhc_site(mhc, None, false)
            .post_pre_c1(ctx.gpu, on, x, partials, norm_w, bufs, k, dumping, stream)
    }
}
