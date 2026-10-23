// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The all-reduce and post-mix steps of an mHC site (moved unchanged from `steps.rs`,
//! with the decode's post fold): which post path a site takes, the exchange that hands the
//! partials to a fused post, and the post-mix itself. 2026-10-23: The bodies moved to
//! `steps/mhc_site.rs` ([`MhcSite`], [`LayerIo`]); these methods build the site from the layer
//! and the forward context and delegate.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: as `steps.rs`'s.

use super::*;

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
