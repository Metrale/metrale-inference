// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The all-reduce and post-mix steps of an mHC site (moved unchanged from `steps.rs`,
//! with the decode's post fold): which post path a site takes, the exchange that hands the
//! partials to a fused post, and the post-mix itself.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: as `steps.rs`'s.

use super::*;

impl Glm5NextLayer {
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
}
