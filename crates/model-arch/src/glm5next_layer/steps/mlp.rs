// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The MLP site of `Glm5NextLayer` (`mlp_forward`, `mlp_compute`), split out of
//! `steps.rs` (file-size cap) by exact copy.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: those of the parent module.

use super::*;

impl Glm5NextLayer {
    /// 2026-10-10: Whether this mHC site sums its all-reduce inside the post-mix
    /// (`METRALE_GLM_AR_FUSED_POST=1`). Errors when the lever is on and the target lacks
    /// `glm5next_hc_post_rank_sum`, rather than silently running the unfused pair.
    /// 2026-09-25: The MLP over `rows` rows of `normed` into `out`, then one all-reduce of all
    /// rows when `Glm5NextMlpConfig::needs_all_reduce`. 2026-10-09: `row_pieces` splits the rows
    /// into consecutive pieces (`[rows]`: one) whose launches must not depend on each other: a
    /// dense site runs once per piece, a routed site launches its router and shared expert once
    /// per piece (`forward_moe_pieces`) and shares the rest, and the all-reduce runs per piece
    /// (`reduce_pieces`).
    /// 2026-10-10: `fused` hands the all-reduce's sum to the caller's post-mix
    /// ([`Self::reduce_or_exchange`]); the returned partials are then `Some`.
    /// 2026-10-13: `tap` dumps `out` between the MLP and the reduce (`glm5next_layer_dump`).
    /// 2026-10-22: And the routed MLP's top-k ids, before the next MLP (the text layers share
    /// one workspace) overwrites them.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mlp_forward(
        &self,
        normed: DevicePtr,
        out: DevicePtr,
        rows: usize,
        row_pieces: &[usize],
        fused: bool,
        ctx: &ForwardContext,
        stream: u64,
        tap: Option<&dump::Tap<'_>>,
    ) -> Result<Option<metrale_comm::PeerPartials>> {
        self.mlp_compute(normed, out, rows, row_pieces, ctx, stream)?;
        dump::put(tap, dump::Point::MlpOut, out)?;
        dump::moe_ids(tap)?;
        // 2026-09-25: One all-reduce covers both partial sums, the EP-split routed experts and
        // the TP-split shared expert: `forward_moe` adds them together before returning.
        let mut partials = None;
        if self.mlp_cfg.needs_all_reduce() {
            // 2026-10-10: Forked before the reduce is enqueued, so the warm runs beside it.
            if let Some(l2) = &self.l2_warm {
                l2.fork(ctx.gpu, self.layer_idx, rows, stream)?;
            }
            self.reduce_probe(profile::REDUCE_MLP_BAR, "mlp", ctx, stream);
            let t = self.levers.profile.start_hot();
            partials = self.reduce_or_exchange(out, row_pieces, fused, ctx, stream)?;
            profile::end_nosync(profile::REDUCE_MLP_ENQ, t);
            // 2026-09-25: This span times only the `synchronize` in `profile::end`.
            let t = self.levers.profile.start_hot();
            profile::end(profile::REDUCE_MLP, t, ctx.gpu, stream);
        }
        Ok(partials)
    }

    /// 2026-10-12: The MLP launches of [`Self::mlp_forward`] without its all-reduce: `out`
    /// holds this rank's partial (or the whole output when no reduce is needed). SP-mHC
    /// (`forward_rows_sp`) reduce-scatters it instead.
    pub(in crate::glm5next_layer) fn mlp_compute(
        &self,
        normed: DevicePtr,
        out: DevicePtr,
        rows: usize,
        row_pieces: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        crate::glm5next_mlp::forward::check_pieces(rows, row_pieces)?;
        // 2026-10-23: An installed plan runs a routed site at its row count as one piece.
        if let Glm5NextMlpSite::Moe(w) = &self.mlp
            && row_pieces == [rows]
            && self.moe_plans.run(
                ctx.gpu,
                &self.mlp_kernels,
                &self.mlp_cfg,
                w,
                &self.mlp_ws,
                normed,
                out,
                rows,
                stream,
            )?
        {
            return Ok(());
        }
        let t_dense = matches!(self.mlp, Glm5NextMlpSite::Dense(_))
            .then(|| self.levers.profile.start())
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
        Ok(())
    }
}
