// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The MTP drafter's entry points into its `Glm5NextLayer`: one decode token, and
//! one context row's DSA cache writes.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::layer::DsaRowSpan;
use metrale_model_layers::layer::AttnMetadataDev;

impl Glm5NextLayer {
    /// 2026-09-25: One token through this block for the MTP drafter; errors on a block with a
    /// hyper-connection. `hidden` is both input and residual and is updated in place. No disk
    /// tiers: the disk block lists passed down are empty.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_one_for_drafter(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.mhc.is_some() {
            bail!(
                "GLM layer {}: decode_one_for_drafter is the MTP block's path; this layer has a \
                 hyper-connection",
                self.layer_idx
            );
        }
        let (mut disk_a, mut disk_b) = (Vec::new(), Vec::new());
        self.forward_one_plain(
            hidden,
            hidden,
            state,
            kv_cache,
            seq_len,
            block_table,
            &mut disk_a,
            &mut disk_b,
            ctx,
            stream,
        )
    }

    /// 2026-10-11: One drafter token for each of `states.len()` sequences in one pass: row `r`
    /// of `hidden` (`[rows, hidden]`, input and residual, updated in place) is sequence `r` at
    /// drafter row `positions[r]`, its KV slot and block table in row `r` of `meta`. The same
    /// steps as [`Self::decode_one_for_drafter`] over all rows: input norm, the DSA rows
    /// (`decode_spans`, one one-row span per sequence), the row-parallel `o_proj` reduce, the
    /// residual, the post-attention norm, the routed MoE over all rows, the residual. Errors on
    /// a block with a hyper-connection or whose mixer is not DSA.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_rows_for_drafter(
        &self,
        hidden: DevicePtr,
        states: &mut [&mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        positions: &[usize],
        meta: &AttnMetadataDev,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let layer = match (&self.mixer, &self.mhc) {
            (Glm5NextMixer::Dsa(l), None) => l,
            _ => bail!(
                "GLM layer {}: decode_rows_for_drafter is the MTP block's path (DSA, no \
                 hyper-connection)",
                self.layer_idx
            ),
        };
        let rows = states.len();
        if rows == 0 || positions.len() != rows {
            bail!(
                "GLM layer {}: {rows} drafter states with {} positions",
                self.layer_idx,
                positions.len()
            );
        }
        let gpu = ctx.gpu;
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();
        self.norm(gpu, hidden, self.input_norm, normed, rows, stream)?;
        let spans: Vec<DsaRowSpan> = positions
            .iter()
            .map(|&p| DsaRowSpan {
                first_pos: p,
                rows: 1,
            })
            .collect();
        // 2026-10-11: Writes its output projection over its input buffer, as `decode` does.
        layer.decode_spans(normed, states, &spans, kv_cache, meta, 0, ctx, stream)?;
        if self.mixer_all_reduce {
            self.reduce_partial(normed, rows, ctx, stream)?;
        }
        self.add_inplace(gpu, hidden, normed, rows * self.hidden, stream)?;
        self.norm(gpu, hidden, self.post_attn_norm, normed, rows, stream)?;
        // 2026-10-11: No post-mix here to take the sum, so the all-reduce stays whole.
        self.mlp_forward(normed, ffn_out, rows, &[rows], false, ctx, stream)?;
        self.add_inplace(gpu, hidden, ffn_out, rows * self.hidden, stream)
    }

    /// 2026-10-11: The most context rows [`Self::drafter_write_kv_rows`] takes in one pass: the
    /// DSA workspace's rows; 0 for a block whose mixer is not DSA.
    pub fn drafter_context_rows_max(&self) -> usize {
        match &self.mixer {
            Glm5NextMixer::Dsa(l) => l.drafter_rows_max(),
            _ => 0,
        }
    }

    /// 2026-10-11: `m` consecutive drafter context rows in one pass: `input_norm` over the rows
    /// of `x` (`[m, hidden]`) into `normed` (the caller's `[m, hidden]` scratch, so no shared
    /// forward buffer is written), then `Glm5NextDsaLayer::write_kv_rows_flat` at drafter row
    /// `seq_len`, row `r` to KV slot `slots_dev[r]`. The rows [`Self::drafter_write_kv_row`]
    /// writes one at a time.
    #[allow(clippy::too_many_arguments)]
    pub fn drafter_write_kv_rows(
        &self,
        x: DevicePtr,
        normed: DevicePtr,
        m: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        slots_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Glm5NextMixer::Dsa(layer) = &self.mixer else {
            bail!(
                "GLM layer {}: drafter_write_kv_rows is the MTP block's path; this layer is not \
                 a DSA layer",
                self.layer_idx
            );
        };
        self.norm(ctx.gpu, x, self.input_norm, normed, m, stream)?;
        layer.write_kv_rows_flat(normed, m, state, kv_cache, seq_len, slots_dev, ctx, stream)
    }

    /// 2026-09-25: One drafter context row: `input_norm`, then `Glm5NextDsaLayer::write_kv_row`,
    /// which writes the row's KV latent and indexer entry without computing the block output.
    /// Errors on a block whose mixer is not DSA.
    ///
    /// 2026-10-11: The normed row goes to the caller's `normed` (`[hidden]` BF16), not the
    /// shared forward scratch, so the row can be written at the end of the target's prefill
    /// while the target still owns that scratch (`DraftProposer::prefill_uses_shared_buffers`).
    #[allow(clippy::too_many_arguments)]
    pub fn drafter_write_kv_row(
        &self,
        x: DevicePtr,
        normed: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let layer = match &self.mixer {
            Glm5NextMixer::Dsa(l) => l,
            _ => bail!(
                "GLM layer {}: drafter_write_kv_row is the MTP block's path; this layer is not \
                 a DSA layer",
                self.layer_idx
            ),
        };
        self.norm(ctx.gpu, x, self.input_norm, normed, 1, stream)?;
        layer.write_kv_row(normed, state, kv_cache, seq_len, block_table, ctx, stream)
    }
}
