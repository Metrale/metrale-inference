// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The attention step of a single-stream prefill pass whose rows from
//! `cl` on are a replay tail (`MidchunkCapture::replay_tail`): rows `[0, cl)` take the
//! pass's own route, and rows `[cl, n)` are one paged call starting at `cl`, the call a
//! request that restores the state captured at `cl` makes. Both calls run their Q/K/V
//! projection, attention and O projection over their own rows, so the tail rows get
//! that request's bits.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - The two calls' outputs are assembled in `moe_output`, which nothing in the
//!   attention step uses; the FFN overwrites it after the residual add has read it.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use super::AttnRoute;
use crate::layer::{ForwardContext, LayerState};

impl Qwen3AttentionLayer {
    /// 2026-10-01: The split row `cl` of this pass when it carries a replay-tail capture
    /// strictly inside its `num_tokens` rows; `None` otherwise.
    pub(super) fn replay_tail_split(ctx: &ForwardContext, num_tokens: usize) -> Option<usize> {
        let cap = ctx.midchunk_capture.as_ref()?;
        (cap.replay_tail && cap.cap_local > 0 && cap.cap_local < num_tokens)
            .then_some(cap.cap_local)
    }

    /// 2026-10-01: The attention output `[num_tokens, hidden]` of a pass split at `cl`
    /// (module doc). `route` is the pass's own route for the head rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_replay_tail(
        &self,
        state: &mut dyn LayerState,
        normed: DevicePtr,
        num_tokens: usize,
        cl: usize,
        route: AttnRoute,
        seq_len_start: usize,
        kv_cache: &mut PagedKvCache,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let row_bytes = ctx.config.hidden_size * 2;
        let meta = ctx.attn_metadata.ok_or_else(|| {
            anyhow::anyhow!("replay-tail attention split: the pass has no attention metadata")
        })?;
        if meta.block_table.is_null() {
            anyhow::bail!(
                "replay-tail attention split: the pass uploaded no block table, which the \
                 tail's paged call reads"
            );
        }
        let head = match route {
            AttnRoute::Contiguous => self.prefill_attention_with_cache_skip(
                state,
                normed,
                cl,
                kv_write_start,
                block_table,
                kv_cache,
                None,
                ctx,
                stream,
            )?,
            AttnRoute::Paged => self.prefill_attention_paged(
                state,
                normed,
                cl,
                seq_len_start,
                kv_cache,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                None,
                kv_write_start,
                ctx,
                stream,
            )?,
            AttnRoute::Refuse => {
                anyhow::bail!("replay-tail attention split: a refused route has no head call")
            }
        };
        let assembled = ctx.buffers.moe_output();
        ctx.gpu
            .copy_d2d_async(head, assembled, cl * row_bytes, stream)?;

        let tail_ctx = ForwardContext {
            buffers: ctx.buffers,
            hc_row_offset: ctx.hc_row_offset,
            gpu: ctx.gpu,
            config: ctx.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: Some(meta.rows_from(cl)),
            profile: ctx.profile,
            comm: ctx.comm,
            graph_capture: ctx.graph_capture,
            decode_step: ctx.decode_step,
            gdn_exact_replay: ctx.gdn_exact_replay,
            gdn_write_on_accept: ctx.gdn_write_on_accept,
            token_ids: ctx.token_ids,
            host_token_ids: ctx.host_token_ids,
            routed_lora_layers: ctx.routed_lora_layers,
            midchunk_capture: None,
            moe_lora_route: ctx.moe_lora_route,
        };
        let tail_rows = num_tokens - cl;
        let tail = self.prefill_attention_paged(
            state,
            normed.offset(cl * row_bytes),
            tail_rows,
            seq_len_start + cl,
            kv_cache,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            None,
            kv_write_start.saturating_sub(cl),
            &tail_ctx,
            stream,
        )?;
        ctx.gpu.copy_d2d_async(
            tail,
            assembled.offset(cl * row_bytes),
            tail_rows * row_bytes,
            stream,
        )?;
        Ok(assembled)
    }
}
