// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: `LayerSplitPrefill` for the GDN layer (moved out of trait_layer.rs unchanged, plus
//! the wave prefill entry points `prefill_mixer` and `prefill_ffn_rows`).
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::Qwen3SsmLayer;
use crate::layer::{ForwardContext, GdnPrefillBuffers, LayerSplitPrefill, LayerState};

impl LayerSplitPrefill for Qwen3SsmLayer {
    fn supports_wave_prefill(&self) -> bool {
        self.hc.is_none()
    }

    fn prefill_mixer(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.hc.is_none(),
            "prefill_mixer: the mHC body has no wave split"
        );
        self.prefill_inner_ex(hidden, residual, num_tokens, state, true, ctx, stream)
    }

    fn prefill_ffn_rows(
        &self,
        hidden: DevicePtr,
        ffn_in: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_ffn_tail(hidden, ffn_in, num_tokens, None, ctx, stream)
    }

    fn prefill_phase1_proj_batched(
        &self,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase1_proj_batched_inner(
            hidden_stacked,
            residual_stacked,
            total_tokens,
            gdn_bufs,
            ctx,
            stream,
        )
    }

    fn prefill_phase1_conv1d_one(
        &self,
        state: &mut dyn LayerState,
        token_offset: usize,
        len: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase1_conv1d_one_inner(state, token_offset, len, gdn_bufs, ctx, stream)
    }

    fn prefill_phase1_l2_batched(
        &self,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase1_l2_batched_inner(total_tokens, gdn_bufs, ctx, stream)
    }

    fn prefill_gdn_full(
        &self,
        state: &mut dyn LayerState,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_gdn_full_inner(state, gdn_bufs, ctx, stream)
    }

    fn prefill_gdn_full_batched(
        &self,
        h_state_ptrs: DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        chunk_len: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_gdn_full_batched_inner(
            h_state_ptrs,
            gdn_bufs,
            batch_size,
            chunk_len,
            ctx,
            stream,
        )
    }

    fn prefill_gdn_full_batched_fla_varlen(
        &self,
        h_state_ptrs: DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        cu_seqlens: DevicePtr,
        max_num_chunks: u32,
        total_nt: usize,
        max_seqlen: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        self.prefill_gdn_full_batched_fla_varlen_inner(
            h_state_ptrs,
            gdn_bufs,
            batch_size,
            cu_seqlens,
            max_num_chunks,
            total_nt,
            max_seqlen,
            ctx,
            stream,
        )
    }

    fn prefill_phase3(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase3_inner(
            hidden,
            residual,
            num_tokens,
            gdn_bufs,
            token_offset,
            ctx,
            stream,
        )
    }

    fn supports_wave_split_mixer(&self) -> bool {
        self.hc.is_none()
    }

    fn wave_core_row_bytes(&self, config: &metrale_config::ModelConfig) -> usize {
        config.linear_num_value_heads * config.linear_value_head_dim * 2
    }

    fn prefill_mixer_pre(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let idx =
            super::debug::SSM_LAYER_CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::layers::ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            ctx.buffers.norm_output(),
            residual,
            num_rows as u32,
            ctx.config.hidden_size as u32,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        self.prefill_block_in(ctx.buffers.norm_output(), num_rows, idx, ctx, stream)
    }

    fn prefill_mixer_core(
        &self,
        row0: usize,
        num_tokens: usize,
        state: &mut dyn LayerState,
        core_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let idx = super::debug::SSM_LAYER_CALL_COUNTER
            .load(std::sync::atomic::Ordering::Relaxed)
            .wrapping_sub(1);
        let out = core_out.offset(row0 * self.wave_core_row_bytes(ctx.config));
        self.prefill_block_core(row0, num_tokens, state, idx, out, ctx, stream)
    }

    fn prefill_mixer_post(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_rows: usize,
        core_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let idx = super::debug::SSM_LAYER_CALL_COUNTER
            .load(std::sync::atomic::Ordering::Relaxed)
            .wrapping_sub(1);
        let out_proj = self.prefill_block_out(core_out, num_rows, idx, ctx, stream)?;
        crate::layers::ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            out_proj,
            &self.post_attn_norm,
            ctx.buffers.norm_output(),
            residual,
            num_rows as u32,
            ctx.config.hidden_size as u32,
            ctx.config.rms_norm_eps as f32,
            stream,
        )
    }
}
