// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The two halves of `prefill_inner` for a wave prefill: `prefill_inner` (both
//! halves, the single-stream prefill), and `prefill_ffn_tail`, the FFN half on rows whose
//! post-attention norm output is `ffn_in`. A wave runs the first half per stream
//! (`prefill_inner_ex` with `defer_ffn`), then the FFN half once over all streams' rows, so
//! each row runs the arithmetic of a single-stream prefill.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - `prefill_inner` is `prefill_inner_ex` without deferral, and `prefill_inner_ex`'s FFN half
//!   is `prefill_ffn_tail` on `norm_output`: the single-stream bits are unchanged.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use super::super::diag_norm;
use crate::layer::{BatchedAttnMetadata, ForwardContext, LayerState};

impl Qwen3AttentionLayer {
    /// 2026-10-05: The whole prefill body (attention and FFN halves).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn prefill_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        batched_meta: Option<&BatchedAttnMetadata>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_inner_ex(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            batched_meta,
            false,
            ctx,
            stream,
        )
    }

    /// 2026-10-05: Whether the FFN half can be deferred: no mHC body, an FFN present, and no
    /// LongCat shortcut carry (its producer writes a per-pass carry buffer).
    pub(in crate::layers::qwen3_attention) fn supports_wave_prefill_body(&self) -> bool {
        self.hc.is_none() && !self.ffn.is_none() && self.shortcut_carry_out.is_none()
    }

    /// 2026-10-05: The FFN half of `prefill_inner` over `num_tokens` rows: `ffn_in` holds the
    /// post-attention norm output of those rows and `hidden` their residual stream; ends with
    /// the layer's output in `hidden`.
    pub(in crate::layers::qwen3_attention) fn prefill_ffn_tail(
        &self,
        hidden: DevicePtr,
        ffn_in: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;
        let bf16 = 2usize;
        let is_mistral_diag = ctx.profile
            && ctx.config.model_type == "mistral"
            && (self.attn_layer_idx == 0 || self.attn_layer_idx == 35);
        // 2026-09-25: `METRALE_PREFILL_HOST_TIMING=1`: host wall-clock time of this
        // layer's FFN half, taken with no synchronize.
        let t_ffn = (std::env::var("METRALE_PREFILL_HOST_TIMING").as_deref() == Ok("1"))
            .then(std::time::Instant::now);
        // 2026-09-25: LongCat shortcut MoE (producer): runs before the dense FFN
        // (both write `moe_output`), folds the zero experts in, and stashes the
        // result in the carry buffer.
        if let (Some(moe_ffn), Some((carry, cap))) = (&self.moe_ffn, self.shortcut_carry_out)
            && self.pre_moe_norm.is_none()
        {
            anyhow::ensure!(
                num_tokens <= cap,
                "shortcut carry capacity {cap} < prefill chunk {num_tokens}"
            );
            moe_ffn
                .forward_prefill(ffn_in, num_tokens, ctx, stream)
                .map_err(|e| anyhow::anyhow!("shortcut moe forward_prefill failed: {e}"))?;
            let moe_out = ctx.buffers.moe_output();
            if let crate::layers::FfnComponent::Moe(m) = moe_ffn {
                m.apply_zero_expert(
                    moe_out,
                    ffn_in,
                    num_tokens as u32,
                    ctx,
                    stream,
                )?;
            }
            // 2026-09-25: `METRALE_OP_DUMP` hook: the shortcut MoE output (zero
            // experts folded in), captured before the dense FFN reuses this
            // buffer. Not the same as "moe_out" below, the dense FFN output.
            if num_tokens > 0 {
                super::super::super::op_dump::dump_bf16(
                    ctx.gpu,
                    moe_out,
                    (num_tokens - 1) * h * bf16,
                    h,
                    self.attn_layer_idx,
                    "shortcut_moe_out",
                    stream,
                )?;
            }
            ctx.gpu
                .copy_d2d_async(moe_out, carry, num_tokens * h * 2, stream)?;
        }
        self.ffn
            .forward_prefill(ffn_in, num_tokens, ctx, stream)
            .map_err(|e| anyhow::anyhow!("ffn.forward_prefill failed: {e}"))?;
        if let Some(t) = t_ffn {
            crate::layers::qwen3_attention::add_ffn_host_us(t.elapsed().as_micros() as u64);
        }

        let dense_out = ctx.buffers.moe_output();
        // 2026-09-25: `METRALE_OP_DUMP` hook: the FFN output (last token), before
        // any post-FFN norm or residual add.
        if num_tokens > 0 {
            super::super::super::op_dump::dump_bf16(
                ctx.gpu,
                dense_out,
                (num_tokens - 1) * h * bf16,
                h,
                self.attn_layer_idx,
                "moe_out",
                stream,
            )?;
        }

        if is_mistral_diag {
            diag_norm(
                ctx.gpu,
                dense_out,
                h,
                stream,
                &format!("L{} moe_out", self.attn_layer_idx),
            );
        }

        self.prefill_ffn_residual(hidden, dense_out, num_tokens, n, h, eps, ctx, stream)?;

        // 2026-09-25: Gemma-4 `layer_scalar`: scale the whole hidden state at the
        // end of the layer.
        if let Some(scalar) = self.layer_scalar {
            self.apply_layer_scalar(ctx.gpu, hidden, num_tokens * h, scalar, stream)?;
        }

        if is_mistral_diag {
            diag_norm(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{} residual", self.attn_layer_idx),
            );
        }

        Ok(())
    }
}
