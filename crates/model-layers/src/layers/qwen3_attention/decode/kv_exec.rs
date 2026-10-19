// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The attention layer's KV executor: one method per launch-argument family
//! (`kv_abi::KvExecutor`) for a symmetric cache, and the fused pair launch for a split one. The
//! codec's spec picks the method (`kv_abi::decode`/`write`); nothing here names a codec.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - A side is WHT-rotated before the write exactly when its spec rotates at the head dim
//!   (`KvCodecSpec::rotates_at`), unless the weights were pre-rotated at load.
//! - A split pair runs the layer's one fused write and decode kernel with each side's tail.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_kv_codec::{KvCodecId, spec};

use super::super::Qwen3AttentionLayer;
use crate::kv_abi::{self, DecodeCommon, KvExecutor, KvSideArgs, SideSource, Tails, WriteCommon};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// 2026-10-18: The arguments of one `run_paged_decode` call.
pub struct DecodeArgs<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub kv_cache: &'a PagedKvCache,
    pub common: DecodeCommon,
    pub workspace: DevicePtr,
    /// 2026-09-25: `ModelLevers::max_decode_seqs`, which the `legacy` split count reads through
    /// `split_ref_seqs`.
    pub max_decode_seqs: u32,
    pub stream: u64,
}

/// 2026-10-18: The arguments of one `write_kv_cache` call.
pub struct WriteArgs<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub kv_cache: &'a PagedKvCache,
    pub common: WriteCommon,
    pub stream: u64,
    pub ctx: &'a ForwardContext<'a>,
}

impl Qwen3AttentionLayer {
    /// 2026-09-25: An MLA layer with a rope part (`mla.rope > 0`) is treated as V4-Flash: its
    /// `kv_lora_rank + rope`-wide cache rows go to the MLA decode kernels.
    pub(super) fn is_v4_flash(&self) -> bool {
        self.mla.as_ref().map(|m| m.rope > 0).unwrap_or(false)
    }

    /// 2026-10-18: The side arguments of the K and V sides of this layer's cache, each from its
    /// codec's family.
    fn kv_sides(&self, kv_cache: &PagedKvCache) -> Result<(KvSideArgs, KvSideArgs)> {
        let layer = self.attn_layer_idx;
        let (k_scale, v_scale) = self.effective_fp8_scales();
        let side = |codec: KvCodecId, pool, block_stride: usize, scale| {
            kv_abi::side_args(
                spec(codec),
                SideSource {
                    pool,
                    block_stride: block_stride as u64,
                    data_bytes: kv_cache.data_bytes_of(codec) as u64,
                    scale,
                },
            )
        };
        Ok((
            side(
                self.kv_pair.k(),
                kv_cache.k_pool_ptr(layer),
                kv_cache.k_block_stride_bytes_for_layer(layer),
                k_scale,
            )?,
            side(
                self.kv_pair.v(),
                kv_cache.v_pool_ptr(layer),
                kv_cache.v_block_stride_bytes_for_layer(layer),
                v_scale,
            )?,
        ))
    }

    /// 2026-10-18: WHT-rotates each side whose codec rotates at this head dim, before the write.
    /// K and V are `[num_tokens, num_kv_heads, head_dim]` BF16 and the kernel runs one CTA per
    /// head row, so the grid covers every (token, kv_head) pair. The InnerQ post-WHT scale
    /// follows a rotated K: while `d_innerq_calibrating` is 1, block 0 also accumulates K²
    /// statistics; with both device flags 0 the kernel does nothing.
    fn rotate_before_write(&self, a: &WriteArgs<'_>) -> Result<()> {
        if a.ctx.levers.weight_pre_rotated || self.wht_bf16_k.0 == 0 {
            return Ok(());
        }
        let c = &a.common;
        let rows = |kernel: KernelHandle, x: DevicePtr| {
            KernelLaunch::new(a.gpu, kernel)
                .grid([c.num_kv_heads * c.num_tokens, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(x)
                .arg_u32(c.head_dim)
                .launch(a.stream)
        };
        if spec(self.kv_pair.k()).rotates_at(c.head_dim) {
            rows(self.wht_bf16_k, c.key)?;
            if self.innerq_apply_k_k.0 != 0 && c.head_dim == 128 {
                rows(self.innerq_apply_k_k, c.key)?;
            }
        }
        if spec(self.kv_pair.v()).rotates_at(c.head_dim) {
            rows(self.wht_bf16_k, c.value)?;
        }
        Ok(())
    }

    /// 2026-10-18: Decode attention over a split pair: the layer's fused kernel with each side's
    /// tail and the sliding window.
    pub(super) fn decode_pair(&self, a: &DecodeArgs<'_>) -> Result<()> {
        let (k, v) = self.kv_sides(a.kv_cache)?;
        kv_abi::launch_fused_decode(
            a.gpu,
            self.paged_decode_k,
            &a.common,
            k,
            v,
            Tails::PerSide,
            a.stream,
        )
    }

    /// 2026-10-18: Write into a split pair: the rotated sides first, then the layer's fused
    /// kernel with each side's tail.
    pub(super) fn write_pair(&self, a: &WriteArgs<'_>) -> Result<()> {
        self.rotate_before_write(a)?;
        let (k, v) = self.kv_sides(a.kv_cache)?;
        kv_abi::launch_fused_write(
            a.gpu,
            self.reshape_cache_k,
            &a.common,
            k,
            v,
            Tails::PerSide,
            a.stream,
        )
    }

    /// 2026-10-18: The non-split launch of a symmetric block-scaled cache through `kernel`.
    pub(super) fn decode_block_scaled_with(
        &self,
        a: &DecodeArgs<'_>,
        kernel: KernelHandle,
    ) -> Result<()> {
        let (k, v) = self.kv_sides(a.kv_cache)?;
        kv_abi::launch_fused_decode(a.gpu, kernel, &a.common, k, v, Tails::Shared, a.stream)
    }
}

impl KvExecutor for Qwen3AttentionLayer {
    type Decode<'a> = DecodeArgs<'a>;
    type Write<'a> = WriteArgs<'a>;

    fn decode_dense(&self, a: &Self::Decode<'_>) -> Result<()> {
        self.run_paged_decode_bf16(a)
    }

    fn decode_dense_scaled(&self, a: &Self::Decode<'_>) -> Result<()> {
        if self.is_v4_flash() {
            return self.run_mla_decode_fp8(a);
        }
        self.run_paged_decode_fp8(a)
    }

    fn decode_block_scaled(&self, a: &Self::Decode<'_>) -> Result<()> {
        let rotated = spec(self.kv_pair.k()).rotates();
        if self.is_v4_flash() && !rotated {
            return self.run_mla_decode_nvfp4(a);
        }
        match self.paged_decode_splitk_k.zip(self.paged_decode_reduce_k) {
            Some(splitk) => self.run_paged_decode_nvfp4(a, splitk),
            None => {
                let kernel = self.rotated_decode_kernel_for(a.common.head_dim)?;
                self.decode_block_scaled_with(a, kernel)
            }
        }
    }

    fn write_dense(&self, a: &Self::Write<'_>) -> Result<()> {
        let c = &a.common;
        ops::reshape_and_cache(
            a.gpu,
            self.reshape_cache_k,
            c.key,
            c.value,
            a.kv_cache.k_pool_ptr(self.attn_layer_idx),
            a.kv_cache.v_pool_ptr(self.attn_layer_idx),
            c.slot_mapping,
            c.num_tokens,
            c.num_kv_heads,
            c.head_dim,
            c.block_size,
            c.key_stride,
            c.value_stride,
            a.kv_cache.cache_stride() as u64,
            a.stream,
        )
    }

    fn write_dense_scaled(&self, a: &Self::Write<'_>) -> Result<()> {
        let c = &a.common;
        self.write_kv_cache_fp8(
            a.gpu,
            c.key,
            c.value,
            a.kv_cache,
            c.slot_mapping,
            c.num_tokens,
            c.num_kv_heads,
            c.head_dim,
            c.block_size,
            c.key_stride,
            c.value_stride,
            a.stream,
            a.ctx.graph_capture,
        )
    }

    fn write_block_scaled(&self, a: &Self::Write<'_>) -> Result<()> {
        self.rotate_before_write(a)?;
        let (k, v) = self.kv_sides(a.kv_cache)?;
        kv_abi::launch_fused_write(
            a.gpu,
            self.reshape_cache_k,
            &a.common,
            k,
            v,
            Tails::Shared,
            a.stream,
        )
    }
}
