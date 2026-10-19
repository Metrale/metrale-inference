// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Paged decode attention for one layer. 2026-10-18: A symmetric cache goes to its
//! codec family's `KvExecutor` method (`kv_abi::decode`), a split pair to the fused pair launch;
//! the families' bodies are the V4-Flash MLA kernels, split-K or non-split block-scaled, and
//! split-K, GQA-packed or non-split BF16 and FP8.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - A GQA-packed kernel is launched only on a non-split path.
//! - No codec is named here; the pair's specs pick the route.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_kv_codec::spec;

use super::super::Qwen3AttentionLayer;
use super::kv_exec::DecodeArgs;
use crate::kv_abi::{self, DecodeCommon};
use crate::layers::ops;

mod bf16_fp8;
mod block_scaled;

impl Qwen3AttentionLayer {
    /// 2026-10-18: The BF16/FP8 decode kernel for `head_dim`: the pair's head_dim-512 build for
    /// heads wider than 256 when it is loaded, else `paged_decode_k`.
    pub(super) fn decode_kernel_for(&self, head_dim: u32) -> KernelHandle {
        if head_dim > 256 {
            self.paged_decode_512_k.unwrap_or(self.paged_decode_k)
        } else {
            self.paged_decode_k
        }
    }

    /// 2026-10-18: The turbo decode kernel for `head_dim`. `paged_decode_k` is a head_dim-256
    /// build, so a wider head needs the pair's own head_dim-512 build; without one the pair is
    /// refused (`KvPairUnsupported`). Turbo3 and Turbo2 used to run turbo4's 512 build here, and
    /// every turbo dtype the 256 build when the 512 one was absent.
    pub(super) fn rotated_decode_kernel_for(&self, head_dim: u32) -> Result<KernelHandle> {
        if head_dim <= 256 {
            return Ok(self.paged_decode_k);
        }
        self.paged_decode_512_k.ok_or_else(|| {
            metrale_error::error!(
                metrale_error::ErrorCode::KvPairUnsupported,
                self.kv_pair.k().name(),
                self.kv_pair.v().name(),
                head_dim
            )
            .into()
        })
    }

    pub(in super::super) fn run_paged_decode(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        kv_cache: &PagedKvCache,
        output: DevicePtr,
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        max_blocks_per_seq: u32,
        num_seqs: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        q_stride: u32,
        workspace: DevicePtr,
        // 2026-09-25: `ModelLevers::max_decode_seqs`, which the `legacy` split count reads through
        // `split_ref_seqs`.
        max_decode_seqs: u32,
        stream: u64,
    ) -> Result<()> {
        let a = DecodeArgs {
            gpu,
            kv_cache,
            common: DecodeCommon {
                q,
                output,
                block_tables: block_table,
                seq_lens,
                max_blocks_per_seq,
                num_seqs,
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_size,
                inv_sqrt_d,
                q_stride,
                sliding_window: self.sliding_window.unwrap_or(0),
            },
            workspace,
            max_decode_seqs,
            stream,
        };
        if self.kv_pair.is_asymmetric() {
            return self.decode_pair(&a);
        }
        kv_abi::decode(spec(self.kv_pair.k()).abi, self, &a)
    }

    /// 2026-09-25: V4-Flash MLA decode over a block-scaled (NVFP4) cache.
    pub(super) fn run_mla_decode_nvfp4(&self, a: &DecodeArgs<'_>) -> Result<()> {
        let mla = self.mla.as_ref().expect("V4-Flash layers carry MLA dims");
        let c = &a.common;
        let kv_cache_dim = (mla.kv_lora_rank + mla.rope) as u32;
        tracing::info!(
            "V4-Flash MLA decode (NVFP4): q_head_dim={}, kv_cache_dim={}",
            c.head_dim,
            kv_cache_dim
        );
        let layer = self.attn_layer_idx;
        ops::mla_paged_decode_nvfp4(
            a.gpu,
            self.mla_paged_decode_k,
            c.q,
            a.kv_cache.k_pool_ptr(layer),
            a.kv_cache.v_pool_ptr(layer),
            c.output,
            c.block_tables,
            c.seq_lens,
            c.max_blocks_per_seq,
            c.num_q_heads,
            c.num_kv_heads,
            c.head_dim,
            kv_cache_dim,
            c.block_size,
            c.inv_sqrt_d,
            a.kv_cache.block_stride_bytes_for_layer(layer) as u64,
            a.kv_cache.data_bytes_of(self.kv_pair.k()) as u64,
            c.num_seqs,
            a.stream,
        )
    }

    /// 2026-09-25: V4-Flash MLA decode over a scaled (FP8) cache.
    pub(super) fn run_mla_decode_fp8(&self, a: &DecodeArgs<'_>) -> Result<()> {
        let mla = self.mla.as_ref().expect("V4-Flash layers carry MLA dims");
        let c = &a.common;
        let kv_cache_dim = (mla.kv_lora_rank + mla.rope) as u32;
        tracing::info!(
            "V4-Flash MLA decode (FP8): q_head_dim={}, kv_cache_dim={}",
            c.head_dim,
            kv_cache_dim
        );
        let (k_scale, v_scale) = self.effective_fp8_scales();
        // 2026-09-25: `inv_sqrt_d` is the caller's `effective_attn_scale(head_dim)`, with the
        // query head dim `nope + rope` (448 + 64 = 512 in
        // `kernels/gb10/deepseek-v4-flash/MODEL.toml`), not the 576-wide cache row. The
        // compressed arm is a flat FP8 pool and a block count: a layer without a compressor
        // passes a null pool and 0 blocks; a compressor layer passes blocks `[0,
        // v4_comp_pool_filled)`, written by prefill and by the decode-time append in
        // `attention_forward_v4`.
        let (comp_pool, comp_blocks) = match mla.compressor {
            Some(comp) => (
                comp.pool,
                self.v4_comp_pool_filled
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            None => (DevicePtr::NULL, 0u32),
        };
        let layer = self.attn_layer_idx;
        ops::mla_paged_decode_fp8(
            a.gpu,
            self.mla_paged_decode_fp8_k,
            c.q,
            a.kv_cache.k_pool_ptr(layer),
            a.kv_cache.v_pool_ptr(layer),
            c.output,
            c.block_tables,
            c.seq_lens,
            c.max_blocks_per_seq,
            c.num_q_heads,
            c.num_kv_heads,
            c.head_dim,
            kv_cache_dim,
            c.block_size,
            c.inv_sqrt_d,
            k_scale,
            v_scale,
            a.kv_cache.block_stride_bytes_for_layer(layer) as u64,
            c.num_seqs,
            // 2026-09-25: The V4 sliding window; prefill's `V4_WINDOW` (`cache_skip_v4.rs`) is
            // the same 128.
            128,
            mla.attn_sink,
            comp_pool,
            comp_blocks,
            a.stream,
        )
    }
}
