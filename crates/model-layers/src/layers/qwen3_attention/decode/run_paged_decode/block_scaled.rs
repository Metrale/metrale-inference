// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The split-K route of a symmetric block-scaled cache whose codec has a split-K
//! kernel pair (the codec kernel table names one for NVFP4 only): split-K plus reduce when the
//! split count pays, else the non-split fused launch through `paged_decode_k`.
//!
//! Owner: model-layers attention decode.
//! Invariants: the split count comes from `splitk_dispatch::num_splits`, which does not move
//! with the co-batched count under `auto` or a pinned count, and under `legacy` only when
//! `num_seqs` exceeds `max_decode_seqs`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::KernelHandle;

use super::super::super::Qwen3AttentionLayer;
use super::super::kv_exec::DecodeArgs;
use super::super::splitk_dispatch;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-10-18: Decode over a symmetric block-scaled cache with the split-K pair
    /// `(splitk_k, reduce_k)` loaded.
    pub(in super::super) fn run_paged_decode_nvfp4(
        &self,
        a: &DecodeArgs<'_>,
        (splitk_k, reduce_k): (KernelHandle, KernelHandle),
    ) -> Result<()> {
        let c = &a.common;
        let num_splits =
            splitk_dispatch::num_splits(c.num_q_heads, c.head_dim, c.num_seqs, a.max_decode_seqs);
        if !splitk_dispatch::splits_are_worth_it(num_splits) {
            splitk_dispatch::log_decode_route(
                splitk_dispatch::RouteArm::Nvfp4,
                splitk_dispatch::ROUTE_NONSPLIT_NVFP4,
                num_splits,
            );
            return self.decode_block_scaled_with(a, self.paged_decode_k);
        }
        splitk_dispatch::log_decode_route(
            splitk_dispatch::RouteArm::Nvfp4,
            splitk_dispatch::ROUTE_SPLITK_NVFP4,
            num_splits,
        );
        let layer = self.attn_layer_idx;
        ops::paged_decode_attn_splitk_nvfp4(
            a.gpu,
            splitk_k,
            c.q,
            a.kv_cache.k_pool_ptr(layer),
            a.kv_cache.v_pool_ptr(layer),
            a.workspace,
            c.block_tables,
            c.seq_lens,
            c.max_blocks_per_seq,
            c.num_q_heads,
            c.num_kv_heads,
            c.head_dim,
            c.block_size,
            c.inv_sqrt_d,
            num_splits,
            c.q_stride,
            a.kv_cache.block_stride_bytes_for_layer(layer) as u64,
            a.kv_cache.data_bytes_of(self.kv_pair.k()) as u64,
            c.num_seqs,
            a.stream,
        )?;
        ops::paged_decode_attn_reduce_nvfp4(
            a.gpu,
            reduce_k,
            a.workspace,
            c.output,
            c.seq_lens,
            c.num_q_heads,
            c.head_dim,
            num_splits,
            c.num_seqs,
            a.stream,
        )
    }
}
