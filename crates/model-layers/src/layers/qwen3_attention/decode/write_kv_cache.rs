// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The KV-cache write. 2026-10-18: A symmetric cache goes to its codec family's
//! `KvExecutor` method (`kv_abi::write`), a split pair to the fused pair write; each WHT-rotated
//! side is rotated before the write unless the weights were pre-rotated at load (`kv_exec`).
//!
//! Owner: model-layers attention decode.
//! Invariants: no codec is named here; the pair's specs pick the route.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_kv_codec::spec;

use super::super::Qwen3AttentionLayer;
use super::kv_exec::WriteArgs;
use crate::kv_abi::{self, WriteCommon};
use crate::layer::ForwardContext;

impl Qwen3AttentionLayer {
    pub(in super::super) fn write_kv_cache(
        &self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        key_stride: u32,
        value_stride: u32,
        stream: u64,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let a = WriteArgs {
            gpu,
            kv_cache,
            common: WriteCommon {
                key: k,
                value: v,
                slot_mapping: slot,
                num_tokens,
                num_kv_heads,
                head_dim,
                block_size,
                key_stride,
                value_stride,
            },
            stream,
            ctx,
        };
        if self.kv_pair.is_asymmetric() {
            return self.write_pair(&a);
        }
        kv_abi::write(spec(self.kv_pair.k()).abi, self, &a)
    }
}
