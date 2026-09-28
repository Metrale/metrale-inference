// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: `AttnFa128Kernels`: the bit-identical 128-row twins of the BF16 prefill
//! flash-attention kernels (`kernels/gb10/common/attn_prefill_fa128.cu`):
//! `attn_prefill_fa128_paged` for `attn_prefill_paged_64` and `attn_prefill_fa128` for
//! `attn_prefill_64`. Each row gets the original's arithmetic in the original's order; the
//! twins read each K/V tile once per 128 query rows, keep P in registers and let every warp
//! run both QK^T and P V. On GB10 they are 1.5-2.3x faster (microtest
//! `attn_prefill_fa128_microtest`: 0 differing values against both originals).
//!
//! Owner: model-layers ops.
//! Invariants: a launcher returns Ok(false), launching nothing, whenever the twin does not
//! apply (handle 0, head_dim != 256, a sliding window, no causal mask, or a head count that
//! is not a multiple of the KV heads); the caller then runs the original.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

/// 2026-09-27: Query rows per CTA, and the 96 KiB of dynamic shared memory (Q 128 rows, one
/// K and one V tile of 32 rows, 512 bytes per row).
const ROWS: u32 = 128;
const SMEM: u32 = (128 + 32 + 32) * 512;

/// 2026-09-27: The two twins' handles; 0 when the target lacks the module or
/// `METRALE_NO_ATTN_FA128` is present (the originals then run, for A/B measurement).
#[derive(Clone, Copy, Debug)]
pub struct AttnFa128Kernels {
    paged: KernelHandle,
    contiguous: KernelHandle,
}

impl AttnFa128Kernels {
    #[track_caller]
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        const MODULE: &str = "attn_prefill_fa128";
        if std::env::var_os("METRALE_NO_ATTN_FA128").is_some() {
            return Self {
                paged: KernelHandle(0),
                contiguous: KernelHandle(0),
            };
        }
        Self {
            paged: crate::layers::try_target_kernel(gpu, MODULE, "attn_prefill_fa128_paged"),
            contiguous: crate::layers::try_target_kernel(gpu, MODULE, "attn_prefill_fa128"),
        }
    }

    fn applies(
        kernel: KernelHandle,
        head_dim: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        causal: bool,
        sliding_window: u32,
    ) -> bool {
        kernel.0 != 0
            && head_dim == 256
            && causal
            && sliding_window == 0
            && num_kv_heads > 0
            && num_q_heads.is_multiple_of(num_kv_heads)
    }

    /// 2026-09-27: In place of `ops::prefill_attention_paged_64` (causal, BF16 paged cache).
    #[allow(clippy::too_many_arguments)]
    pub fn paged(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        output: DevicePtr,
        block_table: DevicePtr,
        q_len: u32,
        kv_len: u32,
        q_offset: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        cache_block_size: u32,
        sliding_window: u32,
        inv_sqrt_d: f32,
        stream: u64,
    ) -> Result<bool> {
        if q_len == 0
            || !Self::applies(
                self.paged,
                head_dim,
                num_q_heads,
                num_kv_heads,
                true,
                sliding_window,
            )
        {
            return Ok(false);
        }
        KernelLaunch::new(gpu, self.paged)
            .grid([num_q_heads, div_ceil(q_len, ROWS), 1])
            .block([256, 1, 1])
            .shared_mem(SMEM)
            .arg_ptr(q)
            .arg_ptr(k_cache)
            .arg_ptr(v_cache)
            .arg_ptr(output)
            .arg_ptr(block_table)
            .arg_u32(q_len)
            .arg_u32(kv_len)
            .arg_u32(q_offset)
            .arg_u32(num_q_heads)
            .arg_u32(num_kv_heads)
            .arg_u32(cache_block_size)
            .arg_f32(inv_sqrt_d)
            .launch(stream)?;
        Ok(true)
    }

    /// 2026-09-27: In place of `ops::prefill_attention_64` (contiguous BF16 Q/K/V).
    #[allow(clippy::too_many_arguments)]
    pub fn contiguous(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
        output: DevicePtr,
        seq_len: u32,
        batch: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        stream: u64,
    ) -> Result<bool> {
        if seq_len == 0
            || batch == 0
            || !Self::applies(
                self.contiguous,
                head_dim,
                num_q_heads,
                num_kv_heads,
                causal,
                sliding_window,
            )
        {
            return Ok(false);
        }
        KernelLaunch::new(gpu, self.contiguous)
            .grid([num_q_heads, div_ceil(seq_len, ROWS), batch])
            .block([256, 1, 1])
            .shared_mem(SMEM)
            .arg_ptr(q)
            .arg_ptr(k)
            .arg_ptr(v)
            .arg_ptr(output)
            .arg_u32(seq_len)
            .arg_u32(num_q_heads)
            .arg_u32(num_kv_heads)
            .arg_f32(inv_sqrt_d)
            .launch(stream)?;
        Ok(true)
    }
}
