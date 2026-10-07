// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Disjoint small/large NVFP4 expert row tiles with BF16 MMA.
//! The caller owns valid offsets, pointer tables and complete M64 grid coverage.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-07: Handles resolved once per layer; no lookup or host copy per launch.
#[derive(Clone, Copy)]
pub struct Nvfp4SmallRowKernels {
    small: KernelHandle,
    large: KernelHandle,
}

impl Nvfp4SmallRowKernels {
    /// 2026-10-07: Missing kernels, wrong tiles or FP8 activation casts refuse the pair.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let pair = Self {
            small: gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_small16_n32")?,
            large: gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_large64")?,
        };
        for (kernel, expected) in [(pair.small, 32), (pair.large, 64)] {
            ensure!(
                gpu.kernel_n_tile(kernel)? == expected,
                "small-row pair tile mismatch"
            );
            ensure!(
                !gpu.kernel_casts_a_to_e4m3(kernel),
                "small-row pair requires BF16 activations"
            );
        }
        Ok(pair)
    }

    /// 2026-10-07: Experts with 1..16 rows use M16; larger experts use M64.
    /// `max_m_tiles` must cover ceil(maximum expert rows/64), never an average cap.
    /// Packed weights are [N,K/2], E4M3 scales [N,K/16], scale2 is FP32 per expert.
    /// Offsets are monotone i32 and all row/index/weight allocations must be valid.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        packed: DevicePtr,
        scales: DevicePtr,
        scale2: DevicePtr,
        c: DevicePtr,
        offsets: DevicePtr,
        sorted: DevicePtr,
        experts: u32,
        n: u32,
        k: u32,
        max_m_tiles: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            experts > 0 && n > 0 && k > 0 && k.is_multiple_of(16),
            "invalid small-row dimensions"
        );
        ensure!(
            max_m_tiles > 0,
            "small-row fallback requires a complete M64 grid"
        );
        for (ptr, alignment) in [
            (a, 2),
            (packed, 8),
            (scales, 8),
            (scale2, 4),
            (c, 2),
            (offsets, 4),
        ] {
            ensure!(
                ptr.0 != 0 && ptr.0.is_multiple_of(alignment),
                "invalid small-row pointer alignment"
            );
        }
        ensure!(
            sorted.0.is_multiple_of(4),
            "invalid sorted-row pointer alignment"
        );
        for (kernel, rows, threads) in [(self.small, 1, 64), (self.large, max_m_tiles, 128)] {
            KernelLaunch::new(gpu, kernel)
                .grid([super::n_tile_blocks(gpu, kernel, n)?, rows, experts])
                .block([threads, 1, 1])
                .arg_ptr(a)
                .arg_ptr(packed)
                .arg_ptr(scales)
                .arg_ptr(scale2)
                .arg_ptr(c)
                .arg_ptr(offsets)
                .arg_ptr(sorted)
                .arg_u32(experts)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)?;
        }
        Ok(())
    }
}
