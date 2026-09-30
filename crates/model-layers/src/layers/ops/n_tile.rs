// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Grid sizing from the N tile a kernel publishes.
//!
//! Owner: model-layers ops.
//! Invariants: a launcher whose grid covers N in tiles takes the tile from the resolved
//! kernel (`GpuBackend::kernel_n_tile`), never from a width written at the call site.
//! One entry name can carry different tiles in different target trees
//! (`moe_w4a16_grouped_gemm_ptrtable_t` is 64 wide in `gb10/common` and 128 wide in the
//! Qwen trees), and a grid sized for the wrong one leaves columns unwritten.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::div_ceil;

/// 2026-09-29: `ceil(n / t)` blocks, `t` being `kernel`'s published N tile. An error when
/// the kernel publishes none.
pub fn n_tile_blocks(gpu: &dyn GpuBackend, kernel: KernelHandle, n: u32) -> Result<u32> {
    Ok(div_ceil(n, gpu.kernel_n_tile(kernel)?))
}

#[cfg(all(test, feature = "cuda"))]
#[path = "n_tile_gpu_tests.rs"]
mod gpu_tests;
#[cfg(test)]
#[path = "n_tile_tests.rs"]
mod tests;
