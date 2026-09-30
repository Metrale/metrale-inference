// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The tiled launchers size grid.x from the kernel's published N tile.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::{
    moe_w4a16_grouped_gemm_ptrtable, moe_w4a16_grouped_gemm_ptrtable_n128, w4a16_gemm_n128,
};
use crate::weight_map::QuantizedWeight;

const NARROW: KernelHandle = KernelHandle(0x64);
const WIDE: KernelHandle = KernelHandle(0x128);
/// 2026-09-29: Nemotron-3-Nano's routed-expert intermediate size: 29 tiles of 64, 14.5 of
/// 128, so a 128-wide grid over a 64-wide kernel writes 960 of its 1856 columns.
const N: u32 = 1856;

fn gpu() -> MockGpuBackend {
    let gpu = MockGpuBackend::new();
    gpu.set_kernel_n_tile(NARROW, 64);
    gpu.set_kernel_n_tile(WIDE, 128);
    gpu
}

fn grouped(gpu: &MockGpuBackend, kernel: KernelHandle) -> anyhow::Result<u32> {
    let p = DevicePtr::NULL;
    moe_w4a16_grouped_gemm_ptrtable_n128(gpu, kernel, p, p, p, p, p, p, p, 128, N, 2688, 1, 0)?;
    let l = gpu.launches_snapshot();
    Ok(l.last().expect("one launch").grid[0])
}

/// 2026-09-29: The grid covers N in the kernel's own tile, whichever the launcher's name
/// says: the 64-wide common kernel gets 29 blocks, the 128-wide one 15.
#[test]
fn the_grouped_grid_covers_n_in_the_kernels_published_tile() {
    let gpu = gpu();
    assert_eq!(grouped(&gpu, NARROW).unwrap(), 29);
    assert_eq!(grouped(&gpu, WIDE).unwrap(), 15);
    let p = DevicePtr::NULL;
    moe_w4a16_grouped_gemm_ptrtable(&gpu, NARROW, p, p, p, p, p, p, p, 128, N, 2688, 1, 0).unwrap();
    assert_eq!(gpu.launches_snapshot().last().unwrap().grid[0], 29);
}

#[test]
fn the_dense_transposed_grid_covers_n_in_the_kernels_published_tile() {
    let gpu = gpu();
    let w = QuantizedWeight::null();
    for (kernel, blocks) in [(NARROW, 29), (WIDE, 15)] {
        w4a16_gemm_n128(
            &gpu,
            kernel,
            DevicePtr::NULL,
            &w,
            DevicePtr::NULL,
            64,
            N,
            2688,
            0,
        )
        .unwrap();
        assert_eq!(gpu.launches_snapshot().last().unwrap().grid[0], blocks);
    }
}

/// 2026-09-29: A kernel that publishes no tile is not launched with a guessed one.
#[test]
fn a_kernel_without_a_published_tile_is_refused() {
    let gpu = gpu();
    let before = gpu.launch_count();
    let err = grouped(&gpu, KernelHandle(0xDEAD)).unwrap_err().to_string();
    assert!(err.contains("publishes no"), "{err}");
    assert_eq!(gpu.launch_count(), before, "nothing may launch");
}
