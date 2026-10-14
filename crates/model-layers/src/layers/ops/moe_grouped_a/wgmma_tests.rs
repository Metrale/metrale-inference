// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Host-side tests of the wgmma twin's launcher (`moe_grouped_a/wgmma.rs`) on the mock
//! backend: the shapes it admits, its shared memory, its grid and the launch it records.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::*;

/// 2026-10-14: The routed prefill's projections on the 512-expert model: gate/up and down.
const GATE_UP: (u32, u32) = (640, 2560);
const DOWN: (u32, u32) = (2560, 640);
const EXPERTS: u32 = 512;

/// 2026-10-14: A mock with one kernel publishing `n_tile` as its N tile.
fn mock(n_tile: u32) -> (MockGpuBackend, KernelHandle) {
    let gpu = MockGpuBackend::new();
    let kernel = gpu
        .kernel(
            "moe_w4a16_grouped_wgmma_hopper",
            "moe_w4a16_grouped_gemm_ptrtable_wgmma",
        )
        .unwrap();
    gpu.set_kernel_n_tile(kernel, n_tile);
    (gpu, kernel)
}

#[test]
fn shared_memory_is_the_fixed_layout_plus_the_scan() {
    assert_eq!(moe_w4a16_wgmma_smem(0), Some(203_780));
    assert_eq!(moe_w4a16_wgmma_smem(EXPERTS), Some(205_828));
    assert_eq!(moe_w4a16_wgmma_smem(u32::MAX), None);
    assert_eq!(moe_w4a16_wgmma_smem(u32::MAX / 4), None);
}

#[test]
fn the_model_projections_are_admitted() {
    let (gpu, kernel) = mock(128);
    for (n, k) in [GATE_UP, DOWN] {
        assert!(
            moe_w4a16_wgmma_admits(&gpu, kernel, n, k, EXPERTS),
            "N={n} K={k}"
        );
    }
}

#[test]
fn shapes_the_kernel_traps_on_are_not_admitted() {
    let (gpu, kernel) = mock(128);
    // 2026-10-14: K off the 64 step; N a multiple of the pipe's 64 but not of 128.
    assert!(!moe_w4a16_wgmma_admits(&gpu, kernel, 640, 2592, EXPERTS));
    assert!(!moe_w4a16_wgmma_admits(&gpu, kernel, 704, 2560, EXPERTS));
    assert!(!moe_w4a16_wgmma_admits(
        &gpu,
        KernelHandle(0),
        640,
        2560,
        EXPERTS
    ));
    // 2026-10-14: The scan fills the opt-in maximum exactly at 7,167 experts.
    assert!(moe_w4a16_wgmma_admits(&gpu, kernel, 640, 2560, 7_167));
    assert!(!moe_w4a16_wgmma_admits(&gpu, kernel, 640, 2560, 7_168));
}

#[test]
fn a_kernel_publishing_another_n_tile_is_not_admitted() {
    let (gpu, kernel) = mock(64);
    assert!(!moe_w4a16_wgmma_admits(&gpu, kernel, 640, 2560, EXPERTS));
    let gpu = MockGpuBackend::new();
    assert!(!moe_w4a16_wgmma_admits(
        &gpu,
        KernelHandle(7),
        640,
        2560,
        EXPERTS
    ));
}

#[test]
fn the_grid_is_one_cta_per_sm_at_most_and_never_more_than_the_units() {
    // 2026-10-14: 4096 tokens x top-10: 320 full tiles + up to 512 partial, x 5 N tiles.
    assert_eq!(moe_w4a16_wgmma_ctas(132, 5, EXPERTS, 40_960), 132);
    // 2026-10-14: One row: one tile (counted twice by the bound) x 5 N tiles.
    assert_eq!(moe_w4a16_wgmma_ctas(132, 5, EXPERTS, 1), 10);
    assert_eq!(moe_w4a16_wgmma_ctas(132, 20, EXPERTS, 10), 132);
    assert_eq!(moe_w4a16_wgmma_ctas(132, 5, EXPERTS, 0), 0);
    assert_eq!(moe_w4a16_wgmma_ctas(132, u32::MAX, EXPERTS, u32::MAX), 132);
}

fn launch(gpu: &MockGpuBackend, kernel: KernelHandle, nk: (u32, u32), rows: u32) -> Result<()> {
    let p = |i: u64| DevicePtr(0x1000 * i);
    moe_w4a16_grouped_gemm_ptrtable_wgmma(
        gpu,
        kernel,
        p(1),
        p(2),
        p(3),
        p(4),
        p(5),
        p(6),
        p(7),
        EXPERTS,
        nk.0,
        nk.1,
        rows,
        0,
    )
}

#[test]
fn the_launch_carries_the_pipe_arguments_and_the_scan_memory() {
    let (gpu, kernel) = mock(128);
    launch(&gpu, kernel, GATE_UP, 40_960).unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    let l = &launches[0];
    // 2026-10-14: The mock's 48 SMs bound the grid.
    assert_eq!(l.grid, [48, 1, 1]);
    assert_eq!(l.block, [256, 1, 1]);
    assert_eq!(l.shared_mem, 205_828);
    let mut want: Vec<MockArg> = (1..=7)
        .map(|i| MockArg::Buffer(DevicePtr(0x1000 * i)))
        .collect();
    want.extend([EXPERTS, 640, 2560].map(|v| MockArg::Bytes(v.to_le_bytes().to_vec())));
    assert_eq!(l.args, want);
}

#[test]
fn zero_rows_launch_nothing_and_a_rejected_shape_is_an_error() {
    let (gpu, kernel) = mock(128);
    launch(&gpu, kernel, DOWN, 0).unwrap();
    assert!(launch(&gpu, kernel, (704, 2560), 40_960).is_err());
    assert!(gpu.launches_snapshot().is_empty());
}

#[test]
fn the_minimum_is_required_and_must_be_a_positive_row_count() {
    let unset = moe_w4a16_wgmma_min_rows(None).unwrap_err().to_string();
    assert!(
        unset.contains("METRALE_MOE_PREFILL_WGMMA_MIN_ROWS"),
        "{unset}"
    );
    for bad in ["", "0", "-5", "abc", "4096.5", "4294967296"] {
        let err = moe_w4a16_wgmma_min_rows(Some(bad)).unwrap_err().to_string();
        assert!(
            err.contains("METRALE_MOE_PREFILL_WGMMA_MIN_ROWS"),
            "{bad:?}: {err}"
        );
    }
    assert_eq!(moe_w4a16_wgmma_min_rows(Some("40960")).unwrap(), 40960);
    assert_eq!(moe_w4a16_wgmma_min_rows(Some("1")).unwrap(), 1);
}

#[test]
fn a_call_routes_to_the_twin_from_the_minimum_up() {
    assert!(!moe_w4a16_wgmma_routes(40959, 40960));
    assert!(moe_w4a16_wgmma_routes(40960, 40960));
    assert!(moe_w4a16_wgmma_routes(u32::MAX, 40960));
    assert!(!moe_w4a16_wgmma_routes(0, 1));
}
