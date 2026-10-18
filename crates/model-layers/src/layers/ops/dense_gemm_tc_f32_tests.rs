// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The FP32-output projection's plan (which path, which grid, which shapes it
//! refuses) and its launches (entry, geometry, argument order) on the recording mock backend.
//! The mock stands in for the device only: plan and launch run their production code, and the
//! mock records what reached `launch_typed`. The numerics are the GPU test's job
//! (dense_gemm_tc_f32_gpu_tests.rs).
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

use super::*;

/// 2026-10-17: `(N, K)` of the GLM-5.3-Flash cuBLASLt FP32-output sites: router logits (288
/// experts, hidden 4096), DSA wq_b (4096 = 32 index heads x 128, q_lora_rank 1536) and DSA
/// weights_proj (32 index heads, hidden 4096).
const GLM_SITES: [(u32, u32); 3] = [(288, 4096), (4096, 1536), (32, 4096)];

fn gemv(kind: DtcKind, n: u32) -> TcF32Path {
    TcF32Path::Gemv {
        kind,
        grid_x: n.div_ceil(16),
    }
}

fn tile(m: u32, n: u32) -> TcF32Path {
    TcF32Path::Tile {
        grid_x: n.div_ceil(64),
        grid_y: m.div_ceil(16),
    }
}

#[test]
fn m_edges_pick_the_narrowest_covering_gemv_tier_then_the_tile() {
    for (n, k) in GLM_SITES {
        for (m, want) in [
            (1, gemv(DtcKind::M8, n)),
            (8, gemv(DtcKind::M8, n)),
            (9, gemv(DtcKind::M16, n)),
            (16, gemv(DtcKind::M16, n)),
            (17, gemv(DtcKind::M32, n)),
            (32, gemv(DtcKind::M32, n)),
            (33, tile(33, n)),
            (300, tile(300, n)),
        ] {
            let p = TcF32Plan::new(m, n, k).unwrap();
            assert_eq!(p.path(), want, "M={m} N={n} K={k}");
            assert_eq!(p.shape(), (m, n, k));
        }
    }
}

#[test]
fn a_k_off_the_gemv_step_takes_the_tile_at_any_m() {
    for k in [1, 16, 63, 100, 1500] {
        for m in [1, 16, 32, 33] {
            assert_eq!(
                TcF32Plan::new(m, 288, k).unwrap().path(),
                tile(m, 288),
                "M={m} K={k}"
            );
        }
    }
}

#[test]
fn grids_cover_a_ragged_n_with_one_partial_tile() {
    for n in [1, 15, 17, 63, 65, 100] {
        let TcF32Path::Gemv { grid_x, .. } = TcF32Plan::new(4, n, 64).unwrap().path() else {
            panic!("M=4 K=64 is a GEMV shape");
        };
        assert!(
            grid_x * 16 >= n && (grid_x - 1) * 16 < n,
            "N={n} GEMV grid {grid_x}"
        );
        let TcF32Path::Tile { grid_x, grid_y } = TcF32Plan::new(40, n, 100).unwrap().path() else {
            panic!("K=100 is a tile shape");
        };
        assert!(
            grid_x * 64 >= n && (grid_x - 1) * 64 < n,
            "N={n} tile grid {grid_x}"
        );
        assert_eq!(grid_y, 3, "M=40 is two full row tiles and a partial one");
    }
}

#[test]
fn refuses_empty_and_unindexable_shapes() {
    for (m, n, k) in [(0, 288, 4096), (17, 0, 4096), (17, 288, 0)] {
        assert!(TcF32Plan::new(m, n, k).is_err(), "M={m} N={n} K={k}");
    }
    // 2026-10-17: Tile grid-y past 65535 row tiles.
    assert!(TcF32Plan::new(65_535 * 16 + 1, 32, 64).is_err());
    assert!(TcF32Plan::new(65_535 * 16, 32, 64).is_ok());
    // 2026-10-17: M * K and M * N past u32 on the tile's 32-bit row offsets.
    assert!(TcF32Plan::new(1 << 16, 32, 1 << 16).is_err());
    assert!(TcF32Plan::new(1 << 16, 1 << 16, 32).is_err());
    assert!(TcF32Plan::new(1 << 16, 32, (1 << 16) - 1).is_ok());
    // 2026-10-17: The GEMV indexes in 64 bits, so a long K at small M plans.
    assert_eq!(
        TcF32Plan::new(32, 32, 1 << 27).unwrap().path(),
        gemv(DtcKind::M32, 32)
    );
}

#[test]
fn resolve_looks_up_the_four_entries_of_one_module() {
    let gpu = MockGpuBackend::new();
    TcF32Kernels::resolve(&gpu).unwrap();
    let want: Vec<(String, String)> = [
        "dense_gemm_tc_f32out",
        "dense_gemv_bf16_tc8_f32out",
        "dense_gemv_bf16_tc16_f32out",
        "dense_gemv_bf16_tc32_f32out",
    ]
    .iter()
    .map(|e| (MODULE.to_owned(), (*e).to_owned()))
    .collect();
    assert_eq!(gpu.kernel_lookups_snapshot(), want);
}

#[test]
fn resolve_fails_naming_any_missing_entry() {
    for entry in ["dense_gemm_tc_f32out", "dense_gemv_bf16_tc16_f32out"] {
        let gpu = MockGpuBackend::new();
        gpu.deny_kernel(MODULE, entry);
        let err = format!("{:#}", TcF32Kernels::resolve(&gpu).unwrap_err());
        assert!(err.contains(entry), "{err}");
    }
}

fn u32_arg(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

#[test]
fn launches_carry_the_plans_geometry_and_the_cublas_argument_order() {
    let gpu = MockGpuBackend::new();
    let kernels = TcF32Kernels::resolve(&gpu).unwrap();
    let (act, weight, out) = (DevicePtr(0x1000), DevicePtr(0x2000), DevicePtr(0x3000));
    let ptrs = [
        MockArg::Buffer(act),
        MockArg::Buffer(weight),
        MockArg::Buffer(out),
    ];
    dense_proj_f32_out(&gpu, &kernels, act, weight, out, 17, 288, 4096, 7).unwrap();
    dense_proj_f32_out(&gpu, &kernels, act, weight, out, 33, 4096, 1536, 9).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 2);

    assert_eq!(
        (l[0].grid, l[0].block, l[0].stream),
        ([18, 1, 1], [256, 1, 1], 7)
    );
    let mut want = ptrs.to_vec();
    want.extend([17, 288, 4096, 288].map(u32_arg));
    assert_eq!(l[0].args, want, "GEMV: A, W, C, M, N, K, out_stride = N");

    assert_eq!(
        (l[1].grid, l[1].block, l[1].stream),
        ([64, 3, 1], [128, 1, 1], 9)
    );
    let mut want = ptrs.to_vec();
    want.extend([33, 4096, 1536].map(u32_arg));
    assert_eq!(l[1].args, want, "tile: A, B, C, M, N, K");
}

#[test]
fn a_gemv_plan_refuses_a_misaligned_base_and_launches_nothing() {
    let gpu = MockGpuBackend::new();
    let kernels = TcF32Kernels::resolve(&gpu).unwrap();
    let plan = TcF32Plan::new(16, 32, 4096).unwrap();
    let ok = DevicePtr(0x1000);
    for (act, weight) in [(DevicePtr(0x1008), ok), (ok, DevicePtr(0x1002))] {
        assert!(launch(&gpu, &kernels, &plan, act, weight, ok, 0).is_err());
    }
    assert!(gpu.launches_snapshot().is_empty());
    // 2026-10-17: The tile reads BF16 scalars, so the same bases launch there.
    let tile_plan = TcF32Plan::new(33, 32, 4096).unwrap();
    launch(&gpu, &kernels, &tile_plan, DevicePtr(0x1002), ok, ok, 0).unwrap();
    assert_eq!(gpu.launches_snapshot().len(), 1);
}
