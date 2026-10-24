// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of `m16_tc_kernel`, the pure half of the m16 tier's tile lever.
//!
//! Owner: model-layers (dense FFN).
//! Invariants: none beyond the types.
//!
//! Which arm a row count takes is `dense_ffn_m16_tc_tests.rs`; how the
//! variables fold into the resolved bools is `ops/target_defaults_m16_tests.rs`;
//! how `METRALE_FFN_M16_TC_NTILE` picks the tile is `dense_ffn_levers_tests.rs`.

use super::m16_tc_kernel;
use crate::layers::ops::{W8A16_GEMM_M16_N_TILE, W8A16_GEMM_M16_N_TILE_WIDE};
use metrale_gpu_runtime::gpu::KernelHandle;

/// 2026-09-25: Distinct handles, so the returned handle tells the two tiles
/// apart.
const M16TC_K: u64 = 0x167C;
const M16TC_N64_K: u64 = 0x1640;

/// 2026-09-25: The tile picks the entry point, and without the wide handle it
/// falls back to the 32-wide kernel instead of launching a zero handle.
#[test]
fn the_wide_tile_falls_back_when_its_entry_point_is_absent() {
    let (_, kernel, tile) = m16_tc_kernel(
        W8A16_GEMM_M16_N_TILE_WIDE,
        KernelHandle(M16TC_K),
        KernelHandle(M16TC_N64_K),
    );
    assert_eq!(kernel.0, M16TC_N64_K);
    assert_eq!(tile, W8A16_GEMM_M16_N_TILE_WIDE);

    let (_, kernel, tile) = m16_tc_kernel(
        W8A16_GEMM_M16_N_TILE_WIDE,
        KernelHandle(M16TC_K),
        KernelHandle(0),
    );
    assert_eq!(
        kernel.0, M16TC_K,
        "no n64 entry point => the 32-wide kernel"
    );
    assert_eq!(tile, W8A16_GEMM_M16_N_TILE);

    let (_, kernel, tile) = m16_tc_kernel(
        W8A16_GEMM_M16_N_TILE,
        KernelHandle(M16TC_K),
        KernelHandle(M16TC_N64_K),
    );
    assert_eq!(
        kernel.0, M16TC_K,
        "the default tile never reaches the wide arm"
    );
    assert_eq!(tile, W8A16_GEMM_M16_N_TILE);
}
