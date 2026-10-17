// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_W4A16_TC_GEMM_MIN_ROWS=<r>`: under `--dense-quantization w4a16`, a
//! launch of at least `r` rows (a prefill sub-chunk) runs `ops::w4a16_tc_gemm` — the row-tile
//! kernel's body over a 2-D grid, reading the NVFP4 weight directly — instead of the per-64-row
//! tile launches or the dequantize + cuBLASLt arm of `METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS`.
//!
//! Numerics: BIT-IDENTICAL to the row tiles (`tc_rows.cuh`: a row's bits depend on neither the
//! row count, the tile height nor the warps per block), so prefill rows carry the bytes decode's
//! W4A16 tier writes, and the lever is compatible with `METRALE_GLM_BATCH_INVARIANT`. Against the
//! dequantize + cuBLASLt arm the bits change (that arm's FP32 order is cuBLASLt's); set one of the
//! two levers, this one wins when both take a launch.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Unset or `0` launches exactly what the tier launched before this lever.
//! - A launch below `r` rows is unchanged.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::QuantizedWeight;

/// 2026-10-12: The narrowest value the lever takes: above the batched decode and verify groups
/// the row tiles serve (the GEMM is one launch of the same bits; below 17 rows it only changes
/// the grid).
pub const MIN_ROWS_FLOOR: usize = 17;

/// 2026-10-12: Whether a launch of `m` rows takes the GEMM under the lever value `min`.
pub fn takes(min: Option<usize>, m: usize) -> bool {
    min.is_some_and(|r| m >= r)
}

/// 2026-10-12: `c[m, n] = a[m, k] @ w^T` in one `w4a16_tc_gemm` launch (contiguous rows).
#[allow(clippy::too_many_arguments)]
pub fn proj(
    gpu: &dyn GpuBackend,
    w: &QuantizedWeight,
    a: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ops::w4a16_tc_gemm(
        gpu, a, w, c, m as u32, n as u32, k as u32, k as u32, n as u32, stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-12: Unset is off; a launch takes the GEMM from the lever value up.
    #[test]
    fn only_launches_from_the_lever_value_take_the_gemm() {
        assert!(!takes(None, 8208));
        assert!(!takes(Some(256), 255));
        assert!(takes(Some(256), 1036));
    }

    /// 2026-10-12: The GEMM's shape contract is the row tiles' for any row count: K a multiple
    /// of 256, the pitches covering K and N.
    #[test]
    fn the_gemm_takes_any_row_count_of_the_row_tiles_shapes() {
        assert!(ops::w4a16_tc_gemm_shape_ok(1036, 2816, 4096, 4096, 2816));
        assert!(ops::w4a16_tc_gemm_shape_ok(8208, 4096, 768, 768, 4096));
        assert!(!ops::w4a16_tc_gemm_shape_ok(0, 2816, 4096, 4096, 2816));
        assert!(!ops::w4a16_tc_gemm_shape_ok(1036, 2816, 4000, 4000, 2816));
        assert!(!ops::w4a16_tc_gemm_shape_ok(1036, 2816, 4096, 4096, 2000));
    }
}
