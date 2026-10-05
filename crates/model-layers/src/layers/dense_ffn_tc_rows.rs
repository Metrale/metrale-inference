// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The dense FFN's row-tile W4A16 arm. An NVFP4 projection of `m` rows, `m` within the
//! target's `ffn_w4a16_tc_rows_max_m` (`kernels/<hw>/HARDWARE.toml [defaults]`), runs
//! `w4a16_tc_rows` on the row-major weight in calls of at most `W4A16_TC_ROWS_MAX_M` rows, in place
//! of the NVFP4 tile GEMMs of the `w4_gemm!` ladder (`dense_ffn_prefill_nvfp4.rs`).
//!
//! Why: the tile GEMMs launch one 128-thread CTA per 128 output columns with no split of K, so on
//! a class with many SMs a decode-width projection fills few of them; the row tile reads the same
//! weight bytes with a 64-column CTA and keeps BF16 activations (the tile GEMMs round them to
//! E4M3), which is the W4A16 the checkpoint's NVFP4 layers take on a class without FP4 MMA.
//!
//! 2026-10-05: Past that band, a prefill projection wider than the small-M arm runs the
//! BF16-activation tile `w4a16_gemm_t_m128_bf16(_v2)` when the target declares
//! `ffn_w4a16_bf16_tile` ([`bf16_tile_route`]), so the NVFP4 FFN keeps BF16 activations at every
//! width instead of the E4M3 rounding of `w4a16_gemm_t_m128(_v2)`.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - A row's output bits depend on neither `m` nor the call its row falls in (the kernel's own
//!   invariant, `kernels/gb10/common/tc_rows.cuh`), so the 64-row chunking never changes them.
//! - The arm applies `weight_scale_2` in the kernel's store, as the tile GEMMs it replaces do.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

/// 2026-10-05: Whether the row-tile arm serves an `m`-row projection of shape `[n, k]`: the
/// kernel module is present, `1 <= m <= max_m`, and every call's shape is inside the kernel's
/// contract (activations and output packed: `lda = k`, `ldc = n`). `max_m = 0` is off.
pub(crate) fn tc_rows_route(m: u32, n: u32, k: u32, max_m: u32, module_present: bool) -> bool {
    module_present
        && m >= 1
        && m <= max_m
        && ops::w4a16_tc_rows_shape_ok(m.min(ops::W4A16_TC_ROWS_MAX_M), n, k, k, n)
}

/// 2026-10-05: The widest prefill the `w4_gemm!` ladder sends to `w4a16_prefill_gemm`'s small-M
/// kernels (`dense_ffn_prefill_nvfp4.rs`).
pub(crate) const PREFILL_SMALL_M_MAX: u32 = 64;

/// 2026-10-05: Whether the wide BF16 tile serves an `m`-row prefill: the target declares it
/// (`on`) and `m` is past the small-M arm. Rows inside the row-tile band never reach it: that arm
/// comes first on the ladder.
pub(crate) fn bf16_tile_route(m: u32, on: bool) -> bool {
    on && m > PREFILL_SMALL_M_MAX
}

impl DenseFfnLayer {
    /// 2026-10-05: [`bf16_tile_route`] under this binary's resolved `ffn_w4a16_bf16_tile`.
    pub(super) fn bf16_tile_serves(&self, m: u32) -> bool {
        bf16_tile_route(
            m,
            ops::target_defaults::resolved().ffn_w4a16_bf16_tile.value,
        )
    }

    /// 2026-10-05: [`tc_rows_route`] under this binary's resolved `ffn_w4a16_tc_rows_max_m`.
    pub(super) fn tc_rows_serves(&self, ctx: &ForwardContext, m: u32, n: u32, k: u32) -> bool {
        tc_rows_route(
            m,
            n,
            k,
            ops::target_defaults::resolved()
                .ffn_w4a16_tc_rows_max_m
                .value,
            ctx.gpu.has_module(ops::W4A16_TC_ROWS_MODULE),
        )
    }

    /// 2026-10-05: `output [m, n] = input [m, k] x W^T` through `w4a16_tc_rows`, at most
    /// `W4A16_TC_ROWS_MAX_M` rows per call.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn w4a16_tc_rows_chunked(
        &self,
        ctx: &ForwardContext,
        weight: &QuantizedWeight,
        input: DevicePtr,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let mut row = 0u32;
        while row < m {
            let rows = (m - row).min(ops::W4A16_TC_ROWS_MAX_M);
            ops::w4a16_tc_rows(
                ctx.gpu,
                input.offset(row as usize * k as usize * 2),
                weight,
                output.offset(row as usize * n as usize * 2),
                rows,
                n,
                k,
                k,
                n,
                stream,
            )?;
            row += rows;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{PREFILL_SMALL_M_MAX, bf16_tile_route, tc_rows_route};

    #[test]
    fn the_bf16_tile_starts_past_the_small_m_arm_and_only_when_declared() {
        assert!(!bf16_tile_route(PREFILL_SMALL_M_MAX, true));
        assert!(bf16_tile_route(PREFILL_SMALL_M_MAX + 1, true));
        assert!(bf16_tile_route(2048, true));
        assert!(!bf16_tile_route(2048, false));
    }

    /// 2026-10-05: The 27B's FFN shapes: gate/up `[17408, 5120]`, down `[5120, 17408]`.
    const GATE: (u32, u32) = (17408, 5120);
    const DOWN: (u32, u32) = (5120, 17408);

    #[test]
    fn the_band_is_one_to_max_m_and_zero_is_off() {
        for (n, k) in [GATE, DOWN] {
            assert!(tc_rows_route(1, n, k, 128, true));
            assert!(tc_rows_route(64, n, k, 128, true));
            assert!(tc_rows_route(65, n, k, 128, true), "a chunked width");
            assert!(tc_rows_route(128, n, k, 128, true));
            assert!(!tc_rows_route(129, n, k, 128, true));
            assert!(!tc_rows_route(0, n, k, 128, true));
            assert!(!tc_rows_route(16, n, k, 0, true), "0 is off");
        }
    }

    #[test]
    fn an_absent_module_or_a_shape_outside_the_contract_declines() {
        assert!(!tc_rows_route(16, GATE.0, GATE.1, 128, false));
        // 2026-10-05: K must be a multiple of 256 (whole load groups).
        assert!(!tc_rows_route(16, 4096, 5120 + 128, 128, true));
        assert!(!tc_rows_route(16, 0, 5120, 128, true));
    }
}
