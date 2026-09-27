// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: The block-scaled FP8 GDN projections under a row-invariant tier
//! policy (`crate::layers::row_tiers`): one summation order at every row count.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants: `row_tier_fp8_proj` launches nothing when it returns `Ok(false)`.

use super::*;
use crate::layers::RowTiers;
use crate::weight_map::Fp8Weight;

impl Qwen3SsmLayer {
    /// 2026-09-27: `[m, k]` BF16 rows at `input` (pitch `k`) times the FP8 weight
    /// `[n, k]` into `[m, n]` at `output` (pitch `n`), when a row-invariant policy
    /// is on and its kernels are linked:
    /// - `Exact`: `w8a16_gemv_batch16` over chunks of at most 16 rows, the scalar
    ///   `w8a16_gemv` order for every row.
    /// - `Canonical`: `w8a16_gemm_pipelined_by_m`, the tensor-core tile family
    ///   (the 32-, 64- and 128-row tiles sum identically).
    ///
    /// Returns `Ok(false)`, launching nothing, under `RowTiers::ByRows`, for a
    /// weight that is not block-scaled, or without the policy's kernels.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn row_tier_fp8_proj(
        &self,
        ctx: &ForwardContext,
        fp8: &Fp8Weight,
        input: DevicePtr,
        output: DevicePtr,
        m: usize,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        if fp8.scale_format != crate::weight_map::WeightQuantFormat::Fp8BlockScaled {
            return Ok(false);
        }
        match crate::layers::row_tiers() {
            RowTiers::ByRows => Ok(false),
            RowTiers::Exact => {
                if self.w8a16_gemv_batch16_k.0 == 0 {
                    return Ok(false);
                }
                let mut done = 0usize;
                while done < m {
                    let rows = (m - done).min(16);
                    ops::w8a16_gemv_batch16(
                        ctx.gpu,
                        self.w8a16_gemv_batch16_k,
                        input.offset(done * k as usize * 2),
                        fp8.weight,
                        fp8.row_scale,
                        output.offset(done * n as usize * 2),
                        rows as u32,
                        n,
                        k,
                        stream,
                    )?;
                    done += rows;
                }
                Ok(true)
            }
            RowTiers::Canonical => {
                if self.w8a16_gemm_pipelined_k.0 == 0 {
                    return Ok(false);
                }
                ops::w8a16_gemm_pipelined_by_m(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    self.w8a16_gemm_pipelined_m32_k,
                    self.w8a16_gemm_pipelined_m64_k,
                    input,
                    fp8.weight,
                    fp8.row_scale,
                    output,
                    m as u32,
                    n,
                    k,
                    stream,
                )?;
                Ok(true)
            }
        }
    }

    /// 2026-09-27: One row's block-scaled FP8 projection `[1, k] x [n, k]^T`:
    /// the tile family under `RowTiers::Canonical`, else the scalar
    /// `w8a16_gemv` (whose order `RowTiers::Exact` keeps at every row count).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn single_row_fp8_proj(
        &self,
        ctx: &ForwardContext,
        fp8: &Fp8Weight,
        input: DevicePtr,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if crate::layers::row_tiers() == RowTiers::Canonical
            && self.row_tier_fp8_proj(ctx, fp8, input, output, 1, n, k, stream)?
        {
            return Ok(());
        }
        ops::w8a16_gemv(
            ctx.gpu,
            self.w8a16_gemv_k,
            input,
            fp8.weight,
            fp8.row_scale,
            output,
            n,
            k,
            stream,
        )
    }
}
