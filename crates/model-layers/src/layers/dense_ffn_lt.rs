// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The dense FFN's wide-row cuBLASLt arm. From the class's `[defaults]
//! ffn_w4a16_lt_min_rows` rows up (`METRALE_FFN_W4A16_LT_MIN_ROWS` overrides; 0: never), and only
//! while every projection family routes `adaptive`, an NVFP4 projection runs as two launches: an
//! exact BF16 copy of the weight in the arena's `ffn_bf16_weight` (`dequant_nvfp4_to_bf16_g16`:
//! E2M1 x E4M3 group scale, exact in BF16) and a cuBLASLt BF16 GEMM that applies
//! `weight_scale_2` as alpha on the FP32 sum.
//!
//! Why: on the H100 SXM the in-tree NVFP4 tile GEMMs reach about 12 % of the BF16 tensor-core
//! peak at prefill widths (gate N=17408 K=5120: 2243 us at 2048 rows for the BF16 tile against
//! 424 us for the cuBLASLt GEMM), and the row-tile kernel stays latency-bound past 64 rows.
//! The copy costs one read of the packed weight and one write of the BF16 one per projection.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - [`lt_route`] is true only for a width at or past a nonzero threshold, an arena scratch of at
//!   least `n * k * 2` bytes, a resolved dequant kernel, no per-row `weight_scale_2_vec`, and
//!   `k % 16 == 0`; [`DenseFfnLayer::try_ffn_lt`] launches nothing otherwise.
//! - Under a fixed activation format the arm is off: its algorithm can change with the row count,
//!   so a row's bits can depend on how many rows share the launch.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

const DEQUANT_MODULE: &str = "dequant_nvfp4_bf16";
const DEQUANT_ENTRY: &str = "dequant_nvfp4_to_bf16_g16";

/// 2026-10-09: The routing rule, without I/O.
pub(crate) fn lt_route(
    m: u32,
    n: u32,
    k: u32,
    min_rows: u32,
    scratch_bytes: usize,
    dequant_present: bool,
    per_row_scale2: bool,
) -> bool {
    min_rows > 0
        && m >= min_rows
        && dequant_present
        && !per_row_scale2
        && n > 0
        && k > 0
        && k.is_multiple_of(16)
        && scratch_bytes >= n as usize * k as usize * 2
}

/// 2026-10-09: The threshold in force: the resolved class row, 0 under a fixed activation format.
fn min_rows() -> u32 {
    if crate::layers::activation_quantization::any_fixed() {
        return 0;
    }
    ops::target_defaults::resolved().ffn_w4a16_lt_min_rows.value
}

impl DenseFfnLayer {
    /// 2026-10-09: `output [m, n] = input [m, k] x W^T` through the cuBLASLt arm when [`lt_route`]
    /// takes it: `Ok(true)` when it launched, `Ok(false)` to keep the in-tree kernels.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_ffn_lt(
        &self,
        ctx: &ForwardContext,
        weight: &QuantizedWeight,
        input: DevicePtr,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let threshold = min_rows();
        if threshold == 0 || m < threshold {
            return Ok(false);
        }
        let dequant = crate::layers::try_kernel(ctx.gpu, DEQUANT_MODULE, DEQUANT_ENTRY);
        let scratch = ctx.buffers.ffn_bf16_weight();
        if !lt_route(
            m,
            n,
            k,
            threshold,
            ctx.buffers.ffn_bf16_weight_bytes(),
            dequant.0 != 0,
            weight.weight_scale_2_vec != DevicePtr::NULL,
        ) {
            return Ok(false);
        }
        let groups = n as u64 * k as u64 / 16;
        let blocks = groups.div_ceil(256).min(132 * 32) as u32;
        KernelLaunch::new(ctx.gpu, dequant)
            .grid([blocks, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(weight.weight)
            .arg_ptr(weight.weight_scale)
            .arg_ptr(scratch)
            .arg_u64(groups)
            .launch(stream)?;
        metrale_gpu_runtime::cublaslt::bf16_gemm_act_weight_t_alpha(
            input.0,
            scratch.0,
            output.0,
            m,
            n,
            k,
            weight.weight_scale_2,
            stream,
        )?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::lt_route;

    const GATE: (u32, u32) = (17408, 5120);
    const DOWN: (u32, u32) = (5120, 17408);
    const SCRATCH: usize = 17408 * 5120 * 2;

    #[test]
    fn the_arm_starts_at_the_threshold_and_zero_is_off() {
        for (n, k) in [GATE, DOWN] {
            assert!(!lt_route(128, n, k, 129, SCRATCH, true, false));
            assert!(lt_route(129, n, k, 129, SCRATCH, true, false));
            assert!(lt_route(2048, n, k, 129, SCRATCH, true, false));
            assert!(!lt_route(2048, n, k, 0, SCRATCH, true, false), "0 is off");
        }
    }

    #[test]
    fn a_short_scratch_a_missing_kernel_or_a_per_row_scale_declines() {
        assert!(!lt_route(
            256,
            GATE.0,
            GATE.1,
            129,
            SCRATCH - 1,
            true,
            false
        ));
        assert!(!lt_route(256, GATE.0, GATE.1, 129, 0, true, false));
        assert!(!lt_route(256, GATE.0, GATE.1, 129, SCRATCH, false, false));
        assert!(!lt_route(256, GATE.0, GATE.1, 129, SCRATCH, true, true));
        assert!(
            !lt_route(256, 17408, 5120 + 8, 129, usize::MAX, true, false),
            "K % 16"
        );
    }
}
