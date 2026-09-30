// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The batched GDN arm's kernels the executor launches (the strided conv, recurrence
//! and gated norm, and the BA-gates twin), looked up literally as the layer's `init` does.
//!
//! Owner: model-layers circuit executor.
//! Invariants: as the parent's.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::declared::Look;
use crate::layers::{try_kernel, try_target_kernel};

/// 2026-09-30: `(module, func, handle)` of each batched-arm kernel.
pub(super) fn entries(
    gpu: &dyn GpuBackend,
    look: &Look<'_>,
) -> [(&'static str, &'static str, KernelHandle); 4] {
    [
        (
            "causal_conv1d",
            "causal_conv1d_update_l2norm_f32_strided",
            look(
                "causal_conv1d",
                "causal_conv1d_update_l2norm_f32_strided",
                &|| {
                    try_kernel(
                        gpu,
                        "causal_conv1d",
                        "causal_conv1d_update_l2norm_f32_strided",
                    )
                },
            ),
        ),
        (
            "gated_delta_rule",
            "gated_delta_rule_decode_f32_strided",
            look(
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_strided",
                &|| {
                    try_kernel(
                        gpu,
                        "gated_delta_rule",
                        "gated_delta_rule_decode_f32_strided",
                    )
                },
            ),
        ),
        (
            "norm",
            "gated_rms_norm_f32_input_strided",
            look("norm", "gated_rms_norm_f32_input_strided", &|| {
                try_kernel(gpu, "norm", "gated_rms_norm_f32_input_strided")
            }),
        ),
        (
            "ssm_ba_gates_hopper",
            "dense_gemm_ba_gates_prefill_hopper",
            look(
                "ssm_ba_gates_hopper",
                "dense_gemm_ba_gates_prefill_hopper",
                &|| {
                    try_target_kernel(
                        gpu,
                        "ssm_ba_gates_hopper",
                        "dense_gemm_ba_gates_prefill_hopper",
                    )
                },
            ),
        ),
    ]
}
