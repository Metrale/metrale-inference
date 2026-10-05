// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The MoE FFN's kernels the executor launches (the grouped FP8 decode's router,
//! top-k, sort, expert and blend kernels, and the drafter's batched router), looked up literally
//! as `MoeLayer`'s `init` does, so a handle equals the layer's own. 2026-10-05: And the grouped
//! NVFP4 (lean and row-major) and BF16 expert pairs, the one-row BF16 path's kernels, and the
//! declared NVFP4 head's row tiles.
//!
//! Owner: model-layers (MoE) circuit emitters.
//! Invariants: as the parent's.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::declared::Look;
use crate::layers::try_kernel;

/// 2026-10-03: `(module, func, handle)` of each MoE kernel.
pub(super) fn entries(
    gpu: &dyn GpuBackend,
    look: &Look<'_>,
) -> [(&'static str, &'static str, KernelHandle); 26] {
    [
        (
            "moe_topk",
            "moe_topk_softmax_rows",
            look("moe_topk", "moe_topk_softmax_rows", &|| {
                try_kernel(gpu, "moe_topk", "moe_topk_softmax_rows")
            }),
        ),
        (
            "moe_topk",
            "moe_topk_softmax_batched",
            look("moe_topk", "moe_topk_softmax_batched", &|| {
                try_kernel(gpu, "moe_topk", "moe_topk_softmax_batched")
            }),
        ),
        (
            "moe_fp8_grouped_sort",
            "moe_fp8_grouped_sort",
            look("moe_fp8_grouped_sort", "moe_fp8_grouped_sort", &|| {
                try_kernel(gpu, "moe_fp8_grouped_sort", "moe_fp8_grouped_sort")
            }),
        ),
        (
            "moe_fp8_grouped_tc",
            "moe_expert_gate_up_act_fp8_grouped_tc",
            look(
                "moe_fp8_grouped_tc",
                "moe_expert_gate_up_act_fp8_grouped_tc",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_fp8_grouped_tc",
                        "moe_expert_gate_up_act_fp8_grouped_tc",
                    )
                },
            ),
        ),
        (
            "moe_fp8_grouped_tc",
            "moe_expert_down_act_fp8_grouped_tc",
            look(
                "moe_fp8_grouped_tc",
                "moe_expert_down_act_fp8_grouped_tc",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_fp8_grouped_tc",
                        "moe_expert_down_act_fp8_grouped_tc",
                    )
                },
            ),
        ),
        (
            "moe_shared_expert_fused_fp8_grouped",
            "moe_expert_gate_up_act_fp8_grouped",
            look(
                "moe_shared_expert_fused_fp8_grouped",
                "moe_expert_gate_up_act_fp8_grouped",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_shared_expert_fused_fp8_grouped",
                        "moe_expert_gate_up_act_fp8_grouped",
                    )
                },
            ),
        ),
        (
            "moe_shared_expert_fused_fp8_grouped",
            "moe_expert_down_act_fp8_grouped",
            look(
                "moe_shared_expert_fused_fp8_grouped",
                "moe_expert_down_act_fp8_grouped",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_shared_expert_fused_fp8_grouped",
                        "moe_expert_down_act_fp8_grouped",
                    )
                },
            ),
        ),
        (
            "moe_fp8_grouped_tc_w8a8",
            "moe_act_quant_e4m3",
            look("moe_fp8_grouped_tc_w8a8", "moe_act_quant_e4m3", &|| {
                try_kernel(gpu, "moe_fp8_grouped_tc_w8a8", "moe_act_quant_e4m3")
            }),
        ),
        (
            "moe_fp8_grouped_tc_w8a8",
            "moe_expert_gate_up_act_fp8_grouped_tc_w8a8",
            look(
                "moe_fp8_grouped_tc_w8a8",
                "moe_expert_gate_up_act_fp8_grouped_tc_w8a8",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_fp8_grouped_tc_w8a8",
                        "moe_expert_gate_up_act_fp8_grouped_tc_w8a8",
                    )
                },
            ),
        ),
        (
            "moe_fp8_grouped_tc_w8a8",
            "moe_expert_down_act_fp8_grouped_tc_w8a8",
            look(
                "moe_fp8_grouped_tc_w8a8",
                "moe_expert_down_act_fp8_grouped_tc_w8a8",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_fp8_grouped_tc_w8a8",
                        "moe_expert_down_act_fp8_grouped_tc_w8a8",
                    )
                },
            ),
        ),
        (
            "moe_fp8_grouped_blend",
            "moe_weighted_sum_blend_fp8_grouped",
            look(
                "moe_fp8_grouped_blend",
                "moe_weighted_sum_blend_fp8_grouped",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_fp8_grouped_blend",
                        "moe_weighted_sum_blend_fp8_grouped",
                    )
                },
            ),
        ),
        (
            "moe_router_gemm",
            "moe_router_gemm_bf16",
            look("moe_router_gemm", "moe_router_gemm_bf16", &|| {
                try_kernel(gpu, "moe_router_gemm", "moe_router_gemm_bf16")
            }),
        ),
        (
            "gemm",
            "dense_gemm_bf16",
            look("gemm", "dense_gemm_bf16", &|| {
                try_kernel(gpu, "gemm", "dense_gemm_bf16")
            }),
        ),
        (
            "moe_nvfp4_grouped_tc",
            "moe_expert_gate_up_act_nvfp4_grouped_tc",
            look(
                "moe_nvfp4_grouped_tc",
                "moe_expert_gate_up_act_nvfp4_grouped_tc",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_nvfp4_grouped_tc",
                        "moe_expert_gate_up_act_nvfp4_grouped_tc",
                    )
                },
            ),
        ),
        (
            "moe_nvfp4_grouped_tc",
            "moe_expert_down_act_nvfp4_grouped_tc",
            look(
                "moe_nvfp4_grouped_tc",
                "moe_expert_down_act_nvfp4_grouped_tc",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_nvfp4_grouped_tc",
                        "moe_expert_down_act_nvfp4_grouped_tc",
                    )
                },
            ),
        ),
        (
            "moe_nvfp4_grouped_tc",
            "moe_expert_gate_up_act_nvfp4_grouped_tc_lean",
            look(
                "moe_nvfp4_grouped_tc",
                "moe_expert_gate_up_act_nvfp4_grouped_tc_lean",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_nvfp4_grouped_tc",
                        "moe_expert_gate_up_act_nvfp4_grouped_tc_lean",
                    )
                },
            ),
        ),
        (
            "moe_nvfp4_grouped_tc",
            "moe_expert_down_act_nvfp4_grouped_tc_lean",
            look(
                "moe_nvfp4_grouped_tc",
                "moe_expert_down_act_nvfp4_grouped_tc_lean",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_nvfp4_grouped_tc",
                        "moe_expert_down_act_nvfp4_grouped_tc_lean",
                    )
                },
            ),
        ),
        (
            "moe_bf16_grouped_tc",
            "moe_expert_gate_up_act_bf16_grouped_tc",
            look(
                "moe_bf16_grouped_tc",
                "moe_expert_gate_up_act_bf16_grouped_tc",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_bf16_grouped_tc",
                        "moe_expert_gate_up_act_bf16_grouped_tc",
                    )
                },
            ),
        ),
        (
            "moe_bf16_grouped_tc",
            "moe_expert_down_act_bf16_grouped_tc",
            look(
                "moe_bf16_grouped_tc",
                "moe_expert_down_act_bf16_grouped_tc",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_bf16_grouped_tc",
                        "moe_expert_down_act_bf16_grouped_tc",
                    )
                },
            ),
        ),
        (
            "moe_shared_expert_fused_bf16",
            "moe_expert_gate_up_shared_bf16",
            look(
                "moe_shared_expert_fused_bf16",
                "moe_expert_gate_up_shared_bf16",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_shared_expert_fused_bf16",
                        "moe_expert_gate_up_shared_bf16",
                    )
                },
            ),
        ),
        (
            "moe_shared_expert_fused_bf16",
            "moe_expert_silu_down_shared_bf16",
            look(
                "moe_shared_expert_fused_bf16",
                "moe_expert_silu_down_shared_bf16",
                &|| {
                    try_kernel(
                        gpu,
                        "moe_shared_expert_fused_bf16",
                        "moe_expert_silu_down_shared_bf16",
                    )
                },
            ),
        ),
        (
            "moe_topk",
            "moe_topk_softmax",
            look("moe_topk", "moe_topk_softmax", &|| {
                try_kernel(gpu, "moe_topk", "moe_topk_softmax")
            }),
        ),
        (
            "moe_expert_gemv",
            "moe_weighted_sum_blend",
            look("moe_expert_gemv", "moe_weighted_sum_blend", &|| {
                try_kernel(gpu, "moe_expert_gemv", "moe_weighted_sum_blend")
            }),
        ),
        (
            "w4a16_tc_rows",
            "w4a16_tc_rows_16",
            look("w4a16_tc_rows", "w4a16_tc_rows_16", &|| {
                try_kernel(gpu, "w4a16_tc_rows", "w4a16_tc_rows_16")
            }),
        ),
        (
            "w4a16_tc_rows",
            "w4a16_tc_rows_32",
            look("w4a16_tc_rows", "w4a16_tc_rows_32", &|| {
                try_kernel(gpu, "w4a16_tc_rows", "w4a16_tc_rows_32")
            }),
        ),
        (
            "w4a16_tc_rows",
            "w4a16_tc_rows_64",
            look("w4a16_tc_rows", "w4a16_tc_rows_64", &|| {
                try_kernel(gpu, "w4a16_tc_rows", "w4a16_tc_rows_64")
            }),
        ),
    ]
}
