// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The prefill emitters' kernels (`emitters/prefill_*.rs`), looked up literally as
//! the legacy prefill does: a kernel of a model-tree module through `try_target_kernel` (the
//! module may be absent from the served target), the others through `try_kernel`. A handle here
//! only says an emitter can launch the kernel; the `ops::*` call it mirrors issues the launch.
//!
//! Owner: model-layers circuit executor.
//! Invariants: as the parent's.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::declared::Look;
use crate::layers::{try_kernel, try_target_kernel};

/// 2026-10-03: `(module, func, handle)` of each prefill kernel.
pub(super) fn entries(
    gpu: &dyn GpuBackend,
    look: &Look<'_>,
) -> [(&'static str, &'static str, KernelHandle); 14] {
    [
        (
            "attn_prefill_fa128",
            "attn_prefill_fa128",
            look("attn_prefill_fa128", "attn_prefill_fa128", &|| try_target_kernel(gpu, "attn_prefill_fa128", "attn_prefill_fa128")),
        ),
        (
            "attn_prefill_fa128",
            "attn_prefill_fa128_paged",
            look("attn_prefill_fa128", "attn_prefill_fa128_paged", &|| try_target_kernel(gpu, "attn_prefill_fa128", "attn_prefill_fa128_paged")),
        ),
        (
            "causal_conv1d",
            "causal_conv1d_prefill_state",
            look("causal_conv1d", "causal_conv1d_prefill_state", &|| try_kernel(gpu, "causal_conv1d", "causal_conv1d_prefill_state")),
        ),
        (
            "causal_conv1d",
            "causal_conv1d_update_prefill_tp",
            look("causal_conv1d", "causal_conv1d_update_prefill_tp", &|| try_kernel(gpu, "causal_conv1d", "causal_conv1d_update_prefill_tp")),
        ),
        (
            "gated_delta_rule_fla",
            "gated_delta_rule_chunk_delta_h_pipe",
            look("gated_delta_rule_fla", "gated_delta_rule_chunk_delta_h_pipe", &|| try_kernel(gpu, "gated_delta_rule_fla", "gated_delta_rule_chunk_delta_h_pipe")),
        ),
        (
            "gated_delta_rule_fla",
            "gated_delta_rule_recompute_wu",
            look("gated_delta_rule_fla", "gated_delta_rule_recompute_wu", &|| try_kernel(gpu, "gated_delta_rule_fla", "gated_delta_rule_recompute_wu")),
        ),
        (
            "gated_delta_rule_regresident",
            "gated_delta_rule_prefill_regresident",
            look("gated_delta_rule_regresident", "gated_delta_rule_prefill_regresident", &|| try_kernel(gpu, "gated_delta_rule_regresident", "gated_delta_rule_prefill_regresident")),
        ),
        (
            "gdn_chunk_fwd_o_mma8",
            "gated_delta_rule_chunk_fwd_o_mma8",
            look("gdn_chunk_fwd_o_mma8", "gated_delta_rule_chunk_fwd_o_mma8", &|| try_target_kernel(gpu, "gdn_chunk_fwd_o_mma8", "gated_delta_rule_chunk_fwd_o_mma8")),
        ),
        (
            "norm",
            "l2_norm_bf16",
            look("norm", "l2_norm_bf16", &|| try_kernel(gpu, "norm", "l2_norm_bf16")),
        ),
        (
            "prefill_paged",
            "attn_prefill_paged",
            look("prefill_paged", "attn_prefill_paged", &|| try_kernel(gpu, "prefill_paged", "attn_prefill_paged")),
        ),
        (
            "ssm_preprocess",
            "deinterleave_qg_split_qnorm",
            look("ssm_preprocess", "deinterleave_qg_split_qnorm", &|| try_kernel(gpu, "ssm_preprocess", "deinterleave_qg_split_qnorm")),
        ),
        (
            "w4a16",
            "bf16_to_fp8",
            look("w4a16", "bf16_to_fp8", &|| try_kernel(gpu, "w4a16", "bf16_to_fp8")),
        ),
        (
            "w4a16_fp8_ldmab",
            "fp8_fp8_gemm_ldmab",
            look("w4a16_fp8_ldmab", "fp8_fp8_gemm_ldmab", &|| try_target_kernel(gpu, "w4a16_fp8_ldmab", "fp8_fp8_gemm_ldmab")),
        ),
        (
            "w4a16_fp8_ldmab",
            "fp8_predequant_nvfp4_t",
            look("w4a16_fp8_ldmab", "fp8_predequant_nvfp4_t", &|| try_target_kernel(gpu, "w4a16_fp8_ldmab", "fp8_predequant_nvfp4_t")),
        ),
    ]
}
