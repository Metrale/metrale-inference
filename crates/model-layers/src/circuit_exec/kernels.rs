// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The kernels the circuit executor can launch, resolved at boot through literal
//! lookups (so the kernel-lookup audit records each at its line here), and the probe of every
//! kernel any rule names, which is what the fuser treats as available.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - The fuser sees every rule kernel the loaded modules contain, not only the ones an emitter
//!   launches: hiding a present kernel would let a lower-priority rule win and the plan leave
//!   legacy routing. A plan that selects a kernel with no emitter handle is refused instead
//!   ([`KernelTable::handle`]).

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use metrale_circuit::{AvailableKernels, KernelId, Rule};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use crate::layers::{try_kernel, try_target_kernel};

mod declared;

/// 2026-09-28: Emitter handles by kernel id; only present kernels are kept.
#[derive(Debug, Clone, Default)]
pub struct KernelTable {
    handles: BTreeMap<KernelId, KernelHandle>,
}

impl KernelTable {
    /// 2026-09-28: Resolve every kernel an emitter can launch that `available` holds. The
    /// lookups are written out literally; one outside `available` is not issued.
    pub fn resolve(gpu: &dyn GpuBackend, available: &AvailableKernels) -> Self {
        let has = |m: &str, f: &str| {
            available.kernels.contains(&KernelId {
                module: m.to_string(),
                func: f.to_string(),
            })
        };
        let look = |m: &str, f: &str, lookup: &dyn Fn() -> KernelHandle| {
            if has(m, f) { lookup() } else { KernelHandle(0) }
        };
        let entries = [
            (
                "norm",
                "rms_norm",
                look("norm", "rms_norm", &|| try_kernel(gpu, "norm", "rms_norm")),
            ),
            (
                "norm",
                "rms_norm_residual",
                look("norm", "rms_norm_residual", &|| {
                    try_kernel(gpu, "norm", "rms_norm_residual")
                }),
            ),
            (
                "norm",
                "residual_add_rms_norm",
                look("norm", "residual_add_rms_norm", &|| {
                    try_kernel(gpu, "norm", "residual_add_rms_norm")
                }),
            ),
            (
                "norm",
                "gated_rms_norm_f32_input",
                look("norm", "gated_rms_norm_f32_input", &|| {
                    try_kernel(gpu, "norm", "gated_rms_norm_f32_input")
                }),
            ),
            (
                "residual_add_rms_norm_exact",
                "residual_add_rms_norm_exact",
                look(
                    "residual_add_rms_norm_exact",
                    "residual_add_rms_norm_exact",
                    &|| {
                        try_target_kernel(
                            gpu,
                            "residual_add_rms_norm_exact",
                            "residual_add_rms_norm_exact",
                        )
                    },
                ),
            ),
            (
                "residual_add",
                "bf16_residual_add",
                look("residual_add", "bf16_residual_add", &|| {
                    try_kernel(gpu, "residual_add", "bf16_residual_add")
                }),
            ),
            (
                "residual_add",
                "sigmoid_gate_mul",
                look("residual_add", "sigmoid_gate_mul", &|| {
                    try_kernel(gpu, "residual_add", "sigmoid_gate_mul")
                }),
            ),
            (
                "w4a16_gemv",
                "w4a16_gemv_sw",
                look("w4a16_gemv", "w4a16_gemv_sw", &|| {
                    try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw")
                }),
            ),
            (
                "w4a16_gemv",
                "w4a16_gemv_qg",
                look("w4a16_gemv", "w4a16_gemv_qg", &|| {
                    try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_qg")
                }),
            ),
            (
                "w4a16_gemv_fused",
                "w4a16_gemv_dual",
                look("w4a16_gemv_fused", "w4a16_gemv_dual", &|| {
                    try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual")
                }),
            ),
            (
                "w4a16_gemv_fused",
                "w4a16_gemv_dual_sw",
                look("w4a16_gemv_fused", "w4a16_gemv_dual_sw", &|| {
                    try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_sw")
                }),
            ),
            (
                "moe_silu_mul",
                "moe_silu_mul",
                look("moe_silu_mul", "moe_silu_mul", &|| {
                    try_kernel(gpu, "moe_silu_mul", "moe_silu_mul")
                }),
            ),
            (
                "ssm_preprocess",
                "dense_gemv_ba_gates",
                look("ssm_preprocess", "dense_gemv_ba_gates", &|| {
                    try_kernel(gpu, "ssm_preprocess", "dense_gemv_ba_gates")
                }),
            ),
            (
                "causal_conv1d",
                "causal_conv1d_update_l2norm_f32",
                look("causal_conv1d", "causal_conv1d_update_l2norm_f32", &|| {
                    try_kernel(gpu, "causal_conv1d", "causal_conv1d_update_l2norm_f32")
                }),
            ),
            (
                "gated_delta_rule",
                "gated_delta_rule_decode_f32",
                look("gated_delta_rule", "gated_delta_rule_decode_f32", &|| {
                    try_kernel(gpu, "gated_delta_rule", "gated_delta_rule_decode_f32")
                }),
            ),
            (
                "rope_mrope_interleaved",
                "rope_forward_mrope_interleaved",
                look(
                    "rope_mrope_interleaved",
                    "rope_forward_mrope_interleaved",
                    &|| {
                        try_target_kernel(
                            gpu,
                            "rope_mrope_interleaved",
                            "rope_forward_mrope_interleaved",
                        )
                    },
                ),
            ),
            (
                "reshape_and_cache",
                "reshape_and_cache_flash",
                look("reshape_and_cache", "reshape_and_cache_flash", &|| {
                    try_kernel(gpu, "reshape_and_cache", "reshape_and_cache_flash")
                }),
            ),
            (
                "paged_decode",
                "paged_decode_attn",
                look("paged_decode", "paged_decode_attn", &|| {
                    try_kernel(gpu, "paged_decode", "paged_decode_attn")
                }),
            ),
            (
                "gemv",
                "dense_gemv_bf16",
                look("gemv", "dense_gemv_bf16", &|| {
                    try_kernel(gpu, "gemv", "dense_gemv_bf16")
                }),
            ),
            (
                "residual_add",
                "sigmoid_gate_mul_batched",
                look("residual_add", "sigmoid_gate_mul_batched", &|| {
                    try_kernel(gpu, "residual_add", "sigmoid_gate_mul_batched")
                }),
            ),
            (
                "ssm_preprocess",
                "deinterleave_qg",
                look("ssm_preprocess", "deinterleave_qg", &|| {
                    try_kernel(gpu, "ssm_preprocess", "deinterleave_qg")
                }),
            ),
            (
                "norm",
                "rms_norm_strided",
                look("norm", "rms_norm_strided", &|| {
                    try_kernel(gpu, "norm", "rms_norm_strided")
                }),
            ),
            (
                "rope",
                "rope_forward_strided",
                look("rope", "rope_forward_strided", &|| {
                    try_kernel(gpu, "rope", "rope_forward_strided")
                }),
            ),
            (
                "w4a16_gemv_tc",
                "w4a16_gemv_tc8",
                look("w4a16_gemv_tc", "w4a16_gemv_tc8", &|| {
                    try_kernel(gpu, "w4a16_gemv_tc", "w4a16_gemv_tc8")
                }),
            ),
            (
                "w4a16_gemv",
                "w4a16_gemv_batch2",
                look("w4a16_gemv", "w4a16_gemv_batch2", &|| {
                    try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch2")
                }),
            ),
            (
                "w4a16_gemv",
                "w4a16_gemv_dual_batch2",
                look("w4a16_gemv", "w4a16_gemv_dual_batch2", &|| {
                    try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_dual_batch2")
                }),
            ),
            (
                "w4a16_gemv",
                "w4a16_gemv_qg_batch2",
                look("w4a16_gemv", "w4a16_gemv_qg_batch2", &|| {
                    try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_qg_batch2")
                }),
            ),
            (
                "w4a16",
                "w4a16_gemm_t_p3",
                look("w4a16", "w4a16_gemm_t_p3", &|| {
                    try_kernel(gpu, "w4a16", "w4a16_gemm_t_p3")
                }),
            ),
            (
                "w4a16",
                "w4a16_gemm_t_k64_p3",
                look("w4a16", "w4a16_gemm_t_k64_p3", &|| {
                    try_kernel(gpu, "w4a16", "w4a16_gemm_t_k64_p3")
                }),
            ),
            (
                "w4a16",
                "w4a16_gemm_t_k64_n64_p3",
                look("w4a16", "w4a16_gemm_t_k64_n64_p3", &|| {
                    try_kernel(gpu, "w4a16", "w4a16_gemm_t_k64_n64_p3")
                }),
            ),
            (
                "w4a16",
                "w4a16_gemm_t_m128",
                look("w4a16", "w4a16_gemm_t_m128", &|| {
                    try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128")
                }),
            ),
            (
                "nvfp4_mmq",
                "metrale_nvfp4_quantize_bf16",
                look("nvfp4_mmq", "metrale_nvfp4_quantize_bf16", &|| {
                    try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_quantize_bf16")
                }),
            ),
            (
                "nvfp4_mmq",
                "metrale_nvfp4_mmq16_nc",
                look("nvfp4_mmq", "metrale_nvfp4_mmq16_nc", &|| {
                    try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_mmq16_nc")
                }),
            ),
            (
                "nvfp4_mmq",
                "metrale_nvfp4_mmq32_nc",
                look("nvfp4_mmq", "metrale_nvfp4_mmq32_nc", &|| {
                    try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_mmq32_nc")
                }),
            ),
            (
                "nvfp4_mmq",
                "metrale_nvfp4_mmq64_nc",
                look("nvfp4_mmq", "metrale_nvfp4_mmq64_nc", &|| {
                    try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_mmq64_nc")
                }),
            ),
            (
                "nvfp4_mmq",
                "metrale_nvfp4_gemm_pipe",
                look("nvfp4_mmq", "metrale_nvfp4_gemm_pipe", &|| {
                    try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_gemm_pipe")
                }),
            ),
            (
                "nvfp4_mmq",
                "metrale_nvfp4_silu_mul_quant",
                look("nvfp4_mmq", "metrale_nvfp4_silu_mul_quant", &|| {
                    try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_silu_mul_quant")
                }),
            ),
            (
                "nvfp4_mmq",
                "metrale_nvfp4_scale_bf16",
                look("nvfp4_mmq", "metrale_nvfp4_scale_bf16", &|| {
                    try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_scale_bf16")
                }),
            ),
            (
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
                look("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm", &|| {
                    try_kernel(gpu, "dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")
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
                "argmax",
                "argmax_bf16",
                look("argmax", "argmax_bf16", &|| {
                    try_kernel(gpu, "argmax", "argmax_bf16")
                }),
            ),
            (
                "causal_conv1d",
                "causal_conv1d_update_l2norm",
                look("causal_conv1d", "causal_conv1d_update_l2norm", &|| {
                    try_kernel(gpu, "causal_conv1d", "causal_conv1d_update_l2norm")
                }),
            ),
            (
                "gated_delta_rule_wy",
                "gated_delta_rule_wy2",
                look("gated_delta_rule_wy", "gated_delta_rule_wy2", &|| {
                    try_kernel(gpu, "gated_delta_rule_wy", "gated_delta_rule_wy2")
                }),
            ),
            (
                "gated_delta_rule_wy3",
                "gated_delta_rule_wy3",
                look("gated_delta_rule_wy3", "gated_delta_rule_wy3", &|| {
                    try_kernel(gpu, "gated_delta_rule_wy3", "gated_delta_rule_wy3")
                }),
            ),
            (
                "gated_delta_rule_wy4",
                "gated_delta_rule_wy4",
                look("gated_delta_rule_wy4", "gated_delta_rule_wy4", &|| {
                    try_kernel(gpu, "gated_delta_rule_wy4", "gated_delta_rule_wy4")
                }),
            ),
            (
                "norm",
                "gated_rms_norm_prefill",
                look("norm", "gated_rms_norm_prefill", &|| {
                    try_kernel(gpu, "norm", "gated_rms_norm_prefill")
                }),
            ),
            (
                "ssm_preprocess",
                "dense_gemm_ba_gates_prefill",
                look("ssm_preprocess", "dense_gemm_ba_gates_prefill", &|| {
                    try_kernel(gpu, "ssm_preprocess", "dense_gemm_ba_gates_prefill")
                }),
            ),
            (
                "w4a16_gemv",
                "w4a16_gemv_batch3",
                look("w4a16_gemv", "w4a16_gemv_batch3", &|| {
                    try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch3")
                }),
            ),
            (
                "w4a16_gemv",
                "w4a16_gemv_dual_batch3",
                look("w4a16_gemv", "w4a16_gemv_dual_batch3", &|| {
                    try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_dual_batch3")
                }),
            ),
            (
                "w4a16_gemv",
                "w4a16_gemv_qg_batch3",
                look("w4a16_gemv", "w4a16_gemv_qg_batch3", &|| {
                    try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_qg_batch3")
                }),
            ),
            (
                "rope",
                "rope_forward",
                look("rope", "rope_forward", &|| {
                    try_kernel(gpu, "rope", "rope_forward")
                }),
            ),
            (
                "residual_add",
                "bf16_concat",
                look("residual_add", "bf16_concat", &|| {
                    try_kernel(gpu, "residual_add", "bf16_concat")
                }),
            ),
        ];
        let handles = entries
            .into_iter()
            .chain(declared::entries(gpu, &look))
            .filter(|(_, _, h)| h.0 != 0)
            .map(|(m, f, h)| {
                (
                    KernelId {
                        module: m.to_string(),
                        func: f.to_string(),
                    },
                    h,
                )
            })
            .collect();
        KernelTable { handles }
    }

    /// 2026-09-28: The handle of `k`; an error when no emitter launches it or it is absent.
    pub fn handle(&self, k: &KernelId) -> Result<KernelHandle> {
        self.handles.get(k).copied().ok_or_else(|| {
            anyhow!(
                "the plan selects {}::{}, which no circuit emitter launches on this target",
                k.module,
                k.func
            )
        })
    }
}

/// 2026-09-28: Whether PTX text defines the entry point `func` (`.entry func(`).
pub fn ptx_defines(ptx: &[u8], func: &str) -> bool {
    let needle = format!(".entry {func}(");
    ptx.windows(needle.len()).any(|w| w == needle.as_bytes())
}

/// 2026-09-28: Every kernel `rules` name that the target's modules (`(name, PTX)`) define. Read
/// from the PTX, not by lookup: a lookup of an entry the target lacks would count as an
/// unresolved kernel at the boot gate. Capability bits are not probed yet, so a rule that
/// requires one is refused rather than assumed present or silently skipped.
pub fn available_in(rules: &[Rule], modules: &[(&str, &[u8])]) -> Result<AvailableKernels> {
    if let Some(r) = rules.iter().find(|r| !r.requires.is_empty()) {
        anyhow::bail!(
            "rule `{}` requires capabilities {:?}; the circuit executor does not probe \
             capabilities yet",
            r.id,
            r.requires
        );
    }
    let mut out = AvailableKernels::default();
    for k in rules.iter().flat_map(|r| r.kernels.iter()) {
        if modules
            .iter()
            .any(|(m, ptx)| *m == k.module && ptx_defines(ptx, &k.func))
        {
            out.kernels.insert(k.clone());
        }
    }
    Ok(out)
}
