// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The Nemotron-H loader levers, resolved from the startup settings once per load.
//!
//! Owner: model-arch weight loader (Nemotron-H).
//! Invariants: `load_layers` resolves this once before its layer loop; each field reads one
//! settings field with the parse its old reader applied.

use metrale_settings::InferenceServerSettings;

/// 2026-10-17: The levers the Nemotron-H loader and its SSM and MoE builders read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NemotronLevers {
    /// 2026-10-17: `METRALE_NEMOTRON_BF16_ATTN`: dense attention keeps the checkpoint BF16
    /// Q/K/V/O (off: quantized to NVFP4).
    pub(super) bf16_attn: bool,
    /// 2026-10-17: `METRALE_NEMOTRON_NATIVE_FP8_SSM` is unset, `1`, `both` or `decode`: an FP8
    /// SSM layer decodes in native FP8.
    pub(super) native_fp8_ssm_decode: bool,
    /// 2026-10-17: The same variable is unset, `1` or `both`: it also prefills in native FP8.
    pub(super) native_fp8_ssm_prefill: bool,
    /// 2026-10-17: `METRALE_NEMOTRON_NATIVE_BF16_SSM`: a BF16 SSM layer stays BF16.
    pub(super) native_bf16_ssm: bool,
    /// 2026-10-17: `METRALE_NO_SSM_FP8_PREFILL` set: transposed NVFP4 SSM prefill copies
    /// instead of pre-dequantized FP8 ones.
    pub(super) no_ssm_fp8_prefill: bool,
    /// 2026-10-17: `METRALE_NO_SSM_PREFILL_T` set as well: no SSM prefill copy.
    pub(super) no_ssm_prefill_t: bool,
    /// 2026-10-17: `METRALE_SHARED_FP8_PREFILL` set: the MoE shared expert's pre-dequantized
    /// FP8 prefill copies (`NemotronMoeLayer::prepare_prefill_weights`).
    pub(super) shared_fp8_prefill: bool,
}

impl NemotronLevers {
    /// 2026-10-17: Resolve from the startup settings.
    pub(super) fn from_settings(s: &InferenceServerSettings) -> Self {
        let k = &s.kernels;
        let mode = k.nemotron_native_fp8_ssm.as_deref().unwrap_or("1");
        Self {
            bf16_attn: k.nemotron_bf16_attn,
            native_fp8_ssm_decode: matches!(mode, "1" | "both" | "decode"),
            native_fp8_ssm_prefill: matches!(mode, "1" | "both"),
            native_bf16_ssm: k.nemotron_native_bf16_ssm,
            no_ssm_fp8_prefill: s.scheduler.no_ssm_fp8_prefill,
            no_ssm_prefill_t: s.scheduler.no_ssm_prefill_t,
            shared_fp8_prefill: s.scheduler.shared_fp8_prefill,
        }
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
