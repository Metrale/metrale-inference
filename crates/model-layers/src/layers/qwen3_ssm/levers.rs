// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The dispatch levers of the GDN layer, resolved once from the startup settings
//! ([`SsmLevers::from_settings`]) and carried on `ModelLevers::ssm`. The forward paths read
//! `ctx.levers.ssm`; nothing in `qwen3_ssm` outside `gdn_flags` and `debug` reads the process
//! environment.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants:
//! - `from_settings(&defaults)` equals [`SsmLevers::declared`]
//!   (`the_declared_defaults_are_the_old_unset_values`).
//! - The text levers keep their readers' predicate or parse: `METRALE_GDN_BF16_WEIGHTS`,
//!   `METRALE_GDN_FUSED_VERIFY` and `METRALE_FP8_W8A8` are on at exactly `1`,
//!   `METRALE_GDN_FUSED_CONV17` is off at exactly `0`, and the two row thresholds parse as in
//!   [`ffn_prefill_min_n_of`] and [`tc_proj_min_n_of`].

use metrale_settings::InferenceServerSettings;

use super::gdn_flags::SSM_M128_MIN_M;
use super::trait_decode_multi_seq::{SSM_FFN_PREFILL_MIN_N, SSM_TC_PROJ_MIN_N};

/// 2026-10-17: The GDN layer's dispatch levers. Each field names its variable; the forward path
/// that takes it documents what it changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SsmLevers {
    /// 2026-09-25: `METRALE_GDN_BF16_WEIGHTS` is exactly `1` (`trait_prefill_proj.rs`).
    pub bf16_weights: bool,
    /// 2026-09-25: `METRALE_GDN_FUSED_VERIFY` is exactly `1`
    /// (`Qwen3SsmLayer::fused_verify_k2_enabled`).
    pub fused_verify: bool,
    /// 2026-09-25: On unless `METRALE_NO_GDN_WY2_RESIDENT` is present at any value
    /// (`wy_select::wy2_kernel`).
    pub wy2_resident: bool,
    /// 2026-09-25: On unless `METRALE_NO_GDN_WY3_RESIDENT` is present at any value
    /// (`wy_select::wy3_kernel`).
    pub wy3_resident: bool,
    /// 2026-09-25: `METRALE_FP8_W8A8` is exactly `1` (`trait_prefill_helper.rs`).
    pub force_w8a8: bool,
    /// 2026-09-25: `METRALE_QWEN4EXP_PREFILL_PROF` (`trait_prefill_hc.rs`).
    pub hc_prefill_prof: bool,
    /// 2026-09-25: `METRALE_QWEN4EXP_DECODE_PROF` (`trait_decode_hc.rs`).
    pub hc_decode_prof: bool,
    /// 2026-09-25: `METRALE_SSM_OUT_W8A16_ONLY` is present at any value
    /// (`prefill_out_w8a8::prefill_out_proj_w8a8_selected`).
    pub out_w8a16_only: bool,
    /// 2026-09-25: On unless `METRALE_NO_GDN_FLA` is switched on (`trait_prefill_recur.rs`).
    pub fla: bool,
    /// 2026-09-25: The batched dense FFN's minimum width, `METRALE_SSM_FFN_PREFILL_MIN_N`
    /// (`trait_decode_multi_seq.rs`).
    pub ffn_prefill_min_n: usize,
    /// 2026-09-25: On unless `METRALE_NO_SSM_FFN_PREFILL` is switched on
    /// (`trait_decode_multi_seq.rs`).
    pub ffn_prefill: bool,
    /// 2026-09-25: On unless `METRALE_NO_BATCHED_BA_GATES` is present at any value
    /// (`trait_decode_batched/gates_norm.rs`).
    pub batched_ba_gates: bool,
    /// 2026-09-25: On unless `METRALE_NO_BATCHED_GDN_NORM` is present at any value
    /// (`trait_decode_batched/gates_norm.rs`).
    pub batched_gdn_norm: bool,
    /// 2026-09-25: On unless `METRALE_NO_QKVZ_NVFP4_DECODE` is present at any value
    /// (`trait_decode_batched::qkvz_verify_nvfp4_wins`).
    pub qkvz_nvfp4_decode: bool,
    /// 2026-09-25: On unless `METRALE_NO_VERIFY_OUTPROJ_TGEMM` is present at any value
    /// (`trait_decode_batched/out_proj.rs`).
    pub verify_outproj_tgemm: bool,
    /// 2026-09-25: On unless `METRALE_NO_VERIFY_GDN_BATCH` is present at any value
    /// (`trait_decode_batched_conv_gdn_multi.rs`).
    pub verify_gdn_batch: bool,
    /// 2026-09-25: The tensor-core projection's minimum width, `None` when off,
    /// `METRALE_SSM_TC_PROJ` (`ssm_batched::ssm_tc_proj_min_n`).
    pub tc_proj_min_n: Option<usize>,
    /// 2026-09-25: On unless `METRALE_GDN_FUSED_CONV17` is exactly `0`
    /// (`trait_decode_batched_conv_gdn_wyn.rs`).
    pub fused_conv17: bool,
    /// 2026-09-25: On unless `METRALE_NO_GDN_HALF_REG` is switched on
    /// (`ssm_batched_recurrent.rs`).
    pub half_reg: bool,
    /// 2026-09-25: `METRALE_GDN_SMEM_STAGE` is present at any value
    /// (`ssm_batched_recurrent.rs`).
    pub smem_stage: bool,
    /// 2026-10-18: The multi-sequence projections' 128-row M-tile width, `None` when
    /// `METRALE_NO_SSM_M128` is present at any value (`kernel_select::ms_proj_gemm`).
    pub m128_min_m: Option<u32>,
}

/// 2026-10-17: `METRALE_SSM_FFN_PREFILL_MIN_N`: a count of at least 2, else 5.
fn ffn_prefill_min_n_of(v: Option<&str>) -> usize {
    v.and_then(|v| v.parse().ok())
        .filter(|&v| v >= 2)
        .unwrap_or(SSM_FFN_PREFILL_MIN_N)
}

/// 2026-10-17: `METRALE_SSM_TC_PROJ`: 9 when unset or `1`, off at `0`, else a count of at
/// least 2, off when it is not one.
fn tc_proj_min_n_of(v: Option<&str>) -> Option<usize> {
    match v {
        None | Some("1") => Some(SSM_TC_PROJ_MIN_N),
        Some("0") => None,
        Some(v) => v.parse::<usize>().ok().filter(|&x| x >= 2),
    }
}

impl SsmLevers {
    /// 2026-10-17: The levers of the settings resolved once at startup.
    pub fn from_settings(s: &InferenceServerSettings) -> Self {
        let (k, sc, sp) = (&s.kernels, &s.scheduler, &s.speculative);
        let is_one = |v: &Option<String>| v.as_deref() == Some("1");
        Self {
            bf16_weights: is_one(&k.gdn_bf16_weights),
            fused_verify: is_one(&sp.gdn_fused_verify),
            wy2_resident: !k.no_gdn_wy2_resident,
            wy3_resident: !k.no_gdn_wy3_resident,
            force_w8a8: is_one(&k.fp8_w8a8),
            hc_prefill_prof: sc.qwen4exp_prefill_prof,
            hc_decode_prof: s.model.qwen4exp_decode_prof,
            out_w8a16_only: k.ssm_out_w8a16_only,
            fla: !k.no_gdn_fla,
            ffn_prefill_min_n: ffn_prefill_min_n_of(sc.ssm_ffn_prefill_min_n.as_deref()),
            ffn_prefill: !sc.no_ssm_ffn_prefill,
            batched_ba_gates: !sc.no_batched_ba_gates,
            batched_gdn_norm: !sc.no_batched_gdn_norm,
            qkvz_nvfp4_decode: !s.cache.no_qkvz_nvfp4_decode,
            verify_outproj_tgemm: !sp.no_verify_outproj_tgemm,
            verify_gdn_batch: !sp.no_verify_gdn_batch,
            tc_proj_min_n: tc_proj_min_n_of(k.ssm_tc_proj.as_deref()),
            fused_conv17: k.gdn_fused_conv17.as_deref() != Some("0"),
            half_reg: !k.no_gdn_half_reg,
            smem_stage: k.gdn_smem_stage,
            m128_min_m: if k.no_ssm_m128 {
                None
            } else {
                Some(SSM_M128_MIN_M)
            },
        }
    }

    /// 2026-10-17: What [`Self::from_settings`] returns at the declared defaults.
    pub const fn declared() -> Self {
        Self {
            bf16_weights: false,
            fused_verify: false,
            wy2_resident: true,
            wy3_resident: true,
            force_w8a8: false,
            hc_prefill_prof: false,
            hc_decode_prof: false,
            out_w8a16_only: false,
            fla: true,
            ffn_prefill_min_n: SSM_FFN_PREFILL_MIN_N,
            ffn_prefill: true,
            batched_ba_gates: true,
            batched_gdn_norm: true,
            qkvz_nvfp4_decode: true,
            verify_outproj_tgemm: true,
            verify_gdn_batch: true,
            tc_proj_min_n: Some(SSM_TC_PROJ_MIN_N),
            fused_conv17: true,
            half_reg: true,
            smem_stage: false,
            m128_min_m: Some(SSM_M128_MIN_M),
        }
    }
}

/// 2026-10-17: [`SsmLevers::declared`], so `ModelLevers` can keep deriving `Default`.
impl Default for SsmLevers {
    fn default() -> Self {
        Self::declared()
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
