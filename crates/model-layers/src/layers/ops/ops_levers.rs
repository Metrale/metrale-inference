// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The launcher-level levers of `layers::ops`, resolved once from the startup
//! settings ([`OpsLevers::from_settings`]) and carried on [`super::ModelLevers::ops`]. A launcher
//! takes the field it needs as an argument; nothing here or in the launchers reads the process
//! environment.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - `from_settings(&defaults)` equals [`OpsLevers::declared`]
//!   (`the_declared_defaults_are_the_old_unset_values`).
//! - `METRALE_LORA_GEMV_MAX_M` keeps its reader's parse: an unsigned row count, and the declared
//!   48 when unset or unparseable.
//! - `METRALE_CUBLAS_SCALE_LAYOUT` keeps its reader's text predicate: exactly `rowmajor` turns
//!   the k-major layout off.
//! - `from_settings` refuses exactly what [`W4a4Levers::from_settings`] refuses.
//! - `METRALE_NO_W4A16_TC`, `METRALE_W4A16_TC_WIDE` and `METRALE_NO_MTP_TC` keep their readers'
//!   text predicate: a non-empty value is set, an empty one reads as unset ([`unset_or_empty`]).

use super::gemv_tc::W4a16TcPlan;
use super::hyper_connection_lowrank::HcLevers;
use super::lora_delta::LoraLevers;
use super::w4a4_proj::W4a4Levers;
use super::w8a16_plan::W8a16Plan;
use anyhow::Result;
use metrale_settings::InferenceServerSettings;

/// 2026-10-17: The rows at or below which a LoRA delta runs row-wise GEMVs when
/// `METRALE_LORA_GEMV_MAX_M` is unset or unparseable.
const LORA_GEMV_MAX_M: u32 = 48;

/// 2026-10-17: The `layers::ops` launcher levers. Each field names its variable; the launcher
/// that takes it documents what it changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpsLevers {
    /// 2026-09-27: On unless `METRALE_NO_FP8_GEMM_PIPE` is present at any value
    /// (`gemm_quant::fp8_gemm_t_blockscaled`).
    pub fp8_gemm_pipe: bool,
    /// 2026-09-25: `METRALE_GEMM_SHAPE_LOG` (`dispatch_helpers::log_gemm_shape`).
    pub gemm_shape_log: bool,
    /// 2026-09-25: On unless `METRALE_RMS_NORM_WARP_ROW` is switched off
    /// (`norm::rms_norm_short_row_eligible`).
    pub rms_norm_warp_row: bool,
    /// 2026-09-27: On unless `METRALE_NO_ATTN_FA128` is present at any value
    /// (`prefill_attn_fa128::AttnFa128Kernels::resolve`).
    pub attn_fa128: bool,
    /// 2026-09-25: `METRALE_LORA_NO_FFN`: the dense-FFN and GDN out_proj deltas are skipped
    /// (`dense_ffn_overlays.rs`, `qwen3_ssm/lora.rs`).
    pub lora_no_ffn: bool,
    /// 2026-09-25: The levers `lora_delta::apply_lora_delta` runs under, carried by each
    /// adapter's `LoraKernels`.
    pub lora: LoraLevers,
    /// 2026-09-25: The low-rank hyper-connection collapse paths (`hyper_connection_lowrank.rs`).
    pub hc: HcLevers,
    /// 2026-09-25: On unless `METRALE_NO_W8A8_DECODE_PROJ` is present at any value
    /// (`dispatch_proj_decode::decode_w8a8_selected`).
    pub w8a8_decode_proj: bool,
    /// 2026-09-25: `METRALE_GGUF_NATIVE_Q2_MMQ`: packed Q2 prefill weights run through
    /// `q2_0_mmq_gemm` where its kernels resolved (`q2_0_mmq.rs`).
    pub native_q2_mmq: bool,
    /// 2026-09-25: On unless `METRALE_ATTN_512_TC` is switched off
    /// (`wide_prefill::wide_prefill_kernel`).
    pub attn_512_tc: bool,
    /// 2026-09-25: On unless `METRALE_CONV1D_TP` is switched off
    /// (`ssm_mamba::conv1d_update_prefill`).
    pub conv1d_tp: bool,
    /// 2026-09-28: On unless `METRALE_FP8_LDMAB` is switched off
    /// (`gemm_fp8_prefill::fp8_gemm_n128`).
    pub fp8_ldmab: bool,
    /// 2026-09-25: On unless `METRALE_CUBLAS_SCALE_LAYOUT` is exactly `rowmajor`: the VEC128
    /// activation scales go to the cuBLASLt block-scaled arm as `[K/128, ceil16(M)]`, tokens
    /// contiguous, the layout `metrale_gpu_runtime::cublaslt::scale_layout` quotes from the
    /// cuBLAS manual. Off passes the quantizer's `[M, K/128]` as is, a control arm
    /// (`dispatch_proj::cublas_fp8_proj_prequant`, `DecodeW8a8Scratch::kmajor`).
    pub cublas_kmajor: bool,
    /// 2026-10-17: The W4A4 projection levers (`w4a4_proj::prepare`).
    pub w4a4: W4a4Levers,
    /// 2026-10-17: The W8A16 tile levers (`w8a16_plan.rs`).
    pub w8a16: W8a16Plan,
    /// 2026-10-17: The W4A16 tensor-core levers (`gemv_tc.rs`).
    pub w4a16_tc: W4a16TcPlan,
    /// 2026-09-25: On unless `METRALE_NO_BATCH8_RT` is switched on: the width-8 W4A16 tier
    /// resolves `w4a16_gemv_batch8_rt2` where the target has it (`gemv_tier::batch8_kernel`).
    pub batch8_rt: bool,
    /// 2026-09-25: On unless `METRALE_NO_GEMV_EXACT_M_TIERS` is present at any value: the
    /// exact-M W4A16 tiers 5/6/7 take part in the tier decision (`w4a16_gemv_tiers.rs`).
    pub gemv_exact_m_tiers: bool,
    /// 2026-09-25: On unless `METRALE_NO_MTP_TC` holds a non-empty value, `0` included: the MTP
    /// drafter's BF16 small-M GEMMs run on the tensor-core entries (`dense_gemv_tc.rs`).
    pub mtp_tc: bool,
}

/// 2026-09-25: The rule of the tensor-core kill switches and the widening: a variable that is
/// unset or empty reads as unset.
fn unset_or_empty(v: Option<&str>) -> bool {
    v.is_none_or(str::is_empty)
}

impl OpsLevers {
    /// 2026-10-17: The levers of the settings resolved once at startup, refusing what
    /// [`W4a4Levers::from_settings`] refuses.
    pub fn from_settings(s: &InferenceServerSettings) -> Result<Self> {
        let (k, t) = (&s.kernels, &s.telemetry);
        Ok(Self {
            fp8_gemm_pipe: !k.no_fp8_gemm_pipe,
            gemm_shape_log: t.gemm_shape_log,
            rms_norm_warp_row: k.rms_norm_warp_row,
            attn_fa128: !k.no_attn_fa128,
            lora_no_ffn: k.lora_no_ffn,
            lora: LoraLevers {
                no_apply: s.model.lora_no_apply,
                gemv_max_m: k
                    .lora_gemv_max_m
                    .as_deref()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(LORA_GEMV_MAX_M),
            },
            hc: HcLevers {
                gemm_disabled: k.qwen4exp_no_hc_gemm,
                decode_split_forced: k.hc_decode_split,
            },
            w8a8_decode_proj: !k.no_w8a8_decode_proj,
            native_q2_mmq: k.gguf_native_q2_mmq,
            attn_512_tc: k.attn_512_tc,
            conv1d_tp: k.conv1d_tp,
            fp8_ldmab: k.fp8_ldmab,
            cublas_kmajor: s.runtime.cublas_scale_layout.as_deref() != Some("rowmajor"),
            w4a4: W4a4Levers::from_settings(s)?,
            w8a16: W8a16Plan {
                pipe128: !k.no_w8a16_pipe128,
                tc_rows: !k.no_w8a16_tc_rows,
            },
            w4a16_tc: {
                let tc = unset_or_empty(k.no_w4a16_tc.as_deref());
                W4a16TcPlan {
                    tc,
                    wide: tc && !unset_or_empty(k.w4a16_tc_wide.as_deref()),
                }
            },
            mtp_tc: unset_or_empty(s.speculative.no_mtp_tc.as_deref()),
            batch8_rt: !s.scheduler.no_batch8_rt,
            gemv_exact_m_tiers: !s.storage.no_gemv_exact_m_tiers,
        })
    }

    /// 2026-10-17: The plan `gemm_quant::fp8_gemm_t_blockscaled` launches under.
    pub fn fp8_blockscaled(&self) -> super::Fp8BlockscaledPlan {
        super::Fp8BlockscaledPlan {
            pipe: self.fp8_gemm_pipe,
            shape_log: self.gemm_shape_log,
        }
    }

    /// 2026-10-17: What [`Self::from_settings`] returns at the declared defaults: every
    /// opt-out on, every opt-in off, the row-wise LoRA bound at 48.
    pub const fn declared() -> Self {
        Self {
            fp8_gemm_pipe: true,
            gemm_shape_log: false,
            rms_norm_warp_row: true,
            attn_fa128: true,
            lora_no_ffn: false,
            lora: LoraLevers {
                no_apply: false,
                gemv_max_m: LORA_GEMV_MAX_M,
            },
            hc: HcLevers {
                gemm_disabled: false,
                decode_split_forced: false,
            },
            w8a8_decode_proj: true,
            native_q2_mmq: false,
            attn_512_tc: true,
            conv1d_tp: true,
            fp8_ldmab: true,
            cublas_kmajor: true,
            w4a4: W4a4Levers::declared(),
            w8a16: W8a16Plan::declared(),
            w4a16_tc: W4a16TcPlan::declared(),
            mtp_tc: true,
            batch8_rt: true,
            gemv_exact_m_tiers: true,
        }
    }
}

/// 2026-10-17: [`OpsLevers::declared`], so `ModelLevers` can keep deriving `Default`.
impl Default for OpsLevers {
    fn default() -> Self {
        Self::declared()
    }
}

#[cfg(test)]
#[path = "ops_levers_tests.rs"]
mod tests;
