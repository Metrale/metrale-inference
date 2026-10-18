// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The dispatch and diagnostic levers of the full-attention layer, resolved once
//! from the startup settings ([`AttnLevers::from_settings`]) and carried on `ModelLevers::attn`.
//! `init` copies them onto the layer (`Qwen3AttentionLayer::levers`), so the paths without a
//! forward context (the split-K trace, the circuit binding) read the same value as the forward
//! paths. `op_dump` reads its three text levers from `ForwardContext::settings`.
//!
//! Owner: model-layers (attention).
//! Invariants:
//! - `from_settings(&defaults)` equals [`AttnLevers::declared`]
//!   (`the_declared_defaults_are_the_old_unset_values`).
//! - The text levers keep their readers' predicate: `METRALE_ATTN_NCOL_WIDTH` picks four columns
//!   at exactly `4`, and `METRALE_ATTN_PREFILL_FUSED_QROPE` is on at exactly `1`.
//! - `multi_seq/attn.rs` keeps its two readers (`METRALE_NO_ATTN_BATCH_CACHE_WRITE`,
//!   `METRALE_NO_ROPE_STRIDED`) and `multi_seq/qkv.rs` the two above its fallback loop
//!   (`METRALE_BF16_QKV_BATCHM`, `METRALE_NO_FUSED_QKV`): kernel-rule citations pin those
//!   files' line numbers.

use metrale_settings::InferenceServerSettings;

use super::attn_ncol_gemv::NcolWidth;

/// 2026-10-17: The attention layer's levers. Each field names its variable and the reader it
/// replaced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnLevers {
    /// 2026-09-25: `METRALE_DIAG_V4_ALL_LAYERS`: the DeepSeek-V4 and mHC diagnostics on every
    /// layer (`decode/attention_forward_v4.rs`, `prefill/cache_skip_v4.rs`, the `hc.rs` paths).
    pub diag_v4_all_layers: bool,
    /// 2026-09-25: `METRALE_MLA_HSD`: the layer-0 MLA output scan (`multi_seq/mla.rs`).
    pub mla_hsd: bool,
    /// 2026-09-25: On unless `METRALE_NO_FP8_QKV_BATCH` is present at any value
    /// (`multi_seq/qkv_fp8_batch.rs`).
    pub fp8_qkv_batch: bool,
    /// 2026-09-25: `METRALE_ATTN_QKV_W8A16_ONLY` is present at any value
    /// (`prefill_qkv_w8a8.rs`).
    pub qkv_w8a16_only: bool,
    /// 2026-09-25: `METRALE_PREFILL_HOST_TIMING`: the host timers of the prefill attention and
    /// FFN halves (`prefill/cache_skip.rs`, `trait_impl/prefill_inner.rs`).
    pub prefill_host_timing: bool,
    /// 2026-09-25: The N-column tier's width, four at exactly `METRALE_ATTN_NCOL_WIDTH=4`, else
    /// two (`attn_ncol_gemv.rs`).
    pub ncol_width: NcolWidth,
    /// 2026-09-25: `METRALE_ATTN_DBG` is present at any value: the split-K trace
    /// (`decode/splitk_dispatch.rs`).
    pub attn_dbg: bool,

    /// 2026-09-25: On unless `METRALE_NO_QK_NORM_STRIDED` is switched on (`multi_seq/qkv.rs`).
    pub qk_norm_strided: bool,
    /// 2026-09-25: `METRALE_ATTN_PREFILL_T_PIPE` (`prefill/cache_skip_qkv.rs`).
    pub prefill_t_pipe: bool,
    /// 2026-09-25: `METRALE_ATTN_PREFILL_FUSED_QROPE` is exactly `1` (`prefill/cache_skip.rs`).
    pub prefill_fused_qrope: bool,
    /// 2026-09-25: `METRALE_FUSED_KV` (`prefill/paged.rs`).
    pub fused_kv: bool,
    /// 2026-09-25: `METRALE_FLASHINFER_PREFILL` (`prefill/paged.rs`).
    pub flashinfer_prefill: bool,
}

impl AttnLevers {
    /// 2026-10-17: The levers of the settings resolved once at startup.
    pub fn from_settings(s: &InferenceServerSettings) -> Self {
        let (k, sc, c, t) = (&s.kernels, &s.scheduler, &s.cache, &s.telemetry);
        let is_one = |v: &Option<String>| v.as_deref() == Some("1");
        Self {
            diag_v4_all_layers: t.diag_v4_all_layers,
            mla_hsd: k.mla_hsd,
            fp8_qkv_batch: !sc.no_fp8_qkv_batch,
            qkv_w8a16_only: c.attn_qkv_w8a16_only,
            prefill_host_timing: t.prefill_host_timing,
            ncol_width: match k.attn_ncol_width.as_deref() {
                Some("4") => NcolWidth::Four,
                _ => NcolWidth::Two,
            },
            attn_dbg: k.attn_dbg,

            qk_norm_strided: !k.no_qk_norm_strided,
            prefill_t_pipe: sc.attn_prefill_t_pipe,
            prefill_fused_qrope: is_one(&sc.attn_prefill_fused_qrope),
            fused_kv: c.fused_kv,
            flashinfer_prefill: sc.flashinfer_prefill,
        }
    }

    /// 2026-10-17: What [`Self::from_settings`] returns at the declared defaults.
    pub const fn declared() -> Self {
        Self {
            diag_v4_all_layers: false,
            mla_hsd: false,
            fp8_qkv_batch: true,
            qkv_w8a16_only: false,
            prefill_host_timing: false,
            ncol_width: NcolWidth::Two,
            attn_dbg: false,

            qk_norm_strided: true,
            prefill_t_pipe: false,
            prefill_fused_qrope: false,
            fused_kv: false,
            flashinfer_prefill: false,
        }
    }
}

/// 2026-10-17: [`AttnLevers::declared`], so `ModelLevers` can keep deriving `Default`.
impl Default for AttnLevers {
    fn default() -> Self {
        Self::declared()
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
