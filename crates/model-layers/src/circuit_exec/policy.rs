// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The live policy a plan is fused under: every setting the rules read, taken from
//! the switch the legacy dispatch reads for it (the sources INSTANCES.toml cites), plus the
//! switches no rule models, which the executor refuses when they are away from their default.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Each setting is read from the one place legacy dispatch reads it; nothing here restates a
//!   default.
//! - A switch that changes legacy decode but that no rule reads is reported by
//!   [`unmodelled_switches`], never silently ignored.
//! - The environment switches of the multi-sequence dispatch ([`MULTI_SEQ_ENV_SWITCHES`]) are
//!   refused when present at all, whatever their value: over-refusal, never a misrun. Their
//!   readers parse them; this list only names them.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_circuit::Policy;
use metrale_config::activation_quantization::Ladder;
use metrale_config::{ActQuantFormat, ProjFamily};

use crate::layers::ops::ModelLevers;
use crate::layers::{RowTiers, row_tiers};

fn on_off(b: bool) -> String {
    if b { "on" } else { "off" }.to_string()
}

/// 2026-09-28: The spelling the rules use for a KV-cache dtype; an error for one the circuit
/// has no rule for.
pub fn kv_dtype_name(d: KvCacheDtype) -> Result<&'static str> {
    Ok(match d {
        KvCacheDtype::Bf16 => "bf16",
        KvCacheDtype::Fp8 => "fp8",
        other => bail!("KV-cache dtype {other:?} has no circuit spelling"),
    })
}

/// 2026-09-28: The policy of this process: `kv_cache_dtype` and `lm_head_dtype` come from the
/// model being built, the rest from the process-wide switches.
/// 2026-10-03: Fails for a fixed `ffn` activation format the rules do not plan.
pub fn live_policy(
    levers: &ModelLevers,
    kv_cache_dtype: &str,
    lm_head_dtype: &str,
) -> Result<Policy> {
    let tiers = match row_tiers() {
        RowTiers::ByRows => "by_rows",
        RowTiers::Exact => "exact",
        RowTiers::Canonical => "canonical",
    };
    let ssm_h = if crate::layers::qwen3_ssm::ssm_h_fp16_enabled() {
        "f16"
    } else {
        "f32"
    };
    let settings = BTreeMap::from([
        ("row_tiers".to_string(), tiers.to_string()),
        ("kv_cache_dtype".to_string(), kv_cache_dtype.to_string()),
        ("lm_head_dtype".to_string(), lm_head_dtype.to_string()),
        ("ssm_h_dtype".to_string(), ssm_h.to_string()),
        // 2026-09-30: The GDN flag cell, which a pinned `--ssm-batched-recurrent` sets and the
        // multi-sequence dispatch reads; the target default alone misses the pin.
        (
            "ssm_batched_recurrent".to_string(),
            on_off(crate::layers::qwen3_ssm::ssm_batched_recurrent_enabled()),
        ),
        (
            "ssm_ba_gates_hopper".to_string(),
            on_off(crate::layers::ops::ssm_ba_gates_hopper_enabled()),
        ),
        ("gemv_sw".to_string(), on_off(levers.gemv_sw)),
        (
            "w4a16_tc".to_string(),
            on_off(crate::layers::ops::gemv_tc::tc_enabled()),
        ),
        (
            "decode_split_silu".to_string(),
            on_off(levers.decode_split_silu),
        ),
        // 2026-09-30: No fused norm-quantize launch exists in the engine.
        ("rms_norm_act_quant".to_string(), "off".to_string()),
        // 2026-10-03: The GDN prefill arm after a prefix-cache restore; every plan but a prefill
        // one is planned off, and the prefill build plans each arm (`prefill::EXACT_REPLAY`).
        (super::prefill::EXACT_REPLAY.to_string(), "off".to_string()),
        // 2026-10-03: The exact MTP verify chain (`--exact-verify`, or a fixed GDN activation
        // format): the verify's GDN conv, recurrence and norm run the decode's FP32 chain per
        // row (`gdn_flags.rs` `verify_exact_enabled`).
        (
            VERIFY_EXACT.to_string(),
            on_off(crate::layers::qwen3_ssm::verify_exact_enabled()),
        ),
        (
            FFN_ACT_FIXED.to_string(),
            ffn_act_fixed(crate::layers::activation_quantization().ladder(ProjFamily::Ffn))?
                .to_string(),
        ),
    ]);
    Ok(Policy {
        opt_in_levers: Default::default(),
        settings,
    })
}

/// 2026-10-03: The policy setting of the dense FFN's fixed-activation path
/// (`DenseFfnLayer::forward_fixed`): `off`, or the format the `ffn` family runs at every row count.
pub const FFN_ACT_FIXED: &str = "ffn_act_fixed";

/// 2026-10-03: `ffn_act_fixed` for the `ffn` family's ladder: `off` when every rung is
/// `adaptive` (today's arms), `declared` when every rung is `declared` (the rules plan that
/// path, `ffn_fixed_*` in FUSIONS.toml). Any other ladder, a fixed format the rules do not plan
/// or a ladder fixed at some row counts only, is refused rather than planned as another path.
pub fn ffn_act_fixed(ladder: &Ladder) -> Result<&'static str> {
    let formats: Vec<ActQuantFormat> = ladder.rungs().iter().map(|r| r.format).collect();
    if formats.iter().all(|f| *f == ActQuantFormat::Adaptive) {
        return Ok("off");
    }
    if formats.iter().all(|f| *f == ActQuantFormat::Declared) {
        return Ok("declared");
    }
    bail!(
        "the circuit plans the ffn family adaptive or declared at every row count, not {}",
        formats
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// 2026-09-28: Environment switches of the multi-sequence decode and (2026-09-29) MTP verify
/// dispatch whose defaults the rules' row bands and arms encode, with the file that reads
/// each.
pub const MULTI_SEQ_ENV_SWITCHES: [(&str, &str); 16] = [
    (
        "METRALE_GDN_FUSED_VERIFY",
        "qwen3_ssm/trait_decode_batched_conv_gdn.rs",
    ),
    (
        "METRALE_NO_BATCHED_BA_GATES",
        "qwen3_ssm/trait_decode_batched.rs",
    ),
    (
        "METRALE_NO_BATCHED_GDN_NORM",
        "qwen3_ssm/trait_decode_batched.rs",
    ),
    (
        "METRALE_SSM_TC_PROJ",
        "qwen3_ssm/trait_decode_multi_seq/ssm_batched.rs",
    ),
    ("METRALE_NO_SSM_M128", "qwen3_ssm/gdn_flags.rs"),
    (
        "METRALE_SSM_FFN_PREFILL_MIN_N",
        "qwen3_ssm/trait_decode_multi_seq.rs",
    ),
    (
        "METRALE_NO_SSM_FFN_PREFILL",
        "qwen3_ssm/trait_decode_multi_seq.rs",
    ),
    (
        "METRALE_NO_QK_NORM_STRIDED",
        "qwen3_attention/trait_impl/multi_seq/qkv.rs",
    ),
    (
        "METRALE_NO_ROPE_STRIDED",
        "qwen3_attention/trait_impl/multi_seq/attn.rs",
    ),
    (
        "METRALE_NO_ATTN_BATCH_CACHE_WRITE",
        "qwen3_attention/trait_impl/multi_seq/attn.rs",
    ),
    ("METRALE_W4A16_K64_MIN_K", "layers/mod.rs (w4a16_k64_min_k)"),
    ("METRALE_NO_W4A16_K64", "layers/mod.rs (w4a16_k64_min_k)"),
    ("METRALE_NO_MMQ_SMALL_TILE", "layers/dense_ffn.rs"),
    ("METRALE_NO_MMQ_TILE64", "layers/dense_ffn.rs"),
    ("METRALE_W4A16_TC_WIDE", "ops/gemv_tc.rs"),
    // 2026-09-30: A diagnostic that runs the W4A16 twin of every W4A4 projection.
    ("METRALE_W4A4_PROJ_AUDIT", "ops/w4a4_proj.rs"),
];

/// 2026-10-03: The policy setting of the exact MTP verify chain, `on` or `off`.
pub const VERIFY_EXACT: &str = "gdn_verify_exact";

/// 2026-09-28: Switches that change legacy decode which no rule reads, when they are set.
pub fn unmodelled_switches(levers: &ModelLevers) -> Vec<String> {
    let mut out: Vec<String> = MULTI_SEQ_ENV_SWITCHES
        .iter()
        .filter(|(var, _)| std::env::var_os(var).is_some())
        .map(|(var, reader)| format!("{var} (read in {reader})"))
        .collect();
    if !levers.ffn_small_m {
        out.push("the small-M projection GEMMs off (METRALE_FFN_SMALLM)".to_string());
    }
    if levers.ssm_ms_profile || levers.ssm_detail_profile || levers.ms_profile || levers.conc_hsd {
        out.push(
            "multi-sequence profiling or hidden dumps (METRALE_MS_PROFILE, METRALE_CONC_HSD, \
             METRALE_SSM_*_PROFILE)"
                .to_string(),
        );
    }
    if crate::layers::qwen3_ssm::gdn_fused_norm_enabled() {
        out.push("fused GDN output norm (METRALE_GDN_FUSED_NORM)".to_string());
    }
    if levers.decode_ffn_via_gemm {
        out.push("dense FFN decode through the GEMM (METRALE_DECODE_FFN_VIA_GEMM)".to_string());
    }
    if levers.fp32_routing {
        out.push("FP32 MoE routing (METRALE_FP32_ROUTING)".to_string());
    }
    if levers.ssm_save_dump {
        out.push("SSM save dump (METRALE_SSM_SAVE_DUMP)".to_string());
    }
    out
}

#[cfg(test)]
#[path = "policy_tests.rs"]
mod tests;
