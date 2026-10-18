// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Resolution equivalence for every lever `ModelLevers::from_settings` reads: each is
//! probed (`lever_equivalence::PROBES`) through a frozen copy of the env reader it replaced and
//! through the settings path, and the whole `ModelLevers` is compared.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - `old_from_values` is the pre-migration `from_values`, frozen; it is never edited to match.
//! - `CASES` names each variable `old_from_values` reads, with the reader it used.

use super::*;
use crate::lever_equivalence::{Old, Outcome, check};

/// 2026-10-17: The pre-migration resolver (`model_levers_resolve.rs` at the settings base), with
/// `shadow_topk` 0, the default drafter context and no pre-rotated weights, the three inputs that
/// came from elsewhere. `value` holds at most the one probed variable, so `present` is
/// `value(..).is_some()`.
fn old_from_values(value: impl Fn(&str) -> Option<String>) -> ModelLevers {
    fn opt_in(value: Option<&str>) -> bool {
        value == Some("1")
    }
    fn opt_out(value: Option<&str>) -> bool {
        value != Some("0")
    }
    fn opt_in_truthy(value: Option<&str>) -> bool {
        value.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    }
    fn opt_in_truthy_exact(value: Option<&str>) -> bool {
        matches!(value, Some("1") | Some("true"))
    }
    let present = |name: &str| value(name).is_some();
    ModelLevers {
        max_decode_seqs: 1,
        shadow_topk: 0,
        kv_poison: opt_in(value("METRALE_KV_POISON").as_deref()),
        drafter: crate::drafter_context::resolve(false, None),
        // 2026-10-17: A field added after the freeze; the launchers read these variables
        // themselves then, and `ops_levers_tests.rs` probes them against those readers.
        ops: crate::layers::ops::OpsLevers::declared(),
        // 2026-10-17: Likewise, probed in `qwen3_ssm/levers_tests.rs`, `moe/levers_tests.rs`,
        // `qwen3_attention/levers_tests.rs`, `mtp_head/levers_tests.rs` and
        // `dense_ffn_levers_tests.rs`.
        ssm: crate::layers::qwen3_ssm::SsmLevers::declared(),
        moe: crate::layers::moe::MoeLevers::declared(),
        attn: crate::layers::qwen3_attention::AttnLevers::declared(),
        mtp: crate::layers::mtp_head::MtpLevers::declared(),
        ffn: crate::layers::dense_ffn::DenseFfnLevers::declared(),
        gdn_regresident: value("METRALE_NO_GDN_REGRESIDENT").as_deref() != Some("1"),
        gdn_batched_fla: opt_in(value("METRALE_GDN_BATCHED_FLA").as_deref()),
        gdn_wy17: opt_out(value("METRALE_GDN_WY17").as_deref()),
        gdn_wyn: opt_out(value("METRALE_GDN_WYN").as_deref()),
        ffn_small_m: opt_out(value("METRALE_FFN_SMALLM").as_deref()),
        gemv_sw: crate::layers::ops::gemv_sw::gemv_sw_from(value("METRALE_NO_GEMV_SW").as_deref()),
        decode_ffn_via_gemm: opt_in(value("METRALE_DECODE_FFN_VIA_GEMM").as_deref()),
        holo_moe_down_fp4: opt_in_truthy(value("METRALE_HOLO_MOE_DOWN_FP4").as_deref()),
        holo_moe_gateup_fp4: opt_in_truthy(value("METRALE_HOLO_MOE_GATEUP_FP4").as_deref()),
        moe_union_stats: opt_in(value("METRALE_MOE_UNION_STATS").as_deref()),
        fp32_routing: opt_in(value("METRALE_FP32_ROUTING").as_deref()),
        fp32_gate: opt_in(value("METRALE_FP32_GATE").as_deref()),
        frankenstein_decode_via_prefill: opt_in(
            value("METRALE_FRANKENSTEIN_DECODE_VIA_PREFILL").as_deref(),
        ),
        k2_diag: opt_in(value("METRALE_K2_DIAG").as_deref()),
        dflash_debug_dump_full: opt_in(value("METRALE_DFLASH_DEBUG_DUMP_FULL").as_deref()),
        mtp_debug_norms: opt_in(value("METRALE_MTP_DEBUG_NORMS").as_deref()),
        mtp_chain_postnorm: opt_in(value("METRALE_MTP_CHAIN_POSTNORM").as_deref()),
        mtp_target_postnorm: opt_in(value("METRALE_MTP_TARGET_POSTNORM").as_deref()),
        mtp_kv_exact: opt_in(value("METRALE_MTP_KV_EXACT").as_deref()),
        mtp_accepted_rows: opt_in(value("METRALE_MTP_ACCEPTED_ROWS").as_deref()),
        mtp_adopt_capture: opt_in(value("METRALE_MTP_ADOPT_CAPTURE").as_deref()),
        mtp_zero_pos0: opt_in(value("METRALE_MTP_ZERO_POS0").as_deref()),
        one_row_prefill: opt_in(value("METRALE_ONE_ROW_PREFILL").as_deref()),
        one_row_tail_decode: opt_in(value("METRALE_ONE_ROW_TAIL_DECODE").as_deref()),
        seq_alloc_async: opt_in(value("METRALE_SEQ_ALLOC_ASYNC").as_deref()),
        glm_keep_slot_graphs: opt_in(value("METRALE_GLM_KEEP_SLOT_GRAPHS").as_deref()),
        glm_mtp_state_pool: opt_in(value("METRALE_GLM_MTP_STATE_POOL").as_deref()),
        moe_fp8_grouped_decode_target: opt_in(value("METRALE_FP8_MOE_GROUPED_DECODE").as_deref()),
        fp8_attn_m32: opt_in(value("METRALE_FP8_ATTN_M32").as_deref()),
        // 2026-09-25: The target's declaration (`kernels/<hw>/HARDWARE.toml` `[defaults]
        // decode_split_silu`; undeclared is on, `crates/kernels/build_defaults.rs`) arrives as
        // `default_split_silu`. The presence of `METRALE_NO_DECODE_SPLIT_SILU`, any value,
        // turns it off (`resolve_toggle`'s `legacy_off`).
        decode_split_silu: crate::layers::ops::target_defaults::resolve_toggle(
            true,
            None,
            present("METRALE_NO_DECODE_SPLIT_SILU"),
        )
        .value,
        bf16_tc_prefill: present("METRALE_BF16_TC_PREFILL"),
        fp8_m64_prefill: present("METRALE_FP8_M64_PREFILL"),
        int8_prefill: present("METRALE_INT8_PREFILL"),
        int8_faith5: present("METRALE_INT8_FAITH5"),
        ffn_nvfp4_mmq: !present("METRALE_NO_FFN_NVFP4_MMQ"),
        ffn_nvfp4_mmq_down: !present("METRALE_NO_FFN_NVFP4_MMQ_DOWN"),
        ffn_mmq: present("METRALE_FFN_MMQ"),
        ffn_mmq_down_q4k: present("METRALE_FFN_MMQ_DOWN_Q4K"),
        fp4_prefill: present("METRALE_FP4_PREFILL"),
        prefill_v2: !present("METRALE_DISABLE_PREFILL_V2"),
        moe_grouped_cutlass: opt_in(value("METRALE_HOLO_MOE_GROUPED_CUTLASS").as_deref()),
        moe_grouped_down: opt_in(value("METRALE_HOLO_MOE_GROUPED_DOWN").as_deref()),
        moe_prefill_exact_tiles: match value("METRALE_MOE_PREFILL_EXACT_TILES").as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        },
        moe_prefill_max_load_factor: value("METRALE_MOE_PREFILL_MAX_LOAD_FACTOR")
            .as_deref()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&factor| factor > 0),
        moe_prefill_zero: opt_in(value("METRALE_MOE_PREFILL_ZERO").as_deref()),
        moe_prefill_fp8_down: opt_in(value("METRALE_MOE_PREFILL_FP8_DOWN").as_deref()),
        ssm_w4a4: !present("METRALE_NO_SSM_W4A4"),
        ssd: !present("METRALE_NO_SSD"),
        ssm_persistent: !present("METRALE_NO_SSM_PERSISTENT"),
        moe_zero_intermediates: !present("METRALE_MOE_NO_ZERO_INTERMEDIATES"),
        moe_max_m_tiles_estimate: present("METRALE_MOE_MAX_M_TILES_ESTIMATE"),
        moe_w4a4: present("METRALE_MOE_W4A4"),
        shared_w4a4: !present("METRALE_NO_SHARED_W4A4"),
        shared_w4a4_down: present("METRALE_SHARED_W4A4_DOWN"),
        dflash_contig_attn: opt_in(value("METRALE_DFLASH_CONTIG_ATTN").as_deref()),
        lora_eager: opt_in_truthy(value("METRALE_LORA_EAGER").as_deref()),
        lora_rotate: opt_in_truthy(value("METRALE_LORA_ROTATE").as_deref()),
        k4_diag: opt_in(value("METRALE_K4_DIAG").as_deref()),
        gemma4_diag: opt_in_truthy(value("METRALE_DIAG_GEMMA4").as_deref()),
        mla_perseq_fallback: opt_in_truthy_exact(value("METRALE_MLA_PERSEQ_FALLBACK").as_deref()),
        hc_perseq_decode: opt_in(value("METRALE_HC_PERSEQ_DECODE").as_deref()),
        decode_batch_log: opt_in(value("METRALE_DECODE_BATCH_LOG").as_deref()),
        ms_profile: opt_in(value("METRALE_MS_PROFILE").as_deref()),
        conc_hsd: opt_in_truthy_exact(value("METRALE_CONC_HSD").as_deref()),
        ssm_save_dump: present("METRALE_SSM_SAVE_DUMP"),
        ep_graphs: opt_in_truthy_exact(value("METRALE_EP_GRAPHS").as_deref()),
        gdn_decode_graph: opt_in_truthy_exact(value("METRALE_GDN_DECODE_GRAPH").as_deref()),
        bf16_tc_proj: present("METRALE_BF16_TC_PROJ"),
        weight_pre_rotated: false,
        ssm_ms_profile: opt_in(value("METRALE_SSM_MS_PROFILE").as_deref()),
        ssm_detail_profile: opt_in(value("METRALE_SSM_DETAIL_PROFILE").as_deref()),
        ssm_gemv_batch4: opt_out(value("METRALE_SSM_GEMV_BATCH4").as_deref()),
        gdn_fused_conv: opt_in(value("METRALE_GDN_FUSED_CONV").as_deref()),
        moe_legacy_pertoken_decode: opt_in(value("METRALE_MOE_LEGACY_PERTOKEN_DECODE").as_deref()),
    }
}

/// 2026-10-17: Each variable `old_from_values` reads and the reader it used. A `NO_`/`DISABLE_`
/// variable whose field is the negation is listed with the reader of the variable itself.
const CASES: [(&str, Old); 74] = [
    ("METRALE_BF16_TC_PREFILL", Old::Presence),
    ("METRALE_BF16_TC_PROJ", Old::Presence),
    ("METRALE_CONC_HSD", Old::TruthyExact),
    ("METRALE_DECODE_BATCH_LOG", Old::OptIn),
    ("METRALE_DECODE_FFN_VIA_GEMM", Old::OptIn),
    ("METRALE_DFLASH_CONTIG_ATTN", Old::OptIn),
    ("METRALE_DFLASH_DEBUG_DUMP_FULL", Old::OptIn),
    ("METRALE_DIAG_GEMMA4", Old::Truthy),
    ("METRALE_DISABLE_PREFILL_V2", Old::Presence),
    ("METRALE_EP_GRAPHS", Old::TruthyExact),
    ("METRALE_FFN_MMQ", Old::Presence),
    ("METRALE_FFN_MMQ_DOWN_Q4K", Old::Presence),
    ("METRALE_FFN_SMALLM", Old::OptOut),
    ("METRALE_FP32_GATE", Old::OptIn),
    ("METRALE_FP32_ROUTING", Old::OptIn),
    ("METRALE_FP4_PREFILL", Old::Presence),
    ("METRALE_FP8_ATTN_M32", Old::OptIn),
    ("METRALE_FP8_M64_PREFILL", Old::Presence),
    ("METRALE_FP8_MOE_GROUPED_DECODE", Old::OptIn),
    ("METRALE_FRANKENSTEIN_DECODE_VIA_PREFILL", Old::OptIn),
    ("METRALE_GDN_BATCHED_FLA", Old::OptIn),
    ("METRALE_GDN_DECODE_GRAPH", Old::TruthyExact),
    ("METRALE_GDN_FUSED_CONV", Old::OptIn),
    ("METRALE_GDN_WY17", Old::OptOut),
    ("METRALE_GDN_WYN", Old::OptOut),
    ("METRALE_GLM_KEEP_SLOT_GRAPHS", Old::OptIn),
    ("METRALE_GLM_MTP_STATE_POOL", Old::OptIn),
    ("METRALE_HC_PERSEQ_DECODE", Old::OptIn),
    ("METRALE_HOLO_MOE_DOWN_FP4", Old::Truthy),
    ("METRALE_HOLO_MOE_GATEUP_FP4", Old::Truthy),
    ("METRALE_HOLO_MOE_GROUPED_CUTLASS", Old::OptIn),
    ("METRALE_HOLO_MOE_GROUPED_DOWN", Old::OptIn),
    ("METRALE_INT8_FAITH5", Old::Presence),
    ("METRALE_INT8_PREFILL", Old::Presence),
    ("METRALE_K2_DIAG", Old::OptIn),
    ("METRALE_K4_DIAG", Old::OptIn),
    ("METRALE_KV_POISON", Old::OptIn),
    ("METRALE_LORA_EAGER", Old::Truthy),
    ("METRALE_LORA_ROTATE", Old::Truthy),
    ("METRALE_MLA_PERSEQ_FALLBACK", Old::TruthyExact),
    ("METRALE_MOE_LEGACY_PERTOKEN_DECODE", Old::OptIn),
    ("METRALE_MOE_MAX_M_TILES_ESTIMATE", Old::Presence),
    ("METRALE_MOE_NO_ZERO_INTERMEDIATES", Old::Presence),
    ("METRALE_MOE_PREFILL_EXACT_TILES", Old::Verbatim),
    ("METRALE_MOE_PREFILL_FP8_DOWN", Old::OptIn),
    ("METRALE_MOE_PREFILL_MAX_LOAD_FACTOR", Old::Number),
    ("METRALE_MOE_PREFILL_ZERO", Old::OptIn),
    ("METRALE_MOE_UNION_STATS", Old::OptIn),
    ("METRALE_MOE_W4A4", Old::Presence),
    ("METRALE_MS_PROFILE", Old::OptIn),
    ("METRALE_MTP_ACCEPTED_ROWS", Old::OptIn),
    ("METRALE_MTP_ADOPT_CAPTURE", Old::OptIn),
    ("METRALE_MTP_CHAIN_POSTNORM", Old::OptIn),
    ("METRALE_MTP_DEBUG_NORMS", Old::OptIn),
    ("METRALE_MTP_KV_EXACT", Old::OptIn),
    ("METRALE_MTP_TARGET_POSTNORM", Old::OptIn),
    ("METRALE_MTP_ZERO_POS0", Old::OptIn),
    ("METRALE_NO_DECODE_SPLIT_SILU", Old::Presence),
    ("METRALE_NO_FFN_NVFP4_MMQ", Old::Presence),
    ("METRALE_NO_FFN_NVFP4_MMQ_DOWN", Old::Presence),
    ("METRALE_NO_GDN_REGRESIDENT", Old::OptIn),
    ("METRALE_NO_GEMV_SW", Old::Verbatim),
    ("METRALE_NO_SHARED_W4A4", Old::Presence),
    ("METRALE_NO_SSD", Old::Presence),
    ("METRALE_NO_SSM_PERSISTENT", Old::Presence),
    ("METRALE_NO_SSM_W4A4", Old::Presence),
    ("METRALE_ONE_ROW_PREFILL", Old::OptIn),
    ("METRALE_ONE_ROW_TAIL_DECODE", Old::OptIn),
    ("METRALE_SEQ_ALLOC_ASYNC", Old::OptIn),
    ("METRALE_SHARED_W4A4_DOWN", Old::Presence),
    ("METRALE_SSM_DETAIL_PROFILE", Old::OptIn),
    ("METRALE_SSM_GEMV_BATCH4", Old::OptOut),
    ("METRALE_SSM_MS_PROFILE", Old::OptIn),
    ("METRALE_SSM_SAVE_DUMP", Old::Presence),
];

/// 2026-10-17: Every probe of every variable gives the outcome its settings rule predicts against
/// the frozen reader (`check` asserts it); the differences print for the behaviour-change list.
#[test]
fn every_model_lever_resolves_as_its_old_reader_or_as_a_listed_change() {
    let mut changes = Vec::new();
    for (env, kind) in CASES {
        changes.extend(check(
            env,
            kind,
            |probe| {
                old_from_values(|name| (name == env).then(|| probe.map(str::to_owned)).flatten())
            },
            |s| ModelLevers::from_settings(s, true, false).expect("one lever resolves"),
        ));
    }
    for (env, probe, outcome) in &changes {
        eprintln!("{env}={probe:?}: {outcome:?}");
    }
    // 2026-10-17: The garbage probe is refused by every switch and number row, so a run with no
    // rejection at all would mean the probes never reached the settings resolve.
    assert!(changes.iter().any(|&(_, _, o)| o == Outcome::Rejected));
}

/// 2026-10-17: Every settings row this resolver owns is probed above, so a lever added to the
/// resolver's rows cannot skip the equivalence check.
#[test]
fn the_cases_cover_every_row_this_resolver_owns() {
    for spec in metrale_settings::groups::by_name() {
        if spec.reader == "crates/model-layers/src/layers/ops/model_levers_resolve.rs" {
            assert!(
                CASES.iter().any(|&(env, _)| env == spec.env),
                "{} is read by this resolver but not probed",
                spec.env
            );
        }
    }
}
