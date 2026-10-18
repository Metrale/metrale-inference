// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `OpsLevers::from_settings` against the environment readers it replaced, frozen
//! here (`old_*`, unchanged but for taking the raw value as a parameter).
//!
//! Owner: model-layers ops.
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a `1`
//! or `0` switch now also reads the other switch spellings and refuses a value the switch rule
//! cannot read; the presence levers, the LoRA row bound and the scale layout read as before.

use super::*;
use crate::lever_equivalence::{Old, Outcome, PROBES, check, defaults, settings_from};

/// 2026-10-17: `lora_delta::lora_gemv_max_m`.
fn old_gemv_max_m(v: Option<&str>) -> u32 {
    v.and_then(|v| v.parse().ok()).unwrap_or(48)
}

/// 2026-10-17: The levers of `env` at `probe` and nothing else; `None` when the settings refuse
/// it.
fn new_at(env: &str, probe: &str) -> Option<OpsLevers> {
    settings_from(&[(env, probe)])
        .ok()
        .and_then(|s| OpsLevers::from_settings(&s).ok())
}

/// 2026-10-17: The levers of settings that hold one lever this file probes, which resolve.
fn ops(s: &metrale_settings::InferenceServerSettings) -> OpsLevers {
    OpsLevers::from_settings(s).expect("one non-W4A4 lever resolves")
}

type Field = (&'static str, fn(&OpsLevers) -> bool);

/// 2026-10-17: `gemm_quant::fp8_gemm_pipe_kernel`, `AttnFa128Kernels::resolve`,
/// `dispatch_proj_decode::w8a8_decode_proj_disabled`, `gemm_quant_w8a16::w8a16_pipe128_kernel`
/// `w8a16_tc_rows::w8a16_tc_rows_enabled` and `w4a16_gemv_tiers::exact_m_tiers_enabled`: off
/// when the variable is present at any value, `0` included.
#[test]
fn the_presence_opt_outs_read_as_before() {
    let cases: [Field; 6] = [
        ("METRALE_NO_FP8_GEMM_PIPE", |l| l.fp8_gemm_pipe),
        ("METRALE_NO_ATTN_FA128", |l| l.attn_fa128),
        ("METRALE_NO_W8A8_DECODE_PROJ", |l| l.w8a8_decode_proj),
        ("METRALE_NO_W8A16_PIPE128", |l| l.w8a16.pipe128),
        ("METRALE_NO_W8A16_TC_ROWS", |l| l.w8a16.tc_rows),
        ("METRALE_NO_GEMV_EXACT_M_TIERS", |l| l.gemv_exact_m_tiers),
    ];
    for (env, field) in cases {
        let changes = check(env, Old::Presence, |p| p.is_none(), |s| field(&ops(s)));
        assert!(changes.is_empty(), "{changes:?}");
        for probe in ["", " ", "no"] {
            assert_eq!(
                new_at(env, probe).map(|l| field(&l)),
                Some(false),
                "{probe:?}"
            );
        }
    }
}

/// 2026-10-17: The `== "1"` switches: the LoRA skips, the hyper-connection paths, the GEMM
/// shape log, `q2_0_mmq::native_q2_mmq_enabled`, `gemv_tier::batch8_kernel`'s classic arm and
/// the dense FFN's two MMQ tile kill switches (`dense_ffn.rs`'s `mmq_*_enabled`).
#[test]
fn the_opt_in_switches_read_as_before_except_the_switch_spellings() {
    let cases: [Field; 9] = [
        ("METRALE_GGUF_NATIVE_Q2_MMQ", |l| l.native_q2_mmq),
        ("METRALE_LORA_NO_APPLY", |l| l.lora.no_apply),
        ("METRALE_LORA_NO_FFN", |l| l.lora_no_ffn),
        ("METRALE_QWEN4EXP_NO_HC_GEMM", |l| l.hc.gemm_disabled),
        ("METRALE_HC_DECODE_SPLIT", |l| l.hc.decode_split_forced),
        ("METRALE_GEMM_SHAPE_LOG", |l| l.gemm_shape_log),
        ("METRALE_NO_BATCH8_RT", |l| !l.batch8_rt),
        ("METRALE_NO_MMQ_SMALL_TILE", |l| !l.mmq_small_tile),
        ("METRALE_NO_MMQ_TILE64", |l| !l.mmq_tile64),
    ];
    for (env, field) in cases {
        let changes = check(env, Old::OptIn, |p| p == Some("1"), |s| field(&ops(s)));
        assert_eq!(
            changes,
            [
                (env, Some("true"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ]
        );
    }
}

/// 2026-10-17: The `!= "0"` switches: `norm::rms_norm_short_row_eligible`,
/// `wide_prefill::wide_prefill_kernel`, `ssm_mamba::conv1d_update_prefill` and
/// `gemm_fp8_prefill::fp8_gemm_n128`.
#[test]
fn the_opt_out_switches_read_as_before_except_the_switch_spellings() {
    let cases: [Field; 4] = [
        ("METRALE_RMS_NORM_WARP_ROW", |l| l.rms_norm_warp_row),
        ("METRALE_ATTN_512_TC", |l| l.attn_512_tc),
        ("METRALE_CONV1D_TP", |l| l.conv1d_tp),
        ("METRALE_FP8_LDMAB", |l| l.fp8_ldmab),
    ];
    for (env, field) in cases {
        let changes = check(env, Old::OptOut, |p| p != Some("0"), |s| field(&ops(s)));
        assert_eq!(
            changes,
            [
                (env, Some("off"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ]
        );
    }
}

/// 2026-10-17: The row bound keeps its reader's parse: an untrimmed unsigned count, 48 otherwise.
#[test]
fn the_lora_row_bound_reads_as_before() {
    let env = "METRALE_LORA_GEMV_MAX_M";
    let changes = check(env, Old::Verbatim, old_gemv_max_m, |s| {
        ops(s).lora.gemv_max_m
    });
    assert!(changes.is_empty(), "{changes:?}");
    for probe in ["12", "0", "+7", " 12", "12 ", "-1", "1.5", "", "4294967296"] {
        assert_eq!(
            new_at(env, probe).map(|l| l.lora.gemv_max_m),
            Some(old_gemv_max_m(Some(probe))),
            "{probe:?}"
        );
    }
}

/// 2026-10-17: `dispatch_proj::cublas_scale_layout_kmajor`: k-major unless the variable is
/// exactly `rowmajor`.
fn old_kmajor(v: Option<&str>) -> bool {
    !matches!(v, Some("rowmajor"))
}

/// 2026-10-17: The scale layout keeps its reader's text predicate.
#[test]
fn the_cublas_scale_layout_reads_as_before() {
    let env = "METRALE_CUBLAS_SCALE_LAYOUT";
    let changes = check(env, Old::Verbatim, old_kmajor, |s| ops(s).cublas_kmajor);
    assert!(changes.is_empty(), "{changes:?}");
    for probe in [
        "rowmajor",
        "kmajor",
        "RowMajor",
        " rowmajor",
        "rowmajor ",
        "",
    ] {
        assert_eq!(
            new_at(env, probe).map(|l| l.cublas_kmajor),
            Some(old_kmajor(Some(probe))),
            "{probe:?}"
        );
    }
}

/// 2026-10-17: Every lever at the value its reader read unset as.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    assert_eq!(ops(&defaults()), OpsLevers::declared());
    assert_eq!(
        OpsLevers::declared(),
        OpsLevers {
            fp8_gemm_pipe: true,
            gemm_shape_log: false,
            rms_norm_warp_row: true,
            attn_fa128: true,
            lora_no_ffn: false,
            lora: LoraLevers {
                no_apply: false,
                gemv_max_m: old_gemv_max_m(None),
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
            cublas_kmajor: old_kmajor(None),
            w4a4: crate::layers::ops::w4a4_proj::W4a4Levers::declared(),
            w8a16: crate::layers::ops::W8a16Plan {
                pipe128: true,
                tc_rows: true,
            },
            w4a16_tc: crate::layers::ops::W4a16TcPlan {
                tc: old_unset_or_empty(None),
                wide: old_wide(None, None),
            },
            mtp_tc: old_unset_or_empty(None),
            batch8_rt: true,
            gemv_exact_m_tiers: true,
            mmq_small_tile: true,
            mmq_tile64: true,
        }
    );
    assert_eq!(
        crate::layers::ops::ModelLevers::defaults().ops,
        OpsLevers::declared()
    );
}

/// 2026-10-17: The blockscaled plan carries the two levers it names, not each other's.
#[test]
fn the_blockscaled_plan_takes_its_two_levers() {
    for (pipe, shape_log) in [(true, false), (false, true)] {
        let l = OpsLevers {
            fp8_gemm_pipe: pipe,
            gemm_shape_log: shape_log,
            ..OpsLevers::declared()
        };
        let plan = l.fp8_blockscaled();
        assert_eq!((plan.pipe, plan.shape_log), (pipe, shape_log));
    }
}

/// 2026-10-17: `gemv_tc::tc_enabled` and `dense_gemv_tc::mtp_tc_from`, one predicate over the
/// looked-up value.
fn old_unset_or_empty(v: Option<&str>) -> bool {
    v.is_none_or(|v| v.is_empty())
}

/// 2026-10-17: `gemv_tc::wide_rows_enabled`.
fn old_wide(kill: Option<&str>, wide: Option<&str>) -> bool {
    old_unset_or_empty(kill) && wide.is_some_and(|v| !v.is_empty())
}

/// 2026-10-17: `gemv_tc::narrow_gemv_max_rows`.
fn old_narrow_rows(kill: Option<&str>, wide: Option<&str>) -> u32 {
    if old_wide(kill, wide) { 16 } else { 8 }
}

/// 2026-10-17: The two tensor-core kill switches keep their text predicate: any non-empty value,
/// `0` and the switch spellings included, turns the path off; an empty one leaves it on.
#[test]
fn the_tensor_core_kill_switches_keep_their_text_predicate() {
    let cases: [Field; 2] = [
        ("METRALE_NO_W4A16_TC", |l| l.w4a16_tc.tc),
        ("METRALE_NO_MTP_TC", |l| l.mtp_tc),
    ];
    for (env, field) in cases {
        let changes = check(env, Old::Verbatim, old_unset_or_empty, |s| field(&ops(s)));
        assert!(changes.is_empty(), "{changes:?}");
        for probe in ["", " ", "no", "unset"] {
            assert_eq!(
                new_at(env, probe).map(|l| field(&l)),
                Some(old_unset_or_empty(Some(probe))),
                "{probe:?}"
            );
        }
    }
}

/// 2026-10-17: The widening and the narrow-GEMV row edge over every pair of the kill switch's and
/// the widening's probes, the empty value included: as `wide_rows_enabled` and
/// `narrow_gemv_max_rows` read them.
#[test]
fn the_widening_composes_with_the_kill_switch_as_before() {
    const KILL: &str = "METRALE_NO_W4A16_TC";
    const WIDE: &str = "METRALE_W4A16_TC_WIDE";
    let probes = PROBES.into_iter().chain([Some("")]);
    for kill in probes.clone() {
        for wide in probes.clone() {
            let mut pairs = Vec::new();
            pairs.extend(kill.map(|v| (KILL, v)));
            pairs.extend(wide.map(|v| (WIDE, v)));
            let plan = settings_from(&pairs).ok().map(|s| ops(&s).w4a16_tc);
            assert_eq!(
                plan.map(|p| (p.tc, p.wide, p.narrow_max_rows())),
                Some((
                    old_unset_or_empty(kill),
                    old_wide(kill, wide),
                    old_narrow_rows(kill, wide)
                )),
                "{kill:?} {wide:?}"
            );
        }
    }
}
