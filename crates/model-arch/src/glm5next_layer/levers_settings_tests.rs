// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `GlmLevers::from_settings` and `arena_lean` against the environment readers they
//! replaced, frozen here as `from_values` and `old_arena_lean`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! strict-`1` switch now also arms at `true`, an opt-out now also turns off at `off`,
//! a value the switch rule cannot read is rejected at startup, and a row width outside its range
//! is an error instead of the default.

use super::*;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

const DEFAULT_ROW_GROUP: usize =
    metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize;

/// 2026-10-17: The readers `from_settings` replaced, unchanged but for their paths: `value` is a
/// variable's text.
fn from_values(mut value: impl FnMut(&str) -> Option<String>) -> GlmLevers {
    let cublas_wide_proj = value("METRALE_GLM_CUBLAS_PROJ").as_deref() != Some("0");
    let dsa_ws_shared = value("METRALE_GLM_DSA_WS_SHARED").as_deref() != Some("0");
    let mlp_ws_shared = value("METRALE_GLM_MLP_WS_SHARED").as_deref() != Some("0");
    let dsa_select_rows = value("METRALE_DSA_SELECT_ROWS").as_deref() != Some("0");
    // 2026-10-21: No reader preceded it; written as the opt-out readers above are.
    let pdl = value("METRALE_GLM_PDL").as_deref() != Some("0");
    let dsa_verify_select_once =
        value("METRALE_GLM_DSA_VERIFY_SELECT_ONCE").as_deref() != Some("0");
    let dsa_proj_in_merged = value("METRALE_GLM_DSA_PROJ_IN_MERGED").as_deref() != Some("0");
    let dsa_selector_pair = value("METRALE_GLM_DSA_SELECTOR_PAIR").as_deref() != Some("0");
    // 2026-10-22: Default on since today (byte-identical, GLM-only): the reader the settings now
    // match is the opt-out one, `!= "0"`.
    let dsa_indexer_rows = value("METRALE_GLM_DSA_INDEXER_ROWS").as_deref() != Some("0");
    let dsa_idx_proj_whole = value("METRALE_GLM_DSA_IDX_PROJ_WHOLE").as_deref() != Some("0");
    let mhc_fused = value("METRALE_GLM_MHC_FUSED").as_deref() != Some("0");
    let prefill_rows = value("METRALE_GLM_PREFILL_ROWS")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|r| *r >= 1)
        .unwrap_or(PREFILL_ROWS);
    let row_group = value("METRALE_GLM_ROW_GROUP")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|r| (1..=MAX_ROW_GROUP).contains(r))
        .unwrap_or(DEFAULT_ROW_GROUP);
    let mut opt_in = |var: &str| value(var).as_deref() == Some("1");
    GlmLevers {
        cublas_wide_proj,
        dsa_batch_qidx: opt_in("METRALE_GLM_DSA_BATCH_QIDX"),
        router_rows: opt_in("METRALE_GLM_ROUTER_ROWS"),
        router_topk_rank: opt_in("METRALE_GLM_ROUTER_TOPK_RANK"),
        moe_combine_split: opt_in("METRALE_GLM_MOE_COMBINE_SPLIT"),
        moe_combine_vec8: opt_in("METRALE_GLM_MOE_COMBINE_VEC8"),
        dsa_indexer_rows,
        dsa_idx_proj_whole,
        dsa_verify_rows: opt_in("METRALE_GLM_DSA_VERIFY_ROWS"),
        dsa_verify_select_once,
        dsa_proj_in_merged,
        dsa_selector_pair,
        ar_fused_post: opt_in("METRALE_GLM_AR_FUSED_POST"),
        mhc_post_fold: opt_in("METRALE_GLM_MHC_POST_FOLD"),
        mhc_fused,
        kda_chunk_prefill: opt_in("METRALE_GLM_KDA_CHUNK_PREFILL"),
        prefill_rows,
        row_group,
        dsa_ws_shared,
        mlp_ws_shared,
        dsa_select_rows,
        dsa_alloc_per_step: opt_in("METRALE_GLM_DSA_ALLOC_PER_STEP"),
        // 2026-10-17: Probed in `glm5next_mhc_levers_tests.rs`, `glm5next_kda/levers_tests.rs`,
        // `glm5next_dsa/levers_tests.rs`, `glm5next_mlp/forward/levers_tests.rs` and
        // `glm5next_dense_levers_tests.rs`; at their defaults here.
        mhc: crate::glm5next_mhc_levers::MhcLevers::declared(),
        kda: crate::glm5next_kda::KdaLevers::declared(),
        dsa: crate::glm5next_dsa::levers::DsaLevers::declared(),
        mlp: crate::glm5next_mlp::forward::MlpLevers::declared(),
        dense: crate::glm5next_dense_levers::DenseLevers::declared(),
        // 2026-10-17: Probed in `profile_tests.rs`, `glm5next_prefill_piece_tests.rs` and
        // `l2_warm.rs`'s tests; at their defaults here.
        profile: crate::glm5next_layer::profile::Profile::from_settings(&defaults()),
        piece_rows: None,
        l2_warm_budget: None,
        pdl,
        // 2026-10-24: Command-line only (no variable); unpublished in a test, so the default.
        decode: Default::default(),
    }
}

/// 2026-10-17: The `arena_lean` reader that was replaced, as a value (`Err` as `None`).
fn old_arena_lean(value: Option<&str>) -> Option<bool> {
    match value {
        None | Some("0") => Some(false),
        Some("1") => Some(true),
        Some(_) => None,
    }
}

// 2026-10-22: `METRALE_GLM_DSA_INDEXER_ROWS`, `METRALE_GLM_DSA_IDX_PROJ_WHOLE` and
// `METRALE_GLM_MHC_FUSED` moved to `OPT_OUT` (default on); `METRALE_GLM_ARENA_LEAN` stays last,
// probed by its own test.
const OPT_IN: [&str; 11] = [
    "METRALE_GLM_DSA_ALLOC_PER_STEP",
    "METRALE_GLM_DSA_BATCH_QIDX",
    "METRALE_GLM_ROUTER_ROWS",
    "METRALE_GLM_ROUTER_TOPK_RANK",
    "METRALE_GLM_MOE_COMBINE_SPLIT",
    "METRALE_GLM_MOE_COMBINE_VEC8",
    "METRALE_GLM_DSA_VERIFY_ROWS",
    "METRALE_GLM_AR_FUSED_POST",
    "METRALE_GLM_MHC_POST_FOLD",
    "METRALE_GLM_KDA_CHUNK_PREFILL",
    "METRALE_GLM_ARENA_LEAN",
];

const OPT_OUT: [&str; 11] = [
    "METRALE_GLM_PDL",
    "METRALE_GLM_CUBLAS_PROJ",
    "METRALE_GLM_DSA_WS_SHARED",
    "METRALE_GLM_MLP_WS_SHARED",
    "METRALE_DSA_SELECT_ROWS",
    "METRALE_GLM_DSA_VERIFY_SELECT_ONCE",
    "METRALE_GLM_DSA_PROJ_IN_MERGED",
    "METRALE_GLM_DSA_SELECTOR_PAIR",
    "METRALE_GLM_DSA_INDEXER_ROWS",
    "METRALE_GLM_DSA_IDX_PROJ_WHOLE",
    "METRALE_GLM_MHC_FUSED",
];

const WIDTHS: [&str; 2] = ["METRALE_GLM_PREFILL_ROWS", "METRALE_GLM_ROW_GROUP"];

/// 2026-10-17: The whole lever set with `env` at `probe` and nothing else set, so a field read
/// from the wrong variable shows as a difference.
fn changes(env: &'static str, old: Old) -> Vec<(&'static str, Option<&'static str>, Outcome)> {
    check(
        env,
        old,
        |probe| from_values(|var| (var == env).then(|| probe.map(str::to_string)).flatten()),
        |s| GlmLevers::from_settings(s).expect("one switch set resolves"),
    )
}

#[test]
fn every_switch_reads_as_before_except_the_switch_spellings() {
    for env in &OPT_IN[..10] {
        assert_eq!(
            changes(env, Old::OptIn),
            [
                (*env, Some("true"), Outcome::Changed),
                (*env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
    for env in OPT_OUT {
        assert_eq!(
            changes(env, Old::OptOut),
            [
                (env, Some("off"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
}

/// 2026-10-17: The old reader refused anything but unset, `0` and `1` at first use; the switch
/// also reads `true` and `off`, and still refuses garbage, now at startup. 2026-10-22: on a
/// `glm5_next` load unset now reads on (default on since today).
#[test]
fn arena_lean_reads_as_before_and_takes_the_switch_spellings() {
    let env = "METRALE_GLM_ARENA_LEAN";
    let new = |probe: Option<&str>| {
        let pairs: Vec<(&str, &str)> = probe.map(|v| (env, v)).into_iter().collect();
        settings_from(&pairs)
            .ok()
            .map(|s| arena_lean(&s, "glm5_next").expect("a glm5_next load takes the lever"))
    };
    for probe in [Some("0"), Some("1"), Some("garbage"), Some("")] {
        assert_eq!(new(probe), old_arena_lean(probe), "{env}={probe:?}");
    }
    assert_eq!((old_arena_lean(None), new(None)), (Some(false), Some(true)));
    assert_eq!(
        (old_arena_lean(Some("true")), new(Some("true"))),
        (None, Some(true))
    );
    assert_eq!(
        (old_arena_lean(Some("off")), new(Some("off"))),
        (None, Some(false))
    );
}

/// 2026-10-17: Each width probe, read by the old reader and by `from_settings`; `None` from the
/// new path is an error (a malformed integer at resolve, or a width out of range).
fn width_reads(env: &'static str, probe: Option<&str>) -> (GlmLevers, Option<GlmLevers>) {
    let old = from_values(|var| (var == env).then(|| probe.map(str::to_string)).flatten());
    let pairs: Vec<(&str, &str)> = probe.map(|v| (env, v)).into_iter().collect();
    let new = settings_from(&pairs)
        .ok()
        .and_then(|s| GlmLevers::from_settings(&s).ok());
    (old, new)
}

#[test]
fn a_width_in_range_reads_as_before_and_one_outside_is_refused() {
    let above = (MAX_ROW_GROUP + 1).to_string();
    for env in WIDTHS {
        for probe in [None, Some("1"), Some("8"), Some("16")] {
            let (old, new) = width_reads(env, probe);
            assert_eq!(new, Some(old), "{env}={probe:?}");
        }
        for probe in [Some("0"), Some("-1"), Some("garbage"), Some("")] {
            assert_eq!(width_reads(env, probe).1, None, "{env}={probe:?} accepted");
        }
        // 2026-10-17: The old reader parsed without trimming; the integer rule trims.
        let (old, new) = width_reads(env, Some(" 8"));
        assert_ne!(Some(old), new, "{env}=\" 8\"");
    }
    // 2026-10-17: The prefill width has no upper bound; the row group stops at `MAX_ROW_GROUP`.
    let (old, new) = width_reads(WIDTHS[0], Some(&above));
    assert_eq!(new, Some(old));
    assert_eq!(width_reads(WIDTHS[1], Some(&above)).1, None);
    let max = MAX_ROW_GROUP.to_string();
    let (old, new) = width_reads(WIDTHS[1], Some(&max));
    assert_eq!(
        (new.map(|l| l.row_group), old.row_group),
        (Some(MAX_ROW_GROUP), MAX_ROW_GROUP)
    );
}

/// 2026-10-17: The declared defaults resolve to the old readers' unset values, the row group's
/// declared `16` included.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    let levers = GlmLevers::from_settings(&defaults()).expect("defaults resolve");
    assert_eq!(levers, from_values(|_| None));
    assert_eq!(levers.row_group, DEFAULT_ROW_GROUP);
    // 2026-10-22: The lean arena is on by default on a `glm5_next` load.
    assert!(arena_lean(&defaults(), "glm5_next").unwrap());
}

/// 2026-10-22: `METRALE_GLM_ARENA_LEAN` is gated by model type: on another model the default is
/// off (no refusal), a set `1` is refused as before, and `0` is off everywhere.
#[test]
fn arena_lean_is_gated_by_model_type() {
    let at = |pairs: &[(&str, &str)], model: &str| {
        arena_lean(&settings_from(pairs).unwrap(), model).map_err(|e| e.to_string())
    };
    let env = "METRALE_GLM_ARENA_LEAN";
    assert_eq!(at(&[], "glm5_next"), Ok(true));
    assert_eq!(at(&[], "qwen3_5"), Ok(false));
    assert_eq!(at(&[(env, "0")], "glm5_next"), Ok(false));
    assert_eq!(at(&[(env, "0")], "qwen3_5"), Ok(false));
    assert_eq!(at(&[(env, "1")], "glm5_next"), Ok(true));
    let e = at(&[(env, "1")], "qwen3_5").unwrap_err();
    assert!(e.contains("is for glm5_next models"), "{e}");
}

/// 2026-10-17: The probe lists cover every variable the old readers read.
#[test]
fn the_probe_lists_cover_every_lever_the_old_readers_read() {
    let mut read = Vec::new();
    let _ = from_values(|var| {
        read.push(var.to_string());
        None
    });
    read.push("METRALE_GLM_ARENA_LEAN".to_string());
    read.sort();
    read.dedup();
    let mut listed: Vec<String> = OPT_IN
        .iter()
        .chain(&WIDTHS)
        .chain(&OPT_OUT)
        .map(|v| (*v).to_string())
        .collect();
    listed.sort();
    assert_eq!(read, listed);
}
