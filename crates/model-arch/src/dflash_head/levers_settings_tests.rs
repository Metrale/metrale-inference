// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `DFlashLevers::from_settings` against the environment reader it replaced, frozen
//! here as `from_values`, probed lever by lever through `lever_equivalence::check`.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! strict-`1` switch now also arms at `true`, an opt-out now also turns off at `off`, and a
//! value the switch rule cannot read is rejected at startup.

use super::*;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check};

/// 2026-10-17: The reader `from_settings` replaced, unchanged but for its paths: `value` is
/// a variable's text, `present` whether it is set at all.
fn from_values(
    mut value: impl FnMut(&str) -> Option<String>,
    mut present: impl FnMut(&str) -> bool,
) -> DFlashLevers {
    fn opt_in(value: Option<&str>) -> bool {
        value == Some("1")
    }
    fn parsed<T: core::str::FromStr>(value: Option<&str>) -> Option<T> {
        value.and_then(|v| v.parse().ok())
    }

    DFlashLevers {
        any_diagnostic_armed: GRAPH_SUPPRESSING_DIAGNOSTICS.iter().any(|var| present(var)),

        debug_dump: opt_in(value("METRALE_DFLASH_DEBUG_DUMP").as_deref()),
        debug_dump_full: opt_in(value("METRALE_DFLASH_DEBUG_DUMP_FULL").as_deref()),
        log_drafts: opt_in(value("METRALE_DFLASH_LOG_DRAFTS").as_deref()),
        block_dump: opt_in(value("METRALE_DFLASH_BLOCK_DUMP").as_deref()),
        block_dump_at_pos: parsed(value("METRALE_DFLASH_BLOCK_DUMP_AT_POS").as_deref())
            .unwrap_or(0),
        option_b_diag: opt_in(value("METRALE_DFLASH_OPTION_B_DIAG").as_deref()),

        force_pattern: opt_in(value("METRALE_DFLASH_DEBUG_FORCE_PATTERN").as_deref()),
        force_noise_pattern: opt_in(value("METRALE_DFLASH_DEBUG_FORCE_NOISE_PATTERN").as_deref()),
        force_no_ctx: opt_in(value("METRALE_DFLASH_DEBUG_CTX_OFF").as_deref()),
        force_ctx_used: parsed(value("METRALE_DFLASH_DEBUG_CTX_USED").as_deref()),

        precompute: opt_in(value("METRALE_DFLASH_PRECOMPUTE").as_deref()),
        precompute_commit: opt_in(value("METRALE_DFLASH_PRECOMPUTE_COMMIT").as_deref()),

        option_b: crate::dflash_head::option_b_from(value("METRALE_DFLASH_OPTION_B").as_deref()),
        option_b_no_ctx: opt_in(value("METRALE_DFLASH_OPTION_B_NO_CTX").as_deref()),
        dflash2: value("METRALE_DFLASH2").as_deref() != Some("0"),
        topk16_split: value("METRALE_DFLASH2_TOPK_SPLIT").as_deref() != Some("0"),
        tp_graphs: opt_in(value("METRALE_DFLASH_TP_GRAPHS").as_deref()),
        batch_propose_width: parsed(value("METRALE_DFLASH_BATCH_PROPOSE").as_deref())
            .unwrap_or(usize::MAX),
        draft_cap: parsed(value("METRALE_DFLASH_DRAFT_CAP").as_deref()),
        full_block: opt_in(value("METRALE_DFLASH_FULL_BLOCK").as_deref()),

        verify_trace: opt_in(value("METRALE_DFLASH_VERIFY_TRACE").as_deref()),
        precompute_dump: opt_in(value("METRALE_DFLASH_PRECOMPUTE_DUMP").as_deref()),
        ctx_parity_dump: opt_in(value("METRALE_DFLASH_CTX_PARITY_DUMP").as_deref()),
        no_decode_append: opt_in(value("METRALE_DFLASH_DEBUG_NO_DECODE_APPEND").as_deref()),
        full_precompute: opt_in(value("METRALE_DFLASH_DEBUG_FULL_PRECOMPUTE").as_deref()),
        ctxlen_probe: opt_in(value("METRALE_DFLASH_CTXLEN_PROBE").as_deref()),

        dspark_markov: value("METRALE_DSPARK_MARKOV").as_deref() != Some("0"),
        conf_tau: parsed(value("METRALE_DSPARK_CONF_TAU").as_deref()).unwrap_or(0.0),

        propose_warmup_n: parsed(value("METRALE_DFLASH_PROPOSE_WARMUP_N").as_deref()).unwrap_or(2),

        dspark_shift: match value("METRALE_DSPARK_SHIFT").as_deref() {
            Some("1") => Some(true),
            Some("0") => Some(false),
            _ => None,
        },
        dspark_anchor_bias: value("METRALE_DSPARK_ANCHOR_BIAS").as_deref() != Some("0"),
        dspark_conf_trace: opt_in(value("METRALE_DSPARK_CONF_TRACE").as_deref()),
    }
}

const OPT_IN: [&str; 20] = [
    "METRALE_DFLASH_TP_GRAPHS",
    "METRALE_DFLASH_DEBUG_DUMP",
    "METRALE_DFLASH_DEBUG_DUMP_FULL",
    "METRALE_DFLASH_LOG_DRAFTS",
    "METRALE_DFLASH_BLOCK_DUMP",
    "METRALE_DFLASH_OPTION_B_DIAG",
    "METRALE_DFLASH_DEBUG_FORCE_PATTERN",
    "METRALE_DFLASH_DEBUG_FORCE_NOISE_PATTERN",
    "METRALE_DFLASH_DEBUG_CTX_OFF",
    "METRALE_DFLASH_PRECOMPUTE",
    "METRALE_DFLASH_PRECOMPUTE_COMMIT",
    "METRALE_DFLASH_OPTION_B_NO_CTX",
    "METRALE_DFLASH_FULL_BLOCK",
    "METRALE_DFLASH_VERIFY_TRACE",
    "METRALE_DFLASH_PRECOMPUTE_DUMP",
    "METRALE_DFLASH_CTX_PARITY_DUMP",
    "METRALE_DFLASH_DEBUG_NO_DECODE_APPEND",
    "METRALE_DFLASH_DEBUG_FULL_PRECOMPUTE",
    "METRALE_DFLASH_CTXLEN_PROBE",
    "METRALE_DSPARK_CONF_TRACE",
];

const OPT_OUT: [&str; 4] = [
    "METRALE_DFLASH2",
    "METRALE_DFLASH2_TOPK_SPLIT",
    "METRALE_DSPARK_MARKOV",
    "METRALE_DSPARK_ANCHOR_BIAS",
];

/// 2026-10-17: Text levers; the head applies the same parse to the text as before.
const VERBATIM: [&str; 9] = [
    "METRALE_DFLASH_BLOCK_DUMP_AT_POS",
    "METRALE_DFLASH_DEBUG_CTX_USED",
    "METRALE_DFLASH_OPTION_B",
    "METRALE_DFLASH_BATCH_PROPOSE",
    "METRALE_DFLASH_DRAFT_CAP",
    "METRALE_DFLASH_PROPOSE_WARMUP_N",
    "METRALE_DSPARK_CONF_TAU",
    "METRALE_DSPARK_SHIFT",
    "METRALE_DFLASH_PROPOSE_NO_GRAPH",
];

/// 2026-10-17: The whole lever set with `env` at `probe` and nothing else set, so a field read
/// from the wrong lever, or a lost `any_diagnostic_armed`, shows as a difference.
fn changes(env: &'static str, old: Old) -> Vec<(&'static str, Option<&'static str>, Outcome)> {
    check(
        env,
        old,
        |probe| {
            from_values(
                |var| (var == env).then(|| probe.map(str::to_string)).flatten(),
                |var| var == env && probe.is_some(),
            )
        },
        DFlashLevers::from_settings,
    )
}

#[test]
fn every_lever_reads_as_before_except_the_switch_spellings() {
    for env in OPT_IN {
        assert_eq!(
            changes(env, Old::OptIn),
            [
                (env, Some("true"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
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
    for env in VERBATIM {
        assert_eq!(changes(env, Old::Verbatim), [], "{env}");
    }
}

/// 2026-10-17: The three lists cover every variable the old reader read, so no lever is left
/// unprobed.
#[test]
fn the_probe_lists_cover_every_lever_the_old_reader_read() {
    let read = core::cell::RefCell::new(Vec::new());
    let _ = from_values(
        |var| {
            read.borrow_mut().push(var.to_string());
            None
        },
        |var| {
            read.borrow_mut().push(var.to_string());
            false
        },
    );
    let mut read = read.into_inner();
    read.sort();
    read.dedup();
    let mut listed: Vec<String> = OPT_IN
        .iter()
        .chain(&OPT_OUT)
        .chain(&VERBATIM)
        .map(|v| (*v).to_string())
        .collect();
    listed.sort();
    assert_eq!(read, listed);
}
