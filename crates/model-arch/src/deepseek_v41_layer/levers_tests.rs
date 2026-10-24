// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `Ds41LoadLevers::from_settings` against the environment readers it replaced,
//! frozen here as `from_values`, probed lever by lever through `lever_equivalence::check`.
//!
//! Owner: model-arch (DeepSeek-V4.1).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! strict-`1` switch now also arms at `true`, an opt-out now also turns off at `off`, a value
//! the switch rule cannot read is rejected at startup, and a count the integer rule cannot read
//! is rejected instead of falling back to its default.

use super::*;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-17: The levers at the settings defaults, for the GPU tests that build a layer.
pub(crate) fn at_defaults() -> Ds41Levers {
    Ds41Levers::from_settings(&defaults())
}

/// 2026-10-17: The readers `from_settings` replaced (the `OnceLock`s of the step, MoE and
/// attention files, `env_usize`, and the loader's own reads), unchanged but for their paths:
/// `value` is a variable's text, `None` when unset.
fn from_values(value: impl Fn(&str) -> Option<String>) -> Ds41LoadLevers {
    let is_one = |k: &str| value(k).is_some_and(|v| v == "1");
    let run = Ds41Levers {
        diag: is_one("METRALE_DS41_DIAG"),
        trace: is_one("METRALE_DS41_TRACE"),
        step_graph: is_one("METRALE_DS41_STEP_GRAPH"),
        step_graph_log: is_one("METRALE_DS41_STEP_GRAPH_LOG"),
        graph_mode: if is_one("METRALE_DS41_GRAPH_ORACLE") {
            GraphMode::Oracle
        } else if is_one("METRALE_DS41_GRAPH") {
            GraphMode::On
        } else {
            GraphMode::Off
        },
        no_engram: is_one("METRALE_DS41_NO_ENGRAM"),
        prefetch_k: parsed(value("METRALE_DS41_PREFETCH_K").as_deref()).unwrap_or(0),
        shared_side: value("METRALE_DS41_SHARED_SIDE").is_none_or(|v| v != "0"),
        router_staged: value("METRALE_DS41_ROUTER_STAGED").is_none_or(|v| v != "0"),
        m1_loop: value("METRALE_DS41_M1_LOOP").is_some_and(|v| v == "1"),
        prefill_gemv: value("METRALE_DS41_PREFILL_GEMV").is_some_and(|v| v == "1"),
        device_route: value("METRALE_DS41_DEVICE_ROUTE").is_some_and(|v| v == "1"),
        device_route_diag: value("METRALE_DS41_DEVICE_ROUTE_DIAG").is_some_and(|v| v == "1"),
        sparse_split: parsed(value("METRALE_DS41_SPARSE_SPLIT").as_deref())
            .filter(|s| (1..=8).contains(s))
            .unwrap_or(2),
        expert_warps: parsed(value("METRALE_DS41_EXPERT_WARPS").as_deref())
            .filter(|w| matches!(w, 2 | 4 | 8))
            .unwrap_or(8),
    };
    let env_usize = |k: &str, d: usize| parsed(value(k).as_deref()).unwrap_or(d);
    Ds41LoadLevers {
        max_seq: env_usize("METRALE_DS41_MAX_SEQ", 8192),
        max_tokens: env_usize("METRALE_DS41_MAX_TOKENS", 2048),
        reader_threads: env_usize("METRALE_DS41_READER_THREADS", 16),
        staging_slots: env_usize("METRALE_DS41_STAGING_SLOTS", 16),
        candidate_source: parsed(value("METRALE_DS41_CANDIDATE_SOURCE").as_deref()),
        expert_cache_gib: parsed(value("METRALE_DS41_EXPERT_CACHE_GIB").as_deref())
            .filter(|g: &f64| g.is_finite() && *g > 0.0)
            .unwrap_or(88.0),
        arena_device: value("METRALE_DS41_ARENA_DEVICE").is_none_or(|v| v != "0"),
        reader_pool: value("METRALE_DS41_READER_POOL").is_some_and(|v| v == "1"),
        engram_q2k: value("METRALE_DS41_ENGRAM_Q2K").is_none_or(|v| v != "0"),
        route_trace: value("METRALE_DS41_ROUTE_TRACE"),
        predict_trace: value("METRALE_DS41_PREDICT_TRACE"),
        run,
    }
}

const OPT_IN: [&str; 12] = [
    "METRALE_DS41_DIAG",
    "METRALE_DS41_TRACE",
    "METRALE_DS41_STEP_GRAPH",
    "METRALE_DS41_STEP_GRAPH_LOG",
    "METRALE_DS41_GRAPH",
    "METRALE_DS41_GRAPH_ORACLE",
    "METRALE_DS41_NO_ENGRAM",
    "METRALE_DS41_M1_LOOP",
    "METRALE_DS41_PREFILL_GEMV",
    "METRALE_DS41_DEVICE_ROUTE",
    "METRALE_DS41_DEVICE_ROUTE_DIAG",
    "METRALE_DS41_READER_POOL",
];

const OPT_OUT: [&str; 4] = [
    "METRALE_DS41_SHARED_SIDE",
    "METRALE_DS41_ROUTER_STAGED",
    "METRALE_DS41_ARENA_DEVICE",
    "METRALE_DS41_ENGRAM_Q2K",
];

/// 2026-10-17: Text levers; `from_settings` applies the same parse to the text as before.
const VERBATIM: [&str; 6] = [
    "METRALE_DS41_PREFETCH_K",
    "METRALE_DS41_SPARSE_SPLIT",
    "METRALE_DS41_EXPERT_WARPS",
    "METRALE_DS41_EXPERT_CACHE_GIB",
    "METRALE_DS41_ROUTE_TRACE",
    "METRALE_DS41_PREDICT_TRACE",
];

/// 2026-10-17: Int levers; an unparseable value is now refused at startup.
const NUMBER: [&str; 5] = [
    "METRALE_DS41_MAX_SEQ",
    "METRALE_DS41_MAX_TOKENS",
    "METRALE_DS41_READER_THREADS",
    "METRALE_DS41_STAGING_SLOTS",
    "METRALE_DS41_CANDIDATE_SOURCE",
];

/// 2026-10-17: One value through the frozen readers, nothing else set.
fn old_at(env: &str, probe: Option<&str>) -> Ds41LoadLevers {
    from_values(|var| (var == env).then(|| probe.map(str::to_string)).flatten())
}

/// 2026-10-17: The whole lever set with `env` at `probe` and nothing else set, so a field read
/// from the wrong lever shows as a difference.
fn changes(env: &'static str, old: Old) -> Vec<(&'static str, Option<&'static str>, Outcome)> {
    check(
        env,
        old,
        |probe| old_at(env, probe),
        Ds41LoadLevers::from_settings,
    )
}

#[test]
fn every_lever_reads_as_before_except_the_switch_spellings_and_unreadable_counts() {
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
    for env in NUMBER {
        assert_eq!(
            changes(env, Old::Number),
            [
                (env, Some("true"), Outcome::Rejected),
                (env, Some("off"), Outcome::Rejected),
                (env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
}

/// 2026-10-17: The four lists cover every variable the old readers read, so no lever is left
/// unprobed.
#[test]
fn the_probe_lists_cover_every_lever_the_old_readers_read() {
    let read = core::cell::RefCell::new(Vec::new());
    let _ = from_values(|var| {
        read.borrow_mut().push(var.to_string());
        None
    });
    let mut read = read.into_inner();
    read.sort();
    read.dedup();
    let mut listed: Vec<String> = OPT_IN
        .iter()
        .chain(&OPT_OUT)
        .chain(&VERBATIM)
        .chain(&NUMBER)
        .map(|v| (*v).to_string())
        .collect();
    listed.sort();
    assert_eq!(read, listed);
}

/// 2026-10-17: The text levers at the values where a parse could part from the old one: the
/// empty value, out-of-range and negative numbers, a fraction and a padded number.
#[test]
fn text_levers_read_as_before_at_edge_values() {
    for env in VERBATIM {
        for v in ["", "0", "3", "4", "9", "-1", "2.5", " 4", "inf", "NaN"] {
            let s = settings_from(&[(env, v)]).expect("a text lever takes any value");
            assert_eq!(
                Ds41LoadLevers::from_settings(&s),
                old_at(env, Some(v)),
                "{env}={v:?}"
            );
        }
    }
}

/// 2026-10-17: With both graph switches set the oracle still wins.
#[test]
fn the_graph_oracle_wins_over_the_graph_switch() {
    let s = settings_from(&[
        ("METRALE_DS41_GRAPH", "1"),
        ("METRALE_DS41_GRAPH_ORACLE", "1"),
    ])
    .expect("two switches resolve");
    assert_eq!(Ds41Levers::from_settings(&s).graph_mode, GraphMode::Oracle);
}

/// 2026-10-17: The negative-value fallbacks are the rows' declared defaults, so a negative count
/// still reads as unset did.
#[test]
fn count_fallbacks_are_the_declared_defaults() {
    for (env, fallback) in [
        ("METRALE_DS41_MAX_SEQ", MAX_SEQ),
        ("METRALE_DS41_MAX_TOKENS", MAX_TOKENS),
        ("METRALE_DS41_READER_THREADS", READER_THREADS),
        ("METRALE_DS41_STAGING_SLOTS", STAGING_SLOTS),
    ] {
        let spec = metrale_settings::groups::lookup(env).expect("a declared lever");
        assert_eq!(spec.default, fallback.to_string(), "{env}");
    }
}

/// 2026-10-17: A negative count reads as its default (the candidate source as unset) as under
/// the `usize` parse; the empty value, which that parse read as unset, is now refused, and a
/// padded number, which it read as unset, now reads as the number.
#[test]
fn counts_keep_their_negative_fallback_refuse_empty_and_trim() {
    for env in NUMBER {
        let s = settings_from(&[(env, "-5")]).expect("an integer resolves");
        assert_eq!(
            Ds41LoadLevers::from_settings(&s),
            old_at(env, Some("-5")),
            "{env}"
        );
        assert!(settings_from(&[(env, "")]).is_err(), "{env}");
    }
    let padded = settings_from(&[("METRALE_DS41_READER_THREADS", " 4")]).expect("trimmed");
    assert_eq!(Ds41LoadLevers::from_settings(&padded).reader_threads, 4);
    assert_eq!(
        old_at("METRALE_DS41_READER_THREADS", Some(" 4")).reader_threads,
        READER_THREADS
    );
}
