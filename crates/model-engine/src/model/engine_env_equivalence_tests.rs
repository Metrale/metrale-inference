// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Every model-engine environment read that became a plain settings field, against
//! a frozen copy of the reader it replaced, through `MapEnv`-resolved settings (never the
//! process environment). The rule-bearing levers are in `engine_levers_tests.rs`.
//!
//! Owner: model-engine.
//! Invariants:
//! - Each old reader is frozen in the polarity of its settings field: a `NO_*` kill switch whose
//!   call site used to compute "enabled" is frozen as "armed", and the call site negates the
//!   field exactly as it negated the old value.
//! - Every probe the settings rule reads differently is pinned by name below.

use metrale_model_layers::lever_equivalence::{Old, Outcome, check, settings_from};
use metrale_settings::InferenceServerSettings as S;

type Field = fn(&S) -> bool;

/// 2026-10-17: Frozen switch readers by kind, as the old call sites wrote them.
fn old_switch(kind: Old) -> impl Fn(Option<&str>) -> bool {
    move |v: Option<&str>| match kind {
        Old::OptIn => v == Some("1"),
        Old::OptOut => v != Some("0"),
        Old::Truthy => v.is_some_and(|v| v == "1" || v.to_lowercase() == "true"),
        Old::TruthyExact => v.is_some_and(|v| v == "1" || v == "true"),
        Old::Presence => v.is_some(),
        Old::Verbatim | Old::Number => unreachable!("{kind:?} is not a switch"),
    }
}

/// 2026-10-17: `== "1"` readers, now Switch rows.
const OPT_IN: [(&str, Field); 22] = [
    ("METRALE_CODISPATCH_BTCHECK", |s| {
        s.scheduler.codispatch_btcheck
    }),
    ("METRALE_EP_DEVICE_ARGMAX", |s| s.runtime.ep_device_argmax),
    ("METRALE_EP_DEVICE_FEED", |s| s.runtime.ep_device_feed),
    ("METRALE_MLA_PREFIX_SKIP", |s| s.cache.mla_prefix_skip),
    ("METRALE_NO_LMHEAD_VOCAB_TP", |s| {
        s.kernels.no_lmhead_vocab_tp
    }),
    ("METRALE_NO_DECODE_GRAPHS_MULTISEQ", |s| {
        s.kernels.no_decode_graphs_multiseq
    }),
    ("METRALE_NO_LM_HEAD_BATCH_GEMV", |s| {
        s.scheduler.no_lm_head_batch_gemv
    }),
    ("METRALE_NO_ARGMAX_BATCH", |s| s.scheduler.no_argmax_batch),
    ("METRALE_NO_TAIL_SPLIT", |s| s.scheduler.no_tail_split),
    ("METRALE_Q12_EFFECTIVE_ARENA", |s| {
        s.kernels.q12_effective_arena
    }),
    ("METRALE_DUMP_EMBED", |s| s.telemetry.dump_embed),
    ("METRALE_MARCONI_EXACT", |s| s.server.marconi_exact),
    ("METRALE_PREFILL_HOST_TIMING", |s| {
        s.telemetry.prefill_host_timing
    }),
    ("METRALE_DUMP_HYPER_RMS", |s| s.telemetry.dump_hyper_rms),
    ("METRALE_K2_DIAG", |s| s.telemetry.k2_diag),
    ("METRALE_K4_DIAG", |s| s.telemetry.k4_diag),
    ("METRALE_GLM_VERIFY_GRAPH_TRACE", |s| {
        s.telemetry.glm_verify_graph_trace
    }),
    ("METRALE_GLM_VERIFY_GRAPH_NOCACHE", |s| {
        s.speculative.glm_verify_graph_nocache
    }),
    ("METRALE_MTP_TIMING", |s| s.telemetry.mtp_timing),
    ("METRALE_NO_MAPPED_ARGMAX", |s| s.runtime.no_mapped_argmax),
    ("METRALE_VERIFY_D2H_DEFAULT_STREAM", |s| {
        s.speculative.verify_d2h_default_stream
    }),
    ("METRALE_NO_PINNED_VERIFY_D2H", |s| {
        s.speculative.no_pinned_verify_d2h
    }),
];

/// 2026-10-17: `!= "0"` readers, now default-on Switch rows.
const OPT_OUT: [(&str, Field); 4] = [
    ("METRALE_COMM_DECODE_GRAPHS", |s| {
        s.kernels.comm_decode_graphs
    }),
    ("METRALE_DFLASH_EAGLE_FIX", |s| {
        s.speculative.dflash_eagle_fix
    }),
    ("METRALE_DFLASH_UNIFIED_CTX", |s| {
        s.speculative.dflash_unified_ctx
    }),
    ("METRALE_GLM_VERIFY_GRAPHS", |s| {
        s.speculative.glm_verify_graphs
    }),
];

/// 2026-10-17: Set-at-any-value readers (`var_os(..).is_some()`, `var(..).is_ok()`), still
/// Presence rows.
const PRESENCE: [(&str, Field); 12] = [
    ("METRALE_NO_GDN_CARRY", |s| s.speculative.no_gdn_carry),
    ("METRALE_NO_LMHEAD_BATCHED_WIDE", |s| {
        s.scheduler.no_lmhead_batched_wide
    }),
    ("METRALE_SEQ_MEMTRACE", |s| s.telemetry.seq_memtrace),
    ("METRALE_NO_BATCHED_SSM_ROLLBACK", |s| {
        s.scheduler.no_batched_ssm_rollback
    }),
    ("METRALE_NO_MTP_BOOT_GRAPH", |s| {
        s.speculative.no_mtp_boot_graph
    }),
    ("METRALE_NO_MTP_EAGER_DRAFTER", |s| {
        s.speculative.no_mtp_eager_drafter
    }),
    ("METRALE_NO_GRAPH_BORROW", |s| s.scheduler.no_graph_borrow),
    ("METRALE_SSM_H_FP16_DEBUG", |s| s.telemetry.ssm_h_fp16_debug),
    ("METRALE_VISION_TIMING", |s| s.telemetry.vision_timing),
    ("METRALE_SSM_SAVE_DUMP", |s| s.telemetry.ssm_save_dump),
    ("METRALE_NO_MTP_VERIFY_GRAPHS", |s| {
        s.speculative.no_mtp_verify_graphs
    }),
    ("METRALE_NO_VERIFY_WY_CACHE", |s| {
        s.speculative.no_verify_wy_cache
    }),
];

fn run(
    kind: Old,
    rows: &[(&'static str, Field)],
) -> Vec<(&'static str, Option<&'static str>, Outcome)> {
    rows.iter()
        .flat_map(|&(env, field)| check(env, kind, old_switch(kind), field))
        .collect()
}

/// 2026-10-17: `"true"` now arms an opt-in switch and `"off"` / garbage were already off;
/// garbage is now a startup error.
#[test]
fn opt_in_readers_change_only_where_the_switch_rule_says() {
    let changes = run(Old::OptIn, &OPT_IN);
    for (env, probe, outcome) in changes {
        assert!(
            matches!(
                (probe, outcome),
                (Some("true"), Outcome::Changed) | (Some("garbage"), Outcome::Rejected)
            ),
            "{env}={probe:?}: {outcome:?}"
        );
    }
}

/// 2026-10-17: `"off"` now disarms a default-on switch; garbage is now a startup error.
#[test]
fn opt_out_readers_change_only_where_the_switch_rule_says() {
    for (env, probe, outcome) in run(Old::OptOut, &OPT_OUT) {
        assert!(
            matches!(
                (probe, outcome),
                (Some("off"), Outcome::Changed) | (Some("garbage"), Outcome::Rejected)
            ),
            "{env}={probe:?}: {outcome:?}"
        );
    }
}

#[test]
fn presence_readers_are_unchanged() {
    assert_eq!(run(Old::Presence, &PRESENCE), vec![]);
    for (env, field) in PRESENCE {
        assert!(
            field(&settings_from(&[(env, "")]).unwrap()),
            "{env}=\"\" is set"
        );
    }
}

/// 2026-10-17: `METRALE_DIAG_GEMMA4` took exactly `1` or `true`; `off` and garbage are the
/// only differences (garbage now an error).
#[test]
fn diag_gemma4_matches_its_exact_reader() {
    for (_, probe, outcome) in run(
        Old::TruthyExact,
        &[("METRALE_DIAG_GEMMA4", |s| s.telemetry.diag_gemma4)],
    ) {
        assert_eq!((probe, outcome), (Some("garbage"), Outcome::Rejected));
    }
}

/// 2026-10-17: `METRALE_NO_PREFILL_ROW_SHIFT` took `1` or `true` in any case.
#[test]
fn no_prefill_row_shift_matches_its_truthy_reader() {
    let field: Field = |s| s.scheduler.no_prefill_row_shift;
    for (_, probe, outcome) in run(Old::Truthy, &[("METRALE_NO_PREFILL_ROW_SHIFT", field)]) {
        assert_eq!((probe, outcome), (Some("garbage"), Outcome::Rejected));
    }
    assert!(field(
        &settings_from(&[("METRALE_NO_PREFILL_ROW_SHIFT", "TRUE")]).unwrap()
    ));
}

/// 2026-10-17: `METRALE_Q12_BATCHED` was on unless `0` or `false` in any case; `off` and `no`
/// now also disable it, and garbage is now an error.
#[test]
fn q12_batched_matches_its_reader() {
    let old = |v: Option<&str>| v.is_none_or(|v| v != "0" && v.to_lowercase() != "false");
    for v in ["1", "0", "true", "FALSE", "false"] {
        let s = settings_from(&[("METRALE_Q12_BATCHED", v)]).unwrap();
        assert_eq!(s.scheduler.q12_batched, old(Some(v)), "{v:?}");
    }
    assert_eq!(settings_from(&[]).unwrap().scheduler.q12_batched, old(None));
    assert!(
        !settings_from(&[("METRALE_Q12_BATCHED", "off")])
            .unwrap()
            .scheduler
            .q12_batched
    );
    assert!(settings_from(&[("METRALE_Q12_BATCHED", "garbage")]).is_err());
}

/// 2026-10-17: `METRALE_NO_REPLAY_COMMIT_GRAPHS` was a presence switch and is a Switch row:
/// `0` and `off` no longer disable the replay graphs, and garbage is now an error.
#[test]
fn no_replay_commit_graphs_presence_to_switch() {
    let field: Field = |s| s.speculative.no_replay_commit_graphs;
    let changes = run(Old::Presence, &[("METRALE_NO_REPLAY_COMMIT_GRAPHS", field)]);
    let probes: Vec<_> = changes.iter().map(|&(_, p, o)| (p, o)).collect();
    assert_eq!(
        probes,
        [
            (Some("0"), Outcome::Changed),
            (Some("off"), Outcome::Changed),
            (Some("garbage"), Outcome::Rejected)
        ]
    );
}

/// 2026-10-17: `METRALE_PREFILL_ZERO_ROWS` parsed strictly as `0`/`1` at first use; the
/// Switch row also takes true/on/yes and false/off/no, and rejects the rest at startup.
#[test]
fn prefill_zero_rows_matches_its_strict_reader() {
    let old = |v: Option<&str>| match v {
        None | Some("0") => Some(false),
        Some("1") => Some(true),
        Some(_) => None,
    };
    let new = |v: Option<&str>| {
        let pairs: Vec<(&str, &str)> = v
            .map(|v| ("METRALE_PREFILL_ZERO_ROWS", v))
            .into_iter()
            .collect();
        settings_from(&pairs)
            .ok()
            .map(|s| s.runtime.prefill_zero_rows)
    };
    for v in [None, Some("0"), Some("1"), Some("garbage"), Some("")] {
        assert_eq!(new(v), old(v), "{v:?}");
    }
    assert_eq!(
        (new(Some("true")), new(Some("off"))),
        (Some(true), Some(false))
    );
}

/// 2026-10-17: Verbatim text and path rows: every probe, `""` included, reads as before.
#[test]
fn text_and_path_rows_are_verbatim() {
    type TextField = fn(&S) -> Option<String>;
    let rows: [(&str, TextField); 3] = [
        ("METRALE_DFLASH_DEBUG_NO_GRAPH", |s| {
            s.telemetry.dflash_debug_no_graph.clone()
        }),
        ("METRALE_NEMO_DUMP", |s| s.telemetry.nemo_dump.clone()),
        ("METRALE_REDZONE_RANGE_FILE", |s| {
            s.storage.redzone_range_file.clone()
        }),
    ];
    for (env, field) in rows {
        assert!(
            check(
                env,
                Old::Verbatim,
                |v: Option<&str>| v.map(str::to_owned),
                field
            )
            .is_empty()
        );
        let s = settings_from(&[(env, "")]).unwrap();
        assert_eq!(field(&s).as_deref(), Some(""), "{env}");
    }
}

/// 2026-10-17: `METRALE_DECODE_CKPT_BLOCKS` keeps its interval for every integer; a
/// non-integer (was the default interval) is now a startup error, and surrounding spaces
/// now parse.
#[test]
fn decode_ckpt_blocks_matches_its_reader() {
    use crate::model::trait_impl::decode_checkpoint::decode_ckpt_interval;
    let old = |v: Option<&str>, multi: bool| {
        let blocks = v.and_then(|s| s.parse::<usize>().ok()).filter(|&v| v > 0);
        decode_ckpt_interval(blocks.map(|b| b as i64), multi)
    };
    for multi in [false, true] {
        let new = |s: &S| decode_ckpt_interval(s.cache.decode_ckpt_blocks, multi);
        let changes = check(
            "METRALE_DECODE_CKPT_BLOCKS",
            Old::Number,
            |v: Option<&str>| old(v, multi),
            new,
        );
        assert!(
            changes.iter().all(|&(_, _, o)| o == Outcome::Rejected),
            "{changes:?}"
        );
        for v in ["8", "-3", "0"] {
            let s = settings_from(&[("METRALE_DECODE_CKPT_BLOCKS", v)]).unwrap();
            assert_eq!(new(&s), old(Some(v), multi), "{v:?}");
        }
        assert!(settings_from(&[("METRALE_DECODE_CKPT_BLOCKS", "")]).is_err());
        let spaced = settings_from(&[("METRALE_DECODE_CKPT_BLOCKS", " 8 ")]).unwrap();
        assert_eq!(new(&spaced), old(Some("8"), multi));
    }
}
