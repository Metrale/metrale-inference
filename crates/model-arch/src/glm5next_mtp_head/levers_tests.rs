// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `MtpLevers::from_settings` against the environment readers it replaced, frozen
//! here (`old_*`, unchanged but for taking the raw value as a parameter).
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a `0`/
//! `1` switch now also reads the other switch spellings and refuses a value the switch rule
//! cannot read, and the context cap now reads `-0` as 0 and refuses a value past `i64::MAX`.

use super::*;
use metrale_model_layers::lever_equivalence::{
    Old, Outcome, PROBES, check, defaults, settings_from,
};

/// 2026-10-17: `skip_head` / `skip_block` in `glm5next_mtp_head.rs`, as one reading.
fn old_skip(v: Option<&str>) -> MtpSkip {
    if v == Some("head") {
        MtpSkip::Head
    } else if v == Some("block") {
        MtpSkip::Block
    } else {
        MtpSkip::None
    }
}

/// 2026-10-17: `context_cap::parse_drafter_context_cap`.
fn old_cap(raw: Option<&str>) -> Result<usize, String> {
    match raw {
        None => Ok(65_536),
        Some(s) => s.trim().parse::<usize>().map_err(|_| {
            format!(
                "METRALE_GLM_MTP_CONTEXT_CAP={s:?}: expected a row count (0 = no cap, unset = \
                 65536)"
            )
        }),
    }
}

/// 2026-10-17: `proposer::prefill_waits`.
fn old_prefill_waits(full: Option<&str>, eager: Option<&str>) -> bool {
    full == Some("1") || eager == Some("0")
}

/// 2026-10-17: `pairs` and nothing else, read through `field`; `None` when the settings or the
/// levers refuse them.
fn new_with<T>(pairs: &[(&str, &str)], field: impl Fn(&MtpLevers) -> T) -> Option<T> {
    settings_from(pairs)
        .ok()
        .and_then(|s| MtpLevers::from_settings(&s).ok())
        .map(|l| field(&l))
}

/// 2026-10-17: `env` at `probe` and nothing else, as [`new_with`].
fn new_at<T>(env: &str, probe: Option<&str>, field: impl Fn(&MtpLevers) -> T) -> Option<T> {
    let pairs: Vec<(&str, &str)> = probe.map(|v| (env, v)).into_iter().collect();
    new_with(&pairs, field)
}

/// 2026-10-17: The levers of resolved settings that hold one switch.
fn levers(s: &metrale_settings::InferenceServerSettings) -> MtpLevers {
    MtpLevers::from_settings(s).expect("one switch resolves")
}

#[test]
fn the_skip_arms_keep_their_text_predicates() {
    let env = "METRALE_GLM_MTP_SKIP";
    let changes = check(env, Old::Verbatim, old_skip, |s| levers(s).skip);
    assert!(changes.is_empty(), "{changes:?}");
    for probe in ["head", "block", "Head", " head", "block ", "", "both"] {
        assert_eq!(
            new_at(env, Some(probe), |l| (
                l.skip,
                l.skip_head(),
                l.skip_block()
            )),
            Some((
                old_skip(Some(probe)),
                old_skip(Some(probe)) == MtpSkip::Head,
                old_skip(Some(probe)) == MtpSkip::Block
            )),
            "{probe:?}"
        );
    }
}

#[test]
fn the_full_prefill_arm_keeps_its_text_predicate() {
    let env = "METRALE_GLM_MTP_PREFILL_FULL";
    let changes = check(
        env,
        Old::Verbatim,
        |probe| probe == Some("1"),
        |s| levers(s).prefill_full,
    );
    assert!(changes.is_empty(), "{changes:?}");
    for probe in [" 1", "01", "yes", ""] {
        assert_eq!(
            new_at(env, Some(probe), |l| l.prefill_full),
            Some(false),
            "{probe:?}"
        );
    }
}

#[test]
fn the_context_cap_reads_as_before_except_signed_zero_and_past_i64() {
    let env = "METRALE_GLM_MTP_CONTEXT_CAP";
    for probe in [
        None,
        Some("0"),
        Some("131072"),
        Some(" 4096 "),
        Some("+5"),
        Some(""),
        Some("-1"),
        Some("64k"),
        Some("1.5"),
        Some("off"),
        Some("garbage"),
    ] {
        assert_eq!(
            new_at(env, probe, |l| l.context_cap),
            old_cap(probe).ok(),
            "{probe:?}"
        );
    }
    let past_i64 = (i64::MAX as u64 + 1).to_string();
    for (probe, was, now) in [
        ("-0", None, Some(0)),
        (past_i64.as_str(), Some(1usize << 63), None),
    ] {
        assert_eq!(
            (
                old_cap(Some(probe)).ok(),
                new_at(env, Some(probe), |l| l.context_cap)
            ),
            (was, now),
            "{probe:?}"
        );
    }
    // 2026-10-17: A negative cap resolves as settings and is refused by the levers, with the
    // variable named.
    let s = settings_from(&[(env, "-1")]).expect("an Int resolves");
    let err = MtpLevers::from_settings(&s).expect_err("a negative cap is refused");
    assert!(format!("{err:#}").starts_with("METRALE_GLM_MTP_CONTEXT_CAP=-1"));
}

type Switch = (&'static str, fn(&MtpLevers) -> bool);

/// 2026-10-17: The `== "1"` switches: graph, batched propose, its check, `eh_proj` TP and the
/// device exchange.
#[test]
fn the_opt_in_switches_read_as_before_except_the_switch_spellings() {
    let cases: [Switch; 5] = [
        ("METRALE_GLM_MTP_GRAPH", |l| l.graph),
        ("METRALE_GLM_MTP_PROPOSE_BATCH", |l| l.propose_batch),
        ("METRALE_GLM_MTP_PROPOSE_BATCH_CHECK", |l| {
            l.propose_batch_check
        }),
        ("METRALE_GLM_MTP_EH_TP", |l| l.eh_tp),
        ("METRALE_GLM_MTP_XCHG_DEVICE", |l| l.xchg_device),
    ];
    for (env, field) in cases {
        let changes = check(env, Old::OptIn, |p| p == Some("1"), |s| field(&levers(s)));
        assert_eq!(
            changes,
            [
                (env, Some("true"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ]
        );
    }
}

/// 2026-10-17: The `!= "0"` switches: the eager drafter prefill and the FP8 draft head.
#[test]
fn the_opt_out_switches_read_as_before_except_the_switch_spellings() {
    let cases: [Switch; 2] = [
        ("METRALE_GLM_MTP_EAGER_PREFILL", |l| l.eager_prefill),
        ("METRALE_GLM_MTP_HEAD_FP8", |l| l.head_fp8),
    ];
    for (env, field) in cases {
        let changes = check(env, Old::OptOut, |p| p != Some("0"), |s| field(&levers(s)));
        assert_eq!(
            changes,
            [
                (env, Some("off"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ]
        );
    }
}

/// 2026-10-17: `METRALE_GLM_MTP_DEVICE_STEP`, which the engine reads straight from the settings
/// (`device_step`, `comm_setup`), against `device_step::device_step_enabled`.
#[test]
fn the_device_step_reads_as_before_except_the_switch_spellings() {
    let env = "METRALE_GLM_MTP_DEVICE_STEP";
    let changes = check(
        env,
        Old::OptIn,
        |p| p == Some("1"),
        |s| s.speculative.glm_mtp_device_step,
    );
    assert_eq!(
        changes,
        [
            (env, Some("true"), Outcome::Changed),
            (env, Some("garbage"), Outcome::Rejected)
        ]
    );
}

/// 2026-10-17: The block and DSA workspace rows follow the batched-propose switch as
/// `drafter_block_rows` / `drafter_dsa_rows` did.
#[test]
fn the_workspace_rows_follow_the_batched_propose_as_before() {
    let env = "METRALE_GLM_MTP_PROPOSE_BATCH";
    let old_rows = |p: Option<&str>| {
        let block = if p == Some("1") {
            super::super::batch::PROPOSE_BATCH_MAX
        } else {
            1
        };
        (block, block.max(super::super::ctx_rows::CTX_ROWS_CHUNK))
    };
    let changes = check(env, Old::OptIn, old_rows, |s| {
        let l = levers(s);
        (l.block_rows(), l.dsa_rows())
    });
    assert_eq!(
        changes,
        [
            (env, Some("true"), Outcome::Changed),
            (env, Some("garbage"), Outcome::Rejected)
        ]
    );
}

/// 2026-10-17: The prefill-wait composition over every pair of the two levers' probes the
/// settings accept: equal to `prefill_waits` but where the eager switch reads a switch spelling
/// the old `== "0"` did not.
#[test]
fn the_prefill_wait_composes_as_before() {
    const FULL: &str = "METRALE_GLM_MTP_PREFILL_FULL";
    const EAGER: &str = "METRALE_GLM_MTP_EAGER_PREFILL";
    for full in PROBES {
        for eager in PROBES {
            let mut pairs = Vec::new();
            pairs.extend(full.map(|v| (FULL, v)));
            pairs.extend(eager.map(|v| (EAGER, v)));
            let new = new_with(&pairs, MtpLevers::prefill_waits);
            let old = old_prefill_waits(full, eager);
            match eager {
                Some("garbage") => assert_eq!(new, None, "{full:?} {eager:?}"),
                Some("off") if full != Some("1") => {
                    assert_eq!((old, new), (false, Some(true)), "{full:?} {eager:?}")
                }
                _ => assert_eq!(new, Some(old), "{full:?} {eager:?}"),
            }
        }
    }
}

/// 2026-10-21: The A2 levers (no old reader): the early launch on unset and ROW_NOSYNC off
/// unset, each on at `1` and off at `0`, a non-switch refused; the early launch is refused (off)
/// beside the NCCL host hoist.
#[test]
fn the_a2_levers_resolve_as_declared() {
    type Field = fn(&MtpLevers) -> bool;
    let cases: [(&str, Field, bool); 2] = [
        ("METRALE_GLM_GRAPH_EARLY_LAUNCH", |l| l.graph_early_launch, true),
        ("METRALE_GLM_MTP_ROW_NOSYNC", |l| l.row_nosync, false),
    ];
    for (env, field, unset) in cases {
        assert_eq!(new_with(&[], field), Some(unset), "{env} unset");
        assert_eq!(new_at(env, Some("1"), field), Some(true), "{env}=1");
        assert_eq!(new_at(env, Some("0"), field), Some(false), "{env}=0");
        assert_eq!(new_at(env, Some("garbage"), field), None, "{env}=garbage");
    }
    let hoisted = [("METRALE_NCCL_HOST_HOIST", "1")];
    assert_eq!(
        new_with(&hoisted, |l| l.graph_early_launch),
        Some(false),
        "the early launch beside the hoist"
    );
}

/// 2026-10-17: Every lever at the value its reader read unset as.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    assert_eq!(
        MtpLevers::from_settings(&defaults()).expect("defaults resolve"),
        MtpLevers {
            skip: MtpSkip::None,
            prefill_full: false,
            context_cap: 65_536,
            graph: false,
            eager_prefill: true,
            propose_batch: false,
            propose_batch_check: false,
            eh_tp: false,
            xchg_device: false,
            head_fp8: true,
            graph_early_launch: true,
            row_nosync: false,
        }
    );
    assert_eq!(MtpLevers::declared().context_cap, old_cap(None).unwrap());
    assert!(!defaults().speculative.glm_mtp_device_step);
}
