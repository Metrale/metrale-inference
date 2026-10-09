// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The DFlash drafter context after every verify width, from the scheduler's trace:
//! each single-sequence verify on a DFlash serve commits exactly its accepted rows, at the
//! anchor's position, before the recurrent state is committed.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::runner::run_scenario;
use super::scenarios;
use super::tests::SERIAL;

/// 2026-10-09: The single-sequence verify calls, by the label the recording model logs.
const VERIFY_LABELS: [&str; 5] = [
    "decode_verify_graphed(",
    "decode_verify_graphed_k3(",
    "decode_verify_graphed_k4(",
    "decode_and_verify_fused(",
    "decode_verify_dflash(",
];

/// 2026-10-09: One DFlash request whose drafter proposes `num_drafts` drafts, on one rank or
/// on several (`ep`, which takes the graphed K=2/3/4 forwards instead of the fused one).
fn dflash_trace(num_drafts: usize, ep: bool) -> Vec<String> {
    let mut sc = scenarios::dflash("dflash_ctx", 1);
    sc.opts.num_drafts = num_drafts;
    sc.cfg.ep = ep;
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(&sc)
}

/// 2026-10-09: `(k, sequence length after the verify)` of a verify line
/// `label(tokens=[..], s1@0:len=N) -> [..]`.
fn verify_shape(line: &str) -> (usize, usize) {
    let tokens = line
        .split("tokens=[")
        .nth(1)
        .and_then(|r| r.split(']').next())
        .expect("tokens in the verify line");
    let k = tokens.split(", ").count();
    let len = line
        .split(":len=")
        .nth(1)
        .and_then(|r| r.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|d| d.parse().ok())
        .expect("len in the verify line");
    (k, len)
}

/// 2026-10-09: The value of the `, key=` argument in a trace line (the comma keeps `n=` from
/// matching inside `len=`).
fn field(line: &str, key: &str) -> usize {
    line.split(&format!(", {key}="))
        .nth(1)
        .and_then(|r| r.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|d| d.parse().ok())
        .unwrap_or_else(|| panic!("no {key}= in {line}"))
}

/// 2026-10-09: Every verify is followed, before its `commit_accepted_prefix`, by exactly one
/// `commit_ctx` of the rows that prefix commits (`n == na`), based at the anchor's position
/// (the length before the verify). Returns the labels of the verifies it checked.
fn check_every_verify_commits_its_rows(trace: &[String]) -> Vec<&'static str> {
    let mut seen = Vec::new();
    for (i, line) in trace.iter().enumerate() {
        let Some(label) = VERIFY_LABELS.iter().find(|l| line.starts_with(**l)) else {
            continue;
        };
        seen.push(*label);
        let (k, len_after) = verify_shape(line);
        let pre = len_after - k;
        // 2026-10-09: This verify's lines run to the next verify. A verdict whose emit ends
        // the sequence stops before the state commit; its context commit came first.
        let rest = &trace[i + 1..];
        let end = rest
            .iter()
            .position(|l| VERIFY_LABELS.iter().any(|v| l.starts_with(v)))
            .unwrap_or(rest.len());
        let own = &rest[..end];
        let state_commit = own
            .iter()
            .position(|l| l.starts_with("commit_accepted_prefix("));
        let ctx: Vec<&String> = own[..state_commit.unwrap_or(own.len())]
            .iter()
            .filter(|l| l.starts_with("commit_ctx("))
            .collect();
        assert_eq!(
            ctx.len(),
            1,
            "verify at line {i} ({line}) commits the drafter context {} times before its \
             state commit",
            ctx.len()
        );
        assert_eq!(
            field(ctx[0], "base"),
            pre,
            "verify at line {i} ({line}): {} must sit at the anchor's position {pre}",
            ctx[0]
        );
        if let Some(c) = state_commit {
            let rows = field(&own[c], "na");
            assert_eq!(
                field(ctx[0], "n"),
                rows,
                "verify at line {i} ({line}): {} must commit the {rows} rows the state keeps",
                ctx[0]
            );
        }
    }
    seen
}

/// 2026-10-09: One, two and three drafts take the K=2, K=3 and K=4 arms (the fused forward
/// on one rank), and each commits its accepted rows as the K=γ arm does.
#[test]
fn every_single_rank_verify_width_commits_its_accepted_rows() {
    for (nd, label) in [
        (1, "decode_and_verify_fused("),
        (2, "decode_and_verify_fused("),
        (3, "decode_and_verify_fused("),
        (5, "decode_verify_dflash("),
    ] {
        let trace = dflash_trace(nd, false);
        let seen = check_every_verify_commits_its_rows(&trace);
        assert!(
            seen.contains(&label),
            "{nd} drafts ran no {label} verify: {seen:?}"
        );
    }
}

/// 2026-10-09: The same on several ranks, where K=2, K=3 and K=4 run the graphed forwards.
#[test]
fn every_multi_rank_verify_width_commits_its_accepted_rows() {
    for (nd, label) in [
        (1, "decode_verify_graphed("),
        (2, "decode_verify_graphed_k3("),
        (3, "decode_verify_graphed_k4("),
    ] {
        let trace = dflash_trace(nd, true);
        let seen = check_every_verify_commits_its_rows(&trace);
        assert!(
            seen.contains(&label),
            "{nd} drafts ran no {label} verify: {seen:?}"
        );
    }
}
