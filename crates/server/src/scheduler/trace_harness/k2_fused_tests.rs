// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The fused multi-rank K=2 MTP wire (`speculative::k2_wire`) on rank 0, from the
//! scheduler's trace: one announcement and one verdict per K=2 step, the propose told by the
//! verdict, and the client streams unchanged.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::pipeline_tests::{client_lines, sync_run};
use super::runner::Scenario;
use super::scenarios;

/// 2026-10-11: The K=2 scenario (a draft rejected every fourth step, ends on EOS) on a
/// multi-rank model, with the fused wire on or off.
fn k2_ep(fused: bool) -> Scenario {
    let mut sc = scenarios::verify_k(if fused { "k2_fused" } else { "k2_unfused" }, 1);
    sc.cfg.ep = true;
    sc.cfg.k2_fused_wire = fused;
    sc
}

/// 2026-10-11: The trace cut at its tick lines; only ticks that ran a K=2 verify.
fn verify_ticks(lines: &[String]) -> Vec<Vec<String>> {
    let mut ticks: Vec<Vec<String>> = vec![Vec::new()];
    for l in lines {
        if l.starts_with("-- tick") {
            ticks.push(Vec::new());
        } else {
            ticks.last_mut().unwrap().push(l.clone());
        }
    }
    ticks
        .into_iter()
        .filter(|t| t.iter().any(|l| l.starts_with("decode_verify_graphed(")))
        .collect()
}

/// 2026-10-11: The `key=value` field of a trace line, up to `,` or `)`.
fn field<'a>(line: &'a str, key: &str) -> &'a str {
    let rest = line.split(&format!("{key}=")).nth(1).expect(key);
    rest.split([',', ')']).next().unwrap()
}

/// 2026-10-11: The `tokens=[..]` list of a trace line.
fn tokens(line: &str) -> &str {
    let rest = line.split("tokens=[").nth(1).expect("tokens");
    rest.split(']').next().unwrap()
}

fn ep_lines(tick: &[String]) -> Vec<&String> {
    tick.iter().filter(|l| l.starts_with("ep_")).collect()
}

/// 2026-10-11: The clients see the same streams with the fused wire on and off.
#[test]
fn the_fused_wire_leaves_every_stream_unchanged() {
    let on = sync_run(&k2_ep(true));
    let off = sync_run(&k2_ep(false));
    assert_eq!(client_lines(&on), client_lines(&off));
    assert!(!client_lines(&on).is_empty());
}

/// 2026-10-11: Every fused K=2 step sends exactly two things: the announcement right before
/// the verify, naming its tokens, and one verdict after it. The unfused control sends at
/// least five, so the count tells the two apart.
#[test]
fn a_fused_k2_step_sends_one_announcement_and_one_verdict() {
    let ticks = verify_ticks(&sync_run(&k2_ep(true)));
    assert!(ticks.len() >= 4, "too few K=2 steps: {}", ticks.len());
    for t in &ticks {
        let ep = ep_lines(t);
        assert_eq!(ep.len(), 2, "tick sends {ep:?}");
        assert!(ep[0].starts_with("ep_announce_k2_fused("));
        assert!(ep[1].starts_with("ep_send_k2_verdict("));
        let v = t.iter().position(|l| l.starts_with("decode_verify_graphed(")).unwrap();
        assert!(t[v - 1].starts_with("ep_announce_k2_fused("), "{t:?}");
        assert_eq!(
            tokens(&t[v - 1]),
            tokens(&t[v]),
            "announced tokens differ from the verified ones"
        );
    }
    for t in verify_ticks(&sync_run(&k2_ep(false))) {
        assert!(ep_lines(&t).len() >= 5, "unfused control: {t:?}");
        assert!(!t.iter().any(|l| l.starts_with("ep_announce_k2_fused(")));
    }
}

/// 2026-10-11: The verdict tells the workers what rank 0 proposes: a verdict with drafts
/// proposes from its token at its position through the announced path (nothing sent), and a
/// verdict without drafts (the step that finished the sequence) proposes nothing.
#[test]
fn the_verdict_is_the_propose_rank_0_runs() {
    let ticks = verify_ticks(&sync_run(&k2_ep(true)));
    let mut finished = 0;
    for t in &ticks {
        let vi = t.iter().position(|l| l.starts_with("ep_send_k2_verdict(")).unwrap();
        let verdict = &t[vi];
        let after = &t[vi + 1..];
        let proposes: Vec<&String> = after
            .iter()
            .filter(|l| l.starts_with("run_mtp_propose_announced("))
            .collect();
        // 2026-10-11: The fake's propose line follows the announced one; a bare
        // `run_mtp_propose_multi` here would be a propose that announced itself again.
        for (i, l) in after.iter().enumerate() {
            if l.starts_with("run_mtp_propose_multi(") {
                assert!(
                    i > 0 && after[i - 1].starts_with("run_mtp_propose_announced("),
                    "{t:?}"
                );
            }
        }
        if field(verdict, "drafts") == "0" {
            finished += 1;
            assert!(proposes.is_empty(), "proposed after a finishing verdict: {t:?}");
        } else {
            assert_eq!(proposes.len(), 1, "{t:?}");
            assert_eq!(field(proposes[0], "token"), field(verdict, "next"));
            assert_eq!(field(proposes[0], "pos"), field(verdict, "pos"));
            let accepted = field(verdict, "accepted") == "true";
            let saved = format!("save_hidden_for_mtp({}, ", usize::from(accepted));
            assert!(after.iter().any(|l| l.starts_with(&saved)), "{t:?}");
        }
    }
    assert_eq!(finished, 1, "the EOS step should be the one verdict without drafts");
    assert!(
        ticks
            .iter()
            .any(|t| t.iter().any(|l| l.contains("ep_send_k2_verdict(accepted=false"))),
        "the scenario rejected no draft"
    );
}
