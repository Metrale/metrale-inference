// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The multi-rank batched DFlash verify's wire order on rank 0, from the scheduler's
//! trace: the announcement right before the forward and the verdict right after it, before the
//! first per-sequence action.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::runner::run_scenario;
use super::scenarios;
use super::tests::SERIAL;

/// 2026-10-09: The two-sequence DFlash scenario on a multi-rank model.
fn ep_trace() -> Vec<String> {
    let mut sc = scenarios::dflash("dflash_batched_ep", 2);
    sc.cfg.ep = true;
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(&sc)
}

/// 2026-10-09: Every batched verify is announced (opcode, n, k, the slots, the n * k tokens) on
/// the lines right before it, and its verdict (one word per sequence) is the line right after
/// it, before `commit_ctx` or any other per-sequence step.
#[test]
fn every_batched_verify_is_announced_before_and_judged_right_after() {
    let trace = ep_trace();
    let verifies: Vec<usize> = trace
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("decode_verify_batched("))
        .map(|(i, _)| i)
        .collect();
    assert!(!verifies.is_empty(), "the scenario ran no batched verify");
    for &i in &verifies {
        assert!(
            i >= 5,
            "verify at line {i} has no room for its announcement"
        );
        let ks = trace[i]
            .split("ks=[")
            .nth(1)
            .and_then(|r| r.split(']').next())
            .expect("ks in the verify line");
        let ks: Vec<usize> = ks.split(", ").map(|k| k.parse().unwrap()).collect();
        let (n, k) = (ks.len(), ks[0]);
        assert_eq!(
            &trace[i - 5..i],
            &[
                "ep_broadcast_cmd_for_seq(seq=0, cmd=0xfffffff9)".to_string(),
                format!("ep_broadcast_cmd({n:#x})"),
                format!("ep_broadcast_cmd({k:#x})"),
                format!("ep_broadcast_tokens(n={n})"),
                format!("ep_broadcast_tokens(n={})", n * k),
            ],
            "announcement before line {i}"
        );
        assert_eq!(
            trace[i + 1],
            format!("ep_broadcast_tokens(n={n})"),
            "verdict after line {i}"
        );
    }
}

/// 2026-10-09: The single-rank trace of the same scenario sends nothing for a batched verify,
/// so the announcement above comes from the multi-rank branch alone.
#[test]
fn a_single_rank_batched_verify_sends_nothing() {
    let sc = scenarios::dflash("dflash_batched", 2);
    let trace = {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        run_scenario(&sc)
    };
    assert!(
        trace
            .iter()
            .any(|l| l.starts_with("decode_verify_batched("))
    );
    assert!(!trace.iter().any(|l| l.contains("cmd=0xfffffff9")));
}
