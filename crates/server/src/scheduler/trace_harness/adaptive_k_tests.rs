// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The single-stream adaptive DFlash draft count (`--dflash-adaptive-k`) through the
//! scheduler (it records the K=γ verify step, `verify_dflash_step`): a step whose drafts all
//! fail, under a table where plain decode is cheaper, stops speculating; a partially accepting stream narrows its proposes below the cap; with the table
//! off the same scenario keeps verifying at the cap.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::runner::run_scenario;
use super::scenarios;
use super::tests::SERIAL;
use metrale_speculative::dflash_adaptive_k::CostModel;

/// 2026-10-10: One DFlash stream with every `wrong`-th draft wrong, run with `table`.
fn trace(wrong: usize, table: Option<&str>) -> Vec<String> {
    let mut sc = scenarios::dflash("dflash_adaptive_k", 1);
    sc.reqs[0].draft_wrong_every = Some(wrong);
    sc.opts.dflash_adaptive_k = table.map(|t| CostModel::parse(t).unwrap());
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(&sc)
}

fn count(trace: &[String], prefix: &str) -> usize {
    trace.iter().filter(|l| l.starts_with(prefix)).count()
}

/// 2026-10-10: The draft counts the stream's proposes asked for, in order.
fn propose_widths(trace: &[String]) -> Vec<usize> {
    trace
        .iter()
        .filter(|l| l.starts_with("run_mtp_propose_multi("))
        .map(|l| {
            l.split(", n=")
                .nth(1)
                .and_then(|r| r.split(',').next())
                .and_then(|n| n.parse().ok())
                .expect("n= in the propose line")
        })
        .collect()
}

/// 2026-10-10: Every draft wrong: with the plain decode step priced below one draft's step,
/// the first K=γ verify (which records the stream's rates) is the last speculative step: no
/// propose or verify follows it, only plain decodes. With the table off the same stream keeps
/// verifying at the cap.
#[test]
fn a_losing_stream_stops_speculating_and_the_control_does_not() {
    let adaptive = trace(1, Some("0:30,1:44,2:50,3:56,4:62,5:68"));
    let first = adaptive
        .iter()
        .position(|l| l.starts_with("decode_verify_dflash"))
        .expect("a K=γ verify");
    let after = &adaptive[first + 1..];
    assert_eq!(count(after, "decode_verify_dflash"), 0, "{adaptive:#?}");
    assert_eq!(count(after, "run_mtp_propose_multi("), 0, "{adaptive:#?}");
    assert!(
        count(after, "decode(") >= 5,
        "plain decode after the suspension"
    );
    let control = trace(1, None);
    assert!(count(&control, "decode_verify_dflash") >= 5, "{control:#?}");
    assert!(propose_widths(&control).iter().all(|&n| n == 5));
}

/// 2026-10-10: Every third draft wrong and no plain-decode entry: the stream keeps proposing
/// but narrows below the cap of 5 once its rates are measured, never to 0.
#[test]
fn a_partial_stream_narrows_and_never_stops() {
    let t = trace(3, Some("1:44,2:50,3:56,4:62,5:68"));
    let widths = propose_widths(&t);
    assert_eq!(widths.first(), Some(&5), "nothing measured: the cap");
    assert!(widths.iter().skip(1).any(|&n| n < 5), "{widths:?}");
    assert!(widths.iter().all(|&n| n >= 1));
    assert!(count(&t, "decode_verify_dflash") >= 2);
}
