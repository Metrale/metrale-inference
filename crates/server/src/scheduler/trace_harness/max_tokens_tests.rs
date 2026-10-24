// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-20: `max_tokens: 0` must not underflow the remaining budget.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::runner::{EOS, ReqSpec, RunOptions, Scenario, run_scenario};
use super::tests::SERIAL;

fn traced(sc: Scenario) -> Vec<String> {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(&sc)
}

fn client_output(t: &[String]) -> Vec<&String> {
    t.iter().filter(|l| l.starts_with("out s")).collect()
}

/// 2026-10-20: `max_tokens: 0` with a spontaneous think start (the first
/// sampled token is the think-start token) skips `immediate_finish`, so the
/// remaining-budget computation used to underflow (`0 - 1`): release builds
/// ran the request essentially forever, debug builds panicked. The sequence
/// must finish at once instead.
#[test]
fn zero_max_tokens_with_spontaneous_think_does_not_underflow() {
    let sc = Scenario {
        name: "zero_max_tokens_spontaneous_think",
        cfg: Default::default(),
        opts: RunOptions {
            think_start_token: Some(10),
            ..RunOptions::default()
        },
        reqs: vec![ReqSpec {
            max_tokens: 0,
            ..ReqSpec::new(1, 8, vec![10, EOS])
        }],
    };
    let t = traced(sc);
    let out = client_output(&t);
    assert!(
        out.iter().any(|l| l.contains("Done")),
        "the request never finished:\n{t:#?}"
    );
}
