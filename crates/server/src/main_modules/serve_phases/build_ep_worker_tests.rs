// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The EP worker loop in `build.rs`, asserted on its source.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

/// 2026-09-26: The worker loop keeps running after an `EpCommandFailed` and
/// exits on any other error. Asserted on the source: the `EpCommandFailed`
/// arm comes before the catch-all `break`.
#[test]
fn a_command_failure_keeps_the_worker_up_and_a_link_failure_does_not() {
    let src = include_str!("build.rs");
    let loop_body = src
        .split_once("match model_owned.ep_worker_step(&mut slots)")
        .expect("the EP worker loop must exist")
        .1;
    let recoverable = loop_body
        .find("EpCommandFailed")
        .expect("the loop must classify command failures");
    let stays_up = loop_body
        .find("worker STAYS UP")
        .expect("the recoverable arm must say so in the log");
    let fatal = loop_body
        .find("break;\n                }\n            }\n        }")
        .expect("the fatal arm must still break");
    assert!(
        recoverable < stays_up && stays_up < fatal,
        "the EpCommandFailed arm must come BEFORE the catch-all break, or every command \
         failure is fatal again"
    );
}
