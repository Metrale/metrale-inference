// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The speculation controller wired into the decode lane (`spec_host`): with the
//! controller armed (no `--mtp-gate force`), every speculative golden scenario must emit
//! byte-identical client streams, because the controller chooses how many drafts a step
//! verifies (or plain decode), never which token is emitted; and the model-call trace must
//! differ from the forced golden, which proves the controller actually chose differently.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::runner::run_scenario;
use super::scenarios;
use super::tests::{SERIAL, golden_dir};

/// 2026-10-10: What the clients saw, without the accepted-drafts count of the `Done` line
/// (`acc=N`): that count is the speculation statistic the controller exists to change.
fn client_lines(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|l| l.starts_with("out s") || l.starts_with("rotation ack"))
        .map(|l| {
            l.split(", ")
                .filter(|f| !f.starts_with("acc="))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .collect()
}

#[test]
fn the_controller_changes_the_steps_and_never_the_client_bytes() {
    let speculative = [
        "verify_k2",
        "verify_k3",
        "verify_k4",
        "batched_verify_k4",
        "dflash",
        "dflash_batched",
    ];
    for name in speculative {
        let mut sc = scenarios::all()
            .into_iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("no scenario {name}"));
        assert!(
            sc.opts.mtp_gate_force,
            "{name}: the golden runs the forced lane"
        );
        sc.opts.mtp_gate_force = false;
        sc.opts.spec_entry_pin_tokens = 0;
        let live = {
            let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
            run_scenario(&sc)
        };
        let golden = std::fs::read_to_string(golden_dir().join(format!("{name}.trace")))
            .unwrap_or_else(|e| panic!("golden for {name}: {e}"));
        let golden: Vec<String> = golden.lines().map(str::to_string).collect();
        assert_eq!(
            client_lines(&live),
            client_lines(&golden),
            "{name}: the controller changed what a client saw"
        );
        assert_ne!(
            live, golden,
            "{name}: the armed controller ran exactly the forced lane's steps (lever not moved)"
        );
    }
}
