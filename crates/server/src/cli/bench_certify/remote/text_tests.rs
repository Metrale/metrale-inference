// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the energy-pin lines of the fleet plan.
//!
//! Owner: server CLI (`met benchmark certify`).
//! Invariants: none beyond the types.

use std::collections::BTreeSet;

use metrale_bench::hardware::equivalence::HardwareFingerprint;

use super::super::node::Node;
use super::super::schedule::EnergyPin;
use super::*;

fn node(addr: &str) -> Node {
    Node {
        addr: addr.into(),
        name: addr.into(),
        node_id: String::new(),
        signer: "s".into(),
        hardware: HardwareFingerprint {
            gpu: "NVIDIA GB10".into(),
            driver_major: Some(580),
            sm_clock_max_mhz: Some(3003.0),
            mem_total_kb: Some(127_601_452),
            thermal_alert: Some(false),
            hottest_chassis_c: Some(40.0),
            postcheck_valid: None,
        },
        free_fraction: Some(0.9),
        built: true,
        local: false,
    }
}

fn fleet(home: usize, pin: Option<(usize, &[&'static str])>) -> Fleet {
    Fleet {
        nodes: vec![node("a:1"), node("b:2")],
        rejected: vec![],
        mode: SpeedMode::Bundle {
            node: home,
            why: vec![],
        },
        energy: pin.map(|(node, gates)| EnergyPin {
            node,
            gates: gates.iter().copied().collect::<BTreeSet<_>>(),
        }),
        envelope: None,
    }
}

/// 2026-10-04: The reference is named as the host of every Speed-class gate,
/// with the energy-bounded gates and why, or that there are none.
#[test]
fn the_plan_names_the_reference_and_the_pinned_gates() {
    assert!(energy_pin_lines(&fleet(0, None)).is_empty());
    let pinned = energy_pin_lines(&fleet(1, Some((1, &["concurrency-sweep"]))));
    assert_eq!(pinned.len(), 1, "{pinned:?}");
    assert!(
        pinned[0]
            .starts_with("reference b:2 (--energy-reference-node) hosts every Speed-class gate")
            && pinned[0].contains("PINNED: concurrency-sweep.")
            && pinned[0].contains("gpu_rail_joules_per_token"),
        "{pinned:?}"
    );
    let none = energy_pin_lines(&fleet(1, Some((1, &[]))));
    assert_eq!(
        none,
        [
            "reference b:2 (--energy-reference-node) hosts every Speed-class gate; no \
          energy-bounded gate in this campaign"
        ]
    );
}
