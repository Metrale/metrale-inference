// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the energy-pin lines of the fleet plan.
//!
//! Owner: server CLI (`met benchmark certify`).
//! Invariants: none beyond the types.

use std::collections::BTreeSet;

use metrale_bench::hardware::equivalence::HardwareFingerprint;

use super::super::super::plan::Estimate;
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

fn unit(id: &'static str, class: Sensitivity) -> Unit {
    Unit {
        id,
        group: None,
        shard: None,
        class,
        estimate: Estimate::Declared(600),
        needs_confirmation: false,
        serve_allowance_s: 600,
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

/// 2026-10-04: The pinned gates and the node are named; the two-signer note
/// appears only when pinned and unpinned Speed gates land on different nodes.
#[test]
fn the_plan_names_the_pinned_gates_and_a_split_speed_class() {
    let units = [
        unit("concurrency-sweep", Sensitivity::Speed),
        unit("decode-floor", Sensitivity::Speed),
        unit("bfcl-subset", Sensitivity::Correctness),
    ];
    assert!(energy_pin_lines(&fleet(0, None), &units).is_empty());

    let split = energy_pin_lines(&fleet(0, Some((1, &["concurrency-sweep"]))), &units);
    assert_eq!(split.len(), 2, "{split:?}");
    assert!(
        split[0].starts_with("energy-bounded gates PINNED on b:2")
            && split[0].contains(": concurrency-sweep.")
            && split[0].contains("gpu_rail_joules_per_token"),
        "{split:?}"
    );
    assert!(
        split[1].contains("other speed-class gates run on a:1"),
        "{split:?}"
    );

    let same_box = energy_pin_lines(&fleet(1, Some((1, &["concurrency-sweep"]))), &units);
    assert_eq!(same_box.len(), 1, "{same_box:?}");
    let all_speed_pinned = energy_pin_lines(
        &fleet(0, Some((1, &["concurrency-sweep", "decode-floor"]))),
        &units,
    );
    assert_eq!(all_speed_pinned.len(), 1, "{all_speed_pinned:?}");

    let none = energy_pin_lines(&fleet(0, Some((1, &[]))), &units);
    assert_eq!(
        none,
        ["no energy-bounded gate in this campaign; nothing PINNED on b:2"]
    );
}
