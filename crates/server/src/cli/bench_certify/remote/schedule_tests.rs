// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the Speed-mode decision, `next_for` and the dry-run simulation.
//!
//! Owner: server CLI (`met benchmark certify`).
//! Invariants: none beyond the types.

use super::super::super::plan::{Estimate, Unit};
use super::*;
use metrale_bench::hardware::equivalence::{EquivalencePolicy, HardwareFingerprint};

fn gb10_policy() -> Option<EquivalencePolicy> {
    Some(EquivalencePolicy {
        clock_spread: 0.01,
        mem_spread: 0.05,
        chassis_delta_c: 15.0,
    })
}

fn unit(id: &'static str, group: Option<&'static str>, class: Sensitivity, secs: u64) -> Unit {
    shard_unit(id, group, None, class, secs)
}

fn shard_unit(
    id: &'static str,
    group: Option<&'static str>,
    shard: Option<(usize, usize)>,
    class: Sensitivity,
    secs: u64,
) -> Unit {
    Unit {
        id,
        group,
        shard,
        class,
        estimate: Estimate::Declared(secs),
        needs_confirmation: false,
        serve_allowance_s: 600,
    }
}

fn gb10(chassis: f64) -> HardwareFingerprint {
    HardwareFingerprint {
        gpu: "NVIDIA GB10".into(),
        driver_major: Some(580),
        driver_full: Some("580.126.09".into()),
        vbios: Some("9A.0B.1E.00.00".into()),
        sm_clock_max_mhz: Some(3003.0),
        mem_total_kb: Some(127_601_452),
        thermal_alert: Some(false),
        hottest_chassis_c: Some(chassis),
        postcheck_valid: None,
    }
}

fn node(addr: &str, chassis: f64, free: f64, built: bool) -> Node {
    Node {
        addr: addr.into(),
        name: addr.into(),
        node_id: String::new(),
        signer: "s".into(),
        hardware: gb10(chassis),
        free_fraction: Some(free),
        built,
        local: false,
    }
}

/// 2026-09-26: Ten plain gates, and the two groups as four shards each.
fn campaign() -> Vec<Unit> {
    let mut v = vec![
        unit("decode-floor", None, Sensitivity::Speed, 180),
        unit("ttft-warm-gate", None, Sensitivity::Speed, 160),
        unit("ttft-cold-gate", None, Sensitivity::Speed, 120),
        unit("agentic-webserver", None, Sensitivity::Speed, 600),
        unit("concurrency-sweep", None, Sensitivity::Speed, 1560),
        unit("concurrency-sweep-dflash2", None, Sensitivity::Speed, 300),
        unit("vision-fidelity", None, Sensitivity::Correctness, 120),
        unit("video-fidelity", None, Sensitivity::Correctness, 70),
        unit(
            "ssm-state-poisoning-gate",
            None,
            Sensitivity::Correctness,
            150,
        ),
        unit("kat-equality-gate", None, Sensitivity::Correctness, 4200),
    ];
    for s in 0..4 {
        v.push(shard_unit(
            "bfcl-subset",
            Some("bfcl-subset"),
            Some((s, 4)),
            Sensitivity::Correctness,
            1500,
        ));
        v.push(shard_unit(
            "bfcl-subset-echolp",
            Some("bfcl-subset-echolp"),
            Some((s, 4)),
            Sensitivity::Correctness,
            1900,
        ));
    }
    v
}

/// 2026-09-26: One node spreads (there is nothing to bundle); two or more always
/// bundle, whether or not they look equivalent now.
#[test]
fn speed_mode_bundles_on_more_than_one_box_whatever_they_look_like_at_rest() {
    assert_eq!(
        speed_mode(&[node("a", 65.0, 0.9, true)], gb10_policy()),
        SpeedMode::Spread
    );
    // 2026-09-26: Equivalent now: still bundled, and the reason says why.
    match speed_mode(
        &[node("a", 65.0, 0.9, true), node("b", 70.0, 0.9, true)],
        gb10_policy(),
    ) {
        SpeedMode::Bundle { why, .. } => {
            assert_eq!(why.len(), 1, "{why:?}");
            assert!(why[0].contains("under load"), "{why:?}");
        }
        other => panic!("{other:?}"),
    }
    // 2026-09-26: A pair 24 °C apart: bundled on the box with more free memory, and
    // the mismatch is named beside the default reason.
    let m = speed_mode(
        &[node("a", 65.0, 0.80, true), node("b", 89.0, 0.90, true)],
        gb10_policy(),
    );
    match m {
        SpeedMode::Bundle { node, why } => {
            assert_eq!(node, 1, "more free memory wins");
            assert!(why[1].contains("chassis 65 vs 89"), "{why:?}");
        }
        other => panic!("{other:?}"),
    }
    // 2026-09-26: Equal memory: the cooler box.
    let m = speed_mode(
        &[node("a", 65.0, 0.9, true), node("b", 89.0, 0.9, true)],
        gb10_policy(),
    );
    assert!(matches!(m, SpeedMode::Bundle { node: 0, .. }), "{m:?}");
    // 2026-09-26: A pair where one node has no chassis reading: `Bundle`.
    let mut blind = node("c", 65.0, 0.9, true);
    blind.hardware.hottest_chassis_c = None;
    assert!(matches!(
        speed_mode(&[node("a", 65.0, 0.9, true), blind], gb10_policy()),
        SpeedMode::Bundle { .. }
    ));
    // 2026-09-26: No policy (`--dangerous-ignore-thermals`): bundled, and the reason
    // says why.
    match speed_mode(
        &[node("a", 65.0, 0.9, true), node("b", 66.0, 0.9, true)],
        None,
    ) {
        SpeedMode::Bundle { why, .. } => assert!(why[1].contains("no thermal envelope"), "{why:?}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn next_for_is_longest_first_with_shard_anti_affinity_and_the_speed_rule() {
    let units = campaign();
    let mut pending = vec![true; units.len()];
    let mut placed = vec![None; units.len()];
    // 2026-09-26: Spread: the longest unit of all first, on any node.
    let first = next_for(1, &units, &pending, &placed, &SpeedMode::Spread, None).unwrap();
    assert_eq!(units[first].id, "kat-equality-gate");
    pending[first] = false;
    placed[first] = Some(1);
    // 2026-09-26: Node 1 again: an echolp shard (1900); once it hosts one, the next
    // echolp shard yields to the longest other unit (bfcl 1500 < sweep 1560,
    // so the sweep).
    let e = next_for(1, &units, &pending, &placed, &SpeedMode::Spread, None).unwrap();
    assert!(
        units[e].label().starts_with("bfcl-subset-echolp["),
        "{}",
        units[e].label()
    );
    pending[e] = false;
    placed[e] = Some(1);
    let next = next_for(1, &units, &pending, &placed, &SpeedMode::Spread, None).unwrap();
    assert_eq!(
        units[next].id, "concurrency-sweep",
        "shard anti-affinity yields"
    );
    // 2026-09-26: Another node takes the next echolp shard freely.
    let e2 = next_for(0, &units, &pending, &placed, &SpeedMode::Spread, None).unwrap();
    assert!(units[e2].label().starts_with("bfcl-subset-echolp["));
    // 2026-09-26: Bundle on node 0: node 1 never gets a Speed unit, even when only
    // Speed units remain.
    let only_speed: Vec<bool> = units
        .iter()
        .map(|u| u.class == Sensitivity::Speed)
        .collect();
    let bundle = SpeedMode::Bundle {
        node: 0,
        why: vec![],
    };
    assert_eq!(
        next_for(1, &units, &only_speed, &placed, &bundle, None),
        None
    );
    let s = next_for(0, &units, &only_speed, &placed, &bundle, None).unwrap();
    assert_eq!(units[s].id, "concurrency-sweep");
    assert_eq!(
        next_for(
            0,
            &units,
            &vec![false; units.len()],
            &placed,
            &SpeedMode::Spread,
            None
        ),
        None
    );
}

#[test]
fn the_simulation_is_work_conserving_and_near_optimal() {
    let units = campaign();
    let total: u64 = units.iter().map(Unit::secs).sum();
    let longest = units.iter().map(Unit::secs).max().unwrap();
    // 2026-09-26: One node: the serial sum.
    let one = simulate(
        &units,
        &[node("a", 65.0, 0.9, true)],
        &SpeedMode::Spread,
        None,
        1800,
    );
    assert_eq!(one.makespan_secs, total);
    assert_eq!(one.queues[0].len(), units.len());
    // 2026-09-26: Three equivalent nodes: every unit placed exactly once, and the
    // makespan at most `lower * 4/3 + longest/3`.
    let nodes = [
        node("a", 65.0, 0.9, true),
        node("b", 66.0, 0.9, true),
        node("c", 67.0, 0.9, true),
    ];
    let three = simulate(&units, &nodes, &SpeedMode::Spread, None, 1800);
    let placed: usize = three.queues.iter().map(Vec::len).sum();
    assert_eq!(placed, units.len());
    let lower = (total / 3).max(longest);
    assert!(
        three.makespan_secs <= lower * 4 / 3 + longest / 3,
        "makespan {} vs lower bound {lower}",
        three.makespan_secs
    );
    assert!(three.makespan_secs < total / 2, "{}", three.makespan_secs);
    // 2026-09-26: A node without the anchor built pays the build allowance once.
    let cold = [node("a", 65.0, 0.9, true), node("b", 66.0, 0.9, false)];
    let p = simulate(&units, &cold, &SpeedMode::Spread, None, 1800);
    let b_work: u64 = p.queues[1].iter().map(|&i| units[i].secs()).sum();
    assert_eq!(p.finish_at[1], b_work + 1800);
    // 2026-09-26: Bundle: every Speed unit is on the home node and nowhere else.
    let bundle = SpeedMode::Bundle {
        node: 0,
        why: vec![],
    };
    let p = simulate(&units, &nodes, &bundle, None, 0);
    for (k, q) in p.queues.iter().enumerate() {
        for &i in q {
            if units[i].class == Sensitivity::Speed {
                assert_eq!(k, 0, "{} landed on node {k}", units[i].label());
            }
        }
    }
    assert_eq!(p.queues.iter().map(Vec::len).sum::<usize>(), units.len());
}

fn three_nodes() -> [Node; 3] {
    [
        node("a", 65.0, 0.9, true),
        node("b", 66.0, 0.9, true),
        node("c", 67.0, 0.9, true),
    ]
}

fn sweep_pinned_to(node: usize) -> EnergyPin {
    EnergyPin {
        node,
        gates: BTreeSet::from(["concurrency-sweep"]),
    }
}

fn bundle_on(node: usize) -> SpeedMode {
    SpeedMode::Bundle { node, why: vec![] }
}

/// 2026-10-04: Under a pin, every Speed unit runs on the reference, whatever
/// mode `next_for` is handed and wherever that mode would bundle; Correctness
/// units still spread.
#[test]
fn every_speed_unit_runs_on_the_reference_under_either_mode() {
    let units = campaign();
    let nodes = three_nodes();
    let pin = sweep_pinned_to(2);
    for mode in [bundle_on(0), SpeedMode::Spread] {
        let p = simulate(&units, &nodes, &mode, Some(&pin), 0);
        for (k, q) in p.queues.iter().enumerate() {
            for &i in q {
                if units[i].class == Sensitivity::Speed {
                    assert_eq!(k, 2, "{} on {k} under {mode:?}", units[i].label());
                }
            }
        }
        assert_eq!(p.queues.iter().map(Vec::len).sum::<usize>(), units.len());
        assert!(
            p.queues[..2].iter().any(|q| !q.is_empty()),
            "Correctness units still spread under {mode:?}"
        );
        let only_sweep: Vec<bool> = units.iter().map(|u| u.id == "concurrency-sweep").collect();
        let placed = vec![None; units.len()];
        assert_eq!(
            next_for(0, &units, &only_sweep, &placed, &mode, Some(&pin)),
            None
        );
        assert_eq!(
            next_for(1, &units, &only_sweep, &placed, &mode, Some(&pin)),
            None
        );
        let s = next_for(2, &units, &only_sweep, &placed, &mode, Some(&pin)).unwrap();
        assert_eq!(units[s].id, "concurrency-sweep");
    }
}

/// 2026-10-04: The campaign's own path (`speed_mode`, then
/// `bundle_on_reference`): for every choice of reference, with or without
/// energy-bounded gates, the Speed class sits on exactly one node, the reference.
#[test]
fn with_a_reference_the_speed_class_never_spans_two_nodes() {
    let units = campaign();
    let nodes = three_nodes();
    for reference in 0..3 {
        for gates in [BTreeSet::new(), BTreeSet::from(["concurrency-sweep"])] {
            let pin = EnergyPin {
                node: reference,
                gates,
            };
            let mode = bundle_on_reference(speed_mode(&nodes, gb10_policy()), &pin);
            assert!(
                matches!(&mode, SpeedMode::Bundle { node, why }
                    if *node == reference && why.last().unwrap().contains("--energy-reference-node")),
                "{mode:?}"
            );
            let p = simulate(&units, &nodes, &mode, Some(&pin), 1800);
            let hosts: BTreeSet<usize> = p
                .queues
                .iter()
                .enumerate()
                .filter(|(_, q)| q.iter().any(|&i| units[i].class == Sensitivity::Speed))
                .map(|(k, _)| k)
                .collect();
            assert_eq!(hosts, BTreeSet::from([reference]), "{:?}", pin.gates);
        }
    }
    let one = [node("a", 65.0, 0.9, true)];
    let pin = sweep_pinned_to(0);
    assert_eq!(
        bundle_on_reference(speed_mode(&one, gb10_policy()), &pin),
        SpeedMode::Spread
    );
}

/// 2026-10-04: For every Correctness unit, on every node and under every mode,
/// the pin changes nothing about where it may run.
#[test]
fn correctness_units_keep_their_placement_rule_under_a_pin() {
    let units = campaign();
    let pin = sweep_pinned_to(1);
    for mode in &[SpeedMode::Spread, bundle_on(0), bundle_on(1)] {
        for u in &units {
            for k in 0..3 {
                let pinned = may_run(u, k, mode, Some(&pin));
                if u.class == Sensitivity::Speed {
                    assert_eq!(pinned, k == 1, "{} on {k} under {mode:?}", u.label());
                } else {
                    assert!(pinned, "{} on {k} under {mode:?}", u.label());
                    assert_eq!(pinned, may_run(u, k, mode, None));
                }
            }
        }
    }
}

/// 2026-10-04: The reference must be admitted, spelled as the fleet spells it.
#[test]
fn a_reference_outside_the_admitted_fleet_is_refused() {
    let nodes = [node("a:1", 65.0, 0.9, true), node("b:2", 66.0, 0.9, true)];
    let rejected = [Rejection {
        addr: "c:3".into(),
        why: "anchor not built".into(),
    }];
    let gates = BTreeSet::from(["concurrency-sweep"]);
    assert_eq!(
        energy_pin(&nodes, &rejected, "b:2", gates.clone()),
        Ok(EnergyPin {
            node: 1,
            gates: gates.clone()
        })
    );
    let e = energy_pin(&nodes, &rejected, "c:3", gates.clone()).unwrap_err();
    assert!(
        e.contains("not admitted") && e.contains("anchor not built"),
        "{e}"
    );
    for absent in ["d:4", "b"] {
        let e = energy_pin(&nodes, &rejected, absent, gates.clone()).unwrap_err();
        assert!(
            e.contains("not in the fleet") && e.contains("a:1, b:2"),
            "{e}"
        );
    }
}
