// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Tests for multi-node units: `gang::take`, and `schedule::simulate` with
//! a unit that holds three boxes.
//!
//! Owner: server CLI (`met benchmark certify`).
//! Invariants: none beyond the types.

use super::super::super::plan::{Estimate, Unit};
use super::super::node::Node;
use super::super::schedule::simulate;
use super::*;
use metrale_bench::hardware::equivalence::HardwareFingerprint;
use metrale_bench::hardware::policy::Sensitivity;

fn unit(id: &'static str, class: Sensitivity, secs: u64, nodes: usize) -> Unit {
    Unit {
        id,
        group: None,
        shard: None,
        class,
        estimate: Estimate::Declared(secs),
        needs_confirmation: false,
        serve_allowance_s: 600,
        nodes,
    }
}

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
            hottest_chassis_c: Some(65.0),
            postcheck_valid: None,
        },
        free_fraction: Some(0.9),
        built: true,
        local: false,
    }
}

/// 2026-10-15: A 3-node GLM gate (unit 0) and three one-box gates.
fn campaign() -> Vec<Unit> {
    vec![
        unit("ttft-cold-glm-512k", Sensitivity::Speed, 600, 3),
        unit("vision-fidelity", Sensitivity::Correctness, 300, 1),
        unit("video-fidelity", Sensitivity::Correctness, 200, 1),
        unit("kat-equality-gate", Sensitivity::Correctness, 100, 1),
    ]
}

fn ask(node: usize, units: &[Unit], slots: &[Slot], mode: &SpeedMode) -> Option<Take> {
    let pending = vec![true; units.len()];
    let placed = vec![None; units.len()];
    take(node, units, &pending, &placed, mode, None, slots)
}

/// 2026-10-15: With every peer idle the asking node heads the unit and holds the other
/// two; with one peer busy it waits rather than start a one-box gate that would keep
/// that box from the wide unit. Control: the same fleet with that peer idle starts it.
#[test]
fn a_three_node_unit_starts_only_with_two_idle_peers_and_otherwise_waits() {
    let units = campaign();
    let all_idle = [Slot::Idle; 3];
    assert_eq!(
        ask(0, &units, &all_idle, &SpeedMode::Spread),
        Some(Take::Run {
            unit: 0,
            peers: vec![1, 2]
        })
    );
    let one_busy = [Slot::Idle, Slot::Busy, Slot::Idle];
    assert_eq!(
        ask(0, &units, &one_busy, &SpeedMode::Spread),
        Some(Take::Wait)
    );
    assert_eq!(
        ask(2, &units, &one_busy, &SpeedMode::Spread),
        Some(Take::Wait)
    );
}

/// 2026-10-15: A wide unit no live set of nodes can hold (one of three gone) is left
/// pending and holds nothing: the live nodes take their one-box gates instead of
/// waiting for a unit that can never start.
#[test]
fn a_unit_wider_than_the_live_fleet_does_not_stall_the_rest() {
    let units = campaign();
    let gone = [Slot::Idle, Slot::Idle, Slot::Gone];
    assert_eq!(
        ask(0, &units, &gone, &SpeedMode::Spread),
        Some(Take::Run {
            unit: 1,
            peers: vec![]
        })
    );
}

/// 2026-10-15: Under a Speed bundle only the home heads a Speed-class wide unit; the
/// other nodes wait for it and are then held as its peers.
#[test]
fn only_the_bundle_home_heads_a_speed_class_wide_unit() {
    let units = campaign();
    let home = SpeedMode::Bundle {
        node: 1,
        why: vec![],
    };
    let idle = [Slot::Idle; 3];
    assert_eq!(ask(0, &units, &idle, &home), Some(Take::Wait));
    assert_eq!(
        ask(1, &units, &idle, &home),
        Some(Take::Run {
            unit: 0,
            peers: vec![0, 2]
        })
    );
}

/// 2026-10-15: With no multi-node unit pending, `take` is `next_for` for every node and
/// every slot state: single-node scheduling is unchanged. And `next_for` alone never
/// hands out a wide unit as one box.
#[test]
fn without_a_wide_unit_take_is_next_for_and_next_for_never_returns_one() {
    let singles: Vec<Unit> = campaign().into_iter().skip(1).collect();
    let pending = vec![true; singles.len()];
    let placed = vec![None; singles.len()];
    for k in 0..3 {
        for slots in [[Slot::Idle; 3], [Slot::Idle, Slot::Busy, Slot::Gone]] {
            let want =
                next_for(k, &singles, &pending, &placed, &SpeedMode::Spread, None).map(|unit| {
                    Take::Run {
                        unit,
                        peers: vec![],
                    }
                });
            assert_eq!(ask(k, &singles, &slots, &SpeedMode::Spread), want);
        }
    }
    let units = campaign();
    let all = vec![true; units.len()];
    let none = vec![None; units.len()];
    let first = next_for(0, &units, &all, &none, &SpeedMode::Spread, None);
    assert_eq!(first, Some(1), "the 600 s wide unit is not a one-box pick");
}

/// 2026-10-15: The simulated campaign runs the wide unit first, on one node, with the
/// other two held: nothing overlaps it, so the makespan is its 600 s plus the longest
/// one-box gate after it (300 s). Control: were the peers not held, the one-box gates
/// would run beside it and the makespan would be 600 s.
#[test]
fn the_simulation_holds_every_peer_for_the_wide_units_whole_run() {
    let units = campaign();
    let nodes = [node("a"), node("b"), node("c")];
    let p = simulate(&units, &nodes, &SpeedMode::Spread, None, 0);
    assert_eq!(p.queues[0].first(), Some(&0));
    assert_eq!(
        p.queues.iter().flatten().filter(|&&i| i == 0).count(),
        1,
        "the wide unit is one unit, queued on its head only"
    );
    assert_eq!(p.makespan_secs, 900);
    assert!(p.finish_at.iter().all(|&t| t >= 600), "{:?}", p.finish_at);
    assert_eq!(p.queues.iter().map(Vec::len).sum::<usize>(), units.len());
}

/// 2026-10-15: On two boxes the 3-node unit is never placed (the plan refuses such a
/// campaign up front, `span::hold`); the one-box gates still all run.
#[test]
fn the_simulation_leaves_an_unplaceable_wide_unit_out() {
    let units = campaign();
    let p = simulate(&units, &[node("a"), node("b")], &SpeedMode::Spread, None, 0);
    assert!(p.queues.iter().flatten().all(|&i| i != 0));
    assert_eq!(p.queues.iter().map(Vec::len).sum::<usize>(), 3);
}

/// 2026-10-15: A finished unit frees its head and its held peers, and a peer whose worker
/// returned meanwhile stays gone (were it set idle, `take` would count a box nobody
/// drives as a peer).
#[test]
fn release_frees_the_head_and_its_peers_but_never_revives_a_gone_node() {
    let mut slots = [Slot::Busy, Slot::Busy, Slot::Gone, Slot::Busy];
    release(&mut slots, 0, vec![1, 2]);
    assert_eq!(slots, [Slot::Idle, Slot::Idle, Slot::Gone, Slot::Busy]);
}

/// 2026-10-15: A multi-node unit is refused without being submitted, as a harness
/// failure that is not retried (a retry would hold the boxes again for nothing); a
/// one-box unit runs.
#[test]
fn a_multi_node_unit_is_refused_unsubmitted_and_a_one_box_unit_runs() {
    let units = campaign();
    match unlaunchable(&units[0]) {
        Some(RunOutcome::Harness { reason, retryable }) => {
            assert!(!retryable);
            assert!(
                reason.contains("ttft-cold-glm-512k holds 3 nodes"),
                "{reason}"
            );
        }
        other => panic!("{other:?}"),
    }
    assert!(unlaunchable(&units[1]).is_none());
}
