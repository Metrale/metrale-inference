// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The fork/join derivation over hand-built group accesses: the legacy shared-expert
//! shape, the scratch hazard with no edge between the groups, the join at the end, coalescing,
//! a side group first in the plan, and the windows the buffer planner widens.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use super::{Access, EventKind, Stream, StreamEvent, derive_events, side_windows};

fn g(stream: Stream, reads: &[usize], writes: &[usize], scratch: &[&str]) -> Access {
    Access {
        stream,
        reads: reads.to_vec(),
        writes: writes.to_vec(),
        scratch: scratch.iter().map(|s| s.to_string()).collect(),
    }
}

fn fork(after: Option<usize>, before: usize) -> StreamEvent {
    StreamEvent {
        kind: EventKind::Fork,
        after,
        before: Some(before),
    }
}

fn join(after: usize, before: Option<usize>) -> StreamEvent {
    StreamEvent {
        kind: EventKind::Join,
        after: Some(after),
        before,
    }
}

use Stream::{Main, Side};

// 2026-10-04: Mutation: a plan with no side group gaining any event changes every golden plan.
#[test]
fn a_plan_without_a_side_group_has_no_event() {
    let gs = [g(Main, &[], &[0], &[]), g(Main, &[0], &[1], &[])];
    assert!(derive_events(&gs).is_empty());
    assert!(side_windows(&gs, &[]).is_empty());
}

// 2026-10-04: The MoE prefill shape: quant (main) -> shared expert (side) beside router and
// top-k (main), joined before the blend that reads the shared output. Mutation: forking after
// an earlier group than the issue point lets the side stream run ahead of main's work; joining
// later than the first reader races it; joining earlier serialises the overlap.
#[test]
fn the_side_group_forks_where_it_is_issued_and_joins_before_its_first_reader() {
    let gs = [
        g(Main, &[], &[0], &[]),     // 0 quant: writes x
        g(Side, &[0], &[1], &[]),    // 1 shared expert: x -> s
        g(Main, &[0], &[2], &[]),    // 2 router: x -> r
        g(Main, &[2], &[3], &[]),    // 3 top-k: r -> t
        g(Main, &[3, 1], &[4], &[]), // 4 blend: t, s -> y
    ];
    let ev = derive_events(&gs);
    assert_eq!(ev, vec![fork(Some(0), 1), join(1, Some(4))]);
    assert_eq!(side_windows(&gs, &ev)[&1], (1, 4));
}

// 2026-10-04: The 8b case: a later main group re-quantizes into the slab the side group used,
// with no edge between them. Mutation: deriving hazards from edges alone drops this join.
#[test]
fn a_shared_scratch_region_joins_without_an_edge() {
    let gs = [
        g(Main, &[], &[0], &[]),
        g(Side, &[0], &[1], &["slab.head"]),
        g(Main, &[0], &[2], &[]),
        g(Main, &[2], &[3], &["slab.head"]),
        g(Main, &[3, 1], &[4], &[]),
    ];
    assert_eq!(derive_events(&gs), vec![fork(Some(0), 1), join(1, Some(3))]);
}

// 2026-10-04: Mutation: dropping the end join leaves a fork unjoined in a captured graph and
// lets the next step's main work overtake the side stream.
#[test]
fn a_side_group_no_main_group_reads_is_joined_at_the_end() {
    let gs = [
        g(Main, &[], &[0], &[]),
        g(Side, &[0], &[1], &[]),
        g(Main, &[0], &[2], &[]),
    ];
    assert_eq!(derive_events(&gs), vec![fork(Some(0), 1), join(1, None)]);
    assert_eq!(side_windows(&gs, &derive_events(&gs))[&1], (1, 2));
}

// 2026-10-04: Two side groups in a row share one fork; one join after the later covers both,
// since the side stream runs them in order. Mutation: a second fork or a join per group.
#[test]
fn consecutive_side_groups_share_their_fork_and_join() {
    let gs = [
        g(Main, &[], &[0], &[]),
        g(Side, &[0], &[1], &[]),
        g(Side, &[1], &[2], &[]),
        g(Main, &[2, 1], &[3], &[]),
    ];
    assert_eq!(derive_events(&gs), vec![fork(Some(0), 1), join(2, Some(3))]);
}

// 2026-10-04: A side group issued before any main group forks at the plan's start, and a
// second side run after more main work forks again. Mutation: no fork for the first (the side
// stream would overtake the previous step) or none for the second.
#[test]
fn each_side_run_forks_after_the_main_work_issued_before_it() {
    let gs = [
        g(Side, &[], &[0], &[]),
        g(Main, &[0], &[1], &[]),
        g(Side, &[1], &[2], &[]),
        g(Main, &[2], &[3], &[]),
    ];
    assert_eq!(
        derive_events(&gs),
        vec![
            fork(None, 0),
            join(0, Some(1)),
            fork(Some(1), 2),
            join(2, Some(3)),
        ]
    );
}

// 2026-10-04: A main group that depends on nothing of the side stream does not wait for it.
#[test]
fn an_independent_main_group_runs_beside_the_side_stream() {
    let gs = [
        g(Main, &[], &[0], &[]),
        g(Side, &[0], &[1], &["a"]),
        g(Main, &[0], &[2], &["b"]),
        g(Main, &[2], &[3], &[]),
    ];
    assert_eq!(derive_events(&gs), vec![fork(Some(0), 1), join(1, None)]);
}

// 2026-10-04: Through the fuser, the renderer, the digest and the buffer planner, on the toy
// circuit with its SiLU on the side stream: the fork follows gate|up, the join precedes down,
// the plan text and digest carry both, and gate|up's output stays live until the join.
// Mutation: dropping the side window from the live ranges lets a main group between the fork
// and the join take the bytes the side group still reads.
#[test]
fn a_side_rule_forks_and_joins_through_the_whole_pipeline() {
    use crate::test_toy::{circuit, fused_with, plan, policy, rules};
    let c = circuit(1);
    let side = rules(&fused_with(
        "act_side",
        r#"{ op = "silu_mul" }"#,
        5,
        (1, 128),
        "numerics = \"reference\"\nstream = \"side\"\nscratch = [\"ws\"]",
    ));
    let p = plan(&c, &side, &policy(), 4);
    let base = plan(&c, &rules(""), &policy(), 4);
    let at = |rule: &str| p.groups.iter().position(|g| g.rule == rule).unwrap();
    let (up, act, down) = (at("up"), at("act_side"), at("down"));
    assert_eq!(p.groups[act].stream, Side);
    assert_eq!(p.events, vec![fork(Some(up), act), join(act, Some(down))]);
    assert!(base.events.is_empty());
    assert_ne!(p.digest, base.digest);
    let text = crate::render::render(&c, &p, &Vec::new());
    assert!(
        text.contains(&format!("-- fork g{up:04} -> g{act:04}")),
        "{text}"
    );
    assert!(
        text.contains(&format!("-- join g{act:04} -> g{down:04}")),
        "{text}"
    );
    assert!(text.contains("stream=side scratch=[ws]"), "{text}");
    let base_text = crate::render::render(&c, &base, &Vec::new());
    assert!(!base_text.contains("stream=") && !base_text.contains("-- "));
    let ranges = crate::planner::live_ranges(&c, &p);
    let gu = c.edges.iter().position(|e| e.id.ends_with(".gu")).unwrap();
    assert!(
        ranges[&gu].1 >= down,
        "gate|up's output dies at {:?}",
        ranges[&gu]
    );
    // 2026-10-04: The same position on the main stream: gate|up's output dies at its reader.
    assert_eq!(crate::planner::live_ranges(&c, &base)[&gu].1, act);
}
