// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The parallel cost model's step (crates/circuit/src/parallel_cost.rs): ranks meet
//! at every collective, so the step is the sum over segments of each segment's slowest rank,
//! not the slowest rank's whole-step total; a collective waits only for its group; ranks that
//! never meet are refused; one rank is its own total; and the rotation pricing finds a better
//! placement exactly when one exists.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;

use std::collections::BTreeMap;

use metrale_circuit::ir::{Circuit, OpKind};
use metrale_circuit::parallel_cost::{
    Link, rank_cost, rotate_floor, segment_of_nodes, segmented, step_us,
};
use metrale_circuit::{Deployment, Instance};

// 2026-10-23: The GB10 values the layout report prices with (kernels/DEVICES.toml 273 GB/s,
// HARDWARE.toml [collective]); any positive values exercise the same arithmetic.
const BW: f64 = 273.0;
const LINK: Link = Link {
    latency_us: 17.57,
    gbps: 25.0,
};

fn glm() -> Instance {
    common::instances()
        .into_iter()
        .find(|i| i.recipe == "glm-5.3/glm-5.3-flash-nvfp4-tp3-512k")
        .expect("the tp3 instance")
}

fn at(inst: &Instance, ranks: u64, degree: BTreeMap<String, u64>) -> Vec<Circuit> {
    let mut i = inst.clone();
    i.parallel = Some(Deployment {
        ranks,
        degree,
        why: "test".into(),
    });
    common::load_texts(&i, &common::Texts::of(&i))
        .unwrap_or_else(|e| panic!("{e}"))
        .ranks
}

fn spec() -> metrale_circuit::parallel::ParallelSpec {
    let t = common::Texts::of(&glm());
    let blocks: Vec<(&str, &str)> = t
        .blocks
        .iter()
        .map(|(n, s)| (n.as_str(), s.as_str()))
        .collect();
    metrale_circuit::parallel_spec(&t.circuit, &blocks).unwrap()
}

fn uniform(d: u64) -> BTreeMap<String, u64> {
    let s = spec();
    metrale_circuit::parallel::parts(&s)
        .map(|k| (k.clone(), d.min(s[k].degree.max)))
        .collect()
}

fn whole_step(ranks: &[Circuit]) -> f64 {
    ranks
        .iter()
        .map(|c| rank_cost(c, BW, LINK).unwrap().total_us())
        .fold(0.0, f64::max)
}

/// 2026-10-23: `c` with the weights of every node in `segs` dropped (that rank reads nothing
/// there).
fn without(c: &Circuit, segs: &[usize]) -> Circuit {
    let mut c = c.clone();
    let seg = segment_of_nodes(&c);
    for (n, s) in c.nodes.iter_mut().zip(seg) {
        if segs.contains(&s) {
            n.weight = None;
        }
    }
    c
}

fn segment_closing(c: &Circuit, block: &str) -> usize {
    let seg = segment_of_nodes(c);
    let i = c
        .nodes
        .iter()
        .position(|n| n.block == block && matches!(n.op, OpKind::AllReduce))
        .expect("a collective in the block");
    seg[i]
}

/// 2026-10-23: Three identical GLM ranks, then rank 0 skips the first KDA segment's reads, rank 1
/// the first MoE segment's, rank 2 both: every segment still has a rank that reads it all, so the
/// step is the full rank's total, while each rank's own total is lower: the whole-step max would
/// undercount by the smaller skipped segment.
#[test]
fn the_step_is_the_sum_of_each_segments_slowest_rank() {
    let r0 = at(&glm(), 3, uniform(3)).swap_remove(0);
    let (a, b) = (segment_closing(&r0, "kda"), segment_closing(&r0, "moe"));
    let full = rank_cost(&r0, BW, LINK).unwrap().total_us();
    let ranks = vec![
        without(&r0, &[a]),
        without(&r0, &[b]),
        without(&r0, &[a, b]),
    ];
    let skipped = |s: usize| full - rank_cost(&without(&r0, &[s]), BW, LINK).unwrap().total_us();
    let (sa, sb) = (skipped(a), skipped(b));
    assert!(
        sa > 1.0 && sb > 1.0,
        "both segments read weights: {sa} {sb}"
    );
    let step = step_us(&ranks, BW, LINK).unwrap();
    assert!(
        (step - full).abs() < 1e-6 * full,
        "segment form: step {step} != the full rank's {full}"
    );
    let lower = whole_step(&ranks);
    assert!(
        step > lower + 0.99 * sa.min(sb),
        "the step {step} must exceed the whole-step max {lower} by the smaller skipped segment {}",
        sa.min(sb)
    );
}

/// 2026-10-23: The shared expert on {0, 1}: its all-reduce synchronises only those two (two waits
/// recorded), rank 2 is not held there, and no rank finishes before its own work.
#[test]
fn a_sub_group_collective_waits_only_for_its_members() {
    let mut degree = uniform(3);
    degree.insert("shared_inter".into(), 2);
    let ranks = at(&glm(), 3, degree);
    let s = segmented(&ranks, BW, LINK).unwrap();
    let sub: Vec<_> = s
        .segments
        .iter()
        .filter(|g| g.collective.ends_with("sdown@all_reduce"))
        .collect();
    assert_eq!(
        sub.len(),
        43,
        "one shared-expert sum per MoE layer (42) and the MTP head's MoE"
    );
    assert!(
        sub.iter()
            .all(|g| g.group == [0, 1] && g.wait_us.len() == 2),
        "over {{0,1}} only"
    );
    assert!(
        s.step_us >= whole_step(&ranks) - 1e-6,
        "no rank finishes before its own work"
    );
}

/// 2026-10-23: Ranks that never meet are refused, not priced: a collective renamed on one rank (the
/// others wait for it forever), and a collective on a rank its group does not name.
#[test]
fn ranks_that_never_meet_are_refused() {
    let ranks = at(&glm(), 3, uniform(3));
    let mut renamed = ranks.clone();
    let i = renamed[1]
        .nodes
        .iter()
        .position(|n| matches!(n.op, OpKind::AllReduce))
        .unwrap();
    renamed[1].nodes[i].id.push_str("-elsewhere");
    let e = segmented(&renamed, BW, LINK).unwrap_err();
    assert!(e.contains("waits forever"), "{e}");
    let mut foreign = ranks.clone();
    let i = foreign[2]
        .nodes
        .iter()
        .position(|n| matches!(n.op, OpKind::AllReduce))
        .unwrap();
    foreign[2].nodes[i]
        .params
        .insert("group".into(), "0,1".into());
    let e = segmented(&foreign, BW, LINK).unwrap_err();
    assert!(e.contains("rank 2 holds collective"), "{e}");
}

/// 2026-10-23: For N in 1..=8, every part at d <= N (capped at its own bound): every layout prices
/// without a deadlock, its step is at least every rank's own total, and every rank's collectives
/// are segments; at N = 1 nothing waits and the step is the rank's total.
#[test]
fn the_step_bounds_every_rank_at_every_rank_count() {
    let inst = glm();
    let mut checked = 0;
    for n in 1..=8u64 {
        for d in 1..=n {
            let mut i = inst.clone();
            i.parallel = Some(Deployment {
                ranks: n,
                degree: uniform(d),
                why: "test".into(),
            });
            let Ok(l) = common::load_texts(&i, &common::Texts::of(&i)) else {
                continue;
            };
            // 2026-10-23: A slice no weight format can size (2456 NVFP4 g16 rows at N = 5) is
            // unpriced by either form.
            let own: Result<Vec<_>, _> = l.ranks.iter().map(|c| rank_cost(c, BW, LINK)).collect();
            let Ok(own) = own else {
                assert!(
                    segmented(&l.ranks, BW, LINK).is_err(),
                    "n={n} d={d}: priced what a rank cannot"
                );
                continue;
            };
            let s = segmented(&l.ranks, BW, LINK).unwrap_or_else(|e| panic!("n={n} d={d}: {e}"));
            let whole = own.iter().map(|r| r.total_us()).fold(0.0, f64::max);
            assert!(
                s.step_us >= whole - 1e-9 * whole,
                "n={n} d={d}: step {} < a rank's total {whole}",
                s.step_us
            );
            for (r, mine_own) in own.iter().enumerate() {
                let mine = s.segments.iter().filter(|g| g.group.contains(&r)).count();
                assert_eq!(mine as u64, mine_own.collectives, "n={n} d={d} rank {r}");
            }
            if n == 1 {
                assert!(
                    s.wait_per_rank().iter().all(|w| *w == 0.0),
                    "one rank never waits"
                );
                assert!(
                    (s.step_us - whole).abs() <= 1e-9 * whole,
                    "N=1: {} vs {whole}",
                    s.step_us
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 30, "the property ran on {checked} layouts");
}

/// 2026-10-23: The rotation finds a better placement exactly when one exists: two parts whose long
/// slices can sit on different ranks drop the slowest rank from 4 to 3; two parts with two long
/// slices each over three ranks cannot (pigeonhole: some rank holds both), so today's offsets stay.
#[test]
fn rotation_prices_only_a_placement_that_exists() {
    let (best, off) =
        rotate_floor(&[0.0; 3], &[vec![2.0, 2.0, 1.0], vec![2.0, 1.0, 1.0]], 100).unwrap();
    assert_eq!(
        best, 3.0,
        "the second part's long slice moves off the ranks holding the first's"
    );
    assert_ne!(off, [0, 0], "found by moving a slice");
    let (best, off) =
        rotate_floor(&[0.5; 3], &[vec![2.0, 2.0, 1.0], vec![3.0, 3.0, 2.0]], 100).unwrap();
    assert_eq!(
        (best, off),
        (5.5, vec![0, 0]),
        "pigeonhole: no gain, today's placement"
    );
    assert!(
        rotate_floor(&[0.0; 3], &[vec![1.0, 1.0]], 100).is_err(),
        "a part of another width"
    );
    assert!(
        rotate_floor(&[0.0; 3], &vec![vec![1.0; 3]; 5], 100).is_err(),
        "3^5 offsets over the limit"
    );
}
