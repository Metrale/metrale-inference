// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Laxity on the toy circuit: a fused `act -> down` group keeps exactly its
//! internal edge out of DRAM (one write, one read) and saves one launch per layer; an unfused
//! plan has no laxity.
//!
//! Owner: metrale-circuit tests.
//! Invariants: every test fails when the behaviour it names is removed (the mutation notes say
//! which change each one catches).

use super::plan_laxity;
use crate::test_toy::{circuit, fused, plan, policy, rules};

const ACT_DOWN: &str = r#"{ op = "silu_mul" }, { op = "linear", role = "down" }"#;
const GBPS: f64 = 1000.0;

// 2026-10-05: Mutation: counting a materialized edge, or only the write (or only the read),
// changes the byte total; dividing by bandwidth in the wrong unit changes the time; counting
// the launch per group instead of per kernel saved changes `launches_saved`.
#[test]
fn a_fused_pair_keeps_its_internal_edge_out_of_dram() {
    let c = circuit(2);
    let rows = 4;
    let p = plan(
        &c,
        &rules(&fused("act_down", ACT_DOWN, 100)),
        &policy(),
        rows,
    );
    let x = plan_laxity(&c, &p, GBPS).unwrap();
    assert_eq!(x.groups.len(), 2, "one fused group per layer: {x:?}");
    let a: f64 = c
        .edges
        .iter()
        .filter(|e| e.id.ends_with(".a"))
        .map(|e| (rows * e.dim_value * 2) as f64)
        .sum();
    assert!(a > 0.0);
    assert_eq!(
        x.bytes(),
        2.0 * a,
        "one write and one read of each `a` edge"
    );
    assert_eq!(x.time_us(), 2.0 * a / (GBPS * 1e3));
    assert_eq!(x.launches_saved(), 2, "two ops, one kernel, per layer");
    assert!(
        x.groups
            .iter()
            .all(|g| g.fused_edges == 1 && g.numerics == "reference")
    );
}

// 2026-10-05: Mutation: reporting single-op groups as zero-gain entries fills the list.
#[test]
fn an_unfused_plan_has_no_laxity() {
    let c = circuit(2);
    let p = plan(&c, &rules(""), &policy(), 1);
    let x = plan_laxity(&c, &p, GBPS).unwrap();
    assert!(x.groups.is_empty(), "{x:?}");
    assert_eq!(x.time_us(), 0.0);
}

// 2026-10-05: Mutation: dropping the shared-input term leaves the merged group at zero bytes;
// counting every reader (not readers minus one) doubles it.
#[test]
fn a_group_reading_one_input_twice_saves_one_read() {
    let c = circuit(1);
    let rows = 2;
    let mut p = plan(&c, &rules(""), &policy(), rows);
    let local = |id: &str| c.nodes.iter().position(|n| n.id.ends_with(id)).unwrap();
    let (norm, add) = (local(".norm"), local(".add"));
    let g_add = p.groups.iter().position(|g| g.nodes == [add]).unwrap();
    p.groups[g_add].nodes.insert(0, norm);
    p.groups.retain(|g| g.nodes != [norm]);
    let x = c.nodes[norm].inputs[0];
    assert!(c.edges[x].consumers.contains(&add), "the stream feeds both");
    let size = (rows * c.edges[x].dim_value * 2) as f64;
    let lax = plan_laxity(&c, &p, GBPS).unwrap();
    assert_eq!(lax.groups.len(), 1);
    assert_eq!(lax.groups[0].fused_edges, 0);
    assert_eq!(lax.bytes(), size, "one read of the shared stream saved");
}
