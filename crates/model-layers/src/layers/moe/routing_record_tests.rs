// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Offsets decode to per-expert counts, and counts accumulate per layer in
//! first-touch order.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

fn le(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn offsets_become_counts() {
    assert_eq!(counts_from_offsets(&le(&[0, 3, 3, 10])), vec![3, 0, 7]);
}

#[test]
fn layers_accumulate_in_first_touch_order() {
    let r = RoutingRecorder::default();
    r.add(0x20, &[1, 2]);
    r.add(0x10, &[5, 0]);
    r.add(0x20, &[1, 1]);
    r.add(0x10, &[9, 9, 9]);
    let rows: Vec<Vec<u64>> = r
        .layers
        .lock()
        .unwrap()
        .iter()
        .map(|(_, c)| c.clone())
        .collect();
    assert_eq!(rows, vec![vec![2, 3], vec![5, 0]]);
}
