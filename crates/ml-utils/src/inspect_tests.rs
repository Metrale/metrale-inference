// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The report's signatures, byte accounting, schemes and expert-traffic table on the
//! toy layouts.
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use super::*;
use crate::testkit;

#[test]
fn the_dense_report_splits_signatures_and_accounts_every_byte() {
    let (config, index) = testkit::dense_ct();
    let r = inspect(&config, None, &index).unwrap();
    assert_eq!(
        (r.arch.as_str(), r.layers, r.layout.as_str()),
        ("qwen3_5", 12, "interval/4")
    );
    assert_eq!(r.signatures.len(), 2);
    assert_eq!((r.signatures[0].units, r.signatures[1].units), (2, 1));
    let in_units: u64 = r
        .signatures
        .iter()
        .map(|s| s.bytes_per_unit * s.units as u64)
        .sum();
    assert_eq!(in_units + r.fixed_bytes, r.bytes);
    assert_eq!(r.schemes["nvfp4 (compressed-tensors global)"], 8 * 3);
    assert!(r.moe.is_none());
}

#[test]
fn the_moe_report_tabulates_distinct_experts() {
    let (config, side, index) = testkit::moe_nvfp4();
    let r = inspect(&config, Some(&side), &index).unwrap();
    let m = r.moe.unwrap();
    assert_eq!((m.experts, m.top_k), (testkit::EXPERTS, testkit::TOP_K));
    assert!(
        (m.unique_experts[0].1 - testkit::TOP_K as f64).abs() < 1e-9,
        "C1 touches top_k"
    );
    assert!(m.unique_experts.windows(2).all(|w| w[0].1 <= w[1].1));
    assert!(m.unique_experts.last().unwrap().1 <= testkit::EXPERTS as f64);
    assert!(r.schemes.contains_key("nvfp4 (ModelOpt global)"));
    assert!(r.schemes.contains_key("fp8 per tensor"));
}
