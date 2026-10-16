// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the MoE gate's descriptor: it shares the dense
//! gates' floor params and driver, and is defined on the Qwen3.6-35B-A3B
//! family only. 2026-10-16: the same for the GLM ladder's, on its family.
//! Listed in `gate::coverage::TEST_ONLY_RUST_MODULES`.
//!
//! Owner: bench (concurrency).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Same metric names as the dense gate, or a MoE floor on
/// `c8_aggregate_tok_s` would gate nothing.
#[test]
fn the_moe_gate_shares_the_drivers_floor_pairing() {
    assert_eq!(MOE_DESCRIPTOR.id, "concurrency-sweep-moe");
    assert_eq!(
        MOE_DESCRIPTOR.threshold_params, DESCRIPTOR.threshold_params,
        "the MoE ladder must gate on the same metric names as the dense one"
    );
    assert_eq!(
        crate::registry::find("concurrency-sweep-moe").map(|d| d.id),
        Some("concurrency-sweep-moe"),
        "an unregistered gate id can be neither run nor owed"
    );
    // 2026-09-26: A `prompt_mode = "essay"` pin selects `Fixture::Essay`, whose
    // prompts `concurrency_essay_tests.rs` pins to the harness's bytes.
    let mut b = ConcurrencySweep::default();
    let mut v = ParamValues::defaults(&b.parameters());
    v.set("prompt_mode", ParamValue::Text("essay".into()));
    v.set("concurrencies", ParamValue::IntList(vec![1, 2, 4, 8, 16]));
    v.set("isls", ParamValue::IntList(vec![128]));
    v.set("osl", ParamValue::Int(1024));
    b.configure(&v).unwrap();
    assert_eq!(b.fixture, Fixture::Essay);
    assert_eq!(b.osl, 1024);
    assert_eq!(
        b.cells,
        vec![(128, 1), (128, 2), (128, 4), (128, 8), (128, 16)]
    );
    assert!(!b.floors.gating(), "an unmeasured entry fills no floor");
}

/// 2026-09-26: `ModelExpectation::accepts` is a case-insensitive substring
/// match on the family name.
#[test]
fn the_moe_gate_is_defined_on_the_moe_family_only() {
    let expect = MOE_DESCRIPTOR
        .intended_for
        .expect("the MoE ladder names the family it is defined on");
    assert!(expect.accepts("Qwen/Qwen3.6-35B-A3B-FP8"));
    assert!(expect.accepts("nvidia/Qwen3.6-35B-A3B-NVFP4"));
    assert!(!expect.accepts("unsloth/Qwen3.8-27B-NVFP4"));
    assert!(!expect.accepts("unsloth/Qwen3.6-27B-NVFP4"));
    assert!(
        DESCRIPTOR.intended_for.is_none(),
        "the plain gate stays family-agnostic; only the MoE, DFlash2 and GLM ids are pinned"
    );
}

/// 2026-10-16: The GLM ladder shares the driver's floor pairing, is registered, is defined on
/// the GLM family only, and its pinned instrument includes the C=12 rung.
#[test]
fn the_glm_ladder_is_registered_on_its_family_with_a_c12_rung() {
    assert_eq!(GLM_DESCRIPTOR.id, "concurrency-sweep-glm");
    assert_eq!(GLM_DESCRIPTOR.threshold_params, DESCRIPTOR.threshold_params);
    assert_eq!(
        crate::registry::find("concurrency-sweep-glm").map(|d| d.id),
        Some("concurrency-sweep-glm")
    );
    let expect = GLM_DESCRIPTOR
        .intended_for
        .expect("the GLM ladder names its family");
    assert!(expect.accepts("nvidia/GLM-5.3-Flash-NVFP4"));
    assert!(!expect.accepts("Qwen/Qwen3.6-35B-A3B-FP8"));
    let mut b = ConcurrencySweep::default();
    let mut v = ParamValues::defaults(&b.parameters());
    v.set(
        "concurrencies",
        ParamValue::IntList(vec![1, 2, 4, 8, 12, 16]),
    );
    v.set("isls", ParamValue::IntList(vec![128]));
    v.set("min_c12", ParamValue::Float(100.0));
    b.configure(&v).unwrap();
    assert_eq!(b.cells.len(), 6);
    assert!(b.cells.contains(&(128, 12)));
    assert!(
        b.floors.per_c.contains(&(12, 100.0)),
        "a C=12 bound gates the C=12 rung: {:?}",
        b.floors.per_c
    );
}
