// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Units are whole periods; a per-layer precision difference splits signatures; a
//! mock keeps the first units of each and renumbers in source order.
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use super::*;
use crate::testkit;

fn sel(config: &str, index: &TensorIndex, counts: Vec<u32>) -> Selection {
    let s = metrale_circuit::layer_schedule(config).unwrap();
    let qc: serde_json::Value = serde_json::from_str(config).unwrap();
    let plan = DeclaredPrecisionPlan::from_quantization_config(&qc["quantization_config"]).unwrap();
    select(&s, index, &plan, |_| Ok(counts)).unwrap()
}

#[test]
fn a_precision_change_between_periods_is_a_new_signature() {
    let (config, index) = testkit::dense_ct();
    let s = sel(&config, &index, vec![1, 1]);
    assert_eq!(s.signatures.len(), 2);
    assert_eq!(
        s.signatures[0].units,
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]
    );
    assert_eq!(s.signatures[1].units, vec![vec![8, 9, 10, 11]]);
    assert!(
        s.signatures[0]
            .precision
            .iter()
            .any(|l| l == ".mlp.gate_proj=W4A4"),
        "{:?}",
        s.signatures[0].precision
    );
    assert!(
        s.signatures[1]
            .precision
            .iter()
            .any(|l| l == ".mlp.gate_proj=W8A8")
    );
    assert_eq!(s.renumber.get(&8), Some(&4));
    assert_eq!(s.renumber.get(&4), None);
}

#[test]
fn identical_periods_share_one_signature_and_expert_indices_are_starred() {
    let (config, index) = testkit::moe_fp8();
    let s = sel(&config, &index, vec![2]);
    assert_eq!(s.signatures.len(), 1);
    assert_eq!(s.kept_layers(), (0..8).collect::<Vec<_>>());
    assert!(
        s.signatures[0]
            .precision
            .iter()
            .any(|l| l.starts_with(".mlp.experts.*.down_proj="))
    );
    assert_eq!(s.signatures[0].kinds[3], "full_attention");
}

#[test]
fn a_layer_count_that_is_not_whole_periods_is_refused() {
    let (config, _) = testkit::moe_fp8();
    let mut s = metrale_circuit::layer_schedule(&config).unwrap();
    s.layer_kinds.truncate(6);
    assert!(units(&s).unwrap_err().to_string().contains("whole number"));
}
