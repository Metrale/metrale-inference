// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Tests for `met circuit diff`'s pure parts: the host argmax that fixes the token
//! stream, the prompt generator, the per-step comparison and the verdict.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::*;

fn bf16(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn run_of(prefill: &[f32], steps: &[&[f32]]) -> Run {
    Run {
        prefill: bf16(prefill),
        steps: steps.iter().map(|s| bf16(s)).collect(),
        tokens: vec![0; steps.len() + 1],
        enqueue_ms: Vec::new(),
        step_ms: Vec::new(),
    }
}

#[test]
fn argmax_takes_the_first_of_equal_maxima_and_never_a_nan() {
    assert_eq!(argmax_bf16(&bf16(&[-3.0, -1.0, -2.0])), 1);
    assert_eq!(argmax_bf16(&bf16(&[1.0, 5.0, 5.0, 2.0])), 1);
    assert_eq!(argmax_bf16(&bf16(&[f32::NAN, 0.5, f32::NAN])), 1);
    assert_eq!(argmax_bf16(&bf16(&[f32::NEG_INFINITY, f32::INFINITY])), 1);
}

#[test]
fn prompts_are_fixed_distinct_lengths_inside_the_vocabulary() {
    let a = prompts(4, 1000);
    assert_eq!(a, prompts(4, 1000), "the generator is deterministic");
    let lens: Vec<usize> = a.iter().map(Vec::len).collect();
    assert_eq!(lens, [23, 64, 105, 146]);
    assert!(a.iter().flatten().all(|&t| (64..1000 - 64).contains(&t)));
    assert_ne!(
        a[0][..10],
        a[1][..10],
        "prompts are not copies of each other"
    );
}

#[test]
fn a_comparison_counts_each_differing_step_and_finds_the_first() {
    let base = run_of(&[1.0, 2.0], &[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
    let same = compare(
        "same",
        &base,
        &run_of(&[1.0, 2.0], &[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]),
    );
    assert_eq!((same.mismatched_steps, same.first_mismatch), (0, None));
    assert!(same.prefill_equal);
    let late = compare(
        "late",
        &base,
        &run_of(&[1.0, 2.5], &[&[1.0, 2.0], &[3.0, 4.5], &[5.5, 6.5]]),
    );
    assert!(!late.prefill_equal);
    assert_eq!((late.mismatched_steps, late.first_mismatch), (2, Some(1)));
    assert_eq!(
        late.max_mismatched_bytes, 2,
        "5.0 -> 5.5 and 6.0 -> 6.5 each change one byte"
    );
}

#[test]
fn the_verdict_needs_matching_circuits_and_a_control_that_saw_the_change() {
    let ok = |variant: &str| Comparison {
        variant: variant.to_string(),
        prefill_equal: true,
        steps: 64,
        mismatched_steps: 0,
        first_mismatch: None,
        max_mismatched_bytes: 0,
    };
    let seen = Comparison {
        mismatched_steps: 32,
        first_mismatch: Some(31),
        ..ok("control")
    };
    assert!(failures(&[vec![ok("circuit")]], &seen).is_empty());
    let blind = failures(&[vec![ok("circuit")]], &ok("control"));
    assert!(
        blind.iter().any(|r| r.contains("detection control")),
        "{blind:?}"
    );
    let off = Comparison {
        mismatched_steps: 1,
        first_mismatch: Some(7),
        ..ok("circuit")
    };
    let bad = failures(&[vec![ok("legacy-repeat"), off]], &seen);
    assert_eq!(bad.len(), 1);
    assert!(
        bad[0].contains("circuit") && bad[0].contains("Some(7)"),
        "{bad:?}"
    );
}

#[test]
fn the_step_median_skips_the_capture_step() {
    assert_eq!(median_after_first(&[500.0, 3.0, 1.0, 2.0]), 2.0);
    assert!(median_after_first(&[500.0]).is_nan());
}
