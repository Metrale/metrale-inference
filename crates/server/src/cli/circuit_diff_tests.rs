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

#[test]
fn batch_prompts_put_rows_at_distinct_positions_and_the_batch_verdict_reads_every_width() {
    let p = batch::batch_prompts(20, 1000);
    assert_eq!(
        p,
        batch::batch_prompts(20, 1000),
        "the generator is deterministic"
    );
    let lens: Vec<usize> = p.iter().map(Vec::len).collect();
    assert_eq!(&lens[..3], &[12, 19, 26]);
    assert_eq!(lens[16], 12, "lengths cycle every 16 rows");
    assert!(p.iter().flatten().all(|&t| (64..1000 - 64).contains(&t)));
    assert_ne!(p[0][..12], p[16][..12], "rows of one length differ");
    let ok = |variant: &str, mismatched: usize| Comparison {
        variant: variant.to_string(),
        prefill_equal: true,
        steps: 8,
        mismatched_steps: mismatched,
        first_mismatch: (mismatched > 0).then_some(4),
        max_mismatched_bytes: mismatched,
    };
    let width = |rows, circuit: usize, control: usize| batch::WidthReport {
        rows,
        changed_row: rows / 2,
        comparisons: vec![ok("circuit", circuit)],
        prefill_rows_differ: vec![Vec::new(), Vec::new()],
        prefill_row_deltas: Vec::new(),
        timings: Vec::new(),
        detection_control: ok("control", control),
    };
    assert!(batch::batch_failures(&[width(2, 0, 4), width(128, 0, 4)]).is_empty());
    let bad = batch::batch_failures(&[width(2, 0, 4), width(128, 1, 4), width(16, 0, 0)]);
    assert_eq!(bad.len(), 2, "{bad:?}");
    assert!(bad[0].starts_with("128 rows, circuit"), "{bad:?}");
    assert!(
        bad[1].starts_with("16 rows: the detection control"),
        "{bad:?}"
    );
    let mut thin = width(16, 0, 4);
    thin.prefill_rows_differ[0] = vec![1, 2, 3];
    thin.prefill_rows_differ[1] = vec![8];
    let bad = batch::batch_failures(&[thin]);
    assert_eq!(bad.len(), 2, "{bad:?}");
    assert!(bad[0].contains("3 rows left out"), "{bad:?}");
    assert!(bad[1].contains("changed row 8 was left out"), "{bad:?}");
}

#[test]
fn a_row_whose_prefill_differs_is_left_out_of_the_step_comparison() {
    let row = |v: f32| [v, v + 1.0];
    let cat = |rows: &[[f32; 2]]| rows.iter().flatten().copied().collect::<Vec<f32>>();
    let reference = run_of(
        &cat(&[row(1.0), row(2.0), row(3.0)]),
        &[&cat(&[row(4.0), row(5.0), row(6.0)])],
    );
    let drifted = run_of(
        &cat(&[row(1.0), row(2.5), row(3.0)]),
        &[&cat(&[row(4.0), row(5.5), row(6.0)])],
    );
    let (c, skipped) = batch::compare_rows("circuit", &reference, &drifted, 4);
    assert_eq!(skipped, vec![1]);
    assert!(c.prefill_equal);
    assert_eq!(c.mismatched_steps, 0);
    let d = batch::row_deltas(&reference, &drifted, 4, &skipped);
    assert_eq!((d[0].row, d[0].max_abs, d[0].argmax_equal), (1, 0.5, true));
    assert!(d[0].bytes >= 1);
    let also_step = run_of(
        &cat(&[row(1.0), row(2.5), row(3.0)]),
        &[&cat(&[row(4.0), row(5.5), row(6.5)])],
    );
    let (c, _) = batch::compare_rows("circuit", &reference, &also_step, 4);
    assert_eq!(
        (c.mismatched_steps, c.first_mismatch),
        (1, Some(0)),
        "row 2 still counts"
    );
}

#[test]
fn verify_drafts_corrupt_one_draft_on_odd_steps_and_accepts_count_the_leading_matches() {
    let g = [10, 11, 12, 13, 14, 15];
    assert_eq!(verify::drafts(&g, 1, 0, 4, 100), vec![12, 13, 14]);
    assert_eq!(verify::drafts(&g, 1, 1, 4, 100), vec![12, 14, 14]);
    assert_eq!(verify::drafts(&g, 1, 3, 4, 100), vec![13, 13, 14]);
    assert_eq!(verify::drafts(&g, 0, 5, 4, 100), vec![11, 12, 14]);
    assert_eq!(verify::drafts(&g, 4, 0, 4, 100), vec![15, 0, 0]);
    assert_eq!(verify::drafts(&[99, 99], 0, 1, 2, 100), vec![0]);
    assert_eq!(verify::accepted(&[1, 2, 3, 4], &[2, 3, 4, 9]), 3);
    assert_eq!(verify::accepted(&[1, 2, 3, 4], &[2, 7, 4, 9]), 1);
    assert_eq!(verify::accepted(&[1, 2], &[5, 9]), 0);
}

#[test]
fn the_verify_verdict_needs_matching_steps_a_seen_control_and_both_commit_paths() {
    let c = |variant: &str, mismatched: usize| Comparison {
        variant: variant.to_string(),
        prefill_equal: true,
        steps: 4,
        mismatched_steps: mismatched,
        first_mismatch: (mismatched > 0).then_some(2),
        max_mismatched_bytes: mismatched,
    };
    let report = |accepted: Vec<usize>, circuit: usize, control: usize| verify::VerifyReport {
        k: 3,
        accepted,
        comparisons: vec![c("circuit", circuit)],
        timings: Vec::new(),
        detection_control: c("control", control),
    };
    assert!(verify::verify_failures(&[report(vec![2, 0, 2, 1], 0, 2)]).is_empty());
    let bad = verify::verify_failures(&[
        report(vec![2, 2, 2, 2], 0, 2),
        report(vec![0, 1, 0, 1], 1, 0),
    ]);
    assert_eq!(bad.len(), 4, "{bad:?}");
    assert!(bad[0].contains("needs both"), "{bad:?}");
    assert!(bad[1].contains("K=3, circuit"), "{bad:?}");
    assert!(bad[2].contains("detection control"), "{bad:?}");
    assert!(bad[3].contains("needs both"), "{bad:?}");
}
