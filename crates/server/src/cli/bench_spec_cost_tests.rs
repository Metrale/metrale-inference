// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for assembling the spec-cost table from runs. The record key names are
//! the bench writer's (its own tests pin writer and reader together); these pin the merge.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants: none beyond the types.

use std::time::Duration;

use metrale_bench::result::BenchmarkResult;

use super::*;

fn key() -> TableKey {
    TableKey {
        schema: SCHEMA,
        box_class: "gb10".into(),
        recipe: "qwen3.6/test".into(),
        plan_digests: BTreeMap::from([("verify".to_string(), "abc123".to_string())]),
    }
}

/// 2026-10-04: A spec-cost run at depth `k` over `widths`; a width's wall time is
/// `10·n + k` ms, its draft share `k` ms (0 at k = 0), its joules `n` and `k/10`.
fn run(k: usize, widths: &[usize], widths_param: &str) -> (String, RunRecord) {
    let mut m = BTreeMap::from([("k".to_string(), k as f64)]);
    for &n in widths {
        let f = n as f64;
        for (field, v) in [
            ("vacuous", 0.0),
            ("steps", 100.0),
            ("wall_ms", 10.0 * f + k as f64),
            ("draft_ms", k as f64),
            ("verify_j", f),
            ("draft_j", k as f64 / 10.0),
            ("tok_per_step", 1.0),
        ] {
            m.insert(format!("n{n}_{field}"), v);
        }
    }
    let record = RunRecord {
        schema: 1,
        run_id: String::new(),
        benchmark_id: "spec-cost".into(),
        benchmark_name: String::new(),
        recorded_at: 0,
        target_url: "http://127.0.0.1:1".into(),
        target_model: "m".into(),
        params: BTreeMap::from([
            ("k".to_string(), k.to_string()),
            ("widths".to_string(), widths_param.to_string()),
        ]),
        serve_overrides: BTreeMap::new(),
        source: Default::default(),
        metrale_version: String::new(),
        frame: BenchmarkResult::completed("done", Duration::ZERO).with_metrics(m),
    };
    (format!("k{k}.json"), record)
}

#[test]
fn one_run_per_depth_makes_a_table_the_serve_loads() {
    let runs: Vec<_> = (0..=2).map(|k| run(k, &[1, 4], "1,4")).collect();
    let text = assemble(&key(), &runs).unwrap();
    let table = CostTable::parse(&text).unwrap();
    assert_eq!(table.key, key());
    assert_eq!(table.max_k(), 2);
    let c = table.cell(4, 2).unwrap();
    assert_eq!(
        (c.verify_ms, c.draft_ms, c.verify_j, c.draft_j),
        (40.0, 2.0, 4.0, 0.2)
    );
    let c0 = table.cell(1, 0).unwrap();
    assert_eq!((c0.verify_ms, c0.draft_ms, c0.draft_j), (10.0, 0.0, 0.0));
}

#[test]
fn a_missing_depth_is_refused() {
    let runs = vec![run(0, &[1], "1"), run(2, &[1], "1")];
    let err = assemble(&key(), &runs).unwrap_err();
    assert!(format!("{err:#}").contains("lacks depth 1"), "{err:#}");
}

#[test]
fn repeats_of_a_depth_combine_into_their_median_and_a_file_twice_is_refused() {
    // 2026-10-04: Three repeats of k = 1 with wall times 11, 11 + 6, 11 + 30 ms at n = 1: the
    // median is the middle one; an even count takes the mean of the middle two.
    let mut runs = vec![run(0, &[1], "1"), run(1, &[1], "1")];
    for (i, extra) in [(2, 6.0), (3, 30.0)] {
        let (_, mut r) = run(1, &[1], "1");
        *r.frame.metrics.get_mut("n1_wall_ms").unwrap() += extra;
        runs.push((format!("k1-rep{i}.json"), r));
    }
    let table = CostTable::parse(&assemble(&key(), &runs).unwrap()).unwrap();
    assert_eq!(
        table.cell(1, 1).unwrap().verify_ms,
        16.0,
        "median of 10, 16, 40"
    );
    let even = CostTable::parse(&assemble(&key(), &runs[..3]).unwrap()).unwrap();
    assert_eq!(
        even.cell(1, 1).unwrap().verify_ms,
        13.0,
        "mean of 10 and 16"
    );
    let twice = vec![run(0, &[1], "1"), run(1, &[1], "1"), run(1, &[1], "1")];
    let err = assemble(&key(), &twice).unwrap_err();
    assert!(format!("{err:#}").contains("given twice"), "{err:#}");
}

#[test]
fn runs_measured_differently_are_refused() {
    let runs = vec![run(0, &[1], "1"), run(1, &[1], "1,2")];
    let err = assemble(&key(), &runs).unwrap_err();
    assert!(format!("{err:#}").contains("not measured alike"), "{err:#}");
    let mut other_model = run(1, &[1], "1");
    other_model.1.target_model = "other".into();
    let err = assemble(&key(), &[run(0, &[1], "1"), other_model]).unwrap_err();
    assert!(format!("{err:#}").contains("not measured alike"), "{err:#}");
}

#[test]
fn another_benchmark_s_result_is_refused() {
    let mut wrong = run(0, &[1], "1");
    wrong.1.benchmark_id = "decode-floor".into();
    let err = assemble(&key(), &[wrong]).unwrap_err();
    assert!(format!("{err:#}").contains("not spec-cost"), "{err:#}");
}

fn drafter() -> DrafterKey {
    DrafterKey {
        weights_sha256: "ab12".into(),
        vocab: 1000,
        quantization: "bf16".into(),
        context: true,
    }
}

/// 2026-10-04: `run` plus acceptance counts: one bucket per `(edge, accepted, rejected)`, and
/// `first` verify steps of one draft accepting it.
fn with_counts(k: usize, buckets: &[(f32, u64, u64)], first: u64) -> (String, RunRecord) {
    let (label, mut record) = run(k, &[1], "1");
    let m = &mut record.frame.metrics;
    for (i, &(edge, acc, rej)) in buckets.iter().enumerate() {
        m.insert(format!("conf{i}_le"), f64::from(edge));
        m.insert(format!("conf{i}_accepted"), acc as f64);
        m.insert(format!("conf{i}_rejected"), rej as f64);
    }
    m.insert("steps_d1_a1".into(), first as f64);
    (label, record)
}

#[test]
fn the_calibration_pools_every_run_with_counts() {
    // 2026-10-04: Neither run alone reaches MIN_OUTCOMES (60 drafts, 60 steps); pooled they do,
    // and the k = 0 run, which has no counts, is skipped.
    let runs = vec![
        run(0, &[1], "1"),
        with_counts(1, &[(0.0, 45, 15)], 60),
        with_counts(2, &[(0.0, 45, 15)], 60),
    ];
    let c = AcceptanceCalibration::parse(&calibrate(drafter(), &runs).unwrap()).unwrap();
    assert_eq!(c.drafter, drafter());
    assert_eq!(c.p_given_lp(-0.5), 0.75);
    assert_eq!(c.prior(1), 1.0);
    let alone = calibrate(drafter(), &runs[1..2]).unwrap_err();
    assert!(
        format!("{alone:#}").contains("60 reached drafts"),
        "{alone:#}"
    );
}

#[test]
fn runs_with_different_buckets_or_no_counts_are_refused() {
    let runs = vec![
        with_counts(1, &[(0.0, 100, 0)], 100),
        with_counts(2, &[(-1.0, 50, 0), (0.0, 50, 0)], 100),
    ];
    let err = calibrate(drafter(), &runs).unwrap_err();
    assert!(format!("{err:#}").contains("confidence buckets"), "{err:#}");
    let err = calibrate(drafter(), &[run(0, &[1], "1")]).unwrap_err();
    assert!(format!("{err:#}").contains("no run carries"), "{err:#}");
}
