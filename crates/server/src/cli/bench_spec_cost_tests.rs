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
fn a_depth_given_twice_is_refused() {
    let runs = vec![run(0, &[1], "1"), run(1, &[1], "1"), run(1, &[1], "1")];
    let err = assemble(&key(), &runs).unwrap_err();
    assert!(
        format!("{err:#}").contains("duplicate cell n=1 k=1"),
        "{err:#}"
    );
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
