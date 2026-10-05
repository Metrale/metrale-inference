// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the spec-cost table: what it must accept, what it must refuse,
//! and its interpolation boundaries.

use super::*;

fn table_text(cells: &str) -> String {
    format!(
        r#"
[key]
schema = 1
box_class = "gb10"
recipe = "qwen3.6/qwen3.6-35b-a3b-fp8-nvfp4head"
plan_digests = {{ verify = "aa", verify_batch = "bb", draft = "cc" }}
{cells}
"#
    )
}

/// 2026-10-04: Widths 1 and 4 at depths 0..=2, with costs that are easy to interpolate.
fn grid() -> String {
    let mut s = String::new();
    for n in [1usize, 4] {
        for k in 0..=2usize {
            let (dm, dj) = if k == 0 {
                (0.0, 0.0)
            } else {
                (k as f64, 0.1 * k as f64)
            };
            s += &format!(
                "[[cells]]\nn = {n}\nk = {k}\nverify_ms = {}\nverify_j = {}\ndraft_ms = {dm}\ndraft_j = {dj}\n",
                10.0 * n as f64 + k as f64,
                n as f64 + 0.5 * k as f64
            );
        }
    }
    s
}

fn key() -> TableKey {
    CostTable::parse(&table_text(&grid())).unwrap().key
}

// 2026-10-04: What it must accept.

#[test]
fn a_complete_grid_parses_and_answers_measured_cells() {
    let t = CostTable::parse(&table_text(&grid())).unwrap();
    assert_eq!(t.max_k(), 2);
    let c = t.cell(4, 2).unwrap();
    assert_eq!((c.verify_ms, c.verify_j, c.draft_ms), (42.0, 5.0, 2.0));
    assert_eq!(c.step_ms(), 44.0);
    assert!((c.step_j() - 5.2).abs() < 1e-12);
}

#[test]
fn widths_between_measured_ones_interpolate_linearly() {
    let t = CostTable::parse(&table_text(&grid())).unwrap();
    // 2026-10-04: n = 2 is a third of the way from 1 to 4.
    let c = t.cell(2, 1).unwrap();
    assert!((c.verify_ms - 21.0).abs() < 1e-9, "{c:?}");
    assert!((c.verify_j - 2.5).abs() < 1e-9, "{c:?}");
    assert_eq!(c.n, 2);
}

#[test]
fn an_equal_key_has_no_mismatch() {
    assert!(key().check(&key()).is_empty());
}

// 2026-10-04: What it must refuse.

#[test]
fn a_missing_depth_at_any_width_is_refused() {
    let partial = grid().replace("n = 4\nk = 1\n", "n = 4\nk = 3\n");
    let e = CostTable::parse(&table_text(&partial)).unwrap_err();
    assert!(e.contains("lacks depth"), "{e}");
}

#[test]
fn bad_costs_are_refused() {
    for bad in [
        grid().replacen("verify_ms = 10", "verify_ms = 0", 1),
        grid().replacen("verify_j = 1", "verify_j = -1", 1),
        grid().replacen("draft_ms = 0", "draft_ms = 0.5", 1),
        grid().replacen("verify_ms = 10", "verify_ms = nan", 1),
    ] {
        assert!(CostTable::parse(&table_text(&bad)).is_err(), "{bad}");
    }
}

#[test]
fn a_duplicate_cell_or_wrong_schema_is_refused() {
    let dup = format!(
        "{}{}",
        grid(),
        "[[cells]]\nn = 1\nk = 0\nverify_ms = 1\nverify_j = 1\ndraft_ms = 0\ndraft_j = 0\n"
    );
    assert!(
        CostTable::parse(&table_text(&dup))
            .unwrap_err()
            .contains("duplicate")
    );
    let v2 = table_text(&grid()).replace("schema = 1", "schema = 2");
    assert!(CostTable::parse(&v2).unwrap_err().contains("schema"));
}

#[test]
fn every_key_difference_is_named() {
    let t = key();
    let mut s = key();
    s.box_class = "hopper".into();
    s.recipe = "other".into();
    s.plan_digests.insert("verify".into(), "zz".into());
    s.plan_digests.remove("draft");
    s.plan_digests.insert("decode".into(), "dd".into());
    let m = t.check(&s);
    assert_eq!(m.len(), 5, "{m:?}");
    assert!(m.contains(&KeyMismatch::PlanDigest {
        mode: "draft".into(),
        table: Some("cc".into()),
        serve: None
    }));
    assert!(m.contains(&KeyMismatch::PlanDigest {
        mode: "decode".into(),
        table: None,
        serve: Some("dd".into())
    }));
}

// 2026-10-04: Boundaries.

#[test]
fn widths_outside_the_measured_range_take_the_nearest_and_depth_is_capped() {
    let t = CostTable::parse(&table_text(&grid())).unwrap();
    assert_eq!(t.cell(128, 0).unwrap().verify_ms, 40.0);
    assert_eq!(t.cell(128, 0).unwrap().n, 128);
    assert!(t.cell(1, 3).is_none());
}

// 2026-10-04: The acceptance calibration.

fn calib(edges: &str, p: &str, prior: &str) -> Result<AcceptanceCalibration, String> {
    AcceptanceCalibration::parse(&format!(
        r#"
[drafter]
weights_sha256 = "ab"
vocab = 100000
quantization = "bf16"
context = true

[acceptance]
edges = {edges}
p_accept = {p}
prior_by_position = {prior}
"#
    ))
}

#[test]
fn a_draft_lands_in_the_first_bucket_that_bounds_it() {
    let c = calib("[-2.0, -0.5, 0.0]", "[0.2, 0.6, 0.9]", "[0.73, 0.57]").unwrap();
    assert_eq!(c.p_given_lp(-3.0), 0.2);
    assert_eq!(c.p_given_lp(-2.0), 0.2, "an edge belongs to its own bucket");
    assert_eq!(c.p_given_lp(-1.0), 0.6);
    assert_eq!(c.p_given_lp(-0.1), 0.9);
    assert_eq!(c.p_given_lp(0.0), 0.9);
}

#[test]
fn a_chain_expectation_is_the_sum_of_running_products() {
    let c = calib("[-0.5, 0.0]", "[0.5, 0.8]", "[0.7, 0.6, 0.5]").unwrap();
    // 2026-10-04: 0.8 + 0.8*0.5 + (0.8*0.5)*prior(3)=0.5.
    let e = c.expected_accepted(&[-0.1, -1.0], 3);
    assert!((e - (0.8 + 0.4 + 0.2)).abs() < 1e-12, "{e}");
    assert_eq!(c.expected_accepted(&[-0.1], 0), 0.0);
    // 2026-10-04: No confidences: the priors alone, positions past the list take the last.
    let p = c.expected_accepted(&[], 4);
    assert!((p - (0.7 + 0.42 + 0.21 + 0.105)).abs() < 1e-12, "{p}");
}

#[test]
fn malformed_calibrations_are_refused() {
    for (e, p, pr) in [
        ("[-0.5, 0.0]", "[0.5]", "[0.7]"),
        ("[0.0, -0.5]", "[0.5, 0.6]", "[0.7]"),
        ("[-0.5, -0.1]", "[0.5, 0.6]", "[0.7]"),
        ("[-0.5, 0.0]", "[0.5, 1.2]", "[0.7]"),
        ("[-0.5, 0.0]", "[0.5, 0.6]", "[]"),
        ("[]", "[]", "[0.7]"),
    ] {
        assert!(calib(e, p, pr).is_err(), "{e} {p} {pr}");
    }
}

// 2026-10-04: Writing a table.

#[test]
fn a_rendered_table_reads_back_cell_for_cell() {
    let t = CostTable::parse(&table_text(&grid())).unwrap();
    let cells: Vec<Cell> = [1usize, 4]
        .iter()
        .flat_map(|&n| (0..=2).map(move |k| (n, k)))
        .map(|(n, k)| t.cell(n, k).unwrap())
        .collect();
    let text = CostTable::render(&t.key, &cells).unwrap();
    assert_eq!(CostTable::parse(&text).unwrap(), t);
}

#[test]
fn rendering_an_incomplete_grid_is_refused() {
    let t = CostTable::parse(&table_text(&grid())).unwrap();
    let cells = vec![t.cell(1, 0).unwrap(), t.cell(1, 2).unwrap()];
    assert!(
        CostTable::render(&t.key, &cells)
            .unwrap_err()
            .contains("lacks depth")
    );
}
