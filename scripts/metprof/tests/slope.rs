// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: per-row cost (pure): the class boundaries, width order and
//! refusals, absent and zero-launch costs, the `[slope]` config, and the
//! joined kernel tree with its JSON read back strictly. Every number is
//! synthetic.

mod common;

use metprof::core::config::parse_toml;
use metprof::core::families::{FamilyTable, classify, table_from_toml};
use metprof::core::model::Kernel;
use metprof::core::slope::{
    Class, Point, SlopeConfig, UNDECLARED, analyze, class_of, order_widths, slope_config,
};
use metprof::core::tree::Tree;
use metprof::core::tree::config::Resolved;
use metprof::core::tree::families::families_tree;
use metprof::core::tree::json::tree_from_json;
use metprof::core::tree::slope::{Width, slope_tree};
use serde_json::{Map, Value, json};

const BAND: f64 = 0.25;

fn p(rows: u64, ms: f64) -> Point {
    Point {
        rows,
        ms_per_step: ms,
        launches_per_step: 1.0,
    }
}

/// The one change of a three-width series 0 ms, 1 ms, `ms3` at rows 1, 2, 3:
/// marginals 1.0 then `ms3 - 1.0`, so the change is `ms3 - 2.0`.
fn change3(ms3: f64) -> (f64, Class) {
    let a = analyze(&[p(1, 0.0), p(2, 1.0), p(3, ms3)], BAND).expect("analysis");
    assert_eq!(a.changes.len(), 1);
    assert_eq!(a.changes[0].class, a.class);
    (a.changes[0].delta_ms_per_row, a.class)
}

#[test]
fn a_change_of_exactly_the_band_is_flat_and_one_ulp_more_is_rising() {
    assert_eq!(change3(2.25), (0.25, Class::Flat));
    let (d, c) = change3(2.25f64.next_up());
    assert!(d > BAND, "{d}");
    assert_eq!(c, Class::Rising);
    let (d, c) = change3(2.25f64.next_down());
    assert!(d < BAND, "{d}");
    assert_eq!(c, Class::Flat);
}

#[test]
fn a_change_of_exactly_minus_the_band_is_flat_and_one_ulp_less_is_falling() {
    assert_eq!(change3(1.75), (-0.25, Class::Flat));
    let (d, c) = change3(1.75f64.next_down());
    assert!(d < -BAND, "{d}");
    assert_eq!(c, Class::Falling);
    let (d, c) = change3(1.75f64.next_up());
    assert!(d > -BAND, "{d}");
    assert_eq!(c, Class::Flat);
}

#[test]
fn a_zero_band_is_flat_only_for_no_change() {
    assert_eq!(class_of(0.0, 0.0), Class::Flat);
    assert_eq!(class_of(f64::MIN_POSITIVE, 0.0), Class::Rising);
    assert_eq!(class_of(-f64::MIN_POSITIVE, 0.0), Class::Falling);
}

#[test]
fn a_cliff_outranks_a_fall_and_a_fall_outranks_flat() {
    let rows = [1, 2, 3, 4];
    let series = |ms: [f64; 4]| {
        let pts: Vec<Point> = rows.iter().zip(ms).map(|(r, m)| p(*r, m)).collect();
        let a = analyze(&pts, BAND).expect("analysis");
        let each: Vec<Class> = a.changes.iter().map(|c| c.class).collect();
        (each, a.class)
    };
    // Marginals 1.0, 0.5, 1.5.
    assert_eq!(
        series([0.0, 1.0, 1.5, 3.0]),
        (vec![Class::Falling, Class::Rising], Class::Rising)
    );
    // Marginals 1.0, 0.5, 0.5.
    assert_eq!(
        series([0.0, 1.0, 1.5, 2.0]),
        (vec![Class::Falling, Class::Flat], Class::Falling)
    );
    // Marginals 1.0, 1.0, 1.0.
    assert_eq!(
        series([0.0, 1.0, 2.0, 3.0]),
        (vec![Class::Flat, Class::Flat], Class::Flat)
    );
}

#[test]
fn a_marginal_divides_by_the_added_rows_and_a_point_by_its_steps() {
    // 8 ms over 4 steps at 2 rows, 12 launches: 2 ms and 3 launches per step.
    let q = Point::of(2, 4, 8_000_000, 12);
    assert_eq!((q.ms_per_step, q.launches_per_step), (2.0, 3.0));
    assert_eq!(q.launches_per_row(), 1.5);
    let pts = [
        Point {
            rows: 1,
            ms_per_step: 1.0,
            launches_per_step: 2.0,
        },
        Point {
            rows: 2,
            ms_per_step: 1.5,
            launches_per_step: 4.0,
        },
        Point {
            rows: 4,
            ms_per_step: 2.0,
            launches_per_step: 8.0,
        },
    ];
    let a = analyze(&pts, BAND).expect("analysis");
    let m: Vec<(u64, u64, f64, f64)> = a
        .marginals
        .iter()
        .map(|m| (m.from_rows, m.to_rows, m.ms_per_row, m.launches_per_row))
        .collect();
    assert_eq!(m, [(1, 2, 0.5, 2.0), (2, 4, 0.25, 2.0)]);
    assert_eq!(a.changes[0].at_rows, 2);
    assert_eq!(a.changes[0].delta_ms_per_row, -0.25);
    assert_eq!(a.class, Class::Flat, "exactly minus the band");
}

#[test]
fn widths_in_any_order_are_sorted_and_points_must_come_sorted() {
    assert_eq!(order_widths(&[4, 1, 2]).unwrap(), [1, 2, 0]);
    assert_eq!(order_widths(&[1, 2, 4, 8]).unwrap(), [0, 1, 2, 3]);
    let e = analyze(&[p(2, 0.0), p(1, 0.0), p(4, 0.0)], BAND).unwrap_err();
    assert!(e.0.contains("ordered by increasing rows"), "{e}");
}

#[test]
fn fewer_than_three_widths_are_refused() {
    for rows in [&[][..], &[1][..], &[1, 2][..]] {
        let e = order_widths(rows).unwrap_err();
        assert!(e.0.contains("at least 3 widths"), "{e}");
        assert!(e.0.contains(&format!("got {}", rows.len())), "{e}");
    }
    let e = analyze(&[p(1, 0.0), p(2, 1.0)], BAND).unwrap_err();
    assert!(e.0.contains("at least 3 widths"), "{e}");
}

#[test]
fn a_width_given_twice_or_of_zero_rows_is_refused() {
    let e = order_widths(&[2, 1, 2]).unwrap_err();
    assert!(e.0.contains("width 2 rows is given twice"), "{e}");
    let e = order_widths(&[0, 1, 2]).unwrap_err();
    assert!(e.0.contains("at least 1 row, got 0"), "{e}");
}

#[test]
fn a_cost_with_no_launches_is_flat_zero_and_absent_everywhere() {
    let pts: Vec<Point> = [1, 2, 4].iter().map(|r| Point::of(*r, 3, 0, 0)).collect();
    let a = analyze(&pts, 0.0).expect("analysis");
    assert_eq!(a.class, Class::Flat);
    assert_eq!(a.absent_at, [1, 2, 4]);
    assert!(
        a.marginals
            .iter()
            .all(|m| m.ms_per_row == 0.0 && m.launches_per_row == 0.0)
    );
    assert!(pts.iter().all(|q| q.launches_per_row() == 0.0));
}

#[test]
fn a_cost_absent_at_one_width_is_zero_there_and_listed() {
    let pts = [
        Point::of(1, 2, 0, 0),
        Point::of(2, 2, 2_000_000, 2),
        Point::of(4, 2, 4_000_000, 4),
    ];
    let a = analyze(&pts, BAND).expect("analysis");
    assert_eq!(a.absent_at, [1]);
    // 0 -> 1 ms at 2 rows, then 1 -> 2 ms at 4 rows: marginals 1.0, 0.5.
    assert_eq!(a.marginals[0].ms_per_row, 1.0);
    assert_eq!(a.marginals[1].ms_per_row, 0.5);
    assert_eq!(a.class, Class::Falling);
}

const TABLE: &str = r#"
schema = 1
coverage_min = 0.5
[api]
host_sync_regex = "^cudaStreamSynchronize"
[[rule]]
family = "gemm"
group = "dense"
class = "compute"
name_regex = "^gemm"
[[rule]]
family = "norm"
group = "misc"
class = "compute"
name_regex = "^norm"
[[rule]]
family = "split"
group = "misc"
class = "compute"
name_regex = "^split"
[tree]
coverage_min = 0.9
"#;

const SLOPE: &str =
    "[slope]\nflat_band_ms_per_row = 0.001\n[slope.cost_driver]\ngemm = \"weight_bytes\"\n";

fn doc(extra: &str) -> toml::Table {
    parse_toml(format!("{TABLE}{extra}").as_bytes()).expect("toml")
}

fn table() -> FamilyTable {
    table_from_toml(&doc(SLOPE)).expect("table")
}

fn config(extra: &str) -> metprof::core::error::Result<SlopeConfig> {
    let d = doc(extra);
    slope_config(&d, &table_from_toml(&d)?)
}

#[test]
fn the_slope_section_states_the_band_and_declared_drivers_only() {
    let c = config(SLOPE).expect("config");
    assert_eq!(c.band_ms_per_row, 0.001);
    assert_eq!(c.driver("gemm"), "weight_bytes");
    assert_eq!(c.driver("norm"), UNDECLARED);
    let c = config("[slope]\nflat_band_ms_per_row = 0\n").expect("integer band");
    assert_eq!(c.band_ms_per_row, 0.0);
    assert!(c.drivers.is_empty());
}

#[test]
fn the_slope_section_has_no_defaults_and_no_guesses() {
    let refused = |extra: &str, want: &str| {
        let e = config(extra).unwrap_err();
        assert!(e.0.contains(want), "{extra:?}: {e}");
    };
    refused("", "[slope] is required");
    refused(
        "[slope]\n",
        "missing required key(s) ['flat_band_ms_per_row']",
    );
    refused(
        "[slope]\nflat_band_ms_per_row = -0.5\n",
        "finite number >= 0",
    );
    refused(
        "[slope]\nflat_band_ms_per_row = \"0.5\"\n",
        "must be a number",
    );
    refused(
        "[slope]\nflat_band_ms_per_row = 0.5\nband = 1\n",
        "unknown key",
    );
    refused(
        "[slope]\nflat_band_ms_per_row = 0.5\n[slope.cost_driver]\ngemm = \"bandwidth\"\n",
        "must be one of",
    );
    refused(
        "[slope]\nflat_band_ms_per_row = 0.5\n[slope.cost_driver]\nattn = \"state_bytes\"\n",
        "'attn' is not a declared family",
    );
}

fn k(name: &str, start_us: i64, dur_us: i64) -> Kernel {
    Kernel {
        start: start_us * 1000,
        end: (start_us + dur_us) * 1000,
        stream: 1,
        correlation: start_us,
        graph_node: None,
        name: name.to_string(),
        demangled: format!("{name}(int)"),
        grid: (1, 1, 1),
        launch: common::LAUNCH0,
    }
}

/// One step at `rows`: gemm (shared bytes: 100, 140, 160 us), norm once per
/// row (50 us each), split only at 4 rows (30 us), and one unclassified
/// 1 us kernel.
fn width(rows: u64) -> Width {
    let gemm = match rows {
        1 => 100,
        2 => 140,
        _ => 160,
    };
    let mut ks = vec![k("gemm_a", 0, gemm), k("odd", 1000, 1)];
    for r in 0..rows as i64 {
        ks.push(k("norm_b", 2000 + 100 * r, 50));
    }
    if rows == 4 {
        ks.push(k("split_c", 3000, 30));
    }
    let t = table();
    let labels = classify(&ks, &t);
    let all: Vec<usize> = (0..ks.len()).collect();
    let tree = families_tree(&ks, &labels, &all, &t, &Resolved::default(), Map::new())
        .expect("family tree");
    Width {
        rows,
        steps: 1,
        graph_share: 1.0,
        capture_sha256: format!("sha-{rows}"),
        tree,
    }
}

fn joined(order: [u64; 3]) -> Tree {
    let cfg = config(SLOPE).expect("config");
    slope_tree(order.iter().map(|r| width(*r)).collect(), &cfg, &table()).expect("slope tree")
}

fn node<'v>(v: &'v Value, path: &str) -> &'v Value {
    let mut n = &v["kernels"];
    for (i, part) in path.split('/').enumerate() {
        if i > 0 {
            n = &n["kernels"];
        }
        n = &n[part];
    }
    n
}

#[test]
fn the_joined_tree_is_the_same_whatever_order_the_widths_come_in() {
    assert_eq!(joined([4, 1, 2]), joined([1, 2, 4]));
    assert_eq!(joined([2, 4, 1]), joined([1, 2, 4]));
}

#[test]
fn every_node_carries_its_slope_and_a_family_its_driver() {
    let v = joined([1, 2, 4]).to_json().expect("json");
    let ms = |us: f64| us / 1000.0;
    let (g1, g2, g4) = (ms(100.0), ms(140.0), ms(160.0));
    let gemm = &node(&v, "dense/gemm")["slope"];
    assert_eq!(
        *gemm,
        json!({
            "class": "FALLING",
            "cost_driver": "weight_bytes",
            "absent_at": [],
            "widths": [
                {"rows": 1, "ms_per_step": g1, "launches_per_step": 1.0, "launches_per_row": 1.0},
                {"rows": 2, "ms_per_step": g2, "launches_per_step": 1.0, "launches_per_row": 0.5},
                {"rows": 4, "ms_per_step": g4, "launches_per_step": 1.0, "launches_per_row": 0.25},
            ],
            "marginals": [
                {"from_rows": 1, "to_rows": 2, "ms_per_row": g2 - g1, "launches_per_row": 0.0},
                {"from_rows": 2, "to_rows": 4, "ms_per_row": (g4 - g2) / 2.0, "launches_per_row": 0.0},
            ],
            "changes": [
                {"at_rows": 2, "delta_ms_per_row": (g4 - g2) / 2.0 - (g2 - g1), "class": "FALLING"},
            ],
        })
    );
    let norm = &node(&v, "misc/norm")["slope"];
    assert_eq!(norm["class"], json!("FLAT"));
    assert_eq!(norm["cost_driver"], json!(UNDECLARED));
    let per_row: Vec<&Value> = norm["widths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| &w["launches_per_row"])
        .collect();
    assert_eq!(per_row, [&json!(1.0), &json!(1.0), &json!(1.0)]);
    // Present at the widest width only: 0 where it did not run, then a cliff.
    let split = &node(&v, "misc/split")["slope"];
    assert_eq!(split["class"], json!("RISING"));
    assert_eq!(split["absent_at"], json!([1, 2]));
    // Leaves and groups are not families: no driver.
    for path in ["dense", "misc", "dense/gemm/gemm_a", "misc/norm/norm_b"] {
        let s = node(&v, path)["slope"].as_object().expect(path);
        assert!(!s.contains_key("cost_driver"), "{path}");
    }
    // The whole step counts the unattributed kernel too.
    let step = &v["slope"]["step"]["widths"];
    assert_eq!(step[0]["ms_per_step"], json!(ms(100.0 + 1.0 + 50.0)));
    assert_eq!(step[2]["launches_per_step"], json!(7.0));
    assert_eq!(v["slope"]["flat_band_ms_per_row"], json!(0.001));
    let rows: Vec<&Value> = v["slope"]["widths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| &w["rows"])
        .collect();
    assert_eq!(rows, [&json!(1), &json!(2), &json!(4)]);
}

#[test]
fn a_node_only_a_narrower_width_ran_is_kept_at_zero_launches() {
    // Mirror the split family: present at 1 row only.
    let cfg = config(SLOPE).expect("config");
    let mut ws = vec![width(1), width(2), width(4)];
    let narrow = {
        let ks = vec![k("gemm_a", 0, 100), k("split_c", 200, 30)];
        let t = table();
        let labels = classify(&ks, &t);
        families_tree(&ks, &labels, &[0, 1], &t, &Resolved::default(), Map::new()).unwrap()
    };
    ws[0].tree = narrow;
    ws[2] = {
        let ks = vec![k("gemm_a", 0, 160)];
        let t = table();
        let labels = classify(&ks, &t);
        Width {
            tree: families_tree(&ks, &labels, &[0], &t, &Resolved::default(), Map::new()).unwrap(),
            ..width(4)
        }
    };
    let t = slope_tree(ws, &cfg, &table()).expect("slope tree");
    let v = t.to_json().expect("json");
    let split = node(&v, "misc/split");
    assert_eq!(split["launches"], json!(0));
    assert_eq!(split["total_time"], json!(0.0));
    assert_eq!(split["slope"]["absent_at"], json!([2, 4]));
    assert_eq!(
        split["slope"]["class"],
        json!("RISING"),
        "marginals -0.03 then 0: a rise in marginal"
    );
    assert_eq!(tree_from_json(&v).expect("reads back"), t);
}

#[test]
fn a_slope_tree_reads_back_exactly() {
    let t = joined([1, 2, 4]);
    assert_eq!(tree_from_json(&t.to_json().unwrap()).unwrap(), t);
}

fn refused(edit: impl Fn(&mut Value), want: &str) {
    let mut v = joined([1, 2, 4]).to_json().unwrap();
    edit(&mut v);
    let e = tree_from_json(&v).unwrap_err();
    assert!(e.0.contains(want), "{e}");
}

#[test]
fn an_edited_slope_is_refused() {
    fn gemm(v: &mut Value) -> &mut Value {
        &mut v["kernels"]["dense"]["kernels"]["gemm"]["slope"]
    }
    refused(
        |v| gemm(v)["class"] = json!("FLAT"),
        "is not what its per-width points imply",
    );
    refused(
        |v| gemm(v)["widths"][0]["ms_per_step"] = json!(0.5),
        "is not what its per-width points imply",
    );
    refused(
        |v| gemm(v)["cost_driver"] = json!("bandwidth"),
        "cost_driver",
    );
    refused(
        |v| {
            gemm(v).as_object_mut().unwrap().remove("widths");
        },
        "widths: must be a list",
    );
    // A consistent slope whose widest point is not the node's own figure.
    refused(
        |v| v["slope"]["widths"][2]["steps"] = json!(2),
        "the widest slope point is not its total_time",
    );
    refused(
        |v| {
            v["kernels"]["misc"]
                .as_object_mut()
                .unwrap()
                .remove("slope");
        },
        "slope must be present exactly when the tree has one",
    );
    refused(
        |v| v["slope"]["widths"][1]["rows"] = json!(1),
        "width 1 rows is given twice",
    );
}

#[test]
fn a_plain_tree_with_a_node_slope_is_refused() {
    let t = joined([1, 2, 4]);
    let mut v = width(4).tree.to_json().unwrap();
    v["kernels"]["dense"]["slope"] = t.to_json().unwrap()["kernels"]["dense"]["slope"].clone();
    let e = tree_from_json(&v).unwrap_err();
    assert!(e.0.contains("slope must be present exactly"), "{e}");
}

#[test]
fn slope_tree_refuses_too_few_or_repeated_widths() {
    let cfg = config(SLOPE).expect("config");
    let e = slope_tree(vec![width(1), width(2)], &cfg, &table()).unwrap_err();
    assert!(e.0.contains("at least 3 widths"), "{e}");
    let e = slope_tree(vec![width(1), width(2), width(2)], &cfg, &table()).unwrap_err();
    assert!(e.0.contains("given twice"), "{e}");
}

#[test]
fn the_shipped_family_table_states_its_band_and_no_guessed_driver() {
    let d = parse_toml(common::SHIPPED_FAMILIES).expect("shipped toml");
    let c = slope_config(&d, &table_from_toml(&d).expect("shipped table")).expect("[slope]");
    assert!(c.band_ms_per_row > 0.0);
    assert!(
        c.drivers.is_empty(),
        "drivers are declared from measurements"
    );
}
