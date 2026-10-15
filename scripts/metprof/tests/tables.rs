// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the per-family and per-group tables: union rows add up, the
//! median is the true median, grid filters restrict the median only.

mod common;

use std::collections::HashMap;

use common::{FAMILIES_TOML, assert_close, assert_err, f};
use metprof::core::config::parse_toml;
use metprof::core::error::Result;
use metprof::core::families::{FamilyTable, classify, table_from_toml};
use metprof::core::model::Kernel;
use metprof::core::tables::{UNCLASSIFIED, family_tables, parse_grid_filters};
use metprof::core::union::union_ns;
use serde_json::{Map, Value};

type Row = Map<String, Value>;

fn table() -> FamilyTable {
    table_from_toml(&parse_toml(FAMILIES_TOML.as_bytes()).unwrap()).unwrap()
}

fn k(name: &str, start: i64, end: i64, grid: (i64, i64, i64), stream: i64) -> Kernel {
    Kernel {
        start,
        end,
        stream,
        correlation: 0,
        graph_node: None,
        name: name.to_string(),
        grid,
    }
}

fn k1(name: &str, start: i64, end: i64) -> Kernel {
    k(name, start, end, (1, 1, 1), 1)
}

fn tables(ks: &[Kernel], filters: &[&str]) -> Result<(HashMap<String, Row>, HashMap<String, Row>)> {
    let t = table();
    let labels: Vec<&str> = classify(ks, &t)
        .into_iter()
        .map(|x| x.unwrap_or(UNCLASSIFIED))
        .collect();
    let specs: Vec<String> = filters.iter().map(|s| s.to_string()).collect();
    let flt = parse_grid_filters(&specs, &t)?;
    let u = union_ns(ks.iter().map(|x| (x.start, x.end)))?;
    let (rows, groups) = family_tables(ks, &labels, &t, &flt, u)?;
    let by = |rs: Vec<Row>, key: &str| {
        rs.into_iter()
            .map(|r| (r[key].as_str().unwrap().to_string(), r))
            .collect()
    };
    Ok((by(rows, "family"), by(groups, "group")))
}

#[test]
fn union_rows_add_up_to_the_busy_union_and_sum_double_counts() {
    let ks = [
        k1("gemm_a", 0, 1_000_000),
        k("nccl_x", 500_000, 1_500_000, (1, 1, 1), 2),
        k1("norm_y", 2_000_000, 2_100_000),
    ];
    let (rows, groups) = tables(&ks, &[]).unwrap();
    assert_close(rows.values().map(|r| f(&r["union_ms"])).sum(), 1.6);
    assert_close(rows.values().map(|r| f(&r["share"])).sum(), 1.0);
    assert_close(f(&rows["gemm"]["union_ms"]), 0.75);
    assert_close(f(&rows["gemm"]["sum_ms"]), 1.0);
    let mut gs: Vec<&str> = groups.keys().map(String::as_str).collect();
    gs.sort();
    assert_eq!(gs, vec!["comm", "dense", "misc"]);
    assert_close(f(&groups["comm"]["union_ms"]), 0.75);
}

#[test]
fn true_median_for_even_count() {
    let ks: Vec<Kernel> = [1_000_000, 4_000_000, 2_000_000, 9_000_000]
        .iter()
        .enumerate()
        .map(|(i, d)| k1("gemm", i as i64 * 10_000_000, i as i64 * 10_000_000 + d))
        .collect();
    let (rows, _) = tables(&ks, &[]).unwrap();
    // the p50 of the sorted list (1, 2, 4, 9) is 3 ms, not the upper middle 4 ms
    assert_close(f(&rows["gemm"]["median_ms"]), 3.0);
    assert_eq!(rows["gemm"]["median_n"], 4);
}

#[test]
fn grid_filter_restricts_the_median_only() {
    let ks = [
        k("gemm", 0, 100_000, (8, 1, 1), 1),
        k("gemm", 1_000_000, 3_000_000, (512, 1, 1), 1),
        k("gemm", 4_000_000, 6_000_000, (600, 1, 1), 1),
    ];
    let (rows, _) = tables(&ks, &["gemm:x>=512"]).unwrap();
    assert_close(f(&rows["gemm"]["median_ms"]), 2.0);
    assert_eq!(
        (&rows["gemm"]["median_n"], &rows["gemm"]["launches"]),
        (&Value::from(2), &Value::from(3))
    );
    assert_close(f(&rows["gemm"]["sum_ms"]), 4.1);
    let (unfiltered, _) = tables(&ks, &[]).unwrap();
    assert_close(f(&unfiltered["gemm"]["median_ms"]), 2.0);
}

#[test]
fn grid_filter_that_admits_nothing_is_an_error() {
    let ks = [k("gemm", 0, 100, (8, 1, 1), 1)];
    assert_err(tables(&ks, &["gemm:y>=2"]), "matched no launches");
}

#[test]
fn unclassified_row() {
    let (rows, groups) = tables(&[k1("gemm", 0, 100), k1("mystery", 100, 300)], &[]).unwrap();
    assert_eq!(rows[UNCLASSIFIED]["launches"], 1);
    assert_close(f(&rows[UNCLASSIFIED]["union_ms"]), 0.0002);
    assert!(groups.contains_key(UNCLASSIFIED));
}

#[test]
fn bad_filters() {
    let t = table();
    for (spec, msg) in [
        ("gemm:w>=1", "must look like"),
        ("gemm:x>1", "must look like"),
        ("ghost:x>=1", "unknown family"),
        ("gemm:x>=-1", "must look like"),
    ] {
        assert_err(parse_grid_filters(&[spec.to_string()], &t), msg);
    }
    assert_err(
        parse_grid_filters(&["gemm:x>=1".to_string(), "gemm:y>=1".to_string()], &t),
        "already has a filter",
    );
}
