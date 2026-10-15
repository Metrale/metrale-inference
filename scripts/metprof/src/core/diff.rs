// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: capture A vs capture B (pure): the ENGAGED proof and where the
//! time moved.
//!
//! Each side is a kernel list, its family labels and its unit count (1 for a
//! whole window, the asserted request count for `--per request`). Every ms and
//! launch column is divided by its side's unit count before the delta is
//! taken, so two windows holding different request counts compare per
//! request. Medians are per launch (us) and are not normalized. Deltas are B
//! minus A.
//!
//! Kernel rows are keyed by (name, grid); a kernel whose name occurs on one
//! side only is listed in only_in_a / only_in_b, which is the proof that a
//! lever put a new kernel on (or took one off) the path.

use std::collections::BTreeSet;
use std::hash::Hash;

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::core::error::Result;
use crate::core::model::{Kernel, KernelKey, ms};
use crate::core::stats::median_i64;
use crate::core::tables::into_map;
use crate::core::union::union_ns;
use crate::fail;

#[derive(Debug, Clone)]
pub struct Side<'a> {
    pub kernels: &'a [Kernel],
    pub labels: Vec<&'a str>,
    pub units: i64,
}

fn aggregate<'a, K: Hash + Eq>(
    side: &Side<'a>,
    key: impl Fn(&'a Kernel, &'a str) -> K,
) -> IndexMap<K, Vec<i64>> {
    let mut agg: IndexMap<K, Vec<i64>> = IndexMap::new();
    for (k, label) in side.kernels.iter().zip(&side.labels) {
        agg.entry(key(k, label)).or_default().push(k.dur());
    }
    agg
}

fn delta_row(
    a: &Side,
    b: &Side,
    da: &[i64],
    db: &[i64],
    mut r: Map<String, Value>,
) -> Map<String, Value> {
    let (ma, mb) = (
        ms(da.iter().sum::<i64>() as f64) / a.units as f64,
        ms(db.iter().sum::<i64>() as f64) / b.units as f64,
    );
    let (la, lb) = (
        da.len() as f64 / a.units as f64,
        db.len() as f64 / b.units as f64,
    );
    let med_a = median_i64(da).map(|m| m / 1e3);
    let med_b = median_i64(db).map(|m| m / 1e3);
    let d_med = match (med_a, med_b) {
        (Some(x), Some(y)) => Some(y - x),
        _ => None,
    };
    for (k, v) in [
        ("a_ms", json!(ma)),
        ("b_ms", json!(mb)),
        ("d_ms", json!(mb - ma)),
        ("a_launches", json!(la)),
        ("b_launches", json!(lb)),
        ("d_launches", json!(lb - la)),
        ("a_median_us", json!(med_a)),
        ("b_median_us", json!(med_b)),
        ("d_median_us", json!(d_med)),
    ] {
        r.insert(k.to_string(), v);
    }
    r
}

fn d_ms(r: &Map<String, Value>) -> f64 {
    r["d_ms"].as_f64().unwrap_or(0.0).abs()
}

fn family_rows(a: &Side, b: &Side) -> Vec<Map<String, Value>> {
    let (ag, bg) = (aggregate(a, |_, f| f), aggregate(b, |_, f| f));
    let keys: BTreeSet<&str> = ag.keys().chain(bg.keys()).copied().collect();
    let mut rows: Vec<(&str, Map<String, Value>)> = keys
        .into_iter()
        .map(|f| {
            let base = into_map(json!({"family": f}));
            let empty = Vec::new();
            (
                f,
                delta_row(
                    a,
                    b,
                    ag.get(f).unwrap_or(&empty),
                    bg.get(f).unwrap_or(&empty),
                    base,
                ),
            )
        })
        .collect();
    rows.sort_by(|x, y| d_ms(&y.1).total_cmp(&d_ms(&x.1)).then_with(|| x.0.cmp(y.0)));
    rows.into_iter().map(|(_, r)| r).collect()
}

fn kernel_rows(a: &Side, b: &Side, top: usize) -> Vec<Map<String, Value>> {
    let key = |k: &Kernel, _: &str| (k.name.clone(), k.grid);
    let (ag, bg) = (aggregate(a, key), aggregate(b, key));
    let keys: BTreeSet<&KernelKey> = ag.keys().chain(bg.keys()).collect();
    let mut rows: Vec<(&KernelKey, Map<String, Value>)> = keys
        .into_iter()
        .map(|kk| {
            let (x, y, z) = kk.1;
            let base = into_map(json!({"kernel": kk.0, "grid": [x, y, z]}));
            let empty = Vec::new();
            (
                kk,
                delta_row(
                    a,
                    b,
                    ag.get(kk).unwrap_or(&empty),
                    bg.get(kk).unwrap_or(&empty),
                    base,
                ),
            )
        })
        .collect();
    // Ties on |delta| order by name, then grid, so the table is deterministic.
    rows.sort_by(|x, y| d_ms(&y.1).total_cmp(&d_ms(&x.1)).then_with(|| x.0.cmp(y.0)));
    rows.into_iter().take(top).map(|(_, r)| r).collect()
}

fn only(x: &Side, other: &Side) -> Vec<Map<String, Value>> {
    let names: BTreeSet<&str> = other.kernels.iter().map(|k| k.name.as_str()).collect();
    let mut agg: IndexMap<&str, Vec<i64>> = IndexMap::new();
    for k in x
        .kernels
        .iter()
        .filter(|k| !names.contains(k.name.as_str()))
    {
        agg.entry(&k.name).or_default().push(k.dur());
    }
    let mut rows: Vec<(&str, f64, f64)> = agg
        .iter()
        .map(|(n, d)| {
            (
                *n,
                ms(d.iter().sum::<i64>() as f64) / x.units as f64,
                d.len() as f64 / x.units as f64,
            )
        })
        .collect();
    rows.sort_by(|p, q| q.1.total_cmp(&p.1).then_with(|| p.0.cmp(q.0)));
    rows.into_iter()
        .map(|(n, m, l)| into_map(json!({"kernel": n, "ms": m, "launches": l})))
        .collect()
}

pub fn capture_diff(a: &Side, b: &Side, top: i64) -> Result<Map<String, Value>> {
    for (name, s) in [("A", a), ("B", b)] {
        if s.units < 1 {
            fail!(
                "side {name}: unit count must be at least 1, got {}",
                s.units
            );
        }
        if s.kernels.is_empty() {
            fail!("side {name}: the window has no kernels");
        }
        if s.labels.len() != s.kernels.len() {
            fail!("side {name}: one family label per kernel is required");
        }
    }
    if top < 1 {
        fail!("--top must be at least 1, got {top}");
    }
    let mut totals = Map::new();
    let mut vals: Vec<[f64; 3]> = Vec::new();
    for (name, s) in [("a", a), ("b", b)] {
        let u = s.units as f64;
        let v = [
            s.kernels.len() as f64 / u,
            ms(s.kernels.iter().map(Kernel::dur).sum::<i64>() as f64) / u,
            ms(union_ns(s.kernels.iter().map(|k| (k.start, k.end)))? as f64) / u,
        ];
        totals.insert(format!("{name}_units"), json!(s.units));
        totals.insert(format!("{name}_kernels"), json!(v[0]));
        totals.insert(format!("{name}_sum_ms"), json!(v[1]));
        totals.insert(format!("{name}_u_all_ms"), json!(v[2]));
        vals.push(v);
    }
    for (i, col) in ["kernels", "sum_ms", "u_all_ms"].iter().enumerate() {
        totals.insert(format!("d_{col}"), json!(vals[1][i] - vals[0][i]));
    }
    let top = usize::try_from(top).unwrap_or(usize::MAX);
    Ok(into_map(json!({
        "totals": totals,
        "only_in_a": only(a, b),
        "only_in_b": only(b, a),
        "families": family_rows(a, b),
        "kernels": kernel_rows(a, b, top),
    })))
}
