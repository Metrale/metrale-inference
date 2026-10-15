// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: per-family and per-group tables (pure).
//!
//! `union_ms` is the family's share of the busy union (overlapping kernels
//! split the overlap equally), so the rows add up to U(all). `sum_ms` is plain
//! kernel time and double-counts overlap. Medians are true medians (the mean
//! of the two middle values for an even count) over the launches that pass
//! the family's grid filter, if one is given.

use std::collections::{BTreeMap, HashMap};
use std::sync::LazyLock;

use indexmap::IndexMap;
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::core::error::Result;
use crate::core::families::FamilyTable;
use crate::core::model::{Kernel, ms};
use crate::core::repr::quote;
use crate::core::stats::median_i64;
use crate::core::union::attribute;
use crate::fail;

pub const UNCLASSIFIED: &str = "(unclassified)";

static FILTER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?P<family>[^:]+):(?P<dim>[xyz])>=(?P<min>[0-9]+)$")
        .expect("the grid filter pattern is a valid regex")
});

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridFilter {
    pub family: String,
    pub dim: char,
    pub minimum: i64,
}

impl GridFilter {
    pub fn admits(&self, k: &Kernel) -> bool {
        let v = match self.dim {
            'x' => k.grid.0,
            'y' => k.grid.1,
            _ => k.grid.2,
        };
        v >= self.minimum
    }

    fn label(&self) -> String {
        format!("{}>={}", self.dim, self.minimum)
    }
}

/// `FAMILY:DIM>=MIN`, at most one per family, the family must be declared.
pub fn parse_grid_filters(
    specs: &[String],
    table: &FamilyTable,
) -> Result<BTreeMap<String, GridFilter>> {
    let declared = table.families();
    let mut out = BTreeMap::new();
    for spec in specs {
        let Some(m) = FILTER.captures(spec) else {
            fail!(
                "grid filter {} must look like FAMILY:x>=N (dims x, y, z)",
                quote(spec)
            );
        };
        let fam = &m["family"];
        if !declared.contains(&fam) {
            fail!("grid filter {}: unknown family {}", quote(spec), quote(fam));
        }
        if out.contains_key(fam) {
            fail!(
                "grid filter {}: family {} already has a filter",
                quote(spec),
                quote(fam)
            );
        }
        let Ok(minimum) = m["min"].parse::<i64>() else {
            fail!(
                "grid filter {}: the minimum does not fit a 64-bit integer",
                quote(spec)
            );
        };
        let dim = m["dim"].chars().next().unwrap_or('x');
        out.insert(
            fam.to_string(),
            GridFilter {
                family: fam.to_string(),
                dim,
                minimum,
            },
        );
    }
    Ok(out)
}

/// Table rows, each a JSON object.
pub type Rows = Vec<Map<String, Value>>;

/// (family rows, group rows), both sorted by union time, largest first.
pub fn family_tables(
    kernels: &[Kernel],
    labels: &[&str],
    table: &FamilyTable,
    filters: &BTreeMap<String, GridFilter>,
    u_all_ns: i64,
) -> Result<(Rows, Rows)> {
    if u_all_ns <= 0 {
        fail!("family table: the window has no busy time");
    }
    let att = attribute(
        kernels
            .iter()
            .zip(labels)
            .map(|(k, f)| (k.start, k.end, *f)),
    )?;
    let mut work: IndexMap<&str, i64> = IndexMap::new();
    let mut durs: HashMap<&str, Vec<i64>> = HashMap::new();
    let mut launches: HashMap<&str, u64> = HashMap::new();
    for (k, f) in kernels.iter().zip(labels) {
        *work.entry(f).or_insert(0) += k.dur();
        *launches.entry(f).or_insert(0) += 1;
        if filters.get(*f).is_none_or(|flt| flt.admits(k)) {
            durs.entry(f).or_default().push(k.dur());
        }
    }
    for (fam, flt) in filters {
        if durs.get(fam.as_str()).is_none_or(|d| d.is_empty()) {
            fail!(
                "grid filter {fam}:{} matched no launches in the window",
                flt.label()
            );
        }
    }
    let a = |f: &str| att.get(f).copied().unwrap_or(0.0);
    let mut fams: Vec<&str> = work.keys().copied().collect();
    fams.sort_by(|x, y| a(y).total_cmp(&a(x)).then_with(|| x.cmp(y)));
    let mut rows = Vec::with_capacity(fams.len());
    for f in fams {
        let unc = f == UNCLASSIFIED;
        let d = durs.get(f).map(Vec::as_slice).unwrap_or(&[]);
        let (group, class) = if unc {
            (UNCLASSIFIED, UNCLASSIFIED)
        } else {
            (table.group(f)?, table.class(f)?.as_str())
        };
        let row = json!({
            "family": f,
            "group": group,
            "class": class,
            "union_ms": ms(a(f)),
            "share": a(f) / u_all_ns as f64,
            "sum_ms": ms(work[f] as f64),
            "launches": launches[f],
            "median_ms": median_i64(d).map(ms),
            "median_n": d.len(),
            "grid_filter": filters.get(f).map(GridFilter::label).unwrap_or_default(),
        });
        rows.push(into_map(row));
    }
    let mut groups: IndexMap<String, (f64, f64, f64, u64)> = IndexMap::new();
    for r in &rows {
        let g = groups
            .entry(r["group"].as_str().unwrap_or_default().to_string())
            .or_insert((0.0, 0.0, 0.0, 0));
        g.0 += r["union_ms"].as_f64().unwrap_or(0.0);
        g.1 += r["share"].as_f64().unwrap_or(0.0);
        g.2 += r["sum_ms"].as_f64().unwrap_or(0.0);
        g.3 += r["launches"].as_u64().unwrap_or(0);
    }
    let mut glist: Vec<(String, (f64, f64, f64, u64))> = groups.into_iter().collect();
    glist.sort_by(|x, y| y.1.0.total_cmp(&x.1.0).then_with(|| x.0.cmp(&y.0)));
    let grows = glist
        .into_iter()
        .map(|(g, (u, s, w, l))| {
            into_map(json!({"group": g, "union_ms": u, "share": s, "sum_ms": w, "launches": l}))
        })
        .collect();
    Ok((rows, grows))
}

/// A `json!` object literal as its map; every caller passes an object.
pub fn into_map(v: Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m,
        other => {
            let mut m = Map::new();
            m.insert("value".to_string(), other);
            m
        }
    }
}
