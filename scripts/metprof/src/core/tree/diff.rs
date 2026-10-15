// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: a kernel tree against an older one (pure), matched by node path.
//!
//! A path is the node keys from the root joined by `/`, so it is stable across
//! captures as long as the plan (circuit) or the family table and overlay
//! (families) name the node the same way. A subtree the older tree has and the
//! newer one does not is reported as fused away (or renamed) with its old
//! time; a subtree only the newer one has is reported as new. Only maximal
//! subtrees are listed: the descendants of a fused-away node are implied.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::core::error::Result;
use crate::core::tree::{BaselineView, Node, Tree, us};
use crate::fail;

fn same_source(a: &Tree, b: &Tree) -> Result<()> {
    if a.source != b.source {
        fail!(
            "kernel trees from different sources ({} vs {}) have no common paths",
            a.source.as_str(),
            b.source.as_str()
        );
    }
    Ok(())
}

fn totals(t: &Tree) -> BTreeMap<String, i64> {
    t.paths()
        .into_iter()
        .map(|(p, n)| (p, n.total().sum))
        .collect()
}

/// Maximal paths of `from` with no node at the same path in `other`.
fn maximal_absent<'t>(from: &'t Tree, other: &BTreeMap<String, i64>) -> Vec<(String, &'t Node)> {
    let mut out: Vec<(String, &Node)> = Vec::new();
    for (p, n) in from.paths() {
        let inside = out.iter().any(|(q, _)| p.starts_with(&format!("{q}/")));
        if !inside && !other.contains_key(&p) {
            out.push((p, n));
        }
    }
    out
}

/// Set every node's baseline from `base` (same path) and record what `base`
/// had that `tree` does not.
pub fn apply_baseline(tree: &mut Tree, base: &Tree) -> Result<()> {
    same_source(tree, base)?;
    fn set(prefix: &str, map: &mut IndexMap<String, Node>, base: &BTreeMap<String, i64>) {
        for (k, n) in map.iter_mut() {
            let p = if prefix.is_empty() {
                k.clone()
            } else {
                format!("{prefix}/{k}")
            };
            n.baseline = base.get(&p).copied();
            set(&p, &mut n.kernels, base);
        }
    }
    set("", &mut tree.kernels, &totals(base));
    let fused_away = maximal_absent(base, &totals(tree))
        .into_iter()
        .map(|(p, n)| (p, n.total().sum))
        .collect();
    tree.baseline = Some(BaselineView {
        total: base.total_ns(),
        fused_away,
    });
    Ok(())
}

fn side(t: &Tree) -> Result<Value> {
    Ok(json!({
        "source": t.source.as_str(),
        "total_time": us(t.total_ns()),
        "coverage": t.coverage()?,
        "unattributed_time": us(t.unattributed_ns()),
    }))
}

fn listed(items: Vec<(String, &Node)>) -> Vec<Value> {
    items
        .into_iter()
        .map(|(p, n)| json!({"path": p, "fuses": n.fuses, "total_time": us(n.total().sum)}))
        .collect()
}

/// Every path of `a` and `b` with both times and the delta (b - a, a missing
/// side counting 0), largest |delta| first; then the maximal subtrees only in
/// `a` (fused away) and only in `b` (new).
pub fn tree_diff(a: &Tree, b: &Tree) -> Result<Map<String, Value>> {
    same_source(a, b)?;
    let (ta, tb) = (totals(a), totals(b));
    let mut paths: Vec<&String> = ta.keys().collect();
    paths.extend(tb.keys().filter(|p| !ta.contains_key(*p)));
    let mut rows: Vec<(i64, &String, Option<i64>, Option<i64>)> = paths
        .into_iter()
        .map(|p| {
            let (x, y) = (ta.get(p).copied(), tb.get(p).copied());
            (y.unwrap_or(0) - x.unwrap_or(0), p, x, y)
        })
        .collect();
    rows.sort_by(|l, r| match r.0.abs().cmp(&l.0.abs()) {
        Ordering::Equal => l.1.cmp(r.1),
        o => o,
    });
    let opt = |v: Option<i64>| v.map_or(Value::Null, |x| json!(us(x)));
    let table: Vec<Value> = rows
        .iter()
        .map(|(d, p, x, y)| {
            let status = match (x, y) {
                (Some(_), Some(_)) => "both",
                (Some(_), None) => "fused_away",
                _ => "new",
            };
            json!({"path": p, "a": opt(*x), "b": opt(*y), "delta": us(*d), "status": status})
        })
        .collect();
    let mut out = Map::new();
    out.insert("a".into(), side(a)?);
    out.insert("b".into(), side(b)?);
    out.insert("delta".into(), json!(us(b.total_ns() - a.total_ns())));
    out.insert(
        "fused_away".into(),
        Value::Array(listed(maximal_absent(a, &tb))),
    );
    out.insert("new".into(), Value::Array(listed(maximal_absent(b, &ta))));
    out.insert("nodes".into(), Value::Array(table));
    Ok(out)
}
