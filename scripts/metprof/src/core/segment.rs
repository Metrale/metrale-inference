// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: request segments and pinned decode steps (pure).
//!
//! Requests: without NVTX request ranges, a request boundary is an idle gap of
//! at least `gap_ns` between kernel islands. The caller states how many
//! requests the window holds; any other count is an error that lists what was
//! found, so a threshold below the largest intra-request host gap cannot pass
//! silently.
//!
//! Decode steps: the PINNED definition. A step is one launch of the largest
//! CUDA graph (most kernel nodes) in the window; graph kernels share their
//! launch's correlation id. There is no other step definition to select.

use std::collections::BTreeMap;

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::core::error::Result;
use crate::core::model::{Kernel, ms};
use crate::core::repr::{float, quote};
use crate::core::stats::median_i64;
use crate::core::tables::into_map;
use crate::core::union::{islands, union_ns};
use crate::fail;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub start: i64,
    pub end: i64,
    /// Indices into the window's kernel list.
    pub kernels: Vec<usize>,
    pub max_intra_gap: i64,
}

fn g_fmt(x: f64) -> String {
    let s = format!("{x}");
    s.strip_suffix(".0").map(str::to_string).unwrap_or(s)
}

pub fn split_requests(kernels: &[Kernel], gap_ns: i64, expect: i64) -> Result<Vec<Segment>> {
    if gap_ns <= 0 {
        fail!("request gap must be positive, got {gap_ns} ns");
    }
    if expect < 1 {
        fail!("expected request count must be at least 1, got {expect}");
    }
    if kernels.is_empty() {
        fail!("the window has no kernels");
    }
    let mut bounds: Vec<(i64, i64, i64)> = Vec::new();
    for (s, e) in islands(kernels.iter().map(|k| (k.start, k.end)))? {
        match bounds.last_mut() {
            Some(b) if s - b.1 < gap_ns => {
                b.2 = b.2.max(s - b.1);
                b.1 = e;
            }
            _ => bounds.push((s, e, 0)),
        }
    }
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); bounds.len()];
    let mut order: Vec<usize> = (0..kernels.len()).collect();
    order.sort_by_key(|&i| kernels[i].start);
    let mut j = 0;
    for i in order {
        while kernels[i].start > bounds[j].1 {
            j += 1;
        }
        members[j].push(i);
    }
    let segs: Vec<Segment> = bounds
        .iter()
        .zip(members)
        .map(|(b, m)| Segment {
            start: b.0,
            end: b.1,
            kernels: m,
            max_intra_gap: b.2,
        })
        .collect();
    if segs.len() as i64 != expect {
        let t0 = segs[0].start;
        let listing: Vec<String> = segs
            .iter()
            .take(12)
            .enumerate()
            .map(|(n, s)| {
                format!(
                    "#{n}: +{:.1} ms span {:.1} ms, {} kernels",
                    ms((s.start - t0) as f64),
                    ms((s.end - s.start) as f64),
                    s.kernels.len()
                )
            })
            .collect();
        let more = if segs.len() > 12 {
            format!("; ... +{} more", segs.len() - 12)
        } else {
            String::new()
        };
        fail!(
            "request count: a {} ms gap splits the window into {} segments, expected {expect} ({}{more})",
            g_fmt(ms(gap_ns as f64)),
            segs.len(),
            listing.join("; ")
        );
    }
    Ok(segs)
}

pub fn inter_gaps(segs: &[Segment]) -> Vec<i64> {
    segs.windows(2).map(|w| w[1].start - w[0].end).collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedSteps {
    /// Kernel indices of each pinned graph launch, ordered by start.
    pub steps: Vec<Vec<usize>>,
    pub nodes: usize,
    pub graph_launches: usize,
}

pub fn pinned_steps(kernels: &[Kernel]) -> Result<PinnedSteps> {
    let mut groups: IndexMap<i64, Vec<usize>> = IndexMap::new();
    for (i, k) in kernels.iter().enumerate() {
        if k.in_graph() {
            groups.entry(k.correlation).or_default().push(i);
        }
    }
    let Some(nodes) = groups.values().map(Vec::len).max() else {
        fail!(
            "no graph-node kernels in the window: a decode step table needs --cuda-graph-trace=node"
        );
    };
    let mut steps: Vec<Vec<usize>> = groups
        .values()
        .filter(|v| v.len() == nodes)
        .map(|v| {
            let mut v = v.clone();
            v.sort_by_key(|&i| (kernels[i].start, kernels[i].end));
            v
        })
        .collect();
    steps.sort_by_key(|idx| kernels[idx[0]].start);
    Ok(PinnedSteps {
        steps,
        nodes,
        graph_launches: groups.len(),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepStats {
    pub span_median_ms: f64,
    pub sum_per_step_ms: f64,
    pub idle_median_ms: f64,
}

/// Median graph span (first start to last end), kernel sum per step, and the
/// median idle inside a step (span minus the union of its kernels).
pub fn step_stats(kernels: &[Kernel], ps: &PinnedSteps) -> Result<StepStats> {
    let (mut spans, mut idles, mut total) = (Vec::new(), Vec::new(), 0i64);
    for idx in &ps.steps {
        let iv: Vec<(i64, i64)> = idx
            .iter()
            .map(|&i| (kernels[i].start, kernels[i].end))
            .collect();
        let span =
            iv.iter().map(|x| x.1).max().unwrap_or(0) - iv.iter().map(|x| x.0).min().unwrap_or(0);
        spans.push(span);
        idles.push(span - union_ns(iv.iter().copied())?);
        total += iv.iter().map(|(s, e)| e - s).sum::<i64>();
    }
    let (Some(span), Some(idle)) = (median_i64(&spans), median_i64(&idles)) else {
        fail!("no steps to tabulate");
    };
    Ok(StepStats {
        span_median_ms: ms(span),
        sum_per_step_ms: ms(per_unit(total as f64, ps.steps.len())),
        idle_median_ms: ms(idle),
    })
}

/// Share of the window's busy union spent inside the pinned graph launches.
pub fn graph_share(kernels: &[Kernel], ps: &PinnedSteps) -> Result<f64> {
    let inside = union_ns(
        ps.steps
            .iter()
            .flatten()
            .map(|&i| (kernels[i].start, kernels[i].end)),
    )?;
    let all = union_ns(kernels.iter().map(|k| (k.start, k.end)))?;
    if all <= 0 {
        fail!("the window has no busy time");
    }
    Ok(inside as f64 / all as f64)
}

/// The pinned steps of a decode window and the share of busy time they
/// cover, refused below `min_share`: a window the decode graph does not
/// dominate has no step, and a per-step figure of it would be a window sum.
pub fn decode_steps(kernels: &[Kernel], min_share: f64) -> Result<(PinnedSteps, f64)> {
    let ps = pinned_steps(kernels)?;
    let share = graph_share(kernels, &ps)?;
    if share < min_share {
        fail!(
            "the pinned decode graph ({} nodes, {} launches) covers {share:.3} of the window's busy time, below \
             --min-graph-share {}: this window is not a decode window, and a per-step table of it would be a \
             window sum. Use `segment requests`.",
            ps.nodes,
            ps.steps.len(),
            float(min_share)
        );
    }
    Ok((ps, share))
}

/// A window sum per unit (request or step): the one normalisation every
/// per-unit figure uses.
pub fn per_unit(sum: f64, units: usize) -> f64 {
    sum / units as f64
}

/// The unit a per-unit table divides by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Request,
    Step,
}

impl Unit {
    pub fn as_str(self) -> &'static str {
        match self {
            Unit::Request => "request",
            Unit::Step => "step",
        }
    }
}

/// Per row: launches, total ns, and ns per unit index.
type Acc = (u64, i64, BTreeMap<usize, i64>);

/// Row key of a per-unit table: a kernel `(name, grid)` or a family label.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RowKey<'a> {
    Kernel(&'a str, (i64, i64, i64)),
    Family(&'a str),
}

impl RowKey<'_> {
    /// The tie-break text: `('name', (x, y, z))` for a kernel, the label for a family.
    fn sort_text(&self) -> String {
        match self {
            RowKey::Kernel(n, (x, y, z)) => format!("({}, ({x}, {y}, {z}))", quote(n)),
            RowKey::Family(f) => f.to_string(),
        }
    }
}

/// Per-unit table: every column is divided by the number of units, and the
/// per-unit median counts a unit without the key as zero. `by_key` groups by
/// (name, grid), otherwise by label (family).
pub fn per_unit_rows(
    kernels: &[Kernel],
    labels: &[&str],
    units: &[Vec<usize>],
    unit: Unit,
    by_key: bool,
    top: usize,
) -> Result<Vec<Map<String, Value>>> {
    let n = units.len();
    let u = unit.as_str();
    if n == 0 {
        fail!("no {u}s to tabulate");
    }
    let mut per: IndexMap<RowKey, Acc> = IndexMap::new();
    for (ui, idx) in units.iter().enumerate() {
        for &i in idx {
            let k = &kernels[i];
            let key = if by_key {
                RowKey::Kernel(&k.name, k.grid)
            } else {
                RowKey::Family(labels[i])
            };
            let row = per.entry(key).or_insert_with(|| (0, 0, BTreeMap::new()));
            row.0 += 1;
            row.1 += k.dur();
            *row.2.entry(ui).or_insert(0) += k.dur();
        }
    }
    let mut items: Vec<(RowKey, Acc)> = per.into_iter().collect();
    sort_desc_by_total(&mut items);
    let mut rows = Vec::new();
    for (key, (count, total, by_unit)) in items.into_iter().take(top) {
        let mut r = match key {
            RowKey::Kernel(name, (x, y, z)) => into_map(json!({"kernel": name, "grid": [x, y, z]})),
            RowKey::Family(f) => into_map(json!({"family": f})),
        };
        let ns_by_unit: Vec<i64> = (0..n)
            .map(|x| by_unit.get(&x).copied().unwrap_or(0))
            .collect();
        r.insert(format!("ms_per_{u}"), json!(ms(per_unit(total as f64, n))));
        r.insert(
            format!("median_ms_per_{u}"),
            json!(median_i64(&ns_by_unit).map(ms)),
        );
        r.insert(
            format!("launches_per_{u}"),
            json!(per_unit(count as f64, n)),
        );
        r.insert(format!("{u}s_with_it"), json!(by_unit.len()));
        r.insert(
            "avg_us".to_string(),
            json!(total as f64 / count as f64 / 1e3),
        );
        rows.push(r);
    }
    Ok(rows)
}

fn sort_desc_by_total<V>(items: &mut [(RowKey<'_>, (u64, i64, V))]) {
    items.sort_by(|a, b| {
        b.1.1
            .cmp(&a.1.1)
            .then_with(|| a.0.sort_text().cmp(&b.0.sort_text()))
    });
}
