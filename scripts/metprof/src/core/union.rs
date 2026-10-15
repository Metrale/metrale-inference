// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: busy time as a union of intervals (gaps-and-islands), exposed
//! comm, idle, host gaps, and union-attributed time per key.
//!
//! Intervals are (start_ns, end_ns). An island starts when an interval starts
//! strictly after the running maximum end of every earlier interval, so
//! touching intervals merge. This is scale-free: no threshold enters the union.

use std::hash::Hash;

use indexmap::IndexMap;

use crate::core::error::Result;
use crate::fail;

/// Kernel class of an interval; `None` in a class list means unclassified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Class {
    Compute,
    Comm,
}

impl Class {
    pub fn as_str(self) -> &'static str {
        match self {
            Class::Compute => "compute",
            Class::Comm => "comm",
        }
    }

    pub const ALL: [Class; 2] = [Class::Compute, Class::Comm];
}

pub type Interval = (i64, i64);

fn reversed(s: i64, e: i64) -> Result<()> {
    if e < s {
        fail!("interval ends before it starts: ({s}, {e})");
    }
    Ok(())
}

pub fn islands(intervals: impl IntoIterator<Item = Interval>) -> Result<Vec<Interval>> {
    let mut sorted: Vec<Interval> = intervals.into_iter().collect();
    sorted.sort_unstable();
    let mut out: Vec<Interval> = Vec::new();
    for (s, e) in sorted {
        reversed(s, e)?;
        match out.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => out.push((s, e)),
        }
    }
    Ok(out)
}

pub fn union_ns(intervals: impl IntoIterator<Item = Interval>) -> Result<i64> {
    Ok(islands(intervals)?.iter().map(|(s, e)| e - s).sum())
}

/// (count, total ns) of idle gaps between islands that are >= `min_gap_ns`.
pub fn gaps(
    intervals: impl IntoIterator<Item = Interval>,
    min_gap_ns: i64,
) -> Result<(usize, i64)> {
    if min_gap_ns <= 0 {
        fail!("gap threshold must be positive, got {min_gap_ns} ns");
    }
    let isl = islands(intervals)?;
    let found: Vec<i64> = isl
        .windows(2)
        .map(|w| w[1].0 - w[0].1)
        .filter(|g| *g >= min_gap_ns)
        .collect();
    Ok((found.len(), found.iter().sum()))
}

/// Union-attributed ns per key: every elementary interval of busy time is
/// split equally among the intervals active in it, so the values sum to the
/// union and concurrent work is never counted twice. Keys appear in the order
/// they first receive time.
pub fn attribute<K: Hash + Eq + Clone>(
    keyed: impl IntoIterator<Item = (i64, i64, K)>,
) -> Result<IndexMap<K, f64>> {
    let mut events: Vec<(i64, i64, K)> = Vec::new();
    for (s, e, key) in keyed {
        reversed(s, e)?;
        if e > s {
            events.push((s, 1, key.clone()));
            events.push((e, -1, key));
        }
    }
    events.sort_by_key(|ev| (ev.0, ev.1));
    let mut active: IndexMap<K, i64> = IndexMap::new();
    let mut n_active: i64 = 0;
    let mut out: IndexMap<K, f64> = IndexMap::new();
    let mut prev_t: Option<i64> = None;
    for (t, delta, key) in events {
        if let Some(p) = prev_t
            && n_active > 0
            && t > p
        {
            let share = (t - p) as f64 / n_active as f64;
            for (k, c) in &active {
                *out.entry(k.clone()).or_insert(0.0) += share * *c as f64;
            }
        }
        prev_t = Some(t);
        let count = active.entry(key.clone()).or_insert(0);
        *count += delta;
        n_active += delta;
        if *count == 0 {
            active.shift_remove(&key);
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BusySummary {
    pub span_ns: i64,
    pub sum_ns: i64,
    pub u_all_ns: i64,
    pub u_compute_ns: i64,
    pub u_comm_ns: i64,
    pub u_unclassified_ns: i64,
    pub exposed_comm_ns: i64,
    pub idle_ns: i64,
    pub gap_count: usize,
    pub gap_ns: i64,
    pub intervals: usize,
}

/// `classes[i]` is the class of `intervals[i]`, `None` = unclassified.
///
/// exposed comm = U(compute | comm) - U(compute): the comm time during which
/// no compute runs. With full classification it equals U(all) - U(compute).
pub fn busy_summary(
    intervals: &[Interval],
    classes: &[Option<Class>],
    min_gap_ns: i64,
) -> Result<BusySummary> {
    if intervals.len() != classes.len() {
        fail!("one class per interval is required");
    }
    if intervals.is_empty() {
        fail!("no intervals in the selected window");
    }
    let of = |want: &dyn Fn(Option<Class>) -> bool| -> Vec<Interval> {
        intervals
            .iter()
            .zip(classes)
            .filter(|(_, c)| want(**c))
            .map(|(iv, _)| *iv)
            .collect()
    };
    let comp = of(&|c| c == Some(Class::Compute));
    let comm = of(&|c| c == Some(Class::Comm));
    let uncl = of(&|c| c.is_none());
    let classified = of(&|c| c.is_some());
    let u_all = union_ns(intervals.iter().copied())?;
    let u_comp = union_ns(comp)?;
    let span = intervals.iter().map(|iv| iv.1).max().unwrap_or(0)
        - intervals.iter().map(|iv| iv.0).min().unwrap_or(0);
    let (gap_count, gap_ns) = gaps(intervals.iter().copied(), min_gap_ns)?;
    Ok(BusySummary {
        span_ns: span,
        sum_ns: intervals.iter().map(|(s, e)| e - s).sum(),
        u_all_ns: u_all,
        u_compute_ns: u_comp,
        u_comm_ns: union_ns(comm)?,
        u_unclassified_ns: union_ns(uncl)?,
        exposed_comm_ns: union_ns(classified)? - u_comp,
        idle_ns: span - u_all,
        gap_count,
        gap_ns,
        intervals: intervals.len(),
    })
}
