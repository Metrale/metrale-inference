// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Plan fidelity: does a fused plan name the kernels a traced serve launched? A trace
//! oracle (`kernels/circuits/plans/fidelity/*.trace.toml`, written by
//! scripts/fidelity/trace_segments.py from an nsys capture) states, per step variant, the ordered
//! launches of each run the step makes (verify, draft) and of what no run plans (the commit
//! machinery, the accepted-row refeed). [`compare`] scores a plan's launch sequence against one
//! such segment.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Kernels compare by function name (the trace's short name); a collective compares by the
//!   library kernel its [`crate::rules::CommLaunch`] names.
//! - The multiset score is `matched / (planned + traced - matched)` (Jaccard over multisets): 1.0
//!   exactly when the plan launches every traced kernel as often as the trace does and nothing
//!   else. The order score is the longest common subsequence over the longer sequence.
//! - A placeholder group (`novel` emitter) launches nothing and is listed, so a plan that cannot
//!   describe a site scores below 1.0 by what it omits, never by a guessed kernel.

use crate::prelude::*;

use alloc::collections::BTreeMap;

use serde::Deserialize;

use crate::fuser::{FusionPlan, Group};
use crate::rules::Mode;

/// 2026-10-23: A trace oracle.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    /// 2026-10-23: File schema (1).
    pub schema: u64,
    /// 2026-10-23: Where the capture came from.
    pub source: String,
    /// 2026-10-23: SHA-256 of the sqlite the oracle was cut from.
    pub sqlite_sha256: String,
    /// 2026-10-23: The TP rank the capture is of.
    pub rank: u64,
    /// 2026-10-23: The kernel that starts a step.
    pub step_kernel: String,
    /// 2026-10-23: Whole steps in the capture.
    pub steps: u64,
    /// 2026-10-23: Each step's variant, in capture order.
    pub sequence: String,
    /// 2026-10-23: The variants, most frequent first.
    pub variant: Vec<Variant>,
}

/// 2026-10-23: Steps that launch the same multiset.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    /// 2026-10-23: `A`, `B`, ...
    pub name: String,
    /// 2026-10-23: How many steps.
    pub steps: u64,
    /// 2026-10-23: Its segments.
    pub segment: Vec<Segment>,
}

/// 2026-10-23: One run of a step (`mode` and `rows` set), or launches no run plans (both unset).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Segment {
    /// 2026-10-23: `verify`, `draft`, `refeed`, `commit`.
    pub name: String,
    /// 2026-10-23: The run's mode.
    pub mode: Option<String>,
    /// 2026-10-23: The run's rows.
    pub rows: Option<u64>,
    /// 2026-10-23: Its launches, in start order.
    pub kernels: Vec<String>,
}

/// 2026-10-23: Why an oracle was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TraceError {
    /// 2026-10-23: Not the oracle schema.
    #[error("trace oracle: {0}")]
    Parse(String),
    /// 2026-10-23: The file is internally inconsistent.
    #[error("trace oracle: {0}")]
    Shape(String),
}

/// 2026-10-23: Parse and check an oracle: the variants' steps add up, the sequence names only
/// stated variants, and a segment states both of mode and rows or neither, with a known mode.
pub fn parse_trace(text: &str) -> Result<Trace, TraceError> {
    let t: Trace = toml::from_str(text).map_err(|e| TraceError::Parse(e.to_string()))?;
    let bad = |m: String| Err(TraceError::Shape(m));
    if t.schema != 1 {
        return bad(format!("schema {} (this reader knows 1)", t.schema));
    }
    if t.variant.iter().map(|v| v.steps).sum::<u64>() != t.steps {
        return bad(format!("variant steps do not add up to {}", t.steps));
    }
    if u64::try_from(t.sequence.chars().count()).ok() != Some(t.steps) {
        return bad("the step sequence is not one variant per step".into());
    }
    for c in t.sequence.chars() {
        if !t.variant.iter().any(|v| v.name.chars().eq(core::iter::once(c))) {
            return bad(format!("step variant `{c}` is not stated"));
        }
    }
    for v in &t.variant {
        for s in &v.segment {
            match (&s.mode, s.rows) {
                (None, None) => {}
                (Some(m), Some(_)) if Mode::parse(m).is_some() => {}
                _ => return bad(format!("variant {} segment {}: mode and rows", v.name, s.name)),
            }
        }
    }
    Ok(t)
}

/// 2026-10-23: The launches one repetition of `g` makes: its kernels with its collectives
/// inserted at their `at`.
fn one_repetition(g: &Group) -> Vec<String> {
    let mut out = Vec::new();
    for (i, k) in g.kernels.iter().enumerate() {
        out.extend(g.comm.iter().filter(|c| c.at == i as u64).map(|c| c.kernel.clone()));
        out.push(k.func.clone());
    }
    let n = g.kernels.len() as u64;
    out.extend(g.comm.iter().filter(|c| c.at == n).map(|c| c.kernel.clone()));
    out
}

/// 2026-10-23: The plan's launches in execution order: each group's repetitions back to back (a
/// `per_run` group per run, each run's launches in order with its collectives first). Placeholder
/// groups launch nothing; their rule ids are returned beside.
pub fn plan_sequence(plan: &FusionPlan) -> (Vec<String>, Vec<String>) {
    let mut seq = Vec::new();
    let mut novel = Vec::new();
    for g in &plan.groups {
        if g.emitter == crate::hardware::plan::NOVEL_EMITTER {
            novel.push(g.rule.clone());
            continue;
        }
        match g.repeat.count(plan.rows) {
            Some(c) => {
                let one = one_repetition(g);
                for _ in 0..c {
                    seq.extend(one.iter().cloned());
                }
            }
            None => {
                for r in &g.runs {
                    seq.extend(g.comm.iter().map(|c| c.kernel.clone()));
                    for (k, t) in &r.launches {
                        for _ in 0..t.count(r.run) {
                            seq.push(k.func.clone());
                        }
                    }
                }
            }
        }
    }
    (seq, novel)
}

/// 2026-10-23: One plan against one traced segment.
#[derive(Debug, Clone, PartialEq)]
pub struct Fidelity {
    /// 2026-10-23: Launches the plan makes.
    pub planned: u64,
    /// 2026-10-23: Launches the trace made.
    pub traced: u64,
    /// 2026-10-23: Launches both make (multiset intersection).
    pub matched: u64,
    /// 2026-10-23: Longest common subsequence of the two orders.
    pub in_order: u64,
    /// 2026-10-23: Traced more often than planned: kernel to the shortfall.
    pub missing: BTreeMap<String, u64>,
    /// 2026-10-23: Planned more often than traced: kernel to the excess.
    pub extra: BTreeMap<String, u64>,
    /// 2026-10-23: Placeholder groups (sites the plan cannot describe).
    pub novel: Vec<String>,
}

impl Fidelity {
    /// 2026-10-23: The multiset score in [0, 1]; 1.0 when both are empty.
    pub fn share(&self) -> f64 {
        let union = self.planned + self.traced - self.matched;
        if union == 0 {
            1.0
        } else {
            self.matched as f64 / union as f64
        }
    }

    /// 2026-10-23: The order score in [0, 1]; 1.0 when both are empty.
    pub fn order_share(&self) -> f64 {
        let longer = self.planned.max(self.traced);
        if longer == 0 {
            1.0
        } else {
            self.in_order as f64 / longer as f64
        }
    }
}

fn counts(seq: &[String]) -> BTreeMap<&str, u64> {
    let mut m = BTreeMap::new();
    for k in seq {
        *m.entry(k.as_str()).or_insert(0) += 1;
    }
    m
}

/// 2026-10-23: Longest common subsequence length (two rolling rows).
fn lcs(a: &[String], b: &[String]) -> u64 {
    let mut prev = vec![0u64; b.len() + 1];
    let mut cur = vec![0u64; b.len() + 1];
    for x in a {
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y {
                prev[j] + 1
            } else {
                cur[j].max(prev[j + 1])
            };
        }
        core::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// 2026-10-23: Score `plan` against `traced`.
pub fn compare(plan: &FusionPlan, traced: &[String]) -> Fidelity {
    let (seq, novel) = plan_sequence(plan);
    let (p, t) = (counts(&seq), counts(traced));
    let mut missing = BTreeMap::new();
    let mut extra = BTreeMap::new();
    let mut matched = 0;
    for k in p.keys().chain(t.keys()) {
        let (a, b) = (p.get(k).copied().unwrap_or(0), t.get(k).copied().unwrap_or(0));
        if a > b {
            extra.insert((*k).to_string(), a - b);
        } else if b > a {
            missing.insert((*k).to_string(), b - a);
        }
    }
    for (k, a) in &p {
        matched += (*a).min(t.get(k).copied().unwrap_or(0));
    }
    Fidelity {
        planned: seq.len() as u64,
        traced: traced.len() as u64,
        matched,
        in_order: lcs(&seq, traced),
        missing,
        extra,
        novel,
    }
}

#[path = "fidelity_render.rs"]
pub mod render;

#[cfg(test)]
#[path = "fidelity_tests.rs"]
mod fidelity_tests;
