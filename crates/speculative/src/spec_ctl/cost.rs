// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The one step-cost model: what a step of `k` drafts over `n` sequences costs in
//! wall time and (when known) energy. A cost has exactly one source, chosen by [`layer`] in a
//! fixed order, times one online calibration ([`super::calib`]). Pure: the host measures, this
//! module only does arithmetic.
//!
//! Owner: speculative.
//! Invariants:
//! - [`layer`] is the only place the source order is decided: a measured table for this box
//!   class and recipe, else the circuit x hardware envelope, else an explicit cold-start prior;
//!   every fallback is recorded in [`Provenance`].
//! - A cost is `None` for a depth its source does not price; the controller never invents one.
//! - Calibration multiplies a source cost; with the calibration off (alpha 0) the cost is the
//!   source's, bit for bit.

use super::calib::Calibration;
use super::online::OnlineTable;
use super::source::DraftCost;
use crate::spec_cost::CostTable;

/// 2026-10-10: One step's cost: wall ms, and GPU-rail joules when the source measured them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepCost {
    pub ms: f64,
    pub j: Option<f64>,
}

/// 2026-10-10: A `k:ms` table of whole-step wall times by draft count (entry 0, when present,
/// the plain decode step), measured at one stream. `--dflash-adaptive-k` spells it.
#[derive(Clone, Debug, PartialEq)]
pub struct StepTable {
    table: Vec<f64>,
    first: usize,
}

impl StepTable {
    /// 2026-10-10: Parse `drafts:ms[,drafts:ms...]`: consecutive draft counts from 0 or 1, at
    /// least one count >= 1, every cost finite and positive.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut table = Vec::new();
        let mut first = None;
        for pair in spec.split(',') {
            let Some((k, ms)) = pair.trim().split_once(':') else {
                return Err(format!("adaptive-K cost pair {pair:?} is not drafts:ms"));
            };
            let k: usize = k.trim().parse().map_err(|e| format!("{pair:?}: {e}"))?;
            let ms: f64 = ms.trim().parse().map_err(|e| format!("{pair:?}: {e}"))?;
            if !(ms.is_finite() && ms > 0.0) {
                return Err(format!(
                    "adaptive-K cost {k}:{ms}: the cost must be positive"
                ));
            }
            let f = *first.get_or_insert(k);
            if f > 1 {
                return Err(format!(
                    "adaptive-K cost table must start at 0 or 1 drafts: {spec:?}"
                ));
            }
            if k != f + table.len() {
                return Err(format!(
                    "adaptive-K cost table draft counts must be consecutive: {spec:?}"
                ));
            }
            table.push(ms);
        }
        let first = first.unwrap_or(0);
        if first + table.len() < 2 {
            return Err(format!(
                "adaptive-K cost table prices no draft count >= 1: {spec:?}"
            ));
        }
        Ok(Self { table, first })
    }

    /// 2026-10-10: The widest priced draft count.
    pub fn max_drafts(&self) -> usize {
        self.first + self.table.len() - 1
    }

    /// 2026-10-10: The table's ms at `k` drafts, `None` when not priced.
    pub fn ms(&self, k: usize) -> Option<f64> {
        k.checked_sub(self.first)
            .and_then(|i| self.table.get(i))
            .copied()
    }
}

/// 2026-10-10: Relative step costs by depth, unitless (`cost(k) / cost(reference)`). A legacy
/// threshold is a cost ratio in disguise: the MTP rung's "two drafts when E(2)/E(1) >= 1.32" is
/// `{1: 1.0, 2: 1.32}`.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthRatio(Vec<(usize, f64)>);

impl DepthRatio {
    /// 2026-10-10: `None` when empty, a depth repeats, or a ratio is not finite and positive.
    pub fn new(mut by_k: Vec<(usize, f64)>) -> Option<Self> {
        by_k.sort_by_key(|e| e.0);
        let ok = !by_k.is_empty()
            && by_k.windows(2).all(|w| w[0].0 < w[1].0)
            && by_k.iter().all(|e| e.1.is_finite() && e.1 > 0.0);
        ok.then_some(Self(by_k))
    }

    fn at(&self, k: usize) -> Option<f64> {
        self.0.iter().find(|e| e.0 == k).map(|e| e.1)
    }
}

/// 2026-10-10: The circuit x hardware-envelope prior: verify cost by total verify rows
/// (`n * (k + 1)`), linear between points, nearest point outside; plus the drafter's cost
/// shape. The host computes the points from the architecture circuit and the box's kernel
/// envelope; this crate only reads them.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvelopeCurve {
    /// 2026-10-10: `(rows, ms, joules)`, ascending in rows, at least one.
    pub points: Vec<(usize, f64, f64)>,
    pub draft: DraftCost,
}

impl EnvelopeCurve {
    fn verify(&self, rows: usize) -> (f64, f64) {
        let p = &self.points;
        let hi = p.iter().position(|q| q.0 >= rows).unwrap_or(p.len() - 1);
        if hi == 0 || p[hi].0 <= rows {
            return (p[hi].1, p[hi].2);
        }
        let (a, b) = (p[hi - 1], p[hi]);
        let t = (rows - a.0) as f64 / (b.0 - a.0) as f64;
        (a.1 + (b.1 - a.1) * t, a.2 + (b.2 - a.2) * t)
    }
}

/// 2026-10-10: Where a cost comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum CostSource {
    /// 2026-10-10: `met bench spec-cost`'s measured (n, k) table, key-checked at boot.
    Measured(CostTable),
    Envelope(EnvelopeCurve),
    StepTable(StepTable),
    DepthRatio(DepthRatio),
    /// 2026-10-10: Measured online only ([`super::online`]); no prior.
    Online(OnlineTable),
}

impl CostSource {
    /// 2026-10-10: The source's own cost of a step of `k` drafts over `n` sequences.
    pub fn cost(&self, n: usize, k: usize) -> Option<StepCost> {
        match self {
            Self::Measured(t) => table_cost(t, n, k),
            Self::Envelope(e) => {
                let (vms, vj) = e.verify(n.max(1) * (k + 1));
                let (dms, dj) = e.draft.of(k);
                Some(StepCost {
                    ms: vms + dms,
                    j: Some(vj + dj),
                })
            }
            Self::StepTable(t) => t.ms(k).map(|ms| StepCost { ms, j: None }),
            Self::DepthRatio(r) => r.at(k).map(|ms| StepCost { ms, j: None }),
            Self::Online(t) => t.cost(n, k),
        }
    }

    /// 2026-10-10: The deepest priced draft count; `None` when the source prices any depth.
    pub fn max_k(&self) -> Option<usize> {
        match self {
            Self::Measured(t) => Some(t.max_k()),
            Self::Envelope(_) => None,
            Self::StepTable(t) => Some(t.max_drafts()),
            Self::DepthRatio(r) => r.0.last().map(|e| e.0),
            Self::Online(_) => None,
        }
    }

    /// 2026-10-10: A batch of `n` sequences verifying `rows` draft rows in total, the drafts of
    /// depth `k_max` already proposed: verify cost interpolated in `rows / n` between the two
    /// neighbouring uniform depths, plus the whole draft cost of `k_max`. For a measured table
    /// this is the planner's batch cost; other sources have no verify/draft split and
    /// interpolate whole steps.
    pub fn batch(&self, n: usize, rows: usize, k_max: usize) -> Option<StepCost> {
        let depth = rows as f64 / n as f64;
        let (lo, hi) = (depth.floor() as usize, depth.ceil() as usize);
        let t = depth - lo as f64;
        if let Self::Measured(tab) = self {
            return table_batch(tab, n, rows, k_max);
        }
        let (a, b) = (self.cost(n, lo)?, self.cost(n, hi)?);
        Some(StepCost {
            ms: a.ms + (b.ms - a.ms) * t,
            j: a.j.zip(b.j).map(|(x, y)| x + (y - x) * t),
        })
    }
}

/// 2026-10-10: A measured table's cost of a whole step (verify + draft) at `(n, k)`.
pub fn table_cost(t: &CostTable, n: usize, k: usize) -> Option<StepCost> {
    t.cell(n, k).map(|c| StepCost {
        ms: c.step_ms(),
        j: Some(c.step_j()),
    })
}

/// 2026-10-10: A measured table's batch cost ([`CostSource::batch`]): verify interpolated in
/// `rows / n` between the neighbouring measured depths, plus the whole draft cost of `k_max`.
pub fn table_batch(t: &CostTable, n: usize, rows: usize, k_max: usize) -> Option<StepCost> {
    let depth = rows as f64 / n as f64;
    let (lo, hi) = (depth.floor() as usize, depth.ceil() as usize);
    let top = t.max_k();
    let (a, b) = (t.cell(n, lo.min(top))?, t.cell(n, hi.min(top))?);
    let s = depth - lo as f64;
    let draft = t.cell(n, k_max)?;
    Some(StepCost {
        ms: a.verify_ms + (b.verify_ms - a.verify_ms) * s + draft.draft_ms,
        j: Some(a.verify_j + (b.verify_j - a.verify_j) * s + draft.draft_j),
    })
}

/// 2026-10-10: Why the serve plans from the source it does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Provenance {
    Measured,
    Envelope,
    /// 2026-10-10: The explicit cold-start prior, and why each better source was not used.
    ColdStart {
        skipped: Vec<String>,
    },
}

/// 2026-10-10: The layering of cost sources, in its one fixed order: a measured table, else
/// the envelope, else the explicit cold-start prior. `envelope` carries the reason it is
/// missing (the circuits envelope plan is not wired yet). Errors when no layer has a source.
pub fn layer(
    measured: Option<CostTable>,
    envelope: Result<EnvelopeCurve, String>,
    cold_start: Option<CostSource>,
) -> Result<(CostSource, Provenance), String> {
    if let Some(t) = measured {
        return Ok((CostSource::Measured(t), Provenance::Measured));
    }
    let skipped = match envelope {
        Ok(e) => return Ok((CostSource::Envelope(e), Provenance::Envelope)),
        Err(why) => vec!["no measured spec-cost table".to_string(), why],
    };
    match cold_start {
        Some(c) => Ok((c, Provenance::ColdStart { skipped })),
        None => Err(format!("no cost source: {}", skipped.join("; "))),
    }
}

/// 2026-10-10: A source and its online calibration: the costs the controller plans with.
#[derive(Clone, Debug, PartialEq)]
pub struct CostModel {
    pub source: CostSource,
    pub calib: Calibration,
}

impl CostModel {
    /// 2026-10-10: The calibrated cost of `k` drafts over `n` sequences.
    pub fn cost(&self, n: usize, k: usize) -> Option<StepCost> {
        self.source.cost(n, k).map(|c| self.calib.apply(n, c))
    }

    /// 2026-10-10: Whether the controller must run this cell before planning from it: only an
    /// online source has cells it has not measured.
    pub fn needs_probe(&self, n: usize, k: usize) -> bool {
        matches!(&self.source, CostSource::Online(t) if t.needs_probe(n, k))
    }

    /// 2026-10-10: The calibrated batch cost ([`CostSource::batch`]).
    pub fn batch(&self, n: usize, rows: usize, k_max: usize) -> Option<StepCost> {
        self.source
            .batch(n, rows, k_max)
            .map(|c| self.calib.apply(n, c))
    }

    /// 2026-10-10: Fold a measured step (`ms`, and `j` when the host read the energy counter)
    /// of `k` drafts over `n` sequences: into the online table when that is the source, else
    /// into the calibration (a depth the source does not price is ignored).
    pub fn observe(&mut self, n: usize, k: usize, ms: f64, j: Option<f64>) {
        if let CostSource::Online(t) = &mut self.source {
            t.observe(n, k, ms, j);
            return;
        }
        if let Some(base) = self.source.cost(n, k) {
            self.calib.observe(n, base, ms, j);
        }
    }
}

#[cfg(test)]
#[path = "cost_tests.rs"]
mod tests;
