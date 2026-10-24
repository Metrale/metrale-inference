// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Adaptive DFlash draft count for a single stream: each step's drafts are the
//! count that maximises expected emitted tokens per millisecond of step, from the sequence's
//! measured per-position acceptance and a step-cost table, or none (plain decode) when no
//! count beats the plain decode step.
//!
//! Owner: speculative.
//! Invariants:
//! - The policy is pure: [`AcceptRates`] and [`CostModel`] hold numbers, [`choose`] reads
//!   them; the scheduler owns when they are updated.
//! - A count is never above the caller's cap, and 0 only when the table prices the plain
//!   decode step (`0:<ms>`).
//!
//! Model. A verify of `k` drafts emits `E(k) = 1 + sum_{j=1..k} prod_{i<j} c_i` tokens in
//! expectation, where `c_i` is the acceptance rate of draft `i` given that drafts `0..i` were
//! accepted (conditional, so a step that verified `d` drafts and accepted `a < d` observes
//! accepts at positions `0..a` and a reject at `a`; a full accept says nothing past `d`). A
//! step with `k` drafts costs `cost(k)` ms: the verify of `k + 1` rows and the propose that
//! follows it, from the table `METRALE_DFLASH_ADAPTIVE_K_COST_MS`, times one scale fitted
//! online to the measured step walls (context growth moves every row of the table alike). The
//! chosen `k` maximises `E(k) / cost(k)` over `1..=cap`; with a plain-decode entry, 0 wins
//! when `1 / cost(0)` beats that by more than [`SUSPEND_MARGIN`].
//!
//! Rates: decayed counts per position ([`DECAY`] per observation, a memory of ~20 steps),
//! blended with the serve-wide counts as a prior worth [`PRIOR_WEIGHT`] observations. A
//! position with no data takes the rate of the deepest one that has some (the reference
//! engine's rule for unverified positions); with no data at all the caller's cap stands.
//! Every [`EXPLORE_EVERY`]-th step verifies one draft more than the choice (within the cap),
//! so a position the choice stopped verifying is measured again.

use anyhow::{Result, bail, ensure};

/// 2026-10-10: Positions tracked; drafts past it share the last position's rate.
pub const MAX_POSITIONS: usize = 16;
/// 2026-10-10: Per-observation decay of a position's counts (memory ~1 / (1 - 0.95) = 20).
pub const DECAY: f64 = 0.95;
/// 2026-10-10: Observations the serve-wide prior is worth in a sequence's rate.
pub const PRIOR_WEIGHT: f64 = 4.0;
/// 2026-10-10: Plain decode must beat the best draft count by this fraction to suspend
/// speculation, so a near-tie keeps speculating (a suspension lasts until the re-probe).
pub const SUSPEND_MARGIN: f64 = 0.03;
/// 2026-10-10: Every this many steps a sequence verifies one draft more than its choice.
pub const EXPLORE_EVERY: u32 = 16;
/// 2026-10-10: Weight of one measured step in the online cost scale.
pub const SCALE_ALPHA: f64 = 0.1;

/// 2026-10-10: Decayed accept and observation counts per draft position (conditional).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AcceptRates {
    acc: [f64; MAX_POSITIONS],
    obs: [f64; MAX_POSITIONS],
}

impl AcceptRates {
    /// 2026-10-10: Fold one verify of `drafts` drafts that accepted `accepted`: accepts at
    /// positions `0..accepted`, a reject at `accepted` when it is below `drafts`. Only the
    /// observed positions decay.
    pub fn observe(&mut self, drafts: usize, accepted: usize) {
        let seen = drafts.min(accepted + 1).min(MAX_POSITIONS);
        for i in 0..seen {
            self.acc[i] = self.acc[i] * DECAY + if i < accepted { 1.0 } else { 0.0 };
            self.obs[i] = self.obs[i] * DECAY + 1.0;
        }
    }

    /// 2026-10-10: Scale every count by `f` (0..=1): older evidence weighs less against the
    /// next observations, as after a re-probe.
    pub fn soften(&mut self, f: f64) {
        for i in 0..MAX_POSITIONS {
            self.acc[i] *= f;
            self.obs[i] *= f;
        }
    }

    /// 2026-10-10: Per-position rates of these counts blended with `prior`'s rates (worth
    /// [`PRIOR_WEIGHT`] observations where the prior has data); a position with neither takes
    /// the deepest informed position's rate. `None` when no position has data.
    pub fn rates(&self, prior: Option<&AcceptRates>) -> Option<[f64; MAX_POSITIONS]> {
        let mut out = [0.0; MAX_POSITIONS];
        let mut last: Option<f64> = None;
        for (i, o) in out.iter_mut().enumerate() {
            let (pa, po) = match prior {
                Some(p) if p.obs[i] > 0.0 => (PRIOR_WEIGHT * p.acc[i] / p.obs[i], PRIOR_WEIGHT),
                _ => (0.0, 0.0),
            };
            let den = self.obs[i] + po;
            let r = if den > 0.0 {
                Some((self.acc[i] + pa) / den)
            } else {
                last
            };
            *o = r?;
            last = r;
        }
        Some(out)
    }
}

/// 2026-10-10: Expected emitted tokens of a verify of `k` drafts at conditional rates `c`.
pub fn expected_tokens(c: &[f64; MAX_POSITIONS], k: usize) -> f64 {
    let mut e = 1.0;
    let mut survive = 1.0;
    for j in 0..k {
        survive *= c[j.min(MAX_POSITIONS - 1)];
        e += survive;
    }
    e
}

/// 2026-10-10: The step-cost table (ms by draft count, entry 0 the plain decode step when
/// present) and the online scale applied to it.
#[derive(Clone, Debug, PartialEq)]
pub struct CostModel {
    /// 2026-10-10: `table[k - first]` is the step cost with `k` drafts.
    table: Vec<f64>,
    /// 2026-10-10: 0 when the table prices the plain decode step, else 1.
    first: usize,
    scale: f64,
}

impl CostModel {
    /// 2026-10-10: Parse `drafts:ms[,drafts:ms...]`: consecutive draft counts from 0 or 1,
    /// at least one count >= 1, every cost finite and positive. Errors otherwise.
    pub fn parse(spec: &str) -> Result<Self> {
        let mut table = Vec::new();
        let mut first = None;
        for pair in spec.split(',') {
            let Some((k, ms)) = pair.trim().split_once(':') else {
                bail!("adaptive-K cost pair {pair:?} is not drafts:ms");
            };
            let (k, ms): (usize, f64) = (k.trim().parse()?, ms.trim().parse()?);
            ensure!(
                ms.is_finite() && ms > 0.0,
                "adaptive-K cost {k}:{ms}: the cost must be positive"
            );
            let f = *first.get_or_insert(k);
            ensure!(
                f <= 1,
                "adaptive-K cost table must start at 0 or 1 drafts: {spec:?}"
            );
            ensure!(
                k == f + table.len(),
                "adaptive-K cost table draft counts must be consecutive: {spec:?}"
            );
            table.push(ms);
        }
        let first = first.unwrap_or(0);
        ensure!(
            first + table.len() >= 2,
            "adaptive-K cost table prices no draft count >= 1: {spec:?}"
        );
        Ok(Self {
            table,
            first,
            scale: 1.0,
        })
    }

    /// 2026-10-10: The widest priced draft count.
    pub fn max_drafts(&self) -> usize {
        self.first + self.table.len() - 1
    }

    /// 2026-10-10: The scaled cost of a step with `k` drafts, `None` when not priced.
    pub fn cost(&self, k: usize) -> Option<f64> {
        k.checked_sub(self.first)
            .and_then(|i| self.table.get(i))
            .map(|ms| ms * self.scale)
    }

    /// 2026-10-10: Fold a measured step of `k` drafts that took `ms`: the scale moves
    /// [`SCALE_ALPHA`] of the way to `ms / table(k)`. A count the table does not price, or a
    /// non-positive wall, is ignored.
    pub fn observe_step(&mut self, k: usize, ms: f64) {
        let Some(base) = k.checked_sub(self.first).and_then(|i| self.table.get(i)) else {
            return;
        };
        if ms.is_finite() && ms > 0.0 {
            self.scale += SCALE_ALPHA * (ms / base - self.scale);
        }
    }

    /// 2026-10-10: The online scale (1 until a step is measured).
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

/// 2026-10-10: The draft count for the next step: the `k` in `1..=min(cap, max)` with the
/// best `E(k) / cost(k)`, or 0 when the plain decode step is priced and beats it by more than
/// [`SUSPEND_MARGIN`]. `cap` 0 returns 0.
pub fn choose(costs: &CostModel, c: &[f64; MAX_POSITIONS], cap: usize) -> usize {
    let top = cap.min(costs.max_drafts());
    let mut best = (0usize, 0.0f64);
    for k in 1..=top {
        let Some(ms) = costs.cost(k) else { continue };
        let tput = expected_tokens(c, k) / ms;
        if tput > best.1 {
            best = (k, tput);
        }
    }
    match costs.cost(0) {
        Some(ms0) if best.0 == 0 || 1.0 / ms0 > best.1 * (1.0 + SUSPEND_MARGIN) => 0,
        _ => best.0,
    }
}

/// 2026-10-10: A sequence's adaptive-K state: its rates and its step counter (exploration).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeqAdaptiveK {
    pub rates: AcceptRates,
    pub steps: u32,
}

/// 2026-10-10: The serve's adaptive-K controller: the cost model and the serve-wide rates.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveK {
    pub costs: CostModel,
    pub global: AcceptRates,
}

impl AdaptiveK {
    pub fn new(costs: CostModel) -> Self {
        Self {
            costs,
            global: AcceptRates::default(),
        }
    }

    /// 2026-10-10: Record a verify of `drafts` that accepted `accepted` for `seq` and the
    /// serve-wide rates.
    pub fn observe(&mut self, seq: &mut SeqAdaptiveK, drafts: usize, accepted: usize) {
        seq.rates.observe(drafts, accepted);
        self.global.observe(drafts, accepted);
        seq.steps = seq.steps.wrapping_add(1);
    }

    /// 2026-10-10: The next step's draft count for `seq` under `cap` (see [`choose`]); `cap`
    /// when nothing is measured yet. Every [`EXPLORE_EVERY`]-th step adds one draft (within
    /// the cap) to a non-zero choice.
    pub fn drafts(&self, seq: &SeqAdaptiveK, cap: usize) -> usize {
        let Some(c) = seq.rates.rates(Some(&self.global)) else {
            return cap;
        };
        let k = choose(&self.costs, &c, cap);
        if k > 0 && seq.steps % EXPLORE_EVERY == EXPLORE_EVERY - 1 {
            (k + 1).min(cap)
        } else {
            k
        }
    }
}

#[cfg(test)]
#[path = "dflash_adaptive_k_tests.rs"]
mod tests;
