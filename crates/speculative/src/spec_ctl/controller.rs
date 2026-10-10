// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The speculation controller: per-stream acceptance x one cost model x one
//! objective -> the next step's draft count, 0 meaning plain decode. Drafter-agnostic: MTP,
//! DFlash, n-gram and prompt lookup differ only in their [`DraftSource`] and cost source.
//!
//! Owner: speculative.
//! Invariants:
//! - Pure: the host owns where the state lives, when steps are observed, and the clock and
//!   energy counter whose readings it feeds to [`SpecController::observe_cost`].
//! - A count is never above the caller's cap, and 0 only when the cost model prices the plain
//!   decode step and it wins by the suspend margin.
//! - With nothing measured for a stream (no acceptance anywhere), the caller's cap stands.
//!
//! Choice: [`super::decide::choose`] over `0..=min(cap, source max)`, each candidate's expected
//! tokens from the stream's rates blended with the serve-wide rates
//! ([`super::accept::AcceptCounts::rates`]), each cost from the calibrated cost model at the
//! live width. Then [`ReprobePolicy::explore`] may add one draft. With a measured-only cost
//! source ([`super::online`]) a depth whose cost is new or stale at this width runs first.

use super::accept::{AcceptCounts, AcceptParams, ColdPrior};
use super::chain::expected_tokens;
use super::cost::CostModel;
use super::decide::{Candidate, Margins, Objective, choose};
use super::reprobe::{ExploreState, ReprobePolicy};
use super::source::DraftSource;

/// 2026-10-10: Everything that shapes a controller's decisions; no field has a default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControllerConfig {
    pub source: DraftSource,
    pub accept: AcceptParams,
    pub objective: Objective,
    pub margins: Margins,
    pub reprobe: ReprobePolicy,
}

/// 2026-10-10: One stream's state: its acceptance counts, decisions taken, the depth it runs
/// (the hysteresis incumbent) and its suspension.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeqState {
    pub rates: AcceptCounts,
    pub steps: u32,
    incumbent: Option<usize>,
    /// 2026-10-10: Plain tokens since the stream was suspended; `None` while speculating.
    suspended: Option<u32>,
    /// 2026-10-10: Speculative steps left in the re-probe after a resume.
    probe_left: u32,
    explore: ExploreState,
}

impl SeqState {
    /// 2026-10-10: Whether the stream is suspended to plain decode.
    pub fn suspended(&self) -> bool {
        self.suspended.is_some()
    }
}

/// 2026-10-10: The serve's controller: config, cost model, serve-wide acceptance prior, and
/// the cold-start prior (`cfg.accept.cold_weight` says what it weighs).
#[derive(Clone, Debug, PartialEq)]
pub struct SpecController {
    pub cfg: ControllerConfig,
    pub cost: CostModel,
    pub global: AcceptCounts,
    pub cold: ColdPrior,
}

impl SpecController {
    pub fn new(cfg: ControllerConfig, cost: CostModel, cold: ColdPrior) -> Self {
        Self {
            cfg,
            cost,
            global: AcceptCounts::default(),
            cold,
        }
    }

    /// 2026-10-10: Record a verify of `drafts` that accepted `accepted` for `seq` and for the
    /// serve-wide prior.
    pub fn observe_step(&mut self, seq: &mut SeqState, drafts: usize, accepted: usize) {
        seq.rates.observe_step(&self.cfg.accept, drafts, accepted);
        self.global.observe_step(&self.cfg.accept, drafts, accepted);
        seq.steps = seq.steps.wrapping_add(1);
        seq.probe_left = seq.probe_left.saturating_sub(1);
    }

    /// 2026-10-10: Fold a measured step of `k` drafts over `n` streams (`ms`, and `j` when the
    /// energy counter was read) into the cost calibration.
    pub fn observe_cost(&mut self, n: usize, k: usize, ms: f64, j: Option<f64>) {
        self.cost.observe(n, k, ms, j);
    }

    fn hysteresis(&self) -> bool {
        let m = &self.cfg.margins;
        m.deeper != 0.0 || m.shallower != 0.0
    }

    /// 2026-10-10: The candidate depths: `allowed` (ascending) up to the drafter's and the cost
    /// source's deepest, without plain decode while any stream is in its re-probe window
    /// (unless nothing else is allowed).
    fn pool(&self, streams: &[&SeqState], allowed: &[usize]) -> Vec<usize> {
        let src = self.cfg.source.max_k;
        let top = self.cost.source.max_k().map_or(src, |m| m.min(src));
        let in_range: Vec<usize> = allowed.iter().copied().filter(|&k| k <= top).collect();
        let no_plain = streams.iter().any(|s| s.probe_left > 0);
        let pool: Vec<usize> = in_range
            .iter()
            .copied()
            .filter(|&k| !(no_plain && k == 0))
            .collect();
        match (pool.is_empty(), in_range.first()) {
            (true, Some(&k)) => vec![k],
            (true, None) => vec![0],
            (false, _) => pool,
        }
    }

    /// 2026-10-10: The shallowest pool depth the cost model has not measured (an online
    /// source's new or stale cell), which runs before anything is planned from it.
    fn probe(&self, n: usize, pool: &[usize]) -> Option<usize> {
        pool.iter().copied().find(|&k| self.cost.needs_probe(n, k))
    }

    /// 2026-10-10: The candidates for `streams` at width `n` over `pool`: every depth the cost
    /// model prices. `None` when a stream has no rates.
    fn candidates(
        &self,
        streams: &[&SeqState],
        n: usize,
        pool: &[usize],
    ) -> Option<Vec<Candidate>> {
        let rates: Vec<_> = streams
            .iter()
            .map(|s| {
                s.rates
                    .rates(&self.cfg.accept, Some(&self.global), &self.cold)
            })
            .collect::<Option<_>>()?;
        let cands = pool
            .iter()
            .filter_map(|&k| {
                let cost = self.cost.cost(n, k)?;
                let es = rates.iter().map(|c| expected_tokens(c, k));
                let (tokens, slowest) =
                    es.fold((0.0, f64::INFINITY), |(t, s), e| (t + e, s.min(e)));
                Some(Candidate {
                    k,
                    tokens,
                    slowest,
                    cost,
                })
            })
            .collect();
        Some(cands)
    }

    /// 2026-10-10: The decision over `pool` for `streams` at width `n`: a probe when a cell is
    /// unmeasured, else the choice, then exploration one deeper when it is due and allowed.
    /// With no rates the caller's `no_data` depth runs.
    #[allow(clippy::too_many_arguments)]
    fn decide(
        &self,
        streams: &[&SeqState],
        n: usize,
        pool: &[usize],
        incumbent: Option<usize>,
        explore: &mut ExploreState,
        steps: u32,
        no_data: usize,
    ) -> usize {
        explore.probing = false;
        if let Some(k) = self.probe(n, pool) {
            explore.probing = true;
            return k;
        }
        let deepest = pool.last().copied().unwrap_or(0);
        let Some(cands) = self.candidates(streams, n, pool) else {
            return no_data;
        };
        let k = choose(&self.cfg.objective, &self.cfg.margins, incumbent, &cands).unwrap_or(0);
        let e = self.cfg.reprobe.explore(explore, k, steps, deepest);
        if pool.contains(&e) { e } else { k }
    }

    /// 2026-10-10: The next step's draft count for one stream at width `n` under `cap` (0:
    /// plain decode; `cap` while nothing is measured). A suspended stream gets 0 until [`Self::note_plain_token`] resumes it.
    /// Advances the stream's exploration schedule when it explores.
    pub fn drafts(&self, seq: &mut SeqState, n: usize, cap: usize) -> usize {
        if seq.suspended.is_some() {
            return 0;
        }
        let allowed: Vec<usize> = (0..=cap).collect();
        let pool = self.pool(&[seq], &allowed);
        let inc = if self.hysteresis() {
            seq.incumbent
        } else {
            None
        };
        let mut explore = std::mem::take(&mut seq.explore);
        let k = self.decide(&[seq], n, &pool, inc, &mut explore, seq.steps, cap);
        seq.explore = explore;
        k
    }

    /// 2026-10-10: One draft count for a whole batch of `streams` (a uniform-depth verify) over
    /// `0..=cap`: the objective summed (or, for Latency, minimised) over the streams. See
    /// [`Self::batch_drafts_in`].
    pub fn batch_drafts(
        &self,
        streams: &[&SeqState],
        explore: &mut ExploreState,
        cap: usize,
    ) -> usize {
        let allowed: Vec<usize> = (0..=cap).collect();
        self.batch_drafts_in(streams, explore, &allowed)
    }

    /// 2026-10-10: One draft count for a batch, among the depths in `allowed` (ascending; the
    /// host's dispatchable depths, e.g. plain decode and the ladder's depth). While every
    /// stream is suspended the batch plain-decodes. `explore` is the batch's exploration
    /// schedule, counted in the first stream's decisions.
    pub fn batch_drafts_in(
        &self,
        streams: &[&SeqState],
        explore: &mut ExploreState,
        allowed: &[usize],
    ) -> usize {
        if !streams.is_empty() && streams.iter().all(|s| s.suspended()) && allowed.contains(&0) {
            return 0;
        }
        let pool = self.pool(streams, allowed);
        let steps = streams.first().map_or(0, |s| s.steps);
        let no_data = allowed.last().copied().unwrap_or(0);
        self.decide(streams, streams.len(), &pool, None, explore, steps, no_data)
    }

    /// 2026-10-10: The host ran `k` drafts for `seq` as [`Self::drafts`] chose: it becomes the
    /// hysteresis incumbent, and a chosen 0 (not a cost probe) suspends the stream (re-probed
    /// after `resume_after_tokens` plain tokens).
    pub fn commit(&self, seq: &mut SeqState, k: usize) {
        let probe = std::mem::take(&mut seq.explore.probing);
        self.settle(seq, k, probe);
    }

    /// 2026-10-10: [`Self::commit`] for a batch that ran `k` as [`Self::batch_drafts`] chose
    /// with `explore`.
    pub fn commit_batch<'a>(
        &self,
        streams: impl IntoIterator<Item = &'a mut SeqState>,
        explore: &mut ExploreState,
        k: usize,
    ) {
        let probe = std::mem::take(&mut explore.probing);
        for s in streams {
            self.settle(s, k, probe);
        }
    }

    fn settle(&self, seq: &mut SeqState, k: usize, probe: bool) {
        if probe {
            return;
        }
        seq.incumbent = Some(k);
        if k == 0 && self.cfg.reprobe.resume_after_tokens.is_some() && seq.suspended.is_none() {
            seq.suspended = Some(0);
        }
    }

    /// 2026-10-10: Count one plain-decoded token of a suspended stream; at
    /// `resume_after_tokens` the stream resumes with its counts softened. Returns whether it
    /// resumed on this token.
    pub fn note_plain_token(&self, seq: &mut SeqState) -> bool {
        let (Some(t), Some(limit)) = (seq.suspended, self.cfg.reprobe.resume_after_tokens) else {
            return false;
        };
        if t + 1 < limit {
            seq.suspended = Some(t + 1);
            return false;
        }
        seq.suspended = None;
        seq.incumbent = None;
        seq.probe_left = self.cfg.reprobe.probe_steps;
        seq.explore = ExploreState::default();
        seq.rates.soften(self.cfg.reprobe.soften);
        true
    }
}

#[cfg(test)]
#[path = "controller_tests.rs"]
mod tests;
