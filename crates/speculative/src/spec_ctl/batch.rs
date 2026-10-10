// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The scheduler's per-step decision for a whole batch: plain decode (0) or one of
//! the speculative depths the host can dispatch this step. It replaces the MTP runtime gate
//! (plain decode vs speculation by measured throughput) and the DFlash gamma resolver
//! (single-stream depth by first-draft hits): both are the controller choosing over a set of
//! allowed depths, with costs measured online (or the measured table) and per-stream
//! acceptance.
//!
//! Owner: speculative.
//! Invariants:
//! - Pure state: the host passes measured walls (and joules when read) and per-stream
//!   outcomes; no clock is read here.
//! - The decision is always one of the allowed depths.
//! - `entered_plain` is reported exactly once per transition from a speculative step to a
//!   chosen (not probing) plain-decode step, so the host drops pending drafts once.

use super::controller::{SeqState, SpecController};
use super::reprobe::ExploreState;

/// 2026-10-10: Weight of a step in the delivered-throughput estimate the snapshot reports.
const DELIVERED_ALPHA: f64 = 0.3;

/// 2026-10-10: What [`BatchSpec::decide`] chose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    pub k: usize,
    /// 2026-10-10: A cost probe (an unmeasured cell), not a choice.
    pub probe: bool,
    /// 2026-10-10: The previous step speculated and this one plain-decodes by choice.
    pub entered_plain: bool,
}

/// 2026-10-10: The batch decision state the scheduler owns.
pub struct BatchSpec {
    pub ctl: SpecController,
    explore: ExploreState,
    last_k: Option<usize>,
    probes: u64,
    last_probe: bool,
    delivered_tps: Option<f64>,
}

impl BatchSpec {
    pub fn new(ctl: SpecController) -> Self {
        Self {
            ctl,
            explore: ExploreState::default(),
            last_k: None,
            probes: 0,
            last_probe: false,
            delivered_tps: None,
        }
    }

    /// 2026-10-10: The depth for the next step of `streams` among `allowed` (ascending).
    pub fn decide(&mut self, streams: &[&SeqState], allowed: &[usize]) -> Decision {
        let k = self
            .ctl
            .batch_drafts_in(streams, &mut self.explore, allowed);
        let probe = self.explore.probing;
        self.last_probe = probe;
        if probe {
            self.probes += 1;
        }
        let entered_plain = k == 0 && !probe && self.last_k.is_some_and(|l| l > 0);
        Decision {
            k,
            probe,
            entered_plain,
        }
    }

    /// 2026-10-10: The host ran `k` for `streams` (the decision, or a forced depth for a pin):
    /// settles the incumbent and suspension of every stream.
    pub fn settle<'a>(&mut self, streams: impl IntoIterator<Item = &'a mut SeqState>, k: usize) {
        self.ctl.commit_batch(streams, &mut self.explore, k);
        self.last_k = Some(k);
    }

    /// 2026-10-10: One stream's verify outcome: `drafts` verified, `accepted` of them.
    pub fn observe_stream(&mut self, seq: &mut SeqState, drafts: usize, accepted: usize) {
        if drafts > 0 {
            self.ctl.observe_step(seq, drafts, accepted.min(drafts));
        }
    }

    /// 2026-10-10: One plain-decoded token of `seq` (counts towards its re-probe).
    pub fn observe_plain(&self, seq: &mut SeqState) {
        self.ctl.note_plain_token(seq);
    }

    /// 2026-10-10: The step's measured wall `ms` (and `j`) at depth `k` over `n` streams, and
    /// the tokens they emitted.
    pub fn observe_step(&mut self, n: usize, k: usize, ms: f64, j: Option<f64>, emitted: usize) {
        self.ctl.observe_cost(n, k, ms, j);
        if ms > 0.0 && ms.is_finite() {
            let tps = emitted as f64 * 1000.0 / ms;
            self.delivered_tps = Some(match self.delivered_tps {
                None => tps,
                Some(p) => p + DELIVERED_ALPHA * (tps - p),
            });
        }
    }

    /// 2026-10-10: The last depth run (0 plain decode), `None` before any step.
    pub fn last_k(&self) -> Option<usize> {
        self.last_k
    }

    /// 2026-10-10: Cost probes so far.
    pub fn probes(&self) -> u64 {
        self.probes
    }

    /// 2026-10-10: Whether the last decision was a cost probe.
    pub fn probing(&self) -> bool {
        self.last_probe
    }

    /// 2026-10-10: Delivered tokens per second over recent steps (EWMA), 0 before any.
    pub fn delivered_tps(&self) -> f64 {
        self.delivered_tps.unwrap_or(0.0)
    }
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
