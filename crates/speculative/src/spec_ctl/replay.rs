// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Trace replay: runs a draft-count policy over a recorded acceptance trace and a
//! step-cost function and reports what it would have delivered (tokens per ms, per joule).
//! Pure; used by the controller's replay tests and by any tool that compares policies offline.
//!
//! Counterfactual rule. A trace logged at `k_logged` drafts fixes each step's accepted run `a`.
//! Drafts are prefix-stable (`DraftSource::prefix_stable`), so a step verifying `k <= k_logged`
//! drafts emits `1 + min(a, k)` tokens; a plain step emits 1 and consumes no trace entry.
//! Acceptance is assumed independent of the depth chosen (a step's run is a draw from the
//! stream's process), which is exact for the first `k` drafts and approximate for the
//! alignment of later steps. Depths above `k_logged` are clamped to it.
//!
//! Owner: speculative.
//! Invariants: the replay stops at the first step whose tokens reach the budget, so every
//! policy is charged for the same work.

use super::accept::AcceptCounts;
use super::controller::{SeqState, SpecController};
use super::cost::StepCost;
use super::reprobe::ExploreState;

/// 2026-10-10: A policy under replay: the batch's next depth, then what it observed.
pub trait Policy {
    fn decide(&mut self, n: usize) -> usize;
    /// 2026-10-10: The step verified `k` drafts per stream (0: plain), the streams accepted
    /// `accepted`, and it cost `cost`.
    fn observe(&mut self, k: usize, accepted: &[usize], cost: StepCost);
}

/// 2026-10-10: What a policy delivered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outcome {
    pub tokens: f64,
    pub ms: f64,
    /// 2026-10-10: `None` when any step's cost had no joules.
    pub j: Option<f64>,
    pub steps: u64,
    pub plain_steps: u64,
    /// 2026-10-10: Steps run at each depth (index = depth).
    pub by_depth: [u64; 17],
}

impl Outcome {
    pub fn tok_per_ms(&self) -> f64 {
        self.tokens / self.ms
    }
    pub fn tok_per_j(&self) -> Option<f64> {
        self.j.map(|j| self.tokens / j)
    }
}

/// 2026-10-10: Replay `policy` at width `n` over `trace` (accepted runs at `k_logged`,
/// cycled) until `budget` tokens; `cost(k)` is the step cost at that width. `None` when the
/// trace is empty or the policy picks a depth `cost` does not price.
pub fn replay(
    trace: &[u8],
    k_logged: usize,
    n: usize,
    budget: f64,
    cost: impl Fn(usize) -> Option<StepCost>,
    policy: &mut dyn Policy,
) -> Option<Outcome> {
    if trace.is_empty() || n == 0 {
        return None;
    }
    let mut o = Outcome {
        tokens: 0.0,
        ms: 0.0,
        j: Some(0.0),
        steps: 0,
        plain_steps: 0,
        by_depth: [0; 17],
    };
    let mut cursor = 0usize;
    let mut accepted = vec![0usize; n];
    while o.tokens < budget {
        let k = policy.decide(n).min(k_logged);
        let c = cost(k)?;
        if k == 0 {
            accepted.iter_mut().for_each(|a| *a = 0);
            o.tokens += n as f64;
            o.plain_steps += 1;
        } else {
            for (i, a) in accepted.iter_mut().enumerate() {
                *a = (trace[(cursor + i) % trace.len()] as usize).min(k);
                o.tokens += 1.0 + *a as f64;
            }
            cursor = (cursor + n) % trace.len();
        }
        o.ms += c.ms;
        o.j = o.j.zip(c.j).map(|(x, y)| x + y);
        o.steps += 1;
        o.by_depth[k.min(16)] += 1;
        policy.observe(k, &accepted, c);
    }
    Some(o)
}

/// 2026-10-10: A fixed depth (a static ladder rung, a pinned gamma).
pub struct Fixed(pub usize);

impl Policy for Fixed {
    fn decide(&mut self, _: usize) -> usize {
        self.0
    }
    fn observe(&mut self, _: usize, _: &[usize], _: StepCost) {}
}

/// 2026-10-10: The controller under replay: one [`SeqState`] per stream, the per-stream choice
/// at one stream and the batch choice above, measured step costs fed back.
pub struct Controlled {
    pub ctl: SpecController,
    pub streams: Vec<SeqState>,
    pub cap: usize,
    explore: ExploreState,
}

impl Controlled {
    pub fn new(ctl: SpecController, n: usize, cap: usize) -> Self {
        Self {
            ctl,
            streams: vec![SeqState::default(); n],
            cap,
            explore: ExploreState::default(),
        }
    }

    /// 2026-10-10: The serve-wide acceptance counts the controller built.
    pub fn global(&self) -> &AcceptCounts {
        &self.ctl.global
    }
}

impl Policy for Controlled {
    fn decide(&mut self, n: usize) -> usize {
        if n == 1 {
            return self.ctl.drafts(&mut self.streams[0], 1, self.cap);
        }
        let refs: Vec<&SeqState> = self.streams.iter().collect();
        self.ctl.batch_drafts(&refs, &mut self.explore, self.cap)
    }

    fn observe(&mut self, k: usize, accepted: &[usize], cost: StepCost) {
        self.ctl.observe_cost(accepted.len(), k, cost.ms, cost.j);
        if self.streams.len() == 1 {
            self.ctl.commit(&mut self.streams[0], k);
        } else {
            self.ctl
                .commit_batch(self.streams.iter_mut(), &mut self.explore, k);
        }
        for (s, &a) in self.streams.iter_mut().zip(accepted) {
            if k == 0 {
                self.ctl.note_plain_token(s);
            } else {
                self.ctl.observe_step(s, k, a);
            }
        }
    }
}
