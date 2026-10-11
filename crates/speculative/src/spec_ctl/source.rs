// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: What the controller needs to know about a drafter, as data: how deep it can
//! propose, whether a shorter verify sees the same leading drafts, and how its propose cost
//! grows with depth. MTP, DFlash, n-gram and prompt lookup are all one [`DraftSource`]; the
//! controller has no drafter-specific branch.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

/// 2026-10-10: The drafter's propose cost as a function of depth `k`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DraftCost {
    /// 2026-10-10: Chained: one propose step per draft (MTP), batched across sequences.
    PerDraft { ms: f64, j: f64 },
    /// 2026-10-10: One propose per verify step whatever `k` (a block drafter: DFlash).
    PerBlock { ms: f64, j: f64 },
    /// 2026-10-10: Host-side lookups (n-gram, prompt lookup).
    Free,
}

impl DraftCost {
    /// 2026-10-10: `(ms, joules)` of proposing `k` drafts; nothing at `k = 0`.
    pub fn of(&self, k: usize) -> (f64, f64) {
        match (*self, k) {
            (_, 0) | (Self::Free, _) => (0.0, 0.0),
            (Self::PerDraft { ms, j }, k) => (ms * k as f64, j * k as f64),
            (Self::PerBlock { ms, j }, _) => (ms, j),
        }
    }
}

/// 2026-10-10: Which drafter family; reported, never branched on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftKind {
    Mtp,
    DFlash,
    NGram,
    PromptLookup,
}

/// 2026-10-10: A drafter as the controller sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftSource {
    pub kind: DraftKind,
    /// 2026-10-10: Deepest proposal (MTP `--num-drafts`, DFlash gamma - 1, lookup length).
    pub max_k: usize,
    /// 2026-10-10: Verifying fewer drafts sees the same leading drafts (true for chained and
    /// block drafters alike). The replay's counterfactual and the per-sequence cut need it.
    pub prefix_stable: bool,
    pub draft_cost: DraftCost,
}
