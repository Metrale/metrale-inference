// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: What a declared state is ([`StateKind`]) and how long a unit of it lives
//! ([`Lifetime`]), split from `state.rs`.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Every kind has one spelling ([`StateKind::parse`] of [`StateKind::name`] is the kind) and
//!   one lifetime ([`Lifetime::of_kind`]); a spelling outside the list is refused by the loader.

#[cfg(doc)]
use super::{Holding, StatePlan};

/// 2026-09-30: What a state is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateKind {
    /// 2026-09-30: A recurrent state, one unit per sequence slot (GatedDeltaNet and Mamba2
    /// `h`, their conv windows).
    Recurrent,
    /// 2026-09-30: One side of a paged KV cache, one unit per token.
    PagedKv,
    /// 2026-10-02: A prefix-cache (Marconi) snapshot of a recurrent state (`of`), or of the
    /// last-token hidden row, one unit per snapshot slot.
    PrefixSnapshot,
    /// 2026-10-02: The decode-rollback ring's copy of a recurrent state (`of`), one unit per ring
    /// slot per sequence slot.
    RingSnapshot,
    /// 2026-10-02: The carried-state verify's per-slot stash of a recurrent layer, one unit per
    /// verify slot (the dummy included).
    CarryStash,
    /// 2026-10-02: The carried-state verify's per-layer pointer and flag tables, one unit per
    /// carry-table row.
    CarryTable,
    /// 2026-10-02: A verify's per-layer WY pointer tables, one unit per verify-table row.
    VerifyTable,
    /// 2026-10-02: The hidden rows of accepted drafts a verify stashes for the next propose, one
    /// unit per verify-table row.
    AcceptStash,
    /// 2026-10-02: Hidden rows captured for the draft head (the prompt's, for the drafter
    /// prefill; DFlash's per-step capture), one unit per captured row.
    HiddenCapture,
    /// 2026-10-02: Host: a sequence's prompt-lookup n-gram index, one entry per history token
    /// (sized by the hash table law in `memory::caches`).
    PromptLookupIndex,
    /// 2026-10-02: A token tree's attention mask, one unit per (tree node, tree node) pair of a
    /// verify slot.
    TokenTreeMask,
    /// 2026-10-02: A speculative step's drafted tokens and their scores, one unit per draft
    /// position of a verify slot (tree nodes under a token tree).
    DraftTokens,
}

/// 2026-10-02: Every kind and its spelling in the circuit TOML.
const KINDS: [(StateKind, &str); 12] = [
    (StateKind::Recurrent, "recurrent"),
    (StateKind::PagedKv, "paged_kv"),
    (StateKind::PrefixSnapshot, "prefix_snapshot"),
    (StateKind::RingSnapshot, "ring_snapshot"),
    (StateKind::CarryStash, "carry_stash"),
    (StateKind::CarryTable, "carry_table"),
    (StateKind::VerifyTable, "verify_table"),
    (StateKind::AcceptStash, "accept_stash"),
    (StateKind::HiddenCapture, "hidden_capture"),
    (StateKind::PromptLookupIndex, "prompt_lookup_index"),
    (StateKind::TokenTreeMask, "token_tree_mask"),
    (StateKind::DraftTokens, "draft_tokens"),
];

impl StateKind {
    /// 2026-09-30: The spelling in the circuit TOML.
    pub fn parse(s: &str) -> Option<Self> {
        KINDS.iter().find(|(_, n)| *n == s).map(|(k, _)| *k)
    }

    /// 2026-10-02: The spelling.
    pub fn name(self) -> &'static str {
        KINDS
            .iter()
            .find(|(k, _)| *k == self)
            .map(|(_, n)| *n)
            .unwrap_or("?")
    }

    /// 2026-10-02: Every spelling, for refusals.
    pub fn spellings() -> Vec<&'static str> {
        KINDS.iter().map(|(_, n)| *n).collect()
    }

    /// 2026-10-02: A cache: a kind no layer node touches, sized by `memory::caches` from the
    /// cache inputs, never by [`StatePlan`]. It may copy another state of its block (`of`).
    pub fn is_cache(self) -> bool {
        !matches!(self, Self::Recurrent | Self::PagedKv)
    }

    /// 2026-10-02: The kinds whose unit copies a declared state (`of`).
    pub fn copies_a_state(self) -> bool {
        matches!(self, Self::PrefixSnapshot | Self::RingSnapshot)
    }

    /// 2026-10-02: Held in host memory rather than device memory (on a unified-memory device it
    /// still competes for the same pool, outside the util budget).
    pub fn is_host(self) -> bool {
        matches!(self, Self::PromptLookupIndex)
    }
}

/// 2026-09-30: How long a unit of state lives (LIFECYCLE-DESIGN.md section 3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lifetime {
    /// 2026-09-30: Allocated once with the model (a KV pool; its blocks are per sequence).
    Model,
    /// 2026-09-30: Claimed with a sequence's slot and released with it.
    Sequence,
    /// 2026-09-30: Valid from a verify's snapshot to its commit or rollback.
    Verify,
    /// 2026-10-02: Owned by the prefix cache or the scheduler's ring, in slots of their own.
    Snapshot,
}

impl Lifetime {
    /// 2026-09-30: The spelling in the circuit TOML.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "model" => Some(Self::Model),
            "sequence" => Some(Self::Sequence),
            "verify" => Some(Self::Verify),
            "snapshot" => Some(Self::Snapshot),
            _ => None,
        }
    }

    /// 2026-09-30: The spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Sequence => "sequence",
            Self::Verify => "verify",
            Self::Snapshot => "snapshot",
        }
    }

    /// 2026-09-30: The one lifetime a state of `kind` may declare: a recurrent state lives with
    /// its sequence, a KV side with the model's pool. For a recurrent state `Verify` belongs to
    /// the verify holdings ([`Holding::lifetime`]), never to its declaration. 2026-10-02: The
    /// caches: snapshots live in their owner's slots, the verify caches across one verify, the
    /// verify tables with the model, a prompt-lookup index and a hidden capture with their
    /// sequence.
    pub fn of_kind(kind: StateKind) -> Self {
        match kind {
            StateKind::Recurrent | StateKind::PromptLookupIndex | StateKind::HiddenCapture => {
                Self::Sequence
            }
            StateKind::PagedKv | StateKind::CarryTable | StateKind::VerifyTable => Self::Model,
            StateKind::PrefixSnapshot | StateKind::RingSnapshot => Self::Snapshot,
            StateKind::CarryStash
            | StateKind::AcceptStash
            | StateKind::TokenTreeMask
            | StateKind::DraftTokens => Self::Verify,
        }
    }
}
