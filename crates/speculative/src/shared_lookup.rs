// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Cross-request token cache for prompt lookup. A bounded index
//! over recent requests' prompt and output tokens, split into scopes keyed by
//! (model, tokenizer, LoRA adapter, tenant). A sequence whose own history has
//! no match may copy the continuation of the same n-gram from what earlier
//! requests of its scope held. Host memory only. The scheduler thread owns the
//! cache, so there is no lock.
//!
//! Each scope is a token ring plus a slot table, both sized once when the scope
//! is created. A document (one request's tokens) is appended to the ring and
//! each of its n-grams is written to the slot its hash selects, holding the
//! ring position of the token that followed it. A later write to the same slot
//! replaces the entry, and the ring overwrites its oldest tokens: both are the
//! eviction. A stale or colliding entry is harmless, because every hit is
//! re-checked token by token against the ring before it is used.
//!
//! Runs a document repeats are not stored again: while an n-gram's existing
//! entry already predicts the document's next token, nothing is written. A
//! re-sent conversation prefix therefore costs no ring space.
//!
//! Owner: speculative.
//! Invariants:
//! - A proposal under scope `S` is a slice of tokens inserted under `S`: the
//!   `key_len` ring tokens before it equal the history's last `key_len`
//!   tokens, it is non-empty, and it stops at its document's end.
//! - No call under one scope reads, writes or evicts another scope's tokens,
//!   except that creating a scope when `max_scopes` exist drops the least
//!   recently used scope whole.
//! - Memory is fixed per scope at creation: at most `max_scopes` scopes of
//!   [`SharedLookupConfig::scope_bytes`] each, within the configured budget.
//! - Deterministic: proposals depend only on the order of `insert` and
//!   `propose` calls and their arguments.

use std::collections::HashMap;

use crate::prompt_lookup::ngram_key;

/// 2026-10-04: Ring marker after each stored run; never a token id, so a
/// history never matches it and a copy stops at it.
const SEP: u32 = u32::MAX;

/// 2026-10-04: Duplicate n-grams in a row, while storing, after which the
/// rest of a repeated run is skipped. Shorter repeats are stored, so a copy is
/// not cut into many short pieces by brief coincidences.
pub const DEDUP_RUN: usize = 32;

/// 2026-10-04: Fewest ring tokens a scope may have.
pub const MIN_SCOPE_TOKENS: usize = 1024;

/// 2026-10-04: What partitions the cache. Two sequences share drafts only when
/// every field is equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeKey {
    pub model: u64,
    pub tokenizer: u64,
    pub adapter: u64,
    pub tenant: u64,
}

/// 2026-10-04: Cache settings (`--prompt-lookup-shared-cache-*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedLookupConfig {
    /// 2026-10-04: Host bytes for all scopes together.
    pub budget_bytes: usize,
    /// 2026-10-04: Most scopes held at once; each gets `budget_bytes / max_scopes`.
    pub max_scopes: usize,
    /// 2026-10-04: Tokens a match must span (the prompt-lookup n-gram length
    /// or minimum match, whichever is longer).
    pub key_len: usize,
}

impl SharedLookupConfig {
    /// 2026-10-04: Slots and ring tokens of one scope: the largest power-of-two
    /// slot table within half the scope's bytes, and the ring in the rest.
    /// `Err` when the ring would hold fewer than [`MIN_SCOPE_TOKENS`].
    pub fn scope_geometry(&self) -> Result<(usize, usize), String> {
        if self.max_scopes == 0 || self.key_len == 0 {
            return Err("shared prompt-lookup cache needs max_scopes >= 1 and key_len >= 1".into());
        }
        let per_scope = self.budget_bytes / self.max_scopes;
        let slot_bytes = std::mem::size_of::<u64>();
        let half_slots = per_scope / 2 / slot_bytes;
        let slots = if half_slots == 0 {
            0
        } else {
            1usize << half_slots.ilog2()
        };
        let ring = (per_scope - slots * slot_bytes) / std::mem::size_of::<u32>();
        if slots == 0 || ring < MIN_SCOPE_TOKENS {
            return Err(format!(
                "shared prompt-lookup cache: {per_scope} bytes per scope holds fewer than \
                 {MIN_SCOPE_TOKENS} tokens; raise the budget or lower the scope count"
            ));
        }
        Ok((slots, ring))
    }

    /// 2026-10-04: Bytes one scope allocates.
    pub fn scope_bytes(&self) -> Result<usize, String> {
        let (slots, ring) = self.scope_geometry()?;
        Ok(slots * std::mem::size_of::<u64>() + ring * std::mem::size_of::<u32>())
    }
}

/// 2026-10-04: One scope: a token ring addressed by logical position (never
/// reused; position 0 is never written, so a 0 slot is empty) and a slot table
/// from n-gram hash to the position of the token after that n-gram.
#[derive(Debug)]
struct Scope {
    ring: Vec<u32>,
    slots: Vec<u64>,
    /// 2026-10-04: Next logical position to write.
    head: u64,
    /// 2026-10-04: Cache clock at this scope's last insert or proposal.
    last_used: u64,
}

impl Scope {
    fn new(slots: usize, ring: usize) -> Self {
        Self {
            ring: vec![0; ring],
            slots: vec![0; slots],
            head: 1,
            last_used: 0,
        }
    }

    fn slot_of(&self, key: &[u32]) -> usize {
        let bits = self.slots.len().trailing_zeros();
        if bits == 0 {
            return 0;
        }
        (ngram_key(key).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - bits)) as usize
    }

    /// 2026-10-04: Oldest logical position still in the ring.
    fn oldest(&self) -> u64 {
        self.head.saturating_sub(self.ring.len() as u64).max(1)
    }

    fn token(&self, pos: u64) -> u32 {
        self.ring[(pos % self.ring.len() as u64) as usize]
    }

    /// 2026-10-04: The token at `pos` when it is still in the ring and not a
    /// run end.
    fn live_token(&self, pos: u64) -> Option<u32> {
        (pos >= self.oldest() && pos < self.head)
            .then(|| self.token(pos))
            .filter(|&t| t != SEP)
    }

    fn push(&mut self, tok: u32) -> u64 {
        let pos = self.head;
        let len = self.ring.len() as u64;
        self.ring[(pos % len) as usize] = tok;
        self.head += 1;
        pos
    }

    /// 2026-10-04: Position after the stored occurrence `key`'s slot points at,
    /// when that occurrence is still in the ring and equals `key`.
    fn find(&self, key: &[u32]) -> Option<u64> {
        let pos = self.slots[self.slot_of(key)];
        let k = key.len() as u64;
        if pos == 0 || pos >= self.head || pos < self.oldest() + k {
            return None;
        }
        key.iter()
            .enumerate()
            .all(|(i, &t)| self.token(pos - k + i as u64) == t)
            .then_some(pos)
    }

    /// 2026-10-04: Stores `doc` with `k`-token keys, skipping runs the scope
    /// already predicts (see [`DEDUP_RUN`]).
    fn insert(&mut self, doc: &[u32], k: usize) {
        // 2026-10-04: The tail that fits, with room for the run marker.
        let doc = &doc[doc.len().saturating_sub(self.ring.len() - 1)..];
        let mut writing = false;
        let mut dup_run = 0usize;
        for e in k..doc.len() {
            let key = &doc[e - k..e];
            let dup = self
                .find(key)
                .is_some_and(|p| self.live_token(p) == Some(doc[e]));
            dup_run = if dup { dup_run + 1 } else { 0 };
            if writing && dup_run >= DEDUP_RUN {
                self.push(SEP);
                writing = false;
            }
            if !writing && dup {
                continue;
            }
            if !writing {
                // 2026-10-04: A new run starts with its key as context.
                for &t in key {
                    self.push(t);
                }
                writing = true;
            }
            let pos = self.push(doc[e]);
            let slot = self.slot_of(key);
            self.slots[slot] = pos;
        }
        if writing {
            self.push(SEP);
        }
    }

    fn propose(&self, history: &[u32], k: usize, max_len: usize) -> Option<Vec<u32>> {
        if max_len == 0 || history.len() < k {
            return None;
        }
        let start = self.find(&history[history.len() - k..])?;
        let copy: Vec<u32> = (start..)
            .take(max_len)
            .map_while(|p| self.live_token(p))
            .collect();
        (!copy.is_empty()).then_some(copy)
    }
}

/// 2026-10-04: The cross-request cache: scopes created on first insert, at
/// most `max_scopes`, each of fixed size.
#[derive(Debug)]
pub struct SharedTokenCache {
    key_len: usize,
    max_scopes: usize,
    slots: usize,
    ring: usize,
    scopes: HashMap<ScopeKey, Scope>,
    clock: u64,
}

impl SharedTokenCache {
    /// 2026-10-04: An empty cache; `Err` when a scope would be too small
    /// ([`SharedLookupConfig::scope_geometry`]).
    pub fn new(cfg: SharedLookupConfig) -> Result<Self, String> {
        let (slots, ring) = cfg.scope_geometry()?;
        Ok(Self {
            key_len: cfg.key_len,
            max_scopes: cfg.max_scopes,
            slots,
            ring,
            scopes: HashMap::new(),
            clock: 0,
        })
    }

    /// 2026-10-04: Tokens a match must span.
    pub fn key_len(&self) -> usize {
        self.key_len
    }

    /// 2026-10-04: Scopes now held.
    pub fn scope_count(&self) -> usize {
        self.scopes.len()
    }

    /// 2026-10-04: Ring tokens written under `scope` so far (0 when absent):
    /// what a document's stored, not skipped, tokens cost.
    pub fn stored_tokens(&self, scope: &ScopeKey) -> u64 {
        self.scopes.get(scope).map_or(0, |s| s.head - 1)
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// 2026-10-04: Adds one request's tokens (prompt then output) under
    /// `scope`, creating the scope if needed. When `max_scopes` already exist
    /// the least recently used one is dropped first.
    pub fn insert(&mut self, scope: ScopeKey, doc: &[u32]) {
        if doc.len() <= self.key_len {
            return;
        }
        let now = self.tick();
        if !self.scopes.contains_key(&scope) && self.scopes.len() >= self.max_scopes {
            // 2026-10-04: Clock values are unique, so the victim is too.
            if let Some(victim) = self
                .scopes
                .iter()
                .min_by_key(|(_, s)| s.last_used)
                .map(|(k, _)| *k)
            {
                self.scopes.remove(&victim);
            }
        }
        let (slots, ring, k) = (self.slots, self.ring, self.key_len);
        let s = self
            .scopes
            .entry(scope)
            .or_insert_with(|| Scope::new(slots, ring));
        s.last_used = now;
        s.insert(doc, k);
    }

    /// 2026-10-04: The continuation, at most `max_len` tokens, of the latest
    /// stored occurrence under `scope` of `history`'s last `key_len` tokens.
    /// `None` when the scope does not exist or holds no such occurrence.
    pub fn propose(
        &mut self,
        scope: &ScopeKey,
        history: &[u32],
        max_len: usize,
    ) -> Option<Vec<u32>> {
        let now = self.tick();
        let s = self.scopes.get_mut(scope)?;
        s.last_used = now;
        s.propose(history, self.key_len, max_len)
    }
}

#[cfg(test)]
#[path = "shared_lookup_tests.rs"]
mod tests;
