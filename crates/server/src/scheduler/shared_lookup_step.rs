// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The scheduler side of the cross-request prompt-lookup cache
//! (`--prompt-lookup-shared-cache-mb`, `metrale_speculative::shared_lookup`):
//! the scope a sequence reads and writes, storing retired sequences, and the
//! shared-copy counters.
//!
//! Threat model. The cache holds earlier requests' prompt and output tokens in
//! host memory, so it is a channel between requests. It is bounded as follows.
//! - Content: a sequence reads only its own scope, keyed by model, tokenizer,
//!   LoRA adapter id and tenant (`auth::LookupTenant`: the bearer token under
//!   `--require-auth`, one tenant without it). A sequence with no tenant
//!   (`SequenceState::lookup_tenant == None`) is neither read nor stored.
//!   The scheduler that owns the cache serves one model load and is dropped
//!   with it, so model and tokenizer separation is also structural.
//! - Output: a copy is a draft; only verified tokens are emitted, so the
//!   cache can change how fast tokens arrive, never which tokens (wherever
//!   verify rows are row-invariant, as for prompt lookup itself).
//! - Timing: what a tenant can learn from its own speed is whether its own
//!   scope's earlier requests contained its continuation. Another tenant's
//!   requests never write, overwrite or evict that scope while no more than
//!   `--prompt-lookup-shared-cache-scopes` scopes are in use; past that, a new
//!   scope drops the least recently used one whole, which reveals only that
//!   other scopes were active (a volume signal, no content). Storing a retired
//!   sequence costs scheduler time in proportion to its length, which, like
//!   batching itself, is a volume signal shared by every request.
//! - The prefix cache is outside this model: it is keyed by adapter, not
//!   tenant, and is unchanged here.
//! - Retention: tokens live in host memory until overwritten or the serve
//!   exits; nothing is written to disk.
//!
//! Owner: scheduler.
//! Invariants:
//! - Only sequences that finished without an error are stored, before
//!   `retire_finished_sequences` releases them.
//! - The cache is used only from the scheduler thread (`RefCell`, no lock).

use std::cell::{Cell, RefCell};

use metrale_model_engine::traits::SequenceState;
use metrale_speculative::shared_lookup::{ScopeKey, SharedLookupConfig, SharedTokenCache};

use super::ActiveSeq;

/// 2026-10-04: What `serve_load` hands the scheduler: the cache settings and
/// fingerprints of the model and tokenizer the serve loaded.
#[derive(Debug, Clone, Copy)]
pub struct SharedLookupSetup {
    pub config: SharedLookupConfig,
    pub model: u64,
    pub tokenizer: u64,
}

/// 2026-10-04: The live cache and its counters.
#[derive(Debug)]
pub struct SharedLookup {
    cache: RefCell<SharedTokenCache>,
    model: u64,
    tokenizer: u64,
    /// 2026-10-04: Shared copies verified, tokens they proposed, tokens
    /// accepted (`prompt_lookup_step::settle_copies`).
    pub stats: Cell<[u64; 3]>,
}

impl SharedLookup {
    /// 2026-10-04: An empty cache; `Err` when a scope would be too small.
    pub fn new(setup: SharedLookupSetup) -> Result<Self, String> {
        Ok(Self {
            cache: RefCell::new(SharedTokenCache::new(setup.config)?),
            model: setup.model,
            tokenizer: setup.tokenizer,
            stats: Cell::new([0; 3]),
        })
    }

    /// 2026-10-04: The scope `seq` reads and writes; `None` without a tenant.
    pub fn scope(&self, seq: &SequenceState) -> Option<ScopeKey> {
        Some(ScopeKey {
            model: self.model,
            tokenizer: self.tokenizer,
            adapter: seq.adapter_id,
            tenant: seq.lookup_tenant?,
        })
    }

    /// 2026-10-04: A copy of at most `max_len` tokens for `history` from
    /// `scope`'s earlier requests.
    pub fn propose(&self, scope: &ScopeKey, history: &[u32], max_len: usize) -> Option<Vec<u32>> {
        self.cache.borrow_mut().propose(scope, history, max_len)
    }

    /// 2026-10-04: Stores every finished, error-free sequence of `active`:
    /// its prompt, then the tokens it emitted. (At a finish `last_token` and
    /// the tail of `seq.tokens` need not be the emitted order, so neither is
    /// read.)
    pub(in crate::scheduler) fn store_finished(&self, active: &[ActiveSeq]) {
        let mut cache = self.cache.borrow_mut();
        for a in active.iter().filter(|a| a.finished && a.error.is_none()) {
            let Some(scope) = self.scope(&a.seq) else {
                continue;
            };
            let prompt = &a.seq.tokens[..a.seq.prompt_len.min(a.seq.tokens.len())];
            let mut doc = Vec::with_capacity(prompt.len() + a.output_tokens.len());
            doc.extend_from_slice(prompt);
            doc.extend_from_slice(&a.output_tokens);
            cache.insert(scope, &doc);
        }
    }

    /// 2026-10-04: Counts one verified shared copy.
    pub fn record(&self, proposed: usize, accepted: usize) {
        let [n, p, acc] = self.stats.get();
        self.stats.set([
            n + 1,
            p + proposed as u64,
            acc + accepted.min(proposed) as u64,
        ]);
    }
}

/// 2026-10-04: Fingerprint of a name or file contents, for [`ScopeKey`].
pub fn fingerprint(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

#[cfg(test)]
#[path = "shared_lookup_step_tests.rs"]
mod tests;
