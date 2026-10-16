// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The warm-cache rule for a concurrency cell: whether the
//! server's snapshot pool can hold the cell's warmed prompts, and whether its
//! measured requests agree about being warm.
//!
//! Owner: bench (concurrency).
//! Invariants: with no warm-up rounds a cell is never cache-uncontrolled
//! or cache-cold.

use super::RequestEvidence;

// 2026-09-26: `stats::make_prompt` puts the prompt's tag at byte zero of the
// message content, so a different prompt shares at most the chat-template
// prefix with a warmed one. 80% cached separates reuse of the warmed prompt
// from that.
pub(super) const WARM_CACHE_FLOOR: f64 = 0.8;

/// 2026-09-26: SSM snapshot slots one warm request holds through a cell: its
/// prompt-tail checkpoint and its finish leaf (`finish_leaf_snapshot` in
/// model-engine's `decode_checkpoint.rs`). The prefill-end leaf is skipped
/// when a tail checkpoint lies within two blocks of the prompt end
/// (`prefill_b::exact_leaf`).
const LIVE_SLOTS_PER_WARM_REQUEST: usize = 2;
/// 2026-09-26: Stale finish leaves each earlier cell of the sweep leaves behind
/// for a prompt it also ran. A finish leaf covers prompt plus generated text,
/// so a warm-up round and a measured round that generate different text leave
/// two. The snapshot pool's eviction is session-aware (`evict_lru` in the
/// cache crate's `radix_tree/snapshot.rs`).
///
/// Measured 2026-09-14 on a 32-slot pool, twice on one node: the conc 8 cell
/// after the 1/2/4 cells lost two, then three, of its eight tails. With one
/// stale leaf per earlier cell [`slots_needed`] gives 31, which fits; with two
/// it gives 38, which does not.
const STALE_LEAVES_PER_EARLIER_CELL: usize = 2;
/// 2026-09-26: Slots the pool must hold for a `conc`-way cell to keep every
/// warmed prompt through its measured round: each prompt's live slots, the
/// stale leaves of every earlier cell that ran the same prompt (`earlier` are
/// those cells' concurrencies; prompt `i` ran in each one wider than `i`),
/// and one spare per in-flight request, because a save that re-homes a prefix
/// takes a new slot before it frees the old one.
///
/// It errs toward "cold": a cell judged cold that hits warm anyway is still
/// measured (only the warm rule is skipped), while a cell judged warm that
/// eviction turns cold would fail the gate with nothing wrong in the engine.
pub(super) fn slots_needed(conc: usize, earlier: &[usize]) -> usize {
    (0..conc)
        .map(|i| {
            let stale_cells = earlier.iter().filter(|&&c| c > i).count();
            LIVE_SLOTS_PER_WARM_REQUEST + STALE_LEAVES_PER_EARLIER_CELL * stale_cells
        })
        .sum::<usize>()
        + conc
}
pub(super) const SSM_CACHE_SLOTS_KEY: &str = "ssm_cache_slots";

/// 2026-09-26: Whether the server's snapshot pool can hold every warmed prompt
/// of a `conc`-way cell. `slots` is the pool size from the serve overrides;
/// `None` (not stated) answers `true`, so the warm rule applies in full.
/// Strictly greater, because the last re-home still needs its spare slot.
pub(super) fn warm_cache_capable(conc: usize, slots: Option<usize>, earlier: &[usize]) -> bool {
    slots.is_none_or(|slots| slots > slots_needed(conc, earlier))
}

/// 2026-10-16: What a cell's measured requests say about the prefix cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CacheState {
    /// No warm-up round sent the measured prompts, so nothing was expected to
    /// be cached. The rule does not apply.
    NotWarmed,
    /// Warmed, but the server reported no cached count for any request. There
    /// is nothing to judge: an absent count is not a 0.
    Unreported,
    /// Every request had at least `WARM_CACHE_FLOOR` of its prompt cached.
    Warm,
    /// Warmed, and no request reached `WARM_CACHE_FLOOR`: the warmed prefix
    /// was not restored. Labelled cache-cold and counted in
    /// `cache_cold_cells`.
    Cold,
    /// The requests disagree, a request reported no prompt usage, or the cell
    /// has no request: two measurements reported as one. Uncontrolled.
    Mixed,
}

/// 2026-10-16: The cache state of a cell whose measured prompts were
/// (`warmed`) or were not sent in a warm-up round first. A request with no
/// cached count is not warm, so it splits a cell whose other requests are.
///
/// A uniformly cold cell is NOT a controlled warm measurement. It used to pass
/// as one: below the server's restore threshold (`--marconi-min-tokens`,
/// `DEFAULT_MARCONI_MIN_TOKENS` = 256 in model-layers `mtp_carry.rs`) no
/// snapshot is restored and every request reports 0 cached, so a recipe whose
/// restore threshold sat above the shared prefix measured a whole decode
/// ladder cold while the record called it controlled. It is now [`Cold`]:
/// still comparable (every request measured the same thing, so the verdict
/// and the `cache_uncontrolled_cells` count are unchanged), but labelled
/// cache-cold in the log and counted in `cache_cold_cells`. A gate that
/// expects a warm ladder bounds that metric at `max = 0`; a gate that is cold
/// by design leaves it unbounded.
///
/// The record also carries `min_cached_prompt_pct` and
/// `min_cached_prompt_tokens`, and the evidence line prints each request's
/// cached count.
///
/// [`Cold`]: CacheState::Cold
pub(super) fn cache_state(requests: &[RequestEvidence], warmed: bool) -> CacheState {
    if !warmed {
        return CacheState::NotWarmed;
    }
    if requests.is_empty() || requests.iter().any(|r| r.prompt_tokens == 0) {
        return CacheState::Mixed;
    }
    if requests.iter().all(|r| r.cached_prompt_tokens.is_none()) {
        return CacheState::Unreported;
    }
    let warm = |r: &RequestEvidence| {
        r.cached_prompt_tokens
            .is_some_and(|c| c as f64 >= WARM_CACHE_FLOOR * r.prompt_tokens as f64)
    };
    match (requests.iter().any(warm), requests.iter().all(warm)) {
        (_, true) => CacheState::Warm,
        (false, _) => CacheState::Cold,
        _ => CacheState::Mixed,
    }
}

/// 2026-10-16: The smallest cached count over `requests`, for a cell or a
/// whole sweep. `None` when there is no request or any request reported no
/// count: a minimum over the reporting requests only would read higher than
/// what was measured.
pub(super) fn min_cached_prompt<'a>(
    requests: impl IntoIterator<Item = &'a RequestEvidence>,
) -> Option<u64> {
    requests
        .into_iter()
        .map(|r| r.cached_prompt_tokens)
        .collect::<Option<Vec<u64>>>()?
        .into_iter()
        .min()
}

/// 2026-10-16: The smallest cached share of the prompt, in percent, over
/// `requests`; `None` on the same terms as [`min_cached_prompt`]. A request
/// with no prompt usage counts as 0%.
pub(super) fn min_cached_prompt_pct<'a>(
    requests: impl IntoIterator<Item = &'a RequestEvidence>,
) -> Option<f64> {
    requests
        .into_iter()
        .map(|r| {
            r.cached_prompt_tokens.map(|c| {
                if r.prompt_tokens == 0 {
                    0.0
                } else {
                    c as f64 / r.prompt_tokens as f64 * 100.0
                }
            })
        })
        .collect::<Option<Vec<f64>>>()?
        .into_iter()
        .reduce(f64::min)
}
