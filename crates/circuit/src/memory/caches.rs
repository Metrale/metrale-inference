// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The caches a circuit declares ([`StateKind::is_cache`]), sized: how many units
//! each kind holds is a plan input ([`CacheInputs`]) the engine decides; the bytes of one unit
//! are the declaration's elements times its element size. A cache whose input is absent holds
//! nothing; nothing here guesses a count.
//!
//! Owner: metrale-circuit (memory).
//! Invariants:
//! - One unit rule per kind ([`units_of`]); every byte count but the prompt-lookup index is
//!   `units x elements x element size`.
//! - The prompt-lookup index is a host hash table per sequence: its bytes follow the table's
//!   growth law ([`hash_table_bytes`]), evaluated at the sequence's history length.
//! - A keyed format must be given ([`crate::state::StateInputs::formats`]); a missing key is
//!   an error, as in [`crate::state::StatePlan`].

use std::collections::BTreeMap;

use crate::state::{StateDecl, StateDtype, StateError, StateFormat, StateKind};

/// 2026-10-02: The prompt-lookup index of the sequences in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookupInputs {
    /// 2026-10-02: Sequences holding an index (the concurrency).
    pub sequences: u64,
    /// 2026-10-02: Tokens of history each one indexes (prompt plus generated).
    pub history_tokens: u64,
}

/// 2026-10-02: How many units each cache kind holds; `None` or 0 for a cache the serve does not
/// keep.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheInputs {
    /// 2026-10-02: Prefix-cache (Marconi) snapshot slots.
    pub prefix_snapshot_slots: u64,
    /// 2026-10-02: Decode-rollback ring slots per sequence slot, and the sequence slots.
    pub ring: (u64, u64),
    /// 2026-10-02: Carried-state verify slots (the dummy included) and carry-table rows; 0
    /// without the carried-state verify.
    pub carry: (u64, u64),
    /// 2026-10-02: Verify-table rows (the WY tables, the accepted-hidden stash); 0 without a
    /// proposer.
    pub verify_table_rows: u64,
    /// 2026-10-02: Hidden rows captured for the draft head.
    pub capture_rows: u64,
    /// 2026-10-02: The prompt-lookup index; `None` without prompt-lookup decoding.
    pub lookup: Option<LookupInputs>,
    /// 2026-10-02: Token-tree verify slots and nodes per tree; `None` without a token tree.
    pub tree: Option<(u64, u64)>,
    /// 2026-10-02: Verify slots and draft positions per slot whose drafted tokens are kept.
    pub drafts: Option<(u64, u64)>,
}

/// 2026-10-02: One sized cache declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheTerm {
    /// 2026-10-02: The declaration's id (`l0.gdn.prefix_h`).
    pub state: String,
    /// 2026-10-02: Its kind.
    pub kind: StateKind,
    /// 2026-10-02: Its element type.
    pub dtype: StateDtype,
    /// 2026-10-02: Units held.
    pub units: u64,
    /// 2026-10-02: Bytes held.
    pub bytes: u64,
    /// 2026-10-02: Held in host memory.
    pub host: bool,
}

/// 2026-10-02: The units `inputs` give a cache of `kind`; `None` for a non-cache kind, for the
/// prompt-lookup index (not linear in its units) and on overflow.
pub fn units_of(kind: StateKind, inputs: &CacheInputs) -> Option<u64> {
    match kind {
        StateKind::Recurrent | StateKind::PagedKv | StateKind::PromptLookupIndex => None,
        StateKind::PrefixSnapshot => Some(inputs.prefix_snapshot_slots),
        StateKind::RingSnapshot => inputs.ring.0.checked_mul(inputs.ring.1),
        StateKind::CarryStash => Some(inputs.carry.0),
        StateKind::CarryTable => Some(inputs.carry.1),
        StateKind::VerifyTable | StateKind::AcceptStash => Some(inputs.verify_table_rows),
        StateKind::HiddenCapture => Some(inputs.capture_rows),
        StateKind::TokenTreeMask => inputs.tree.map_or(Some(0), |(slots, nodes)| {
            slots.checked_mul(nodes)?.checked_mul(nodes)
        }),
        StateKind::DraftTokens => inputs
            .drafts
            .map_or(Some(0), |(slots, pos)| slots.checked_mul(pos)),
    }
}

/// 2026-10-02: Bytes of a Rust `HashMap` (hashbrown) holding `entries` entries of
/// `entry_bytes` each (a multiple of 16), as it grows by inserting them: buckets are the next
/// power of two of `entries * 8 / 7` (4 or 8 below 8 entries; hashbrown `capacity_to_buckets`),
/// each bucket one entry plus one control byte, plus one 16-byte control group. 0 entries hold no
/// allocation.
pub fn hash_table_bytes(entries: u64, entry_bytes: u64) -> Option<u64> {
    if entries == 0 {
        return Some(0);
    }
    let buckets = if entries < 4 {
        4
    } else if entries < 8 {
        8
    } else {
        (entries.checked_mul(8)? / 7).checked_next_power_of_two()?
    };
    buckets
        .checked_mul(entry_bytes.checked_add(1)?)?
        .checked_add(16)
}

/// 2026-10-02: Size every cache declaration of `states` under `inputs`; `formats` resolves keyed
/// formats.
pub fn cache_terms(
    states: &[StateDecl],
    formats: &BTreeMap<String, StateDtype>,
    inputs: &CacheInputs,
) -> Result<Vec<CacheTerm>, StateError> {
    let mut out = Vec::new();
    for s in states.iter().filter(|s| s.kind.is_cache()) {
        let dtype = match &s.format {
            StateFormat::Fixed(d) => *d,
            StateFormat::Keyed(k) => *formats.get(k).ok_or_else(|| StateError::MissingFormat {
                state: s.id.clone(),
                key: k.clone(),
            })?,
        };
        let over = || StateError::Overflow(s.id.clone());
        let unit = s.elements.checked_mul(dtype.size()).ok_or_else(over)?;
        let (units, bytes) = match s.kind {
            StateKind::PromptLookupIndex => match inputs.lookup {
                Some(l) => {
                    let one = hash_table_bytes(l.history_tokens, unit).ok_or_else(over)?;
                    (
                        l.sequences.checked_mul(l.history_tokens).ok_or_else(over)?,
                        l.sequences.checked_mul(one).ok_or_else(over)?,
                    )
                }
                None => (0, 0),
            },
            k => {
                let units = units_of(k, inputs).ok_or_else(over)?;
                (units, units.checked_mul(unit).ok_or_else(over)?)
            }
        };
        out.push(CacheTerm {
            state: s.id.clone(),
            kind: s.kind,
            dtype,
            units,
            bytes,
            host: s.kind.is_host(),
        });
    }
    Ok(out)
}

#[cfg(test)]
#[path = "caches_tests.rs"]
mod caches_tests;
