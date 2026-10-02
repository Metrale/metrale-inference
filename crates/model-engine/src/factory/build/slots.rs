// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: How many sequence slots a model is built with. `--max-batch-size N` asks for N;
//! `--max-batch-size auto` asks for the count that admits the most sequences at the full
//! `--max-seq-len`, sizing the per-slot state (the SSM pools and every other slot-scaled reserve
//! term) and the KV pool together: each slot of state costs KV blocks, so past some count more
//! slots admit fewer sequences, not more.
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - [`balance_slots`] is pure: the reserve and the pool size come in as functions of the slot
//!   count, so the boot and the tests evaluate the same arithmetic.
//! - An explicit count is never changed; `auto` is the only request this module resolves.

use anyhow::{Result, ensure};

/// 2026-10-01: The `--max-batch-size` request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotRequest {
    /// 2026-10-01: Exactly this many slots.
    Count(usize),
    /// 2026-10-01: The count [`balance_slots`] chooses, at most [`AUTO_MAX_SLOTS`].
    Auto,
}

/// 2026-10-01: The most slots `auto` considers: the widest rung of the decode batch ladder
/// (`traits::DECODE_BATCH_LADDER`), the widest batch a decode step is padded to and captured at.
pub const AUTO_MAX_SLOTS: usize =
    crate::traits::DECODE_BATCH_LADDER[crate::traits::DECODE_BATCH_LADDER.len() - 1];

impl SlotRequest {
    /// 2026-10-01: The slots sized before the KV pool is (the pre-load reserve, the buffer
    /// arena): the count, or for `auto` its ceiling [`AUTO_MAX_SLOTS`].
    pub fn ceiling(self) -> usize {
        match self {
            SlotRequest::Count(n) => n,
            SlotRequest::Auto => AUTO_MAX_SLOTS,
        }
    }
}

impl std::str::FromStr for SlotRequest {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        if s == "auto" {
            return Ok(SlotRequest::Auto);
        }
        s.parse::<usize>()
            .map(SlotRequest::Count)
            .map_err(|_| format!("`{s}` is neither a slot count nor `auto`"))
    }
}

impl std::fmt::Display for SlotRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlotRequest::Count(n) => write!(f, "{n}"),
            SlotRequest::Auto => f.write_str("auto"),
        }
    }
}

/// 2026-10-01: The slot request and the pre-load reserve as a function of the slot count (the
/// preflight's plan, `inference_reserve(slots)`), which KV sizing evaluates at the count it
/// builds.
pub struct SlotPlan<'a> {
    pub request: SlotRequest,
    pub reserve_for: &'a dyn Fn(usize) -> Result<usize>,
}

/// 2026-10-01: A built model and the slot count it was built with (the request, or what `auto`
/// resolved to), which the scheduler and the serve's disclosure take from here.
pub struct BuiltModel {
    pub model: Box<dyn crate::traits::Model>,
    pub max_batch_size: usize,
}

/// 2026-10-01: One candidate slot count's outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Balance {
    /// 2026-10-01: The slot count.
    pub slots: usize,
    /// 2026-10-01: KV blocks the pool gets with that many slots.
    pub kv_blocks: usize,
    /// 2026-10-01: Sequences admitted at the full `--max-seq-len`: the slots, or the full-length
    /// sequences the pool holds, whichever is fewer.
    pub admitted: usize,
}

/// 2026-10-01: The slot count in `1..=max_slots` that admits the most sequences at
/// `blocks_per_seq` blocks each, `kv_blocks_at(slots)` giving the pool size for a count. On a tie
/// the smaller count wins, so the chosen count never exceeds what its pool holds at full length
/// (a count above it ties with the count one below, whose pool is at least as large) and the boot's
/// overcommit check passes. An error when `max_slots` is 0, when `kv_blocks_at` fails for every
/// count, or when no count's pool holds one full-length sequence.
pub fn balance_slots(
    max_slots: usize,
    blocks_per_seq: usize,
    kv_blocks_at: impl Fn(usize) -> Result<usize>,
) -> Result<Balance> {
    ensure!(max_slots > 0, "no slot count to balance (ceiling 0)");
    let per_seq = blocks_per_seq.max(1);
    let mut best: Option<Balance> = None;
    for slots in 1..=max_slots {
        let Ok(kv_blocks) = kv_blocks_at(slots) else {
            continue;
        };
        let b = Balance {
            slots,
            kv_blocks,
            admitted: slots.min(kv_blocks / per_seq),
        };
        if best.is_none_or(|x| b.admitted > x.admitted) {
            best = Some(b);
        }
    }
    let best = best.ok_or_else(|| {
        anyhow::anyhow!("no slot count in 1..={max_slots} leaves memory for the KV pool")
    })?;
    ensure!(
        best.admitted > 0,
        "no slot count in 1..={max_slots} leaves a KV pool that holds one sequence at the full \
         --max-seq-len ({per_seq} blocks; {} at 1 slot): lower --max-seq-len",
        best.kv_blocks
    );
    Ok(best)
}

#[cfg(test)]
#[path = "slots_tests.rs"]
mod slots_tests;
