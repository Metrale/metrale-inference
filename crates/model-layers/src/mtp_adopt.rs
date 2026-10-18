// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Drafter context on a prefix-cache hit (`METRALE_MTP_ADOPT_CAPTURE=1`,
//! `ModelLevers::mtp_adopt_capture`).
//!
//! A hit does not recompute its matched rows, so the whole-prompt hidden capture
//! (`mtp_prefill_hidden`, row `i` = `hidden_i`) is short of the prompt, the drafter's prompt
//! prefill is skipped (the model's `ensure_drafter_context`) and the drafter proposes with no
//! prompt context: the GLM-5.3 1k probe accepted 0.409 on the hit against 0.882 on the miss.
//! The capture buffer still holds the rows the last consumed capture wrote. When those rows were
//! computed for the same tokens as the hit's matched prefix, they are the hiddens a cold pass
//! would recompute for it, so the hit may adopt them: its own rows complete the capture and the
//! drafter is prefilled over the whole prompt as on a miss. The target's passes are untouched.
//!
//! The model keeps one [`CaptureRecord`]: the tokens whose hiddens the buffer's leading rows
//! hold, and the adoption claim in flight.
//! - Every capture write from row `s` on calls [`CaptureRecord::note_write`]: rows from `s` no
//!   longer hold the recorded tokens.
//! - A write that neither starts a capture nor extends the sequence's own claims the capture as
//!   an adoption when [`claims_adoption`] says so, recording the matched row count.
//! - The consume prefills the drafter only when [`CaptureRecord::covers`] holds, and then
//!   records the prompt ([`CaptureRecord::note_consumed`]).
//!
//! Owner: model-layers (MTP drafter).
//! Invariants:
//! - [`CaptureRecord::covers`] is true for every capture that is not an adoption claim, so with
//!   the lever off (no claim is ever made) the consume decides as before.
//! - An adoption claim is covered only while the recorded tokens are exactly the prompt's first
//!   `matched` tokens.
//! - The record never claims more rows than the last write left untouched.

/// 2026-10-14: A capture claimed by a sequence whose prompt matched `matched` rows in the prefix
/// cache: its own writes start at row `matched`, rows below come from the record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdoptClaim {
    /// 2026-10-14: The capture generation the claim drew (`SequenceState::mtp_capture_gen`).
    pub generation: u64,
    /// 2026-10-14: The first row the claiming sequence wrote.
    pub matched: usize,
}

/// 2026-10-14: Rows `[0, tokens.len())` of the capture hold the target's final hiddens for
/// `tokens`; `adopted` is the latest adoption claim.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptureRecord {
    /// 2026-10-14: The tokens of the rows a consumed capture left, cut back by later writes.
    pub tokens: Vec<u32>,
    /// 2026-10-14: The latest adoption claim; older claims have stale generations.
    pub adopted: Option<AdoptClaim>,
}

impl CaptureRecord {
    /// 2026-10-14: A capture write of rows from `start` on: those rows no longer hold the
    /// recorded tokens. Rows below `start` keep them.
    pub fn note_write(&mut self, start: usize) {
        self.tokens.truncate(start);
    }

    /// 2026-10-14: The capture under `generation` claims rows `[0, matched)` from the record.
    pub fn note_claim(&mut self, generation: u64, matched: usize) {
        self.adopted = Some(AdoptClaim {
            generation,
            matched,
        });
    }

    /// 2026-10-14: A consume built the drafter from rows `[0, prompt.len())`: they hold the
    /// hiddens of `prompt`.
    pub fn note_consumed(&mut self, prompt: &[u32]) {
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt);
        self.adopted = None;
    }

    /// 2026-10-14: True when the capture under `generation` may be consumed for `prompt`: it is
    /// not an adoption claim, or the record still holds exactly the prompt's first `matched`
    /// tokens.
    pub fn covers(&self, generation: u64, prompt: &[u32]) -> bool {
        match self.adopted {
            Some(c) if c.generation == generation => {
                c.matched <= prompt.len()
                    && self.tokens.len() >= c.matched
                    && self.tokens[..c.matched] == prompt[..c.matched]
            }
            _ => true,
        }
    }
}

/// 2026-10-14: True when a capture write from row `start` that neither starts a capture nor
/// extends the sequence's own claims the capture as an adoption: the lever is on, the drafter
/// carry (which handles a warm turn its own way) is not armed, the write is past row 0, and the
/// record holds at least `start` rows (`recorded`). A claim draws a new capture generation, which
/// ends another sequence's ownership, so a claim the record cannot cover is not made.
pub fn claims_adoption(adopt: bool, carry_armed: bool, start: usize, recorded: usize) -> bool {
    adopt && !carry_armed && start > 0 && recorded >= start
}

#[cfg(test)]
#[path = "mtp_adopt_tests.rs"]
mod tests;
