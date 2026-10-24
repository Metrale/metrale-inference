// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Pre-sampling EOS mask. 2026-10-13: driven by
//! `eos_hold::presample_hold`, the one rule every sampling path shares:
//! `min_tokens` not yet reached, or a required tool call not yet opened
//! outside thinking. The grammar part of the hold is the grammar bitmask
//! stage.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

/// 2026-09-25: Set every EOS logit to `-inf`.
pub fn mask_eos(logits: &mut [f32], eos_tokens: &[u32]) {
    for &eos in eos_tokens {
        if let Some(slot) = logits.get_mut(eos as usize) {
            *slot = f32::NEG_INFINITY;
        }
    }
}

/// 2026-09-25: Pipeline stage: masks EOS while `presample_hold` holds it at
/// `output_tokens.len() + ctx.verify_pos`.
pub struct EosHoldMask;

impl LogitsProcessor for EosHoldMask {
    fn apply(
        &self,
        logits: &mut [f32],
        seq: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        let generated_before = seq.output_tokens.len().saturating_add(ctx.verify_pos);
        mask_eos(
            logits,
            crate::scheduler::eos_hold::eos_mask_ids(seq, generated_before),
        );
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "eos_hold_mask"
    }
}

#[cfg(test)]
mod tests {
    use super::mask_eos;

    #[test]
    fn masks_listed_ids_and_skips_out_of_vocab() {
        let mut logits = [0.5, 1.0, 2.0];
        mask_eos(&mut logits, &[1, 99_999]);
        assert_eq!(logits, [0.5, f32::NEG_INFINITY, 2.0]);
        let mut untouched = [0.5, 1.0];
        mask_eos(&mut untouched, &[]);
        assert_eq!(untouched, [0.5, 1.0]);
    }
}
