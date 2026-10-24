// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The multi-rank batched DFlash verify command: its opcode, the shape bounds and
//! the verdict both ranks apply after the forward.
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - [`EP_CMD_VERIFY_BATCH`] is above the decode-token range and differs from every other
//!   worker opcode (asserted where the opcodes are listed, `decode_checkpoint/plan.rs`).
//! - The verdict is sent whatever rank 0's forward returned, so the word count on the wire
//!   never depends on an error.
//!
//! Wire shape, after the `(0, cmd)` preamble (the slots travel in the payload, as in the
//! batched decode):
//!
//! ```text
//! rank 0                                     worker
//! n, k                 (two words)       ->  checked by verify_batch_shape
//! slots[n]             (one bulk)        ->  the batch's sequences, in this order
//! tokens[n * k]        (one bulk)        ->  sequence-major rows
//! batched verify forward (collectives)  <->  the same forward
//! committed[n]         (one bulk)        ->  verify_batch_verdict: trim, fold, commit each
//! ```
//!
//! `committed[i]` counts sequence `i`'s anchor row (`num_accepted + 1`, in `1..=k`); all zeros
//! means rank 0's forward failed and it finishes the whole batch.

use anyhow::{Result, bail};

use super::KGAMMA_MAX_ROWS;

/// 2026-10-09: EP worker command: the batched DFlash verify of `n` sequences (module doc).
pub const EP_CMD_VERIFY_BATCH: u32 = 0xFFFF_FFF9;

/// 2026-10-09: The most sequences one batched verify carries: the width of the batched
/// verify's per-layer pointer tables (`layer::VERIFY_WY_TABLE_SEQS`).
pub const VERIFY_BATCH_MAX_SEQS: usize = crate::layer::VERIFY_WY_TABLE_SEQS;

/// 2026-10-09: `(n, k)` as a batched verify's sequence count and per-sequence row count, or an
/// error for fewer than 2 sequences, more than [`VERIFY_BATCH_MAX_SEQS`], or `k` outside
/// `2..=KGAMMA_MAX_ROWS`.
pub fn verify_batch_shape(n: u32, k: u32) -> Result<(usize, usize)> {
    let (n, k) = (n as usize, k as usize);
    if !(2..=VERIFY_BATCH_MAX_SEQS).contains(&n) || !(2..=KGAMMA_MAX_ROWS).contains(&k) {
        bail!(
            "batched verify shape: {n} sequences of {k} rows (sequences 2..={VERIFY_BATCH_MAX_SEQS}, \
             rows 2..={KGAMMA_MAX_ROWS})"
        );
    }
    Ok((n, k))
}

/// 2026-10-09: The verdict rank 0 sends for a forward that failed: one zero per sequence.
pub fn verify_batch_failed_verdict(n: usize) -> Vec<u32> {
    vec![0; n]
}

/// 2026-10-09: Read a verdict of `k`-row verifies: `Ok(None)` when every word is 0 (rank 0's
/// forward failed), else each sequence's committed row count. Errors on a mix of zeros and
/// counts, or a count above `k`.
pub fn verify_batch_verdict(words: &[u32], k: usize) -> Result<Option<Vec<usize>>> {
    if words.iter().all(|&w| w == 0) {
        return Ok(None);
    }
    words
        .iter()
        .enumerate()
        .map(|(i, &w)| {
            let c = w as usize;
            if c == 0 || c > k {
                bail!("batched verify verdict: sequence {i} commits {c} of {k} rows");
            }
            Ok(c)
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_verdict_of_zeros_is_a_failed_batch_and_counts_are_kept_in_order() {
        assert_eq!(
            verify_batch_verdict(&verify_batch_failed_verdict(3), 8).unwrap(),
            None
        );
        assert_eq!(
            verify_batch_verdict(&[1, 8, 3], 8).unwrap(),
            Some(vec![1, 8, 3])
        );
    }

    /// 2026-10-09: A zero beside counts (one sequence's verdict lost) or a count past `k`
    /// is refused rather than read as a rejection.
    #[test]
    fn a_partial_or_impossible_verdict_is_refused() {
        assert!(verify_batch_verdict(&[0, 2], 8).is_err());
        assert!(verify_batch_verdict(&[9, 2], 8).is_err());
    }

    #[test]
    fn the_shape_bounds_are_the_batched_verify_bounds() {
        assert!(verify_batch_shape(1, 8).is_err());
        assert!(verify_batch_shape(2, 1).is_err());
        assert!(verify_batch_shape(VERIFY_BATCH_MAX_SEQS as u32 + 1, 8).is_err());
        assert!(verify_batch_shape(2, KGAMMA_MAX_ROWS as u32 + 1).is_err());
        assert_eq!(verify_batch_shape(16, 8).unwrap(), (16, 8));
    }
}
