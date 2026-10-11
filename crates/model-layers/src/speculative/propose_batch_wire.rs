// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The multi-rank batched MTP propose command: its opcode, the header bounds and
//! the payload both ranks read.
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - [`EP_CMD_MTP_PROPOSE_BATCH`] is above the decode-token range and differs from every other
//!   worker opcode (asserted where the opcodes are listed, `decode_checkpoint/plan.rs`).
//! - Rank 0 sends the command only for a batch its proposer has said it takes
//!   (`DraftProposer::propose_batch_max`), so both ranks run the batched drafter forward.
//!
//! Wire shape, after the `(0, cmd)` preamble (the slots travel in the payload, as in the
//! batched verify):
//!
//! ```text
//! rank 0                                        worker
//! n, num_drafts            (one bulk of 2)  ->  checked by propose_batch_shape
//! slots[n] tokens[n] positions[n] stash[n]  ->  propose_batch_payload: sequence i is
//!                          (one bulk of 4n)     slots[i], proposing from tokens[i] at
//!                                               positions[i], its target hidden in stash
//!                                               slot stash[i]
//! batched drafter forward (collectives)    <->  the same forward
//! ```

use anyhow::{Result, bail};

/// 2026-10-11: EP worker command: the batched MTP propose of `n` sequences (module doc).
pub const EP_CMD_MTP_PROPOSE_BATCH: u32 = 0xFFFF_FFFD;

/// 2026-10-11: The most sequences one batched propose carries: the batched verify's stash
/// width, which holds their target hiddens.
pub const PROPOSE_BATCH_MAX_SEQS: usize = crate::layer::VERIFY_WY_TABLE_SEQS;

/// 2026-10-11: The most drafts per sequence one batched propose asks for.
pub const PROPOSE_BATCH_MAX_DRAFTS: usize = 8;

/// 2026-10-11: The header words of a batch of `n` sequences proposing `num_drafts` each.
pub fn propose_batch_header(n: usize, num_drafts: usize) -> Result<[u32; 2]> {
    propose_batch_shape(&[n as u32, num_drafts as u32])?;
    Ok([n as u32, num_drafts as u32])
}

/// 2026-10-11: `(n, num_drafts)` from the header, or an error for fewer than 2 sequences, more
/// than [`PROPOSE_BATCH_MAX_SEQS`], or a draft count outside `1..=PROPOSE_BATCH_MAX_DRAFTS`.
pub fn propose_batch_shape(header: &[u32]) -> Result<(usize, usize)> {
    let [n, d] = header else {
        bail!("batched propose header: {} words, not 2", header.len());
    };
    let (n, d) = (*n as usize, *d as usize);
    if !(2..=PROPOSE_BATCH_MAX_SEQS).contains(&n) || !(1..=PROPOSE_BATCH_MAX_DRAFTS).contains(&d) {
        bail!(
            "batched propose shape: {n} sequences of {d} drafts (sequences \
             2..={PROPOSE_BATCH_MAX_SEQS}, drafts 1..={PROPOSE_BATCH_MAX_DRAFTS})"
        );
    }
    Ok((n, d))
}

/// 2026-10-11: The four columns of a batch, as one payload: slots, tokens, positions, stash.
pub fn propose_batch_payload(
    slots: &[u32],
    tokens: &[u32],
    positions: &[u32],
    stash: &[u32],
) -> Result<Vec<u32>> {
    let n = slots.len();
    if tokens.len() != n || positions.len() != n || stash.len() != n {
        bail!(
            "batched propose payload: {n} slots, {} tokens, {} positions, {} stash rows",
            tokens.len(),
            positions.len(),
            stash.len()
        );
    }
    Ok([slots, tokens, positions, stash].concat())
}

/// 2026-10-11: One batch's columns, as the worker reads them.
#[derive(Debug, PartialEq, Eq)]
pub struct ProposeBatchColumns {
    pub slots: Vec<u32>,
    pub tokens: Vec<u32>,
    pub positions: Vec<usize>,
    pub stash: Vec<usize>,
}

/// 2026-10-11: Split a payload of `n` sequences into its columns. Errors on a payload of any
/// other length.
pub fn propose_batch_columns(words: &[u32], n: usize) -> Result<ProposeBatchColumns> {
    if words.len() != 4 * n {
        bail!(
            "batched propose payload: {} words for {n} sequences",
            words.len()
        );
    }
    let col = |c: usize| &words[c * n..(c + 1) * n];
    Ok(ProposeBatchColumns {
        slots: col(0).to_vec(),
        tokens: col(1).to_vec(),
        positions: col(2).iter().map(|&p| p as usize).collect(),
        stash: col(3).iter().map(|&s| s as usize).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: What rank 0 encodes the worker decodes, column for column.
    #[test]
    fn a_payload_round_trips() {
        let words =
            propose_batch_payload(&[3, 0, 7], &[11, 12, 13], &[40, 41, 900], &[0, 1, 2]).unwrap();
        assert_eq!(words.len(), 12);
        assert_eq!(
            propose_batch_columns(&words, 3).unwrap(),
            ProposeBatchColumns {
                slots: vec![3, 0, 7],
                tokens: vec![11, 12, 13],
                positions: vec![40, 41, 900],
                stash: vec![0, 1, 2],
            }
        );
    }

    /// 2026-10-11: Ragged columns, a payload of the wrong length, and a header out of bounds
    /// are refused.
    #[test]
    fn malformed_batches_are_refused() {
        assert!(propose_batch_payload(&[1, 2], &[1], &[1, 2], &[0, 1]).is_err());
        assert!(propose_batch_columns(&[0; 11], 3).is_err());
        assert!(propose_batch_header(1, 1).is_err());
        assert!(propose_batch_header(2, 0).is_err());
        assert!(propose_batch_header(PROPOSE_BATCH_MAX_SEQS + 1, 1).is_err());
        assert!(propose_batch_header(2, PROPOSE_BATCH_MAX_DRAFTS + 1).is_err());
        assert!(propose_batch_shape(&[2]).is_err());
        assert_eq!(propose_batch_header(16, 2).unwrap(), [16, 2]);
        assert_eq!(propose_batch_shape(&[16, 2]).unwrap(), (16, 2));
    }
}
