// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The multi-rank batched prefill command (`EP_CMD_PREFILL_SPANS`): its opcode, its
//! header codec and the split of rank 0's sub-batches, as pure functions.
//!
//! Owner: model-engine (EP worker protocol).
//! Invariants:
//! - [`EP_CMD_PREFILL_SPANS`] is above the decode-token range and differs from every other
//!   worker opcode (asserted where the opcodes are listed, `decode_checkpoint/plan.rs`).
//! - [`decode_spans_header`] returns what [`encode_spans_header`] was given, and rejects a
//!   header whose chunk lies outside its prompt or is empty.
//!
//! Wire shape, after the `(0, cmd)` preamble (the slots travel in the payload, as in the
//! batched decode):
//!
//! ```text
//! rank 0                                          worker
//! n                       (one word)          ->  1..=slots
//! header[1 + 4n]          (one bulk)          ->  row_base, seq_ids[n], chunk_start[n],
//!                                                 chunk_len[n], full_len[n]
//! prompts[Σ full_len]     (one bulk)          ->  each sequence's full prompt, in order
//! multi-sequence prefill (collectives)       <->  the same pass
//! ```

use anyhow::{Result, bail};

/// 2026-10-09: EP worker command: prefill one chunk of each of `n` sequences in one
/// multi-sequence pass (module doc).
pub(in crate::model) const EP_CMD_PREFILL_SPANS: u32 = 0xFFFF_FFFA;

/// 2026-10-09: What one `EP_CMD_PREFILL_SPANS` carries besides the prompts: the logits row of
/// the first sequence, and per sequence its slot, its chunk and its prompt length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::model) struct SpansHeader {
    pub row_base: usize,
    pub seq_ids: Vec<u32>,
    pub chunk_start: Vec<usize>,
    pub chunk_len: Vec<usize>,
    pub full_len: Vec<usize>,
}

impl SpansHeader {
    /// 2026-10-09: Words of the prompts bulk that follows the header.
    pub(in crate::model) fn prompt_words(&self) -> usize {
        self.full_len.iter().sum()
    }

    /// 2026-10-09: Whether sequence `i`'s chunk ends its prompt: the `is_last` both ranks'
    /// prefill branches on, from the chunk bounds as the single-sequence worker handler
    /// computes it.
    pub(in crate::model) fn is_last(&self, i: usize) -> bool {
        self.chunk_start[i] + self.chunk_len[i] >= self.full_len[i]
    }
}

/// 2026-10-09: Header words for `n` sequences.
pub(in crate::model) fn spans_header_words(n: usize) -> usize {
    1 + 4 * n
}

/// 2026-10-09: The header as rank 0 sends it. Errors on a value that does not fit a word or on
/// per-sequence lists of different lengths.
pub(in crate::model) fn encode_spans_header(h: &SpansHeader) -> Result<Vec<u32>> {
    let n = h.seq_ids.len();
    if h.chunk_start.len() != n || h.chunk_len.len() != n || h.full_len.len() != n {
        bail!("prefill spans header: per-sequence lists differ in length");
    }
    let word = |v: usize| -> Result<u32> {
        u32::try_from(v).map_err(|_| anyhow::anyhow!("prefill spans header: {v} exceeds a word"))
    };
    let mut out = Vec::with_capacity(spans_header_words(n));
    out.push(word(h.row_base)?);
    out.extend_from_slice(&h.seq_ids);
    for list in [&h.chunk_start, &h.chunk_len, &h.full_len] {
        for &v in list.iter() {
            out.push(word(v)?);
        }
    }
    Ok(out)
}

/// 2026-10-09: The header of `n` sequences from its words. Errors on `n` outside
/// `1..=max_seqs`, a word count other than [`spans_header_words`], an empty chunk, or a chunk
/// past its prompt.
pub(in crate::model) fn decode_spans_header(
    n: usize,
    max_seqs: usize,
    words: &[u32],
) -> Result<SpansHeader> {
    if !(1..=max_seqs).contains(&n) {
        bail!("prefill spans: {n} sequences (1..={max_seqs})");
    }
    if words.len() != spans_header_words(n) {
        bail!(
            "prefill spans: a {n}-sequence header has {} words, got {}",
            spans_header_words(n),
            words.len()
        );
    }
    let list = |k: usize| -> Vec<usize> {
        words[1 + k * n..1 + (k + 1) * n]
            .iter()
            .map(|&w| w as usize)
            .collect()
    };
    let h = SpansHeader {
        row_base: words[0] as usize,
        seq_ids: words[1..1 + n].to_vec(),
        chunk_start: list(1),
        chunk_len: list(2),
        full_len: list(3),
    };
    for i in 0..n {
        if h.chunk_len[i] == 0 || h.chunk_start[i] + h.chunk_len[i] > h.full_len[i] {
            bail!(
                "prefill spans: sequence {i} chunk {}+{} of a {}-token prompt",
                h.chunk_start[i],
                h.chunk_len[i],
                h.full_len[i]
            );
        }
    }
    Ok(h)
}

/// 2026-10-09: The prompts bulk cut into each sequence's prompt. Errors when its length is not
/// the header's `prompt_words`.
pub(in crate::model) fn split_prompts(h: &SpansHeader, flat: &[u32]) -> Result<Vec<Vec<u32>>> {
    if flat.len() != h.prompt_words() {
        bail!(
            "prefill spans: {} prompt words for prompts totalling {}",
            flat.len(),
            h.prompt_words()
        );
    }
    let mut at = 0usize;
    Ok(h.full_len
        .iter()
        .map(|&len| {
            let p = flat[at..at + len].to_vec();
            at += len;
            p
        })
        .collect())
}

/// 2026-10-09: Consecutive sub-batches of sequences whose chunks fit `cap` rows together, as
/// index ranges in order. Errors on a single chunk above `cap`, which no pass can hold.
pub(in crate::model) fn spans_batches(
    chunk_lens: &[usize],
    cap: usize,
) -> Result<Vec<std::ops::Range<usize>>> {
    let mut out = Vec::new();
    let (mut lo, mut rows) = (0usize, 0usize);
    for (i, &len) in chunk_lens.iter().enumerate() {
        if len > cap {
            bail!("prefill spans: a {len}-row chunk exceeds the {cap}-row arena");
        }
        if rows + len > cap {
            out.push(lo..i);
            (lo, rows) = (i, 0);
        }
        rows += len;
    }
    if lo < chunk_lens.len() {
        out.push(lo..chunk_lens.len());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> SpansHeader {
        SpansHeader {
            row_base: 3,
            seq_ids: vec![4, 0, 9],
            chunk_start: vec![0, 512, 0],
            chunk_len: vec![198, 100, 1],
            full_len: vec![198, 612, 7],
        }
    }

    /// 2026-10-09: The worker reads back exactly what rank 0 encoded.
    #[test]
    fn header_round_trips() {
        let h = header();
        let w = encode_spans_header(&h).unwrap();
        assert_eq!(w.len(), spans_header_words(3));
        assert_eq!(decode_spans_header(3, 16, &w).unwrap(), h);
        assert!(h.is_last(0) && h.is_last(1) && !h.is_last(2));
    }

    /// 2026-10-09: A header read at the wrong `n`, a chunk past its prompt, an empty chunk, or
    /// more sequences than slots is refused.
    #[test]
    fn malformed_headers_are_refused() {
        let w = encode_spans_header(&header()).unwrap();
        assert!(decode_spans_header(2, 16, &w).is_err());
        assert!(decode_spans_header(3, 2, &w).is_err());
        assert!(decode_spans_header(0, 16, &[0]).is_err());
        let mut past = header();
        past.chunk_len[1] = 101;
        assert!(decode_spans_header(3, 16, &encode_spans_header(&past).unwrap()).is_err());
        let mut empty = header();
        empty.chunk_len[0] = 0;
        assert!(decode_spans_header(3, 16, &encode_spans_header(&empty).unwrap()).is_err());
    }

    /// 2026-10-09: The prompts bulk splits at the header's lengths and nowhere else.
    #[test]
    fn prompts_split_at_the_header_lengths() {
        let h = SpansHeader {
            row_base: 0,
            seq_ids: vec![1, 2],
            chunk_start: vec![0, 0],
            chunk_len: vec![2, 3],
            full_len: vec![2, 3],
        };
        assert_eq!(
            split_prompts(&h, &[10, 11, 20, 21, 22]).unwrap(),
            vec![vec![10, 11], vec![20, 21, 22]]
        );
        assert!(split_prompts(&h, &[10, 11, 20, 21]).is_err());
    }

    /// 2026-10-09: Sub-batches are consecutive, cover every sequence once and stay within the
    /// arena; a chunk larger than the arena is refused.
    #[test]
    fn sub_batches_fit_the_arena() {
        assert_eq!(
            spans_batches(&[198; 16], 8208).unwrap(),
            vec![0..16],
            "the ladder's 16 prompts fit one pass"
        );
        assert_eq!(
            spans_batches(&[5000, 4000, 100, 8192], 8208).unwrap(),
            vec![0..1, 1..3, 3..4]
        );
        assert!(spans_batches(&[8209], 8208).is_err());
        assert!(spans_batches(&[], 8).unwrap().is_empty());
    }
}
