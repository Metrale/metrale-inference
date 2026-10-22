// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Exact encode of a long prompt through a memo of its text pieces.
//!
//! A Hugging Face tokenizer first splits the text at its added tokens
//! (`AddedVocabulary::extract_and_normalize`); every piece between them is then normalized,
//! pre-tokenized and run through the model on its own. A piece's ids therefore depend on the
//! piece alone, and a conversation re-sent with one more turn re-encodes only the new pieces.
//! Measured on GLM-5.3-Flash's tokenizer: a 479 KB user message (128025 tokens) took 120-135 ms
//! on one core, the bulk of a warm 128k request's 153 ms of template and tokenize.
//!
//! Owner: server tokenizer.
//! Invariants:
//! - [`SegmentMemo::encode`] returns exactly `Tokenizer::encode(text, false)`'s ids: the pieces
//!   are the tokenizer's own split, and short runs are re-encoded whole, added tokens included.
//! - The memo holds at most `budget_bytes` of piece text plus ids; the least recently used
//!   pieces leave first.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use parking_lot::Mutex;
use rayon::prelude::*;
use tokenizers::{OffsetReferential, OffsetType, Tokenizer};

/// 2026-10-22: Below this a text is encoded whole: the split and the lookups cost more than
/// they save.
const MIN_TEXT_BYTES: usize = 32 << 10;
/// 2026-10-22: A text piece this long is memoized; shorter pieces and the added tokens between
/// them are re-encoded as one run.
const MIN_PIECE_BYTES: usize = 4 << 10;

/// 2026-10-22: One unit of a split text: a run re-encoded whole, or a memoized piece.
enum Unit<'a> {
    Run(&'a str),
    Piece(&'a str),
}

struct Entry {
    ids: Arc<[u32]>,
    last_use: u64,
}

#[derive(Default)]
struct Table {
    map: HashMap<Arc<str>, Entry>,
    bytes: usize,
    clock: u64,
}

/// 2026-10-22: The piece memo of one tokenizer (`--tokenizer-memo-mb`).
pub struct SegmentMemo {
    budget_bytes: usize,
    table: Mutex<Table>,
}

impl SegmentMemo {
    /// 2026-10-22: A memo of at most `budget_bytes`; 0 encodes every text whole.
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            table: Mutex::new(Table::default()),
        }
    }

    /// 2026-10-22: `tokenizer.encode(text, false)`'s ids, reusing memoized pieces.
    pub fn encode(&self, tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
        if self.budget_bytes == 0 || text.len() < MIN_TEXT_BYTES {
            return encode_whole(tokenizer, text);
        }
        let units = split_units(tokenizer, text);
        let mut ids: Vec<Option<Arc<[u32]>>> = {
            let mut t = self.table.lock();
            t.clock += 1;
            let now = t.clock;
            units
                .iter()
                .map(|u| match u {
                    Unit::Piece(s) => t.map.get_mut(*s).map(|e| {
                        e.last_use = now;
                        e.ids.clone()
                    }),
                    Unit::Run(_) => None,
                })
                .collect()
        };
        let fresh: Vec<(usize, Result<Arc<[u32]>>)> = units
            .par_iter()
            .enumerate()
            .filter(|(i, _)| ids[*i].is_none())
            .map(|(i, u)| {
                let s = match u {
                    Unit::Run(s) | Unit::Piece(s) => s,
                };
                (i, encode_whole(tokenizer, s).map(Arc::from))
            })
            .collect();
        for (i, r) in fresh {
            let v = r?;
            if let Unit::Piece(s) = units[i] {
                self.insert(s, v.clone());
            }
            ids[i] = Some(v);
        }
        Ok(ids.into_iter().flatten().flat_map(|v| v.to_vec()).collect())
    }

    fn insert(&self, piece: &str, ids: Arc<[u32]>) {
        let cost = entry_bytes(piece.len(), ids.len());
        if cost > self.budget_bytes {
            return;
        }
        let mut t = self.table.lock();
        t.clock += 1;
        let now = t.clock;
        if t.map.contains_key(piece) {
            return;
        }
        while t.bytes + cost > self.budget_bytes {
            let Some(oldest) = t
                .map
                .iter()
                .min_by_key(|(_, e)| e.last_use)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            if let Some(e) = t.map.remove(&oldest) {
                t.bytes -= entry_bytes(oldest.len(), e.ids.len());
            }
        }
        t.bytes += cost;
        t.map.insert(Arc::from(piece), Entry { ids, last_use: now });
    }

    /// 2026-10-22: Bytes of piece text plus ids held now.
    pub fn held_bytes(&self) -> usize {
        self.table.lock().bytes
    }
}

fn entry_bytes(text_len: usize, n_ids: usize) -> usize {
    text_len + n_ids * std::mem::size_of::<u32>()
}

fn encode_whole(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {e}"))?
        .get_ids()
        .to_vec())
}

/// 2026-10-22: `text` cut at the tokenizer's own added-token split into memoized pieces of at
/// least [`MIN_PIECE_BYTES`] and the runs between them. A run starts and ends on that split's
/// boundaries, so encoding it alone finds the same added tokens.
fn split_units<'a>(tokenizer: &Tokenizer, text: &'a str) -> Vec<Unit<'a>> {
    let pre = tokenizer
        .get_added_vocabulary()
        .extract_and_normalize(tokenizer.get_normalizer(), text);
    let splits = pre.get_splits(OffsetReferential::Original, OffsetType::Byte);
    // 2026-10-22: A normalizer that drops characters leaves gaps between the splits' original
    // offsets; such a text is one run.
    let contiguous = splits
        .iter()
        .try_fold(0usize, |at, (_, (s, e), _)| (*s == at).then_some(*e))
        == Some(text.len());
    if !contiguous {
        return vec![Unit::Run(text)];
    }
    let mut units = Vec::new();
    let mut run_start = 0usize;
    for (_, (start, end), tokens) in splits {
        if tokens.is_none() && end - start >= MIN_PIECE_BYTES {
            if run_start < start {
                units.push(Unit::Run(&text[run_start..start]));
            }
            units.push(Unit::Piece(&text[start..end]));
            run_start = end;
        }
    }
    if run_start < text.len() {
        units.push(Unit::Run(&text[run_start..]));
    }
    units
}

#[cfg(test)]
#[path = "segment_memo_tests.rs"]
mod tests;
