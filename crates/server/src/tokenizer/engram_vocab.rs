// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-19: The DeepSeek-V4.1 Engram compressed token map, ported from the publisher's
//! `inference/engram.py` (`build_compressed_token_map`). Every token id maps onto a smaller id
//! space in which tokens that normalize alike collapse together, so " The", "the" and "THE"
//! hash the same way. The map is not stored in the checkpoint; it is derived from
//! `tokenizer.json` once at load.
//!
//! Owner: server (tokenizer).
//! Invariants:
//! - Ids are assigned in first-seen order over token ids `0..len`, `len` being the vocab size
//!   with added tokens (the publisher's `len(tokenizer)`).
//! - A token whose decoded text contains U+FFFD (a partial UTF-8 byte token) is keyed by its
//!   raw token string; a text that normalizes to nothing is keyed by itself.
//! - The compressed size must equal the config's `engram_compressed_vocab_size`: every hash
//!   multiplier derives from it, so a mismatch is an error, never a silent rehash.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use tokenizers::normalizers::{
    NFD, NFKC, Sequence, Strip, StripAccents, replace::Replace, utils::Lowercase,
};
use tokenizers::{NormalizedString, Normalizer, NormalizerWrapper, Tokenizer};

/// 2026-10-19: A private-use character standing in for a token that is exactly one space, so
/// `Strip` does not empty it and merge it with unrelated tokens.
const SENTINEL: &str = "\u{e000}";

/// 2026-10-19: The token id -> compressed id lookup and the compressed vocab size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedTokenMap {
    lookup: Vec<u32>,
    size: u32,
}

impl CompressedTokenMap {
    /// 2026-10-19: Build the map from `tokenizer` and require its size to be `expected_size`
    /// (the config's `engram_compressed_vocab_size`).
    pub fn build(tokenizer: &Tokenizer, expected_size: u64) -> Result<Self> {
        let normalizer = engram_normalizer()?;
        let len = u32::try_from(tokenizer.get_vocab_size(true))
            .context("tokenizer vocab does not fit u32 ids")?;
        let mut key_to_new: HashMap<String, u32> = HashMap::new();
        let mut lookup = Vec::with_capacity(len as usize);
        for id in 0..len {
            let text = tokenizer
                .decode(&[id], false)
                .map_err(|e| anyhow::anyhow!("decode token {id}: {e}"))?;
            let key = if text.contains('\u{fffd}') {
                tokenizer
                    .id_to_token(id)
                    .with_context(|| format!("token {id} decodes but has no token string"))?
            } else {
                let mut normalized = NormalizedString::from(text.as_str());
                normalizer
                    .normalize(&mut normalized)
                    .map_err(|e| anyhow::anyhow!("normalize token {id}: {e}"))?;
                match normalized.get() {
                    "" => text,
                    n => n.to_owned(),
                }
            };
            let next = u32::try_from(key_to_new.len()).context("compressed id overflow")?;
            lookup.push(*key_to_new.entry(key).or_insert(next));
        }
        let size = u32::try_from(key_to_new.len()).context("compressed id overflow")?;
        ensure!(
            u64::from(size) == expected_size,
            "Engram compressed vocab is {size} but the config declares {expected_size}: the hash \
             multipliers derive from it, so this tokenizer does not match the checkpoint"
        );
        Ok(Self { lookup, size })
    }

    /// 2026-10-19: The compressed id of every token id, indexed by token id.
    pub fn lookup(&self) -> &[u32] {
        &self.lookup
    }

    /// 2026-10-19: The number of distinct compressed ids.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// 2026-10-19: The compressed id of token `id` (the pad fill is `get(engram_pad_token_id)`).
    pub fn get(&self, id: u32) -> Result<u32> {
        self.lookup.get(id as usize).copied().with_context(|| {
            format!(
                "token id {id} is outside the {}-token vocab",
                self.lookup.len()
            )
        })
    }
}

/// 2026-10-19: The publisher's normalizer sequence, in its order.
fn engram_normalizer() -> Result<Sequence> {
    let replace = |pattern: tokenizers::normalizers::replace::ReplacePattern, content: &str| {
        Replace::new(pattern, content)
            .map(NormalizerWrapper::from)
            .map_err(|e| anyhow::anyhow!("engram normalizer pattern: {e}"))
    };
    use tokenizers::normalizers::replace::ReplacePattern::{Regex, String as Literal};
    Ok(Sequence::new(vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        replace(Regex(r"[ \t\r\n]+".into()), " ")?,
        replace(Regex(r"^ $".into()), SENTINEL)?,
        Strip::new(true, true).into(),
        replace(Literal(SENTINEL.into()), " ")?,
    ]))
}

#[cfg(test)]
#[path = "tests/engram_vocab.rs"]
mod tests;
