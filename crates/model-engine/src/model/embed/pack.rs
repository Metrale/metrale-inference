// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: The pure half of `embed_batch`: input validation and the packing of inputs,
//! in order and without padding, into forwards of at most `budget` token rows.
//!
//! Owner: model-engine (embedding model).
//! Invariants:
//! - Packs are contiguous, in input order, cover every input exactly once, and each holds at
//!   most `budget` tokens.

use std::ops::Range;

use anyhow::Result;
use metrale_error::{ErrorCode, error};

/// 2026-10-24: Refuse an empty input, one longer than `max_tokens`, and a token id outside
/// the embedding table (the gather kernel does not range-check ids).
pub fn validate_inputs(inputs: &[Vec<u32>], max_tokens: usize, vocab_rows: usize) -> Result<()> {
    for (i, ids) in inputs.iter().enumerate() {
        if ids.is_empty() {
            return Err(error!(ErrorCode::EmbedInputInvalid, i, "it has no tokens").into());
        }
        if ids.len() > max_tokens {
            return Err(error!(
                ErrorCode::EmbedInputInvalid,
                i,
                format!("{} tokens exceed the model's {max_tokens}", ids.len())
            )
            .into());
        }
        if let Some(&bad) = ids.iter().find(|&&t| t as usize >= vocab_rows) {
            return Err(error!(
                ErrorCode::EmbedInputInvalid,
                i,
                format!("token id {bad} is outside the {vocab_rows}-row embedding table")
            )
            .into());
        }
    }
    Ok(())
}

/// 2026-10-24: Greedy packing in input order: a pack closes when the next input would take it
/// past `budget`. An input longer than `budget` is an error (validate first).
pub fn plan_packs(lens: &[usize], budget: usize) -> Result<Vec<Range<usize>>> {
    let mut packs = Vec::new();
    let (mut start, mut rows) = (0usize, 0usize);
    for (i, &n) in lens.iter().enumerate() {
        if n > budget {
            return Err(error!(
                ErrorCode::EmbedInputInvalid,
                i,
                format!("{n} tokens exceed the pack budget {budget}")
            )
            .into());
        }
        if rows + n > budget {
            packs.push(start..i);
            start = i;
            rows = 0;
        }
        rows += n;
    }
    if start < lens.len() {
        packs.push(start..lens.len());
    }
    Ok(packs)
}

#[cfg(test)]
#[path = "pack_tests.rs"]
mod tests;
