// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Pure block-causal visibility contract for Qwen Image 2.1.
//! This describes visibility only, not an executable attention kernel or model.
//! Image blocks are bidirectional; text remains causal. No dense mask is allocated.

use std::collections::BTreeSet;

/// 2026-10-07: One position in a flattened, sample-contiguous joint sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    /// 2026-10-07: Request/sample identity. Attention never crosses this boundary.
    pub sample: u32,
    /// 2026-10-07: Image block identity within the sample; `None` means text.
    pub image: Option<u32>,
    /// 2026-10-07: Padding masks keys, not queries, matching the pinned reference.
    pub key_valid: bool,
}

/// 2026-10-07: Validated contiguous sample and image-block layout.
#[derive(Debug)]
pub struct Layout {
    tokens: Vec<Token>,
}

/// 2026-10-07: Stable valid-key gathers plus each query's visible packed prefix.
/// Offsets address the gather array, never the original sequence.
#[derive(Debug, PartialEq, Eq)]
pub struct PackedPrefixes {
    pub gather: Vec<u32>,
    pub spans: Vec<(u32, u32)>,
}

/// 2026-10-07: Malformed layouts and indices must fail before attention planning.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum LayoutError {
    /// 2026-10-07: No token positions supplied.
    #[error("image attention layout is empty")]
    Empty,
    /// 2026-10-07: A sample or image block reappeared after another segment.
    #[error("image attention layout repeats a noncontiguous sample or image block")]
    Noncontiguous,
    /// 2026-10-07: Query or key position does not belong to the layout.
    #[error("image attention index is out of range")]
    Index,
}

impl Layout {
    /// 2026-10-07: Validate segmentation once, without allocating a quadratic mask.
    pub fn new(tokens: Vec<Token>) -> Result<Self, LayoutError> {
        if tokens.is_empty() {
            return Err(LayoutError::Empty);
        }
        let mut samples = BTreeSet::new();
        let mut images = BTreeSet::new();
        let mut previous: Option<Token> = None;
        for token in &tokens {
            if previous.is_none_or(|p| p.sample != token.sample) && !samples.insert(token.sample) {
                return Err(LayoutError::Noncontiguous);
            }
            if let Some(image) = token.image
                && previous.is_none_or(|p| p.sample != token.sample || p.image != token.image)
                && !images.insert((token.sample, image))
            {
                return Err(LayoutError::Noncontiguous);
            }
            previous = Some(*token);
        }
        Ok(Self { tokens })
    }

    /// 2026-10-07: Reference rule: same sample AND valid key AND
    /// (causal position OR same non-text image block). This does not lower a kernel.
    pub fn can_attend(&self, query: usize, key: usize) -> Result<bool, LayoutError> {
        let q = self.tokens.get(query).ok_or(LayoutError::Index)?;
        let k = self.tokens.get(key).ok_or(LayoutError::Index)?;
        Ok(q.sample == k.sample
            && k.key_valid
            && (query >= key || (q.image.is_some() && q.image == k.image)))
    }

    /// 2026-10-07: Linear storage/scan, preserving visibility without a dense mask.
    /// Contiguous image blocks make every row a prefix of its sample's valid keys.
    pub fn packed_prefixes(&self) -> Result<PackedPrefixes, LayoutError> {
        if self.tokens.len() > u32::MAX as usize {
            return Err(LayoutError::Index);
        }
        let mut result = PackedPrefixes {
            gather: Vec::new(),
            spans: Vec::new(),
        };
        let mut start = 0;
        while start < self.tokens.len() {
            let mut end = start + 1;
            while end < self.tokens.len() && self.tokens[end].sample == self.tokens[start].sample {
                end += 1;
            }
            let packed_start = result.gather.len() as u32;
            let mut prefix = vec![0u32];
            for (i, token) in self.tokens[start..end].iter().enumerate() {
                if token.key_valid {
                    result.gather.push((start + i) as u32);
                }
                prefix.push(result.gather.len() as u32 - packed_start);
            }
            let mut query = start;
            while query < end {
                let mut block_end = query + 1;
                if self.tokens[query].image.is_some() {
                    while block_end < end
                        && self.tokens[block_end].image == self.tokens[query].image
                    {
                        block_end += 1;
                    }
                }
                for _ in query..block_end {
                    result.spans.push((packed_start, prefix[block_end - start]));
                }
                query = block_end;
            }
            start = end;
        }
        Ok(result)
    }
}
