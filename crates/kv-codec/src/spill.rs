// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The spill salt: what a cache's codecs contribute to the KV peer-paging namespace
//! (`derive_kv_ns` in metrale-storage). The namespace already folds the model and the block
//! geometry, but not what the bytes of a block mean, and two codecs can share a byte layout
//! (nvfp4 and turbo4 store the same bits per value and the same scales). Folding the salt
//! makes a block spilled under one codec a miss, never a misread, under any other.
//!
//! Owner: kv-codec.
//! Invariants:
//! - The salt is an FNV-1a 64 hash over fixed-width tagged records, so it is a pure function of
//!   the records, in order.
//! - A side record holds the layer, the side, the codec id, the codec's spec digest and the
//!   VBR rung, so a change to any of them changes the salt.

use crate::plan::{KvSide, KvSidePlan, LayerIx};
use crate::spec::Fnv;
use crate::{KvCodecId, spec};

/// 2026-10-18: Encoding version, hashed first. A change to the records or their order must
/// bump it.
pub const SPILL_SALT_VERSION: u8 = 1;

/// 2026-10-18: Builds a spill salt one record at a time.
#[derive(Clone, Copy)]
pub struct SpillSalt(Fnv);

impl SpillSalt {
    /// 2026-10-18: A salt with no records.
    pub const fn new() -> Self {
        SpillSalt(Fnv::new().byte(SPILL_SALT_VERSION))
    }

    /// 2026-10-18: Folds one side of one attention layer: its codec, that codec's layout
    /// digest, and its rung (0 without VBR).
    pub const fn side(self, layer: LayerIx, side: KvSide, plan: KvSidePlan) -> Self {
        SpillSalt(
            self.0
                .byte(0x01)
                .word(layer.0 as u64)
                .byte(side as u8)
                .byte(plan.codec as u8)
                .word(spec(plan.codec).digest())
                .byte(plan.rung.0),
        )
    }

    /// 2026-10-18: Folds the shape of one attention layer, for caches whose layers differ in
    /// shape.
    pub const fn layer_shape(self, layer: LayerIx, num_kv_heads: u64, head_dim: u64) -> Self {
        SpillSalt(
            self.0
                .byte(0x02)
                .word(layer.0 as u64)
                .word(num_kv_heads)
                .word(head_dim),
        )
    }

    /// 2026-10-18: Folds the sparse-attention indexer rows a cache keeps in its blocks: the
    /// indexer codec (`None` when the rows are not in the blocks) and the block bytes they take
    /// (`None` when they do not resize the block).
    pub const fn indexer(self, codec: Option<KvCodecId>, block_bytes: Option<u64>) -> Self {
        let h = match codec {
            None => self.0.byte(0x03).byte(0),
            Some(id) => self
                .0
                .byte(0x03)
                .byte(1)
                .byte(id as u8)
                .word(spec(id).digest()),
        };
        SpillSalt(match block_bytes {
            None => h.byte(0),
            Some(b) => h.byte(1).word(b),
        })
    }

    /// 2026-10-18: The salt.
    pub const fn finish(self) -> u64 {
        self.0.finish()
    }
}

impl Default for SpillSalt {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "spill_tests.rs"]
mod tests;
