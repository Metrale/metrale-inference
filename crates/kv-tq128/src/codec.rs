// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The four 128-group codecs and everything that differs between them: block
//! layout, codebook per side, decode scale per side, trellis shape.
//!
//! Owner: kv-tq128.
//! Invariants: a block is the group's fp16 norm (little-endian) followed by its index payload,
//! zero-padded to an even length; `block_bytes() * 8 / 128` is the codec's bits per value.

use crate::codebooks::{TCQ1_K, TCQ1_V, TCQ2_K, TCQ2_V, TCQ3_K, TCQ3_V, TURBO4_LEVELS};
use crate::trellis::Trellis;

/// 2026-10-18: Elements per group: one norm, one rotation, one block.
pub const GROUP: usize = 128;

/// 2026-10-18: Which half of the KV cache a block holds. The TCQ codecs train a separate book
/// per side and scale V at decode; turbo4_n128 treats both sides alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvSide {
    Key,
    Value,
}

/// 2026-10-18: One of the 128-group rotated codecs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tq128Codec {
    /// turbo3_tcq: 3-bit symbols on a 512-state trellis, 3.25 bits per value.
    Tcq3,
    /// turbo2_tcq: 2-bit symbols on a 256-state trellis, 2.25 bits per value.
    Tcq2,
    /// turbo1_tcq: 1-bit symbols on a 256-state trellis, 1.25 bits per value.
    Tcq1,
    /// turbo4_n128: a 4-bit scalar index per value, 4.125 bits per value.
    Turbo4N128,
}

impl Tq128Codec {
    /// 2026-10-18: Every codec, in the order the docs list them.
    pub const ALL: [Self; 4] = [Self::Tcq3, Self::Tcq2, Self::Tcq1, Self::Turbo4N128];

    /// 2026-10-18: The codec's cache dtype name (`--kv-cache-dtype` spells `_` as `-`).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Tcq3 => "turbo3_tcq",
            Self::Tcq2 => "turbo2_tcq",
            Self::Tcq1 => "turbo1_tcq",
            Self::Turbo4N128 => "turbo4_n128",
        }
    }

    /// 2026-10-18: The trellis a TCQ codec quantizes along; `None` for the scalar codec.
    pub const fn trellis(self) -> Option<Trellis> {
        match self {
            Self::Tcq3 => Some(Trellis::TCQ3),
            Self::Tcq2 => Some(Trellis::TCQ2),
            Self::Tcq1 => Some(Trellis::TCQ1),
            Self::Turbo4N128 => None,
        }
    }

    /// 2026-10-18: Bytes of index payload: the trellis bitstream (prefix plus 128 symbols,
    /// rounded up to a byte) or 128 nibbles.
    pub const fn payload_bytes(self) -> usize {
        match self.trellis() {
            Some(t) => t.stream_bits(GROUP).div_ceil(8),
            None => GROUP / 2,
        }
    }

    /// 2026-10-18: Bytes per encoded group: the 2-byte norm plus the payload, padded to even.
    pub const fn block_bytes(self) -> usize {
        (2 + self.payload_bytes()).next_multiple_of(2)
    }

    /// 2026-10-18: Entries in the codebook an index selects from.
    pub const fn codebook_len(self) -> usize {
        match self.trellis() {
            Some(t) => t.states(),
            None => TURBO4_LEVELS.len(),
        }
    }

    /// 2026-10-18: The codebook for one side (credit spiritbuun; arXiv:2410.13780).
    pub const fn codebook(self, side: KvSide) -> &'static [f32] {
        match (self, side) {
            (Self::Tcq3, KvSide::Key) => &TCQ3_K,
            (Self::Tcq3, KvSide::Value) => &TCQ3_V,
            (Self::Tcq2, KvSide::Key) => &TCQ2_K,
            (Self::Tcq2, KvSide::Value) => &TCQ2_V,
            (Self::Tcq1, KvSide::Key) => &TCQ1_K,
            (Self::Tcq1, KvSide::Value) => &TCQ1_V,
            (Self::Turbo4N128, _) => &TURBO4_LEVELS,
        }
    }

    /// 2026-10-18: The factor decode multiplies the stored norm by. K decodes at 1.0. V of the
    /// TCQ codecs decodes slightly enlarged, undoing the trellis's shrinkage toward the book's
    /// centre; the factors are upstream's KL-tuned `TURBO_TCQ_ALPHA_V_T{1,2,3}`
    /// (turbo-tcq-alpha.cuh at 34ef0ef7c6ea; credit spiritbuun). turbo4_n128 decodes both
    /// sides at 1.0, as upstream's turbo4 does.
    pub const fn decode_scale(self, side: KvSide) -> f32 {
        match (self, side) {
            (_, KvSide::Key) | (Self::Turbo4N128, KvSide::Value) => 1.0,
            (Self::Tcq1, KvSide::Value) => 1.26,
            (Self::Tcq2, KvSide::Value) => 1.06,
            (Self::Tcq3, KvSide::Value) => 1.02,
        }
    }
}
