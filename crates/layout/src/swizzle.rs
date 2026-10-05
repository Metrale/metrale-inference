// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: XOR swizzles: `o ^ (((o >> (m + s)) & (2^b - 1)) << m)`. A field of `b` bits at
//! `m + s` is XORed into the field at `m`. On bits that is an invertible linear map over GF(2)
//! (it is its own inverse), so it permutes offsets inside each aligned block of `2^(m + s + b)`
//! and moves none across blocks. Shared-memory tiles use it so that the rows a warp reads land
//! in different banks.
//!
//! Owner: metrale-layout.
//! Invariants: `s >= b`, so the source field and the target field do not overlap.

use crate::layout::{Layout, LayoutError};

/// 2026-10-05: A swizzle: `b` bits, base `m`, shift `s`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Swizzle {
    /// 2026-10-05: Width of the XORed field, bits.
    pub b: u32,
    /// 2026-10-05: Lowest bit of the target field (units that move together: `2^m` offsets).
    pub m: u32,
    /// 2026-10-05: Distance from the target field to the source field, bits.
    pub s: u32,
}

impl Swizzle {
    /// 2026-10-05: A swizzle whose fields do not overlap.
    pub fn new(b: u32, m: u32, s: u32) -> Result<Self, LayoutError> {
        if s < b {
            return Err(LayoutError(format!(
                "swizzle: shift {s} < width {b} overlaps the fields"
            )));
        }
        Ok(Swizzle { b, m, s })
    }

    /// 2026-10-05: The identity (no bits).
    pub const NONE: Swizzle = Swizzle { b: 0, m: 0, s: 0 };

    /// 2026-10-05: Apply to an offset.
    pub fn apply(self, o: u64) -> u64 {
        let mask = (1u64 << self.b) - 1;
        o ^ (((o >> (self.m + self.s)) & mask) << self.m)
    }

    /// 2026-10-05: The offsets of `l` after the swizzle (negative offsets are refused).
    pub fn offsets(self, l: &Layout) -> Result<Vec<u64>, LayoutError> {
        l.offsets()
            .into_iter()
            .map(|o| {
                u64::try_from(o)
                    .map(|o| self.apply(o))
                    .map_err(|_| LayoutError("swizzle: negative offset".into()))
            })
            .collect()
    }
}
