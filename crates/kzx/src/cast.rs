// SPDX-License-Identifier: MIT OR Apache-2.0

//! The two narrowing casts the hot paths use, each justified once here so the cast lints stay
//! denied everywhere else.

/// Every supported target has a `usize` of at least 32 bits (wasm32 is the floor), so a `u32`
/// index converts losslessly. The const assertion makes a 16-bit target fail to build instead
/// of truncating silently.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub fn ix(i: u32) -> usize {
    const _: () = assert!(usize::BITS >= 32, "kzx needs a 32-bit or wider usize");
    i as usize
}

/// A table index computed in `i32` arithmetic that the caller has already clamped into a
/// non-negative range (the clamp is at the call site, this only reinterprets the bits).
#[inline(always)]
#[allow(clippy::cast_sign_loss)]
pub fn ixi(i: i32) -> usize {
    ix(i as u32)
}

/// The low 32 bits of a `u64` hash: hashing deliberately keeps only these.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub fn lo32(v: u64) -> u32 {
    v as u32
}

/// A `usize` byte count as `u64`: lossless on every target (`usize` is never wider than 64).
#[inline(always)]
pub fn u64_of(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}
