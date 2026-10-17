// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Lower-case hex for the digests this crate computes, spelled exactly as
//! `metrale_closure::hex_lower` spells them (closure needs `std`, so this crate cannot reach
//! it; the parity test pins the two together).
//!
//! Owner: serve identity.
//! Invariants: two lower-case digits per byte, no separator, no prefix.

use alloc::string::String;
use core::fmt::Write as _;

/// 2026-10-17: `bytes` as lower-case hex.
#[must_use]
pub fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        // 2026-10-17: Writing into a `String` cannot fail.
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::hex_lower;
    use sha2::{Digest, Sha256};

    /// 2026-10-17: FIPS 180-2's "abc" vector, and every byte value spelled as metrale-closure
    /// spells it, so a digest from either crate compares equal.
    #[test]
    fn the_spelling_matches_the_published_vector_and_metrale_closure() {
        assert_eq!(
            hex_lower(&Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let every: Vec<u8> = (0..=255u8).collect();
        assert_eq!(hex_lower(&every), metrale_closure::hex_lower(&every));
        assert_eq!(hex_lower(&[]), "");
    }
}
