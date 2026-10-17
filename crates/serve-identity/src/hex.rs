// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Lower-case hex, the one spelling every SHA-256 this workspace records uses.
//! `metrale_closure::hex_lower` re-exports it, so a digest from either crate compares equal.
//!
//! 2026-10-17: Moved from metrale-closure (2026-09-30), which needs `std`, into this no_std
//! crate, the lowest one that hashes. sha2 0.11's output (`hybrid_array::Array`) has no
//! `LowerHex`, so `format!("{:x}", ..)` does not compile on a digest; every recorded digest
//! keeps the spelling it had under 0.10 by going through here.
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

    /// 2026-10-17: FIPS 180-2's "abc" vector, so a record digest is spelled exactly as before;
    /// a leading zero nibble keeps its digit and letters are lower case.
    #[test]
    fn a_sha256_digest_reads_as_the_published_vector() {
        assert_eq!(
            hex_lower(&Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(hex_lower(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
        assert_eq!(hex_lower(&[]), "");
    }
}
