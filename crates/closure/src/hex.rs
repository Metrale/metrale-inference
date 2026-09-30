// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Lower-case hex, the one spelling every SHA-256 this workspace records uses.
//!
//! Owner: metrale-closure (the lowest crate that hashes; metrale-bench reaches it too).
//! Invariants: two lower-case digits per byte, no separator, no prefix. sha2 0.11's output
//! (`hybrid_array::Array`) has no `LowerHex`, so `format!("{:x}", ..)` no longer compiles on a
//! digest; every recorded digest keeps the spelling it had under 0.10 by going through here.

use std::fmt::Write as _;

/// 2026-09-30: `bytes` as lower-case hex.
pub fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::hex_lower;
    use sha2::{Digest, Sha256};

    /// 2026-09-30: FIPS 180-2's "abc" vector, so a record digest is spelled exactly as before.
    #[test]
    fn a_sha256_digest_reads_as_the_published_vector() {
        assert_eq!(
            hex_lower(&Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(hex_lower(&[0x00, 0x0f, 0xa0]), "000fa0");
        assert_eq!(hex_lower(&[]), "");
    }
}
