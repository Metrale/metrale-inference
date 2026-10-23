// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The one hash of the knowledge graph: sha256 (delegated decision 2026-10-23, in
//! place of the design's blake3). A record's `sha`, a node's u64 point id and the bundle digest
//! all come from here.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - [`point_id`] is the first 8 bytes of `sha256(id)`, little-endian; the bundle asserts that no
//!   two ids share one (`KgIdCollision`).

use crate::id::KgId;
use crate::prelude::*;
use sha2::{Digest, Sha256 as Hasher};

/// 2026-10-23: A sha256 digest, shown as 64 lowercase hex digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256(pub [u8; 32]);

impl Sha256 {
    /// 2026-10-23: The 64-digit lowercase hex form.
    pub fn hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push(char::from(HEX[usize::from(b >> 4)]));
            s.push(char::from(HEX[usize::from(b & 0xf)]));
        }
        s
    }
}

/// 2026-10-23: The sha256 of `bytes`.
pub fn sha256(bytes: &[u8]) -> Sha256 {
    let out = Hasher::digest(bytes);
    let mut a = [0u8; 32];
    a.copy_from_slice(&out);
    Sha256(a)
}

/// 2026-10-23: The sha of a record's canonical text.
pub fn record_sha(text: &str) -> Sha256 {
    sha256(text.as_bytes())
}

/// 2026-10-23: The LatticeDB point id of a typed id: `u64::from_le_bytes(sha256(id)[..8])`.
pub fn point_id(id: &KgId) -> u64 {
    let d = sha256(id.to_string().as_bytes());
    let mut b = [0u8; 8];
    b.copy_from_slice(&d.0[..8]);
    u64::from_le_bytes(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_the_published_test_vector() {
        assert_eq!(
            record_sha("abc").hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn point_id_is_the_little_endian_prefix_of_the_id_hash() {
        let id = KgId::parse("ledger:l2-prefetch-6").unwrap();
        let d = record_sha("ledger:l2-prefetch-6");
        let mut b = [0u8; 8];
        b.copy_from_slice(&d.0[..8]);
        assert_eq!(point_id(&id), u64::from_le_bytes(b));
        assert_ne!(point_id(&id), u64::from_be_bytes(b));
    }
}
