// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: point ids from natural keys (DESIGN §2.1): the first 8 bytes,
//! big-endian, of `SHA-256("metrale.kkb/1\0" ‖ label ‖ "\0" ‖ natural_key)`.
//! Re-ingesting the same entity yields the same id; that is the dedup.

use sha2::{Digest, Sha256};

use crate::schema::{KEY_SEP, Kind, NodeRef, SCHEMA};

pub fn point_id(kind: Kind, natural_key: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(SCHEMA.as_bytes());
    h.update([0u8]);
    h.update(kind.label().as_bytes());
    h.update([0u8]);
    h.update(natural_key.as_bytes());
    let d = h.finalize();
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[..8]);
    u64::from_be_bytes(b)
}

pub fn node_id(n: &NodeRef) -> u64 {
    point_id(n.kind, &n.key)
}

/// A composite natural key: the parts joined by `‖`.
pub fn natural_key(parts: &[&str]) -> String {
    parts.join(KEY_SEP)
}

/// A point id as the 16-hex-digit text used inside other natural keys.
pub fn hex_id(id: u64) -> String {
    format!("{id:016x}")
}
