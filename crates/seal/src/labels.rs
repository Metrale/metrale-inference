// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The versioned contexts of sealed kernels (DESIGN-encrypted-kernels.md §1.3) and
//! the suite id.
//!
//! Owner: blueprint (sealed kernels).
//! Invariants:
//! - Every context is versioned; a change of construction is a new `/v2` string, never an edit.
//! - Every variable input that joins a context goes through [`lv_cat`], so no two input lists
//!   share an encoding.
//! - The sealed-manifest signature context (`metrale/kbp-sealed/v1`) is
//!   `metrale_blueprint_trust::Purpose::KbpSealed`, beside the other signing contexts.

use alloc::vec::Vec;

/// 2026-10-22: Expands a 32-byte recipient seed into the 96-byte hybrid secret (HKDF salt).
pub const RECIPIENT: &[u8] = b"metrale/kbp-recipient/v1";
/// 2026-10-22: The HKDF info of the seed expansion: what the 96 bytes are.
pub const RECIPIENT_INFO: &[u8] = b"hybrid-x25519-mlkem768";
/// 2026-10-22: The content-key wrap: HKDF salt of the key-encryption key and first part of the
/// wrap's AAD.
pub const CEK: &[u8] = b"metrale/kbp-cek/v1";
/// 2026-10-22: The per-artifact key and nonce (HKDF salt) and first part of the artifact AAD.
pub const SEAL: &[u8] = b"metrale/kbp-seal/v1";
/// 2026-10-22: The suite id of the one implemented suite (KKB1's `WRAP_SCHEME` spelling).
pub const SUITE_HYBRID_X25519_MLKEM768_HKDF_A256GCM: &str = "hybrid-x25519-mlkem768-hkdf-a256gcm";

/// 2026-10-22: Length-value concatenation: each part as a big-endian `u32` length then its
/// bytes (metrale-auth `encoding::lv_cat`), so no two part lists share an encoding. The one
/// definition: `metrale_blueprint_trust::lv_cat` re-exports it (SSOT).
pub fn lv_cat(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.len() + 4).sum());
    for p in parts {
        // 2026-10-22: Every caller's parts are far below 4 GiB (labels, ids, fingerprints, and
        // blueprint bodies capped at 64 MiB raw).
        let len = u32::try_from(p.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(p);
    }
    out
}
