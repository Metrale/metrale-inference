// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The sealed section's wire types (DESIGN-encrypted-kernels.md §2.2): which modules
//! of the body travel encrypted, to whom their content key is wrapped, and the signed length and
//! hash of every ciphertext.
//!
//! ```text
//! sealed section = postcard SealedBlock { manifest: postcard SealedManifest, sig: SigBlock }
//! signed bytes   = "kbp-sealed-v1\0" ‖ manifest     (Purpose::KbpSealed, scope Blueprints)
//! ```
//!
//! Owner: blueprint.
//! Invariants (the `wire-types` gate covers this file):
//! - Sealed-ness is a property of a module stated here, never a locator variant or flag: module
//!   `i` is sealed iff `artifacts` has an entry with `module == i`.
//! - The key, fingerprint, wrap and suite types are `metrale_seal`'s own (SSOT).
//! - Nothing here names the customer: `customer` is an opaque tag of the id, never the id.

use alloc::vec::Vec;

use metrale_blueprint_trust::SigBlock;
use metrale_seal::{Fingerprint, SealSuite, WrappedCek};
use serde::{Deserialize, Serialize};

use crate::codec::Codec;
use crate::ids::{BlueprintId, Sha256};

/// 2026-10-22: The sealed manifest schema. Bumped with every change of its shape.
pub const SEALED_SCHEMA: u32 = 1;

/// 2026-10-22: The prefix of the bytes the sealed signature covers.
pub const SEALED_SIG_PREFIX: &[u8] = b"kbp-sealed-v1\0";

/// 2026-10-22: The sealed section as it travels: the manifest bytes exactly as signed, and their
/// signature.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedBlock {
    /// 2026-10-22: postcard [`SealedManifest`], canonical.
    pub manifest: Vec<u8>,
    /// 2026-10-22: The signature over [`SEALED_SIG_PREFIX`] ‖ `manifest`.
    pub sig: SigBlock,
}

/// 2026-10-22: Everything a reader without a key may know about the sealed modules.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedManifest {
    /// 2026-10-22: [`SEALED_SCHEMA`]; the first field, so a newer manifest is refused before
    /// the rest decodes.
    pub schema: u32,
    /// 2026-10-22: The body this section belongs to; must equal the file's `BlueprintId`.
    pub blueprint: BlueprintId,
    /// 2026-10-22: The wrap and AEAD suite.
    pub suite: SealSuite,
    /// 2026-10-22: The codec every sealed artifact was compressed with before sealing.
    pub codec: Codec,
    /// 2026-10-22: `sha256("kbp-customer\0" ‖ lv(customer id))`: an opaque tag, never the id.
    pub customer: Sha256,
    /// 2026-10-22: When the pipeline sealed it (its Clock port), Unix ms; reported, never a
    /// decision input.
    pub issued_ms: i64,
    /// 2026-10-22: At least one, strictly sorted by fingerprint.
    pub recipients: Vec<Recipient>,
    /// 2026-10-22: At least one, strictly sorted by module index.
    pub artifacts: Vec<SealedArtifact>,
}

/// 2026-10-22: One recipient: its fingerprint and the content key wrapped to it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recipient {
    /// 2026-10-22: SHA-256 of the recipient's public key bytes.
    pub fingerprint: Fingerprint,
    /// 2026-10-22: The content key, wrapped to that key.
    pub wrap: WrappedCek,
}

/// 2026-10-22: One sealed module's ciphertext, as CI asserts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedArtifact {
    /// 2026-10-22: Index into `KernelSet.modules`.
    pub module: u32,
    /// 2026-10-22: The compressed (pre-AEAD) length.
    pub compressed_len: u64,
    /// 2026-10-22: The ciphertext length, `compressed_len + 16`.
    pub ct_len: u64,
    /// 2026-10-22: SHA-256 of the ciphertext: its content address in the public store.
    pub ct_sha256: Sha256,
}
