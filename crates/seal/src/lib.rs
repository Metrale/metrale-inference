// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Sealed kernels: per-customer encryption of `.kbp` kernel artifacts
//! (`claude-state/glm/blueprint/DESIGN-encrypted-kernels.md`).
//!
//! - [`primitives`]: HKDF-SHA-256, AES-256-GCM, ML-KEM-768 and X25519, thin and public for KATs;
//! - [`hybrid`]: the recipient key ([`HybridSecret`] from a 32-byte seed, [`HybridPublic`],
//!   [`Fingerprint`]): one key type for customers and the store's KeyRing;
//! - [`wrap`]: the content key ([`Cek`]), one run's randomness ([`SealRandom`]) and the hybrid
//!   X25519 + ML-KEM-768 wrap (X-Wing ordering, the KKB1 construction);
//! - [`artifact`]: per-artifact HKDF key and nonce and the AAD-bound AES-256-GCM seal/open;
//! - [`labels`]: the versioned contexts and [`lv_cat`].
//!
//! This crate is the single source of the primitives and the key type (owner decision
//! 2026-10-22); `metprof-envelope` adopts it once its branch lands.
//!
//! Owner: blueprint (sealed kernels).
//! Invariants:
//! - `no_std` + `alloc`, no `unsafe`, no I/O, no clock; builds for both wasm32 targets (NSCC),
//!   so the store opens sealed kernels in a Worker with this same code.
//! - Randomness enters only as values drawn from the `Entropy` port ([`SealRandom::draw`], one
//!   `fill` per run); `getrandom` is never reached.
//! - Secret material is `Zeroizing` and its `Debug` is redacted.

#![no_std]
#![forbid(unsafe_code)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;

pub mod artifact;
mod fixed;
pub mod hybrid;
pub mod labels;
pub mod primitives;
pub mod wrap;

pub use artifact::{ArtifactBinding, ArtifactKey, open_artifact, seal_artifact};
pub use hybrid::{Fingerprint, HybridPublic, HybridSecret, PUBLIC_LEN, SECRET_LEN, SEED_LEN, hex};
pub use labels::lv_cat;
pub use wrap::{
    Cek, SealRandom, WRAP_RANDOM_LEN, WRAPPED_CT_LEN, WrapLabel, WrapRandom, WrappedCek,
    unwrap_key, wrap_key,
};

/// 2026-10-22: The sealed-kernel suites. A new suite (XChaCha20-Poly1305, libcrux ML-KEM) is a
/// new variant, never a format change.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum SealSuite {
    /// 2026-10-22: X25519 + ML-KEM-768 wrap (HKDF-SHA-256 combiner) and AES-256-GCM.
    HybridX25519MlKem768HkdfA256Gcm,
}

impl SealSuite {
    /// 2026-10-22: The suite id (the KKB1 `WRAP_SCHEME` spelling).
    pub const fn name(self) -> &'static str {
        match self {
            SealSuite::HybridX25519MlKem768HkdfA256Gcm => {
                labels::SUITE_HYBRID_X25519_MLKEM768_HKDF_A256GCM
            }
        }
    }
}
