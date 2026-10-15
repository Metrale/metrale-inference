// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the KKB1 envelope (DESIGN §7.1–7.2).
//!
//! A publish draws one data-encryption key ([`Dek`]); each file is sealed
//! under a per-file key derived from it ([`seal`] / [`open`]), and the DEK is
//! wrapped to the KeyRing's hybrid X25519 + ML-KEM-768 public key
//! ([`wrap_dek`] / [`unwrap_dek`]). Randomness and time enter only through
//! the [`Entropy`] and [`Clock`] ports. Pure Rust; builds for
//! `wasm32-unknown-unknown`.

pub mod error;
pub mod format;
pub mod ports;
pub mod primitives;
pub mod seal;
pub mod wrap;

pub use error::{EnvelopeError, Result};
pub use format::{
    ALG, Header, KDF, MAGIC, Parsed, PlaintextInfo, VERSION, WRAP_SCHEME, WrappedDek, info_string,
    parse,
};
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use ports::SystemClock;
pub use ports::{Clock, Entropy, OsEntropy};
pub use seal::{Dek, Expected, Opened, SealRequest, derive_file_key, open, seal};
pub use wrap::{KeyRingPublic, KeyRingSecret, unwrap_dek, wrap_dek};
