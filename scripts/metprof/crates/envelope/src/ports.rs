// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the two inputs that are not plain values, behind ports. Every
//! random byte the crate uses (DEKs, nonces, ephemeral X25519 keys, ML-KEM
//! encapsulation randomness, KeyRing keys) is drawn from an [`Entropy`]; the
//! header's `created_at` comes from a [`Clock`]. Tests pass deterministic
//! fakes; production passes [`OsEntropy`] and [`SystemClock`].

use crate::error::{EnvelopeError, Result};

/// A source of cryptographically secure random bytes.
pub trait Entropy {
    /// Fills `buf` entirely, or fails. A partial fill is never returned as success.
    fn fill(&mut self, buf: &mut [u8]) -> Result<()>;
}

/// The time a header is stamped with.
pub trait Clock {
    /// Whole seconds since the Unix epoch, UTC.
    fn now_unix_seconds(&self) -> Result<u64>;
}

/// The operating system's CSPRNG (`getrandom`; the Web Crypto API on
/// `wasm32-unknown-unknown`).
#[derive(Debug, Clone, Copy)]
pub struct OsEntropy;

impl Entropy for OsEntropy {
    fn fill(&mut self, buf: &mut [u8]) -> Result<()> {
        getrandom::getrandom(buf).map_err(|e| EnvelopeError::Entropy(e.to_string()))
    }
}

/// The system wall clock. Not available on `wasm32-unknown-unknown`, which
/// has no clock in `std`; a browser or Worker caller implements [`Clock`].
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[derive(Debug, Clone, Copy)]
pub struct SystemClock;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl Clock for SystemClock {
    fn now_unix_seconds(&self) -> Result<u64> {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .map_err(|e| EnvelopeError::Clock(e.to_string()))
    }
}
