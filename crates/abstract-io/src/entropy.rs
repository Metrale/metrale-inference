// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The randomness port.
//!
//! Owner: metrale-abstract-io.
//! Invariants: `fill` either fills the whole buffer or returns an error.

use core::fmt;

/// 2026-10-17: Why an entropy source could not fill a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyError {
    /// The host's source of randomness failed or does not exist.
    Unavailable,
}

impl fmt::Display for EntropyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => f.write_str("entropy source unavailable"),
        }
    }
}

impl core::error::Error for EntropyError {}

/// 2026-10-17: A source of random bytes, supplied by a host.
pub trait Entropy {
    fn fill(&self, buf: &mut [u8]) -> Result<(), EntropyError>;
}
