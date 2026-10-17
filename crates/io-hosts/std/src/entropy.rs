// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The operating system's random source as an [`Entropy`].
//!
//! Owner: metrale-io-host-std.
//! Invariants: a failed OS read is returned as `ErrorCode::EntropyUnavailable` with the OS
//! detail as its argument, never ignored.

use metrale_abstract_io::{Entropy, Result};
use metrale_error::{ErrorCode, error};

/// 2026-10-17: Bytes from the OS RNG via `getrandom`.
#[derive(Clone, Copy, Debug)]
pub struct OsEntropy;

impl Entropy for OsEntropy {
    fn fill(&self, buf: &mut [u8]) -> Result<()> {
        getrandom::fill(buf).map_err(|e| error!(ErrorCode::EntropyUnavailable, e.to_string()))
    }
}

#[cfg(test)]
#[path = "entropy_tests.rs"]
mod tests;
