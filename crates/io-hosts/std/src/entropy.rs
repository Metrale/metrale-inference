// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The operating system's random source as an [`Entropy`].
//!
//! Owner: metrale-io-host-std.
//! Invariants: a failed OS read is returned as `EntropyError::Unavailable`, never ignored.

use metrale_abstract_io::{Entropy, EntropyError};

/// 2026-10-17: Bytes from the OS RNG via `getrandom`.
#[derive(Clone, Copy, Debug)]
pub struct OsEntropy;

impl Entropy for OsEntropy {
    fn fill(&self, buf: &mut [u8]) -> Result<(), EntropyError> {
        // 2026-10-17: The port's error is one variant (as in mfs); getrandom's
        // detail is dropped, the failure itself is not.
        getrandom::fill(buf).map_err(|_| EntropyError::Unavailable)
    }
}

#[cfg(test)]
#[path = "entropy_tests.rs"]
mod tests;
