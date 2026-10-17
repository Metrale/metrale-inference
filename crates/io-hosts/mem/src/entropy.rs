// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: A reproducible [`Entropy`] from a ChaCha20 stream.
//!
//! Owner: metrale-io-host-mem.
//! Invariants: the byte stream is a pure function of `(seed, stream)` and the
//! total bytes drawn before; `fill` never errors.

use std::sync::Mutex;

use metrale_abstract_io::{Entropy, EntropyError};
use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};

/// 2026-10-17: ChaCha20 seeded from `seed`, on stream `stream`.
pub struct SeededEntropy {
    rng: Mutex<ChaCha20Rng>,
}

impl SeededEntropy {
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        rng.set_stream(stream);
        Self {
            rng: Mutex::new(rng),
        }
    }
}

impl Entropy for SeededEntropy {
    fn fill(&self, buf: &mut [u8]) -> Result<(), EntropyError> {
        self.rng
            .lock()
            .expect("SeededEntropy lock poisoned")
            .fill_bytes(buf);
        Ok(())
    }
}

#[cfg(test)]
#[path = "entropy_tests.rs"]
mod tests;
