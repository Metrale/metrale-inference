// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: shared test helpers. The fakes implement the crate's ports
//! (no mocking of crypto): a deterministic SHA-256 counter stream for
//! entropy, a fixed clock, and a constant-byte source used as a negative
//! control for the nonce-uniqueness check.

#![allow(dead_code)]

use metprof_envelope::primitives::sha256;
use metprof_envelope::{Clock, Entropy, Result, SealRequest};

/// Decodes hex (either case); panics on malformed input, which is a test bug.
pub fn hex(text: &str) -> Vec<u8> {
    let text = text.trim();
    assert!(text.len().is_multiple_of(2), "odd-length hex");
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex digit"))
        .collect()
}

pub fn hex_array<const N: usize>(text: &str) -> [u8; N] {
    hex(text).try_into().expect("hex length")
}

/// Deterministic entropy: block i of the stream is sha256(seed ‖ i as u64 LE).
#[derive(Debug, Clone)]
pub struct StreamEntropy {
    seed: Vec<u8>,
    counter: u64,
    pending: Vec<u8>,
}

impl StreamEntropy {
    pub fn new(seed: &[u8]) -> Self {
        Self {
            seed: seed.to_vec(),
            counter: 0,
            pending: Vec::new(),
        }
    }
}

impl Entropy for StreamEntropy {
    fn fill(&mut self, buf: &mut [u8]) -> Result<()> {
        while self.pending.len() < buf.len() {
            let mut block = self.seed.clone();
            block.extend_from_slice(&self.counter.to_le_bytes());
            self.pending.extend_from_slice(&sha256(&block));
            self.counter += 1;
        }
        let rest = self.pending.split_off(buf.len());
        buf.copy_from_slice(&self.pending);
        self.pending = rest;
        Ok(())
    }
}

/// A broken RNG that always returns the same byte (negative control only).
#[derive(Debug, Clone, Copy)]
pub struct ConstantEntropy(pub u8);

impl Entropy for ConstantEntropy {
    fn fill(&mut self, buf: &mut [u8]) -> Result<()> {
        buf.fill(self.0);
        Ok(())
    }
}

/// A clock frozen at one instant.
#[derive(Debug, Clone, Copy)]
pub struct FixedClock(pub u64);

impl Clock for FixedClock {
    fn now_unix_seconds(&self) -> Result<u64> {
        Ok(self.0)
    }
}

/// 2026-10-15: synthetic test values (not real kids, paths or builds).
pub const KID: &str = "kid-test-01";
pub const PATH: &str = "lattice/levers.jsonl";
pub const ENCODING: &str = "deflate+lattice-jsonl/1";
pub const BUILD: &str = "test-build-0";
pub const CREATED_AT: u64 = 1_700_000_000;

pub fn request(wrapped: Option<metprof_envelope::WrappedDek>) -> SealRequest {
    SealRequest {
        kid: KID.to_owned(),
        path: PATH.to_owned(),
        encoding: ENCODING.to_owned(),
        publisher_build: BUILD.to_owned(),
        wrapped_dek: wrapped,
    }
}
