// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The randomness port.
//!
//! Owner: metrale-abstract-io.
//! Invariants: `fill` either fills the whole buffer or returns `EntropyUnavailable`.

use crate::Result;

/// 2026-10-17: A source of random bytes, supplied by a host.
pub trait Entropy {
    /// Fills all of `buf`, or fails with `ErrorCode::EntropyUnavailable`.
    fn fill(&self, buf: &mut [u8]) -> Result<()>;

    /// 2026-10-17: One 32-byte seed: the single draw per unseeded request at admission (BPPT).
    fn seed32(&self) -> Result<[u8; 32]> {
        let mut seed = [0u8; 32];
        self.fill(&mut seed)?;
        Ok(seed)
    }
}
