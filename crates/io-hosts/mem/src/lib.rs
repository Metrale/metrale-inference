// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Deterministic hosts for the `metrale-abstract-io` ports: an
//! environment backed by a map, a clock that moves only when told, and a
//! seeded ChaCha20 entropy source. [`MemIo`] bundles one of each.
//!
//! Owner: metrale-io-host-mem.
//! Invariants:
//! - Nothing here reads the process environment, the system clock or the OS RNG.
//! - The same inputs produce the same outputs on every run.
#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]

mod clock;
mod entropy;
mod env;

pub use clock::ManualClock;
pub use entropy::SeededEntropy;
pub use env::MapEnv;

use metrale_abstract_io::Io;

/// 2026-10-17: A [`ManualClock`], a [`SeededEntropy`] and a [`MapEnv`] as one [`Io`].
pub struct MemIo {
    clock: ManualClock,
    entropy: SeededEntropy,
    env: MapEnv,
}

impl MemIo {
    pub fn new(clock: ManualClock, entropy: SeededEntropy, env: MapEnv) -> Self {
        Self {
            clock,
            entropy,
            env,
        }
    }
}

impl Io for MemIo {
    type Clock = ManualClock;
    type Entropy = SeededEntropy;
    type Env = MapEnv;

    fn clock(&self) -> &ManualClock {
        &self.clock
    }

    fn entropy(&self) -> &SeededEntropy {
        &self.entropy
    }

    fn env(&self) -> &MapEnv {
        &self.env
    }
}
