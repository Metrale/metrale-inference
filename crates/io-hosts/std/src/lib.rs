// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Process hosts for the `metrale-abstract-io` ports: the process
//! environment, the system clock and the OS random source. [`StdIo`] bundles
//! one of each and is chosen only by a binary.
//!
//! Owner: metrale-io-host-std.
//! Invariants:
//! - `env.rs` is the only file in the engine that reads `std::env`.
//! - `SystemClock::sleep_ms` needs a running tokio time driver.
#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]

mod clock;
mod entropy;
mod env;

pub use clock::SystemClock;
pub use entropy::OsEntropy;
pub use env::ProcessEnv;

use metrale_abstract_io::Io;

/// 2026-10-17: A [`SystemClock`], [`OsEntropy`] and [`ProcessEnv`] as one [`Io`].
#[derive(Clone, Copy, Debug)]
pub struct StdIo {
    clock: SystemClock,
    entropy: OsEntropy,
    env: ProcessEnv,
}

impl StdIo {
    #[must_use]
    pub fn new() -> Self {
        Self {
            clock: SystemClock::new(),
            entropy: OsEntropy,
            env: ProcessEnv,
        }
    }
}

// 2026-10-17: Default only satisfies clippy::new_without_default; it carries no config value.
impl Default for StdIo {
    fn default() -> Self {
        Self::new()
    }
}

impl Io for StdIo {
    type Clock = SystemClock;
    type Entropy = OsEntropy;
    type Env = ProcessEnv;

    fn clock(&self) -> &SystemClock {
        &self.clock
    }

    fn entropy(&self) -> &OsEntropy {
        &self.entropy
    }

    fn env(&self) -> &ProcessEnv {
        &self.env
    }
}
