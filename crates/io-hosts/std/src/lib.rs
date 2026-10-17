// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Process hosts for the `metrale-abstract-io` ports: the process
//! environment, the system clock, a tokio timer and the OS random source. [`StdIo`]
//! bundles one of each and is chosen only by a binary.
//!
//! Owner: metrale-io-host-std.
//! Invariants:
//! - `env.rs` is the only file in the engine that reads `std::env` (gate: `cargo xtask env-ban`).
//! - `SystemClock` and `TokioTimer` share one process origin.
//! - `TokioTimer::sleep_until` needs a running tokio time driver; the clock does not.
#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]

mod clock;
mod entropy;
mod env;
mod timer;

pub use clock::SystemClock;
pub use entropy::OsEntropy;
pub use env::ProcessEnv;
pub use timer::TokioTimer;

use metrale_abstract_io::Io;

/// 2026-10-17: A [`SystemClock`], [`TokioTimer`], [`OsEntropy`] and [`ProcessEnv`] as one [`Io`].
#[derive(Clone, Copy, Debug)]
pub struct StdIo {
    clock: SystemClock,
    timer: TokioTimer,
    entropy: OsEntropy,
    env: ProcessEnv,
}

impl StdIo {
    /// 2026-10-17: 2a only. 3a replaces `new` and `Default` with `from_settings(&s) ->
    /// Result<StdIo>`, because blob roots are settings, not defaults (PCND).
    #[must_use]
    pub fn new() -> Self {
        Self {
            clock: SystemClock::new(),
            timer: TokioTimer,
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
    type Timer = TokioTimer;
    type Entropy = OsEntropy;
    type Env = ProcessEnv;

    fn clock(&self) -> &SystemClock {
        &self.clock
    }

    fn timer(&self) -> &TokioTimer {
        &self.timer
    }

    fn entropy(&self) -> &OsEntropy {
        &self.entropy
    }

    fn env(&self) -> &ProcessEnv {
        &self.env
    }
}
