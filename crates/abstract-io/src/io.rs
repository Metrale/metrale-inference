// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The bundle of ports one host supplies.
//!
//! Owner: metrale-abstract-io.
//! Invariants: every accessor returns the same instance for the life of the bundle.

use crate::{Clock, Entropy, Env};

/// 2026-10-17: One host's choice of clock, entropy source and environment.
pub trait Io {
    type Clock: Clock;
    type Entropy: Entropy;
    type Env: Env;
    fn clock(&self) -> &Self::Clock;
    fn entropy(&self) -> &Self::Entropy;
    fn env(&self) -> &Self::Env;
}
