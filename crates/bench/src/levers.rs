// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The harness's lever values, resolved once by the binary and passed down. The type
//! lives in `metrale_host_probe::levers`, so `met` resolves it without the `bench` feature; this
//! module re-exports it under the harness's own path and holds its tests.

pub use metrale_host_probe::levers::{
    BenchLevers, MODEL_CARD_SAMPLING, hw_switch, resolve_harness,
};

#[cfg(test)]
#[path = "levers_tests.rs"]
pub(crate) mod tests;
