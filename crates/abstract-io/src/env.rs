// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The environment-variable port.
//!
//! Owner: metrale-abstract-io.
//! Invariants:
//! - `var` returns the value as stored: never trimmed or rewritten.
//! - `names` is sorted ascending in byte order with no duplicates.

use alloc::string::String;
use alloc::vec::Vec;

use crate::Result;

/// 2026-10-17: Read access to a set of named string variables, supplied by a host.
pub trait Env {
    /// `Ok(None)` when unset. Never trims or rewrites the value. A set value that is not
    /// valid unicode fails with `ErrorCode::EnvNotUnicode`.
    fn var(&self, name: &str) -> Result<Option<String>>;
    /// Every variable name present, lossily decoded, sorted ascending (byte order), unique.
    fn names(&self) -> Vec<String>;
}
