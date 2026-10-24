// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-19: The DeepSeek-V4.1 real-checkpoint tests' inputs, read through the `Env` port
//! (the `ProcessEnv` host) rather than `std::env`.
//!
//! Owner: server tests (deepseek-v4.1).
//! Invariants: an unset or non-unicode variable panics with its name.

use std::path::PathBuf;

use metrale_abstract_io::Env as _;
use metrale_io_host_std::ProcessEnv;

/// 2026-10-19: The path the variable `name` holds.
pub fn path(name: &str) -> PathBuf {
    let value = ProcessEnv
        .var(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    PathBuf::from(value.unwrap_or_else(|| panic!("{name} unset")))
}
