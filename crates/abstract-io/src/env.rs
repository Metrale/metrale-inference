// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The environment-variable port.
//!
//! Owner: metrale-abstract-io.
//! Invariants:
//! - `var` returns the value as stored: never trimmed or rewritten.
//! - `names` is sorted ascending in byte order with no duplicates.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// 2026-10-17: Why a set variable could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvError {
    /// The variable is set but its value is not valid UTF-8.
    NotUnicode { name: String },
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotUnicode { name } => {
                write!(f, "environment variable {name} is not valid UTF-8")
            }
        }
    }
}

impl core::error::Error for EnvError {}

/// 2026-10-17: Read access to a set of named string variables, supplied by a host.
pub trait Env {
    /// `Ok(None)` when unset. Never trims or rewrites the value.
    fn var(&self, name: &str) -> Result<Option<String>, EnvError>;
    /// Every variable name present, lossily decoded, sorted ascending (byte order), unique.
    fn names(&self) -> Vec<String>;
}
