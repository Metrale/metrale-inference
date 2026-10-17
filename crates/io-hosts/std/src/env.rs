// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The process environment as an [`Env`].
//!
//! Owner: metrale-io-host-std.
//! Invariants: this is the only file allowed to call `std::env::var*`; every
//! other reader goes through the `Env` port.

use metrale_abstract_io::{Env, EnvError};

/// 2026-10-17: The live process environment, read on every call.
#[derive(Clone, Copy, Debug)]
pub struct ProcessEnv;

impl Env for ProcessEnv {
    fn var(&self, name: &str) -> Result<Option<String>, EnvError> {
        match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => Err(EnvError::NotUnicode {
                name: name.to_owned(),
            }),
        }
    }

    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = std::env::vars_os()
            .map(|(name, _)| name.to_string_lossy().into_owned())
            .collect();
        names.sort_unstable();
        names.dedup();
        names
    }
}

#[cfg(test)]
#[path = "env_tests.rs"]
mod tests;
