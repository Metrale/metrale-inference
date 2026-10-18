// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The process environment as an [`Env`].
//!
//! Owner: metrale-io-host-std.
//! Invariants: this is the only non-build-script file the `env-ban` gate (`cargo xtask
//! env-ban`) lets call `std::env::var*` without an allowlist entry; every other reader goes
//! through the `Env` port.

use metrale_abstract_io::{Env, Result};
use metrale_error::{ErrorCode, error};

/// 2026-10-17: The live process environment, read on every call.
#[derive(Clone, Copy, Debug)]
pub struct ProcessEnv;

impl Env for ProcessEnv {
    fn var(&self, name: &str) -> Result<Option<String>> {
        match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(error!(ErrorCode::EnvNotUnicode, name.to_owned()))
            }
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
