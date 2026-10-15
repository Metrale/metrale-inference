// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the one error type. Every refusal and failed assertion returns
//! it; the CLI turns it into exit status 2 and never swallows it.

use std::fmt;

/// A hard error: bad input, a missing config key, or a refused analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl Error {
    /// The same error with `prefix: ` in front, for errors that cross a layer
    /// that knows more context (a config path, the side of a diff).
    pub fn context(self, prefix: &str) -> Error {
        Error(format!("{prefix}: {}", self.0))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// `fail!(...)` returns `Err(Error(format!(...)))` from the enclosing function.
#[macro_export]
macro_rules! fail {
    ($($arg:tt)*) => {
        return Err($crate::core::error::Error(format!($($arg)*)))
    };
}
