// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the one error type of the crate: a message, never swallowed.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// Return early with an [`Error`] built from a format string.
#[macro_export]
macro_rules! bail {
    ($($t:tt)*) => {
        return Err($crate::Error(format!($($t)*)))
    };
}
