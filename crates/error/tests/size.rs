// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `Error` is one pointer wide: `Error`, `Result<()>` and `Option<Error>` are all
//! the size of a `usize`.

use std::mem::size_of;

use metrale_error::{Error, Result};

#[test]
fn error_is_pointer_sized() {
    assert_eq!(size_of::<Error>(), size_of::<usize>());
}

#[test]
fn result_is_pointer_sized() {
    assert_eq!(size_of::<Result<()>>(), size_of::<usize>());
}

#[test]
fn option_of_error_is_pointer_sized() {
    assert_eq!(size_of::<Option<Error>>(), size_of::<usize>());
}
