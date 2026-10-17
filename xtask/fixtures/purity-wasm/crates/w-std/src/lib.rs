// SPDX-License-Identifier: MIT OR Apache-2.0
use std::os::unix::ffi::OsStrExt;

pub fn bytes(s: &std::ffi::OsStr) -> &[u8] {
    s.as_bytes()
}
