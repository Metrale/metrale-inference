// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: TOML text to a table (pure). The adapters read the bytes; this
//! decodes and parses them, and the validators take the table.

use crate::core::error::Result;
use crate::fail;

pub fn parse_toml(bytes: &[u8]) -> Result<toml::Table> {
    let Ok(text) = std::str::from_utf8(bytes) else {
        fail!("invalid TOML: the file is not UTF-8");
    };
    match text.parse::<toml::Table>() {
        Ok(t) => Ok(t),
        Err(e) => fail!("invalid TOML: {}", e.to_string().trim_end()),
    }
}
