// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The `kbp1.<base64>` text transport: standard alphabet, padded, unwrapped.
//!
//! Owner: blueprint.
//! Invariants: decoding refuses text longer than the full-file cap allows, before decoding.

use alloc::string::String;
use alloc::vec::Vec;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use metrale_error::{Error, ErrorCode, error};

use crate::frame::{HEADER_LEN, MAX_BODY, MAX_LOCATOR, MAX_PAYLOAD, MAX_SIG};

/// 2026-10-22: The transport prefix.
pub const PREFIX: &str = "kbp1.";

/// 2026-10-22: `kbp1.` + base64 of `file`.
pub fn to_text(file: &[u8]) -> String {
    let mut s = String::from(PREFIX);
    STANDARD.encode_string(file, &mut s);
    s
}

/// 2026-10-22: The file bytes of a `kbp1.` text (surrounding whitespace ignored).
pub fn from_text(text: &str) -> Result<Vec<u8>, Error> {
    let body = text.trim().strip_prefix(PREFIX).ok_or_else(|| {
        error!(
            ErrorCode::BlueprintFrame,
            "transport text does not start with kbp1."
        )
    })?;
    let max_file = HEADER_LEN as u64 + MAX_BODY + MAX_LOCATOR + MAX_PAYLOAD + MAX_SIG;
    let len = u64::try_from(body.len()).unwrap_or(u64::MAX);
    if len / 4 * 3 > max_file {
        return Err(error!(
            ErrorCode::BlueprintTooLarge,
            "transport text",
            len,
            max_file / 3 * 4
        ));
    }
    STANDARD.decode(body).map_err(|_| {
        error!(
            ErrorCode::BlueprintFrame,
            "transport text is not standard padded base64"
        )
    })
}
