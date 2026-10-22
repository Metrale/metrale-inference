// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The one compression seam of the `.kbp` container.
//!
//! Every compressed byte in a file (the body section, each inline artifact) is tagged with a
//! [`Codec`] id and passes through [`Codec::compress`] / [`Codec::decompress`]; nothing else in
//! the crate names a compressor. Replacing the compressor adds a variant here and changes
//! [`Codec::WRITE`].
//!
//! Owner: blueprint.
//! Invariants:
//! - Identity never depends on the codec: hashes, `BlueprintId`, `ArtifactId` and the signed
//!   message cover the **raw** canonical bytes, so a file re-encoded under another codec keeps its
//!   id and its signature (`frame::recode`).
//! - Decompression of untrusted input is bounded: the declared raw length is checked against an
//!   explicit cap before any work, the decompressor is limited to that length, and output of any
//!   other length is refused (`BlueprintCodec`).
//! - The wire form of a codec is its one-byte id ([`Codec::id`]); unknown ids are refused.

use alloc::vec::Vec;

use metrale_error::{Error, ErrorCode, error};
use serde::{Deserialize, Serialize};

/// 2026-10-22: How a stored section's bytes encode its raw bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub enum Codec {
    /// 2026-10-22: The stored bytes are the raw bytes.
    Stored,
    /// 2026-10-22: Raw DEFLATE (RFC 1951), no zlib wrapper.
    Deflate,
}

/// 2026-10-22: The DEFLATE level writers use; decompression does not depend on it.
const DEFLATE_LEVEL: u8 = 9;

impl Codec {
    /// 2026-10-22: The codec every writer in this build uses.
    pub const WRITE: Codec = Codec::Deflate;

    /// 2026-10-22: The one-byte wire id.
    pub const fn id(self) -> u8 {
        match self {
            Codec::Stored => 0,
            Codec::Deflate => 1,
        }
    }

    /// 2026-10-22: The codec of a wire id.
    pub fn from_id(id: u8) -> Result<Self, Error> {
        match id {
            0 => Ok(Codec::Stored),
            1 => Ok(Codec::Deflate),
            _ => Err(error!(ErrorCode::BlueprintFormat, "codec id", id, 1u8)),
        }
    }

    /// 2026-10-22: `stored` or `deflate`.
    pub const fn name(self) -> &'static str {
        match self {
            Codec::Stored => "stored",
            Codec::Deflate => "deflate",
        }
    }

    /// 2026-10-22: Encode `raw`.
    pub fn compress(self, raw: &[u8]) -> Vec<u8> {
        match self {
            Codec::Stored => raw.to_vec(),
            Codec::Deflate => miniz_oxide::deflate::compress_to_vec(raw, DEFLATE_LEVEL),
        }
    }

    /// 2026-10-22: Decode `stored`, which must yield exactly `raw_len` bytes; `raw_len` above
    /// `cap` is refused before any work (`BlueprintTooLarge`).
    pub fn decompress(
        self,
        what: &'static str,
        stored: &[u8],
        raw_len: u64,
        cap: u64,
    ) -> Result<Vec<u8>, Error> {
        if raw_len > cap {
            return Err(error!(ErrorCode::BlueprintTooLarge, what, raw_len, cap));
        }
        let limit = usize::try_from(raw_len)
            .map_err(|_| error!(ErrorCode::BlueprintTooLarge, what, raw_len, cap))?;
        let raw = match self {
            Codec::Stored => stored.to_vec(),
            Codec::Deflate => miniz_oxide::inflate::decompress_to_vec_with_limit(stored, limit)
                .map_err(|_| {
                    error!(
                        ErrorCode::BlueprintCodec,
                        what,
                        self.name(),
                        "the stream is corrupt or longer than its declared length"
                    )
                })?,
        };
        if raw.len() != limit {
            return Err(error!(
                ErrorCode::BlueprintCodec,
                what,
                self.name(),
                "the output length differs from the declared length"
            ));
        }
        Ok(raw)
    }
}

impl TryFrom<u8> for Codec {
    type Error = Error;
    fn try_from(id: u8) -> Result<Self, Error> {
        Codec::from_id(id)
    }
}

impl From<Codec> for u8 {
    fn from(c: Codec) -> u8 {
        c.id()
    }
}
