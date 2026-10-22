// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The one compression seam of the `.kbp` container.
//!
//! Every compressed byte in a file (the body section, the inline payload) is tagged with a
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
//!   explicit cap before any work, the decompressor may produce at most that length, a kzx
//!   block's model memory is checked against the caller's [`DecodeMemory`] before it is
//!   allocated, and output of any other length is refused (`BlueprintCodec`).
//! - Ids are append-only: a codec, once written, stays readable (DESIGN.md §2 Versioning: a
//!   reader reads older files). `Deflate` is no longer written by default but is still decoded.
//! - kzx refusals are mapped here, by variant, onto the blueprint range (CERF); nothing
//!   outside this file sees a `kzx::Error`.

use alloc::vec::Vec;

use metrale_error::{Error, ErrorCode, error};
use serde::{Deserialize, Serialize};

/// 2026-10-22: How a stored section's bytes encode its raw bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub enum Codec {
    /// 2026-10-22: The stored bytes are the raw bytes.
    Stored,
    /// 2026-10-22: Raw DEFLATE (RFC 1951), no zlib wrapper. Read for files written before kzx.
    Deflate,
    /// 2026-10-22: A `kzx` stream (context mixing; PTX, ELF and postcard-aware).
    Kzx,
}

/// 2026-10-22: The DEFLATE level writers use; decompression does not depend on it.
const DEFLATE_LEVEL: u8 = 9;

/// 2026-10-22: The most model memory one kzx block may use while it decodes, chosen by the host
/// that runs the decoder (PCND: there is no default; a server passes about 1 GiB, the store's
/// wasm verifier about 64 MiB, which covers a lite body). Stored and deflate need none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeMemory(pub u64);

impl Codec {
    /// 2026-10-22: The codec every writer in this build uses.
    pub const WRITE: Codec = Codec::Kzx;

    /// 2026-10-22: The one-byte wire id.
    pub const fn id(self) -> u8 {
        match self {
            Codec::Stored => 0,
            Codec::Deflate => 1,
            Codec::Kzx => 2,
        }
    }

    /// 2026-10-22: The codec of a wire id.
    pub fn from_id(id: u8) -> Result<Self, Error> {
        match id {
            0 => Ok(Codec::Stored),
            1 => Ok(Codec::Deflate),
            2 => Ok(Codec::Kzx),
            _ => Err(error!(ErrorCode::BlueprintFormat, "codec id", id, 2u8)),
        }
    }

    /// 2026-10-22: `stored`, `deflate` or `kzx`.
    pub const fn name(self) -> &'static str {
        match self {
            Codec::Stored => "stored",
            Codec::Deflate => "deflate",
            Codec::Kzx => "kzx",
        }
    }

    /// 2026-10-22: Encode `raw`. Deterministic for every codec, so equal raw bytes give equal
    /// stored bytes (lite and full share their body section byte for byte).
    pub fn compress(self, raw: &[u8]) -> Vec<u8> {
        match self {
            Codec::Stored => raw.to_vec(),
            Codec::Deflate => miniz_oxide::deflate::compress_to_vec(raw, DEFLATE_LEVEL),
            Codec::Kzx => kzx::compress(raw),
        }
    }

    /// 2026-10-22: Decode `stored`, which must yield exactly `raw_len` bytes; `raw_len` above
    /// `cap` is refused before any work (`BlueprintTooLarge`), and so is a kzx block that needs
    /// more model memory than `memory`.
    pub fn decompress(
        self,
        what: &'static str,
        stored: &[u8],
        raw_len: u64,
        cap: u64,
        memory: DecodeMemory,
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
            Codec::Kzx => {
                let limits = kzx::Limits {
                    max_output: raw_len,
                    max_memory: memory.0,
                };
                kzx::decompress(stored, &limits).map_err(|e| kzx_refusal(what, e))?
            }
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

/// 2026-10-22: A kzx refusal as a blueprint error (kzx DESIGN.md §4.4): a cap is
/// `BlueprintTooLarge`, refused before any work; a malformed stream is `BlueprintCodec`.
fn kzx_refusal(what: &'static str, e: kzx::Error) -> Error {
    let corrupt = |reason: &'static str| error!(ErrorCode::BlueprintCodec, what, "kzx", reason);
    match e {
        kzx::Error::OutputCapExceeded { declared, cap } => {
            error!(ErrorCode::BlueprintTooLarge, what, declared, cap)
        }
        kzx::Error::MemoryCapExceeded { needed, cap } => error!(
            ErrorCode::BlueprintTooLarge,
            alloc::format!("{what} decode model memory"),
            needed,
            cap
        ),
        kzx::Error::BadMagic => corrupt("bad magic"),
        kzx::Error::UnsupportedVersion { .. } => corrupt("unsupported kzx format version"),
        kzx::Error::Truncated { .. } => corrupt("truncated"),
        kzx::Error::UnknownCodec { .. } => corrupt("unknown kzx block codec"),
        kzx::Error::BadMemoryParameter { .. } => corrupt("block memory parameter out of range"),
        kzx::Error::LengthOverflow => corrupt("length overflow"),
        kzx::Error::StoredLengthMismatch => corrupt("stored block length mismatch"),
        kzx::Error::TrailingBytes { .. } => corrupt("trailing bytes after the last block"),
        kzx::Error::TooManyBlocks { .. } => corrupt("too many blocks"),
        kzx::Error::ReservedNonZero => corrupt("reserved header field is not zero"),
        kzx::Error::InvalidParameter => corrupt("encoder parameter out of range"),
        _ => corrupt("refused by kzx"),
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
