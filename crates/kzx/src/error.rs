// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed errors. The code is the identity: callers match on the variant, never on text.

use core::fmt;

/// Why a kzx stream was refused. Every variant is a hard refusal before or during decoding;
/// none is recoverable by retrying with the same bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The stream does not start with the `kzx` magic.
    BadMagic,
    /// The stream names a format version this build does not read (`found`).
    UnsupportedVersion {
        /// The version byte found in the stream.
        found: u8,
    },
    /// The stream ended inside a header or a block (`needed` bytes were not there).
    Truncated {
        /// How many more bytes the reader needed at the point of failure.
        needed: u64,
    },
    /// A block names a codec id this build does not have.
    UnknownCodec {
        /// The codec id found.
        id: u8,
    },
    /// A block's model memory parameter is outside the range the format allows.
    BadMemoryParameter {
        /// The `mem_log2` byte found.
        mem_log2: u8,
    },
    /// The declared total output length exceeds the caller's `max_output` cap.
    OutputCapExceeded {
        /// What the stream declares, in bytes.
        declared: u64,
        /// What the caller allows, in bytes.
        cap: u64,
    },
    /// The model memory a block needs exceeds the caller's `max_memory` cap.
    MemoryCapExceeded {
        /// What the block needs, in bytes.
        needed: u64,
        /// What the caller allows, in bytes.
        cap: u64,
    },
    /// A length field overflowed `u64` arithmetic or `usize` on this target.
    LengthOverflow,
    /// A stored block's packed length differs from its raw length.
    StoredLengthMismatch,
    /// Bytes remain after the last declared block.
    TrailingBytes {
        /// How many bytes follow the last block.
        count: u64,
    },
    /// The number of blocks exceeds [`crate::format::MAX_BLOCKS`].
    TooManyBlocks {
        /// The block count found.
        count: u32,
    },
    /// A reserved header field is not zero.
    ReservedNonZero,
    /// The caller asked for a model memory parameter the encoder does not accept.
    InvalidParameter,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadMagic => f.write_str("not a kzx stream: bad magic"),
            Error::UnsupportedVersion { found } => {
                write!(f, "kzx format version {found} is not supported")
            }
            Error::Truncated { needed } => {
                write!(f, "kzx stream truncated: {needed} bytes missing")
            }
            Error::UnknownCodec { id } => write!(f, "kzx block names unknown codec id {id}"),
            Error::BadMemoryParameter { mem_log2 } => {
                write!(f, "kzx block memory parameter {mem_log2} is out of range")
            }
            Error::OutputCapExceeded { declared, cap } => {
                write!(
                    f,
                    "kzx declares {declared} output bytes, above the cap of {cap}"
                )
            }
            Error::MemoryCapExceeded { needed, cap } => {
                write!(
                    f,
                    "kzx block needs {needed} bytes of model memory, above the cap of {cap}"
                )
            }
            Error::LengthOverflow => f.write_str("kzx length arithmetic overflowed"),
            Error::StoredLengthMismatch => {
                f.write_str("kzx stored block packed length differs from its raw length")
            }
            Error::TrailingBytes { count } => {
                write!(f, "{count} bytes follow the last kzx block")
            }
            Error::TooManyBlocks { count } => write!(f, "kzx stream declares {count} blocks"),
            Error::ReservedNonZero => f.write_str("kzx reserved header field is not zero"),
            Error::InvalidParameter => f.write_str("kzx encoder parameter out of range"),
        }
    }
}

impl core::error::Error for Error {}

/// The crate's result type.
pub type Result<T> = core::result::Result<T, Error>;
