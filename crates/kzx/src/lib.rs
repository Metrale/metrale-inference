// SPDX-License-Identifier: MIT OR Apache-2.0

//! # kzx: kernel compression
//!
//! A `no_std`, zero-dependency context-mixing compressor tuned for compiled GPU kernel
//! artifacts (PTX text, cubin/SASS ELF objects) and small structured bodies (postcard), the
//! content of Metrale `.kbp` blueprint files. It trades compression speed for ratio: the models
//! are bit predictors mixed by a trained logistic mixer and coded by a binary arithmetic coder,
//! so it reaches ratios LZ-family compressors (deflate, zstd, xz, brotli) cannot, and it decodes
//! at a few MB/s, which is fine for a file that is decoded once at load.
//!
//! ## Guarantees
//!
//! - **Deterministic**: the same input bytes give the same output bytes on every target (all
//!   arithmetic is integer), so signatures over compressed files are stable.
//! - **Bounded on untrusted input**: the declared output length is checked against the caller's
//!   [`Limits::max_output`] and the model memory against [`Limits::max_memory`] before any
//!   allocation; every loop is bounded by a declared length; there are no panics for any input.
//! - **Pure**: `no_std` + `alloc`, `#![forbid(unsafe_code)]`, no dependencies, builds for
//!   `wasm32-unknown-unknown` and `wasm32v1-none`.
//!
//! ## Quick start
//!
//! ```
//! let raw = b".version 9.0\n.target sm_90a\n.address_size 64\n\n.visible .entry k(){ret;}\n";
//! let packed = kzx::compress(raw);
//! let limits = kzx::Limits { max_output: 1 << 20, max_memory: 64 << 20 };
//! let back = kzx::decompress(&packed, &limits).unwrap();
//! assert_eq!(&back[..], &raw[..]);
//! ```
//!
//! A stream can hold several independently coded blocks, each with its own codec:
//!
//! ```
//! let body = b"\x01\x05hello\x02\x03abc";
//! let kernel = b".version 9.0\n.target sm_121a\n";
//! let packed = kzx::compress_blocks(&[&body[..], &kernel[..]]);
//! let info = kzx::inspect(&packed).unwrap();
//! assert_eq!(info.blocks.len(), 2);
//! assert_eq!(info.total_raw, (body.len() + kernel.len()) as u64);
//! let limits = kzx::Limits { max_output: 1 << 16, max_memory: 64 << 20 };
//! let parts = kzx::decompress_blocks(&packed, &limits).unwrap();
//! assert_eq!(parts[1], kernel);
//! ```
//!
//! Refusals are typed; a declared length above the cap is refused before any work:
//!
//! ```
//! let packed = kzx::compress(&[7u8; 10_000]);
//! let tight = kzx::Limits { max_output: 100, max_memory: 64 << 20 };
//! assert!(matches!(
//!     kzx::decompress(&packed, &tight),
//!     Err(kzx::Error::OutputCapExceeded { declared: 10_000, cap: 100 })
//! ));
//! ```

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;

mod cast;
mod coder;
mod context;
mod engine;
mod error;
pub mod format;
mod matcher;
mod mixer;
mod predictor;
mod sniff;
mod slots;
mod states;
mod tables;

use alloc::vec::Vec;

pub use error::{Error, Result};
pub use format::{BlockHeader, Profile, StreamInfo};
pub use predictor::{MAX_MEM_LOG2, MIN_MEM_LOG2, memory_bytes};
pub use sniff::sniff;

/// Inputs shorter than this are stored, not modelled.
const STORED_BELOW: usize = 24;

/// What a decoder may spend. Both caps are checked against the stream's declarations before
/// anything is allocated or decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// The most output bytes a stream may declare in total.
    pub max_output: u64,
    /// The most model memory one block may need (see [`memory_bytes`]).
    pub max_memory: u64,
}

/// The `mem_log2` [`compress`] picks for an input of `len` bytes: tables about as large as
/// the input, within `[MIN_MEM_LOG2, 22]`.
pub fn default_mem_log2(len: usize) -> u8 {
    let bits = u64::BITS - cast::u64_of(len.max(1)).leading_zeros();
    u8::try_from(bits.saturating_sub(5))
        .unwrap_or(20)
        .clamp(MIN_MEM_LOG2, 22)
}

fn frame_one(out: &mut Vec<u8>, raw: &[u8], profile: Profile, mem_log2: u8) {
    let (profile, mem_log2, packed) = if profile == Profile::Stored || raw.len() < STORED_BELOW {
        (Profile::Stored, 0u8, raw.to_vec())
    } else {
        (profile, mem_log2, engine::compress_block(raw, profile, mem_log2))
    };
    format::write_block_header(
        out,
        &format::BlockHeader {
            profile,
            mem_log2,
            raw_len: cast::u64_of(raw.len()),
            packed_len: cast::u64_of(packed.len()),
        },
    );
    out.extend_from_slice(&packed);
}

/// Compress `raw` as one block, sniffing its content class and sizing the model from its
/// length. Deterministic: equal inputs give equal outputs.
pub fn compress(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() / 4 + 32);
    format::write_stream_header(&mut out, 1);
    frame_one(&mut out, raw, sniff(raw), default_mem_log2(raw.len()));
    out
}

/// Compress `raw` as one block with an explicit profile and memory parameter
/// (`MIN_MEM_LOG2..=MAX_MEM_LOG2`, else [`Error::InvalidParameter`]).
pub fn compress_with(raw: &[u8], profile: Profile, mem_log2: u8) -> Result<Vec<u8>> {
    if !(MIN_MEM_LOG2..=MAX_MEM_LOG2).contains(&mem_log2) {
        return Err(Error::InvalidParameter);
    }
    let mut out = Vec::with_capacity(raw.len() / 4 + 32);
    format::write_stream_header(&mut out, 1);
    frame_one(&mut out, raw, profile, mem_log2);
    Ok(out)
}

/// Compress several parts as independent blocks of one stream (at most
/// [`format::MAX_BLOCKS`]; extra parts are dropped, so check the count first).
pub fn compress_blocks(parts: &[&[u8]]) -> Vec<u8> {
    let n = u32::try_from(parts.len()).unwrap_or(format::MAX_BLOCKS).min(format::MAX_BLOCKS);
    let mut out = Vec::new();
    format::write_stream_header(&mut out, n);
    for part in parts.iter().take(cast::ix(n)) {
        frame_one(&mut out, part, sniff(part), default_mem_log2(part.len()));
    }
    out
}

/// Read the headers of `packed` without decoding anything.
pub fn inspect(packed: &[u8]) -> Result<StreamInfo> {
    format::inspect(packed)
}

fn check(info: &StreamInfo, limits: &Limits) -> Result<()> {
    if info.total_raw > limits.max_output {
        return Err(Error::OutputCapExceeded {
            declared: info.total_raw,
            cap: limits.max_output,
        });
    }
    usize::try_from(info.total_raw).map_err(|_| Error::LengthOverflow)?;
    for h in &info.blocks {
        match h.profile {
            Profile::Stored => {
                if h.raw_len != h.packed_len || h.mem_log2 != 0 {
                    return Err(Error::StoredLengthMismatch);
                }
            }
            Profile::Generic | Profile::Ptx | Profile::Elf => {
                if !(MIN_MEM_LOG2..=MAX_MEM_LOG2).contains(&h.mem_log2) {
                    return Err(Error::BadMemoryParameter {
                        mem_log2: h.mem_log2,
                    });
                }
                let needed = memory_bytes(h.profile, h.mem_log2);
                if needed > limits.max_memory {
                    return Err(Error::MemoryCapExceeded {
                        needed,
                        cap: limits.max_memory,
                    });
                }
            }
        }
    }
    Ok(())
}

fn decode_into(packed: &[u8], info: &StreamInfo, mut sink: impl FnMut(&[u8])) -> Result<()> {
    let mut at = format::STREAM_HEADER_LEN;
    let mut buf = Vec::new();
    for h in &info.blocks {
        let start = at + format::BLOCK_HEADER_LEN;
        let len = usize::try_from(h.packed_len).map_err(|_| Error::LengthOverflow)?;
        let raw_len = usize::try_from(h.raw_len).map_err(|_| Error::LengthOverflow)?;
        let end = start.checked_add(len).ok_or(Error::LengthOverflow)?;
        let bytes = packed.get(start..end).ok_or(Error::Truncated {
            needed: cast::u64_of(end.saturating_sub(packed.len())),
        })?;
        buf.clear();
        buf.reserve(raw_len);
        match h.profile {
            Profile::Stored => buf.extend_from_slice(bytes),
            Profile::Generic | Profile::Ptx | Profile::Elf => {
                engine::decompress_block(bytes, raw_len, h.profile, h.mem_log2, &mut buf);
            }
        }
        sink(&buf);
        at = end;
    }
    Ok(())
}

/// Decompress every block of `packed` into one vector, under `limits`.
pub fn decompress(packed: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    let info = inspect(packed)?;
    check(&info, limits)?;
    let mut out = Vec::with_capacity(usize::try_from(info.total_raw).map_err(|_| Error::LengthOverflow)?);
    decode_into(packed, &info, |b| out.extend_from_slice(b))?;
    Ok(out)
}

/// Decompress `packed` block by block, under `limits`.
pub fn decompress_blocks(packed: &[u8], limits: &Limits) -> Result<Vec<Vec<u8>>> {
    let info = inspect(packed)?;
    check(&info, limits)?;
    let mut parts = Vec::with_capacity(info.blocks.len());
    decode_into(packed, &info, |b| parts.push(b.to_vec()))?;
    Ok(parts)
}
