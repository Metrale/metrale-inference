// SPDX-License-Identifier: MIT OR Apache-2.0

//! The container: a stream header followed by independent blocks, each tagged with its codec,
//! its model-memory parameter and both lengths, so a reader can bound every allocation and
//! every loop before it decodes a bit.
//!
//! ```text
//! stream  = "kzx" u8:version=1  u32le:block_count  block*
//! block   = u8:codec  u8:mem_log2  u16le:reserved=0  u64le:raw_len  u64le:packed_len  bytes[packed_len]
//! ```

use crate::cast::u64_of;
use crate::error::{Error, Result};
use alloc::vec::Vec;

/// The three magic bytes every stream starts with.
pub const MAGIC: [u8; 3] = *b"kzx";
/// The format version this build writes and reads.
pub const VERSION: u8 = 1;
/// Stream header size in bytes.
pub const STREAM_HEADER_LEN: usize = 8;
/// Block header size in bytes.
pub const BLOCK_HEADER_LEN: usize = 20;
/// The most blocks one stream may declare (a bound on header work for untrusted input).
pub const MAX_BLOCKS: u32 = 65_536;

/// How a block's bytes encode its raw bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Profile {
    /// Raw bytes, no model (used for inputs too small to pay for a model header).
    Stored,
    /// The generic byte-oriented context-mixing model: orders 1–6, word and match models.
    Generic,
    /// The PTX/text model: `Generic` plus token, line-field and opcode contexts.
    Ptx,
    /// The ELF/SASS model: `Generic` plus 16-byte record (instruction word) contexts.
    Elf,
}

impl Profile {
    /// The one-byte wire id.
    pub const fn id(self) -> u8 {
        match self {
            Profile::Stored => 0,
            Profile::Generic => 1,
            Profile::Ptx => 2,
            Profile::Elf => 3,
        }
    }

    /// The profile a wire id names.
    pub fn from_id(id: u8) -> Result<Self> {
        match id {
            0 => Ok(Profile::Stored),
            1 => Ok(Profile::Generic),
            2 => Ok(Profile::Ptx),
            3 => Ok(Profile::Elf),
            _ => Err(Error::UnknownCodec { id }),
        }
    }

    /// The profile's name, for reports.
    pub const fn name(self) -> &'static str {
        match self {
            Profile::Stored => "stored",
            Profile::Generic => "cm-generic",
            Profile::Ptx => "cm-ptx",
            Profile::Elf => "cm-elf",
        }
    }
}

/// One block's header fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockHeader {
    /// The codec that coded the block.
    pub profile: Profile,
    /// log2 of the per-context hash table bucket count (0 for `Stored`).
    pub mem_log2: u8,
    /// The block's uncompressed length.
    pub raw_len: u64,
    /// The block's compressed length.
    pub packed_len: u64,
}

/// Append the stream header for `block_count` blocks.
pub fn write_stream_header(out: &mut Vec<u8>, block_count: u32) {
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&block_count.to_le_bytes());
}

/// Append a block header.
pub fn write_block_header(out: &mut Vec<u8>, h: &BlockHeader) {
    out.push(h.profile.id());
    out.push(h.mem_log2);
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&h.raw_len.to_le_bytes());
    out.extend_from_slice(&h.packed_len.to_le_bytes());
}

/// Parse the stream header; returns the block count.
pub fn parse_stream_header(data: &[u8]) -> Result<u32> {
    let Some(head) = data.get(..STREAM_HEADER_LEN) else {
        return Err(Error::Truncated {
            needed: u64_of(STREAM_HEADER_LEN.saturating_sub(data.len())),
        });
    };
    if head[..3] != MAGIC {
        return Err(Error::BadMagic);
    }
    if head[3] != VERSION {
        return Err(Error::UnsupportedVersion { found: head[3] });
    }
    let count = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
    if count > MAX_BLOCKS {
        return Err(Error::TooManyBlocks { count });
    }
    Ok(count)
}

/// Parse the block header at `at`; returns the header and the offset of the block's bytes.
pub fn parse_block_header(data: &[u8], at: usize) -> Result<(BlockHeader, usize)> {
    let end = at
        .checked_add(BLOCK_HEADER_LEN)
        .ok_or(Error::LengthOverflow)?;
    let Some(h) = data.get(at..end) else {
        return Err(Error::Truncated {
            needed: u64_of(end.saturating_sub(data.len())),
        });
    };
    let profile = Profile::from_id(h[0])?;
    let mem_log2 = h[1];
    if h[2] != 0 || h[3] != 0 {
        return Err(Error::ReservedNonZero);
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&h[4..12]);
    let mut packed = [0u8; 8];
    packed.copy_from_slice(&h[12..20]);
    Ok((
        BlockHeader {
            profile,
            mem_log2,
            raw_len: u64::from_le_bytes(raw),
            packed_len: u64::from_le_bytes(packed),
        },
        end,
    ))
}

/// A stream's declared layout, read from headers alone (nothing is decoded).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamInfo {
    /// Every block header, in order.
    pub blocks: Vec<BlockHeader>,
    /// The sum of the blocks' raw lengths.
    pub total_raw: u64,
    /// The sum of the blocks' packed lengths.
    pub total_packed: u64,
}

/// Walk the headers of `data` and check that the declared packed lengths tile the stream
/// exactly. Work is bounded by [`MAX_BLOCKS`] header reads.
pub fn inspect(data: &[u8]) -> Result<StreamInfo> {
    let count = parse_stream_header(data)?;
    let mut blocks = Vec::new();
    let mut at = STREAM_HEADER_LEN;
    let mut total_raw = 0u64;
    let mut total_packed = 0u64;
    for _ in 0..count {
        let (h, start) = parse_block_header(data, at)?;
        total_raw = total_raw
            .checked_add(h.raw_len)
            .ok_or(Error::LengthOverflow)?;
        total_packed = total_packed
            .checked_add(h.packed_len)
            .ok_or(Error::LengthOverflow)?;
        let packed = usize::try_from(h.packed_len).map_err(|_| Error::LengthOverflow)?;
        let next = start.checked_add(packed).ok_or(Error::LengthOverflow)?;
        if next > data.len() {
            return Err(Error::Truncated {
                needed: u64_of(next - data.len()),
            });
        }
        blocks.push(h);
        at = next;
    }
    if at != data.len() {
        return Err(Error::TrailingBytes {
            count: u64_of(data.len() - at),
        });
    }
    Ok(StreamInfo {
        blocks,
        total_raw,
        total_packed,
    })
}
