// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The payload section's two layouts (unsigned, like the locator table).
//!
//! ```text
//! solid (FLAG_SOLID set; what this build writes for a full file):
//!   [0] codec id u8   [1..9] raw_len u64   [9..17] index_len u64
//!   [17..17+index_len] index: postcard Vec<(ArtifactId, u64 raw length)>, in entry order
//!   then the stored bytes of the raw artifacts concatenated in entry order, under `codec`
//! per-entry (FLAG_SOLID clear; files written before the solid layout, and every lite file):
//!   postcard Vec<Entry { artifact, raw_len, codec, stored }>
//! ```
//!
//! Owner: blueprint.
//! Invariants:
//! - One solid block: the compressor's window spans every artifact, so code shared across
//!   modules is coded once (kzx DESIGN.md §4.1). The index stays stored: it is hashes.
//! - Every length is checked before anything is allocated or decoded: the raw length against
//!   [`MAX_PAYLOAD_RAW`], each artifact against [`MAX_ARTIFACT`], and the index lengths must sum
//!   to the raw length exactly.
//! - Decoding yields raw [`InlineArtifact`]s; their hashes are checked against the signed
//!   module specs by `resolve`, never here.
//! - An empty payload is the per-entry empty list, so a lite file has no payload codec at all.

use alloc::vec::Vec;

use metrale_error::{Error, ErrorCode, error};
use serde::{Deserialize, Serialize};

use super::{FLAG_FULL, FLAG_SOLID, InlineArtifact, MAX_ARTIFACT, MAX_PAYLOAD_RAW, decode, encode};
use crate::codec::{Codec, DecodeMemory};
use crate::ids::ArtifactId;

/// 2026-10-22: The solid layout's fixed prefix length.
const SOLID_PREFIX: usize = 17;

/// 2026-10-22: One artifact of the per-entry layout, stored under its own codec.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadEntry {
    /// 2026-10-22: SHA-256 of the raw bytes.
    pub artifact: ArtifactId,
    /// 2026-10-22: Raw length.
    pub raw_len: u64,
    /// 2026-10-22: How `stored` encodes the raw bytes.
    pub codec: Codec,
    /// 2026-10-22: Stored bytes.
    pub stored: Vec<u8>,
}

fn frame_err(why: &'static str) -> Error {
    error!(ErrorCode::BlueprintFrame, why)
}

fn len64(n: usize) -> Result<u64, Error> {
    u64::try_from(n).map_err(|_| frame_err("payload length"))
}

/// 2026-10-22: The flag bits and section bytes of `payload` under `codec`: the per-entry empty
/// list when there is nothing inline, else one solid block.
pub(super) fn encode_payload(
    payload: &[InlineArtifact],
    codec: Codec,
) -> Result<(u8, Vec<u8>), Error> {
    if payload.is_empty() {
        return Ok((0, encode("payload", &Vec::<PayloadEntry>::new())?));
    }
    let index = payload
        .iter()
        .map(|a| Ok((a.artifact, len64(a.bytes.len())?)))
        .collect::<Result<Vec<_>, Error>>()?;
    let index = encode("payload index", &index)?;
    let raw: Vec<u8> = payload
        .iter()
        .flat_map(|a| a.bytes.iter().copied())
        .collect();
    let stored = codec.compress(&raw);
    let mut out = Vec::with_capacity(SOLID_PREFIX + index.len() + stored.len());
    out.push(codec.id());
    out.extend_from_slice(&len64(raw.len())?.to_le_bytes());
    out.extend_from_slice(&len64(index.len())?.to_le_bytes());
    out.extend_from_slice(&index);
    out.extend_from_slice(&stored);
    Ok((FLAG_FULL | FLAG_SOLID, out))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

fn decode_solid(section: &[u8], memory: DecodeMemory) -> Result<Vec<InlineArtifact>, Error> {
    if section.len() < SOLID_PREFIX {
        return Err(frame_err("truncated solid payload prefix"));
    }
    let codec = Codec::from_id(section[0])?;
    let raw_len = u64_at(section, 1);
    if raw_len > MAX_PAYLOAD_RAW {
        return Err(error!(
            ErrorCode::BlueprintTooLarge,
            "raw payload", raw_len, MAX_PAYLOAD_RAW
        ));
    }
    let index_end = usize::try_from(u64_at(section, 9))
        .ok()
        .and_then(|n| n.checked_add(SOLID_PREFIX))
        .filter(|end| *end <= section.len())
        .ok_or_else(|| frame_err("the payload index extends past the section"))?;
    let index: Vec<(ArtifactId, u64)> = decode("payload index", &section[SOLID_PREFIX..index_end])?;
    if index.is_empty() {
        return Err(frame_err("a solid payload has no entries"));
    }
    let mut total = 0u64;
    for (_, len) in &index {
        if *len > MAX_ARTIFACT {
            return Err(error!(
                ErrorCode::BlueprintTooLarge,
                "inline artifact", *len, MAX_ARTIFACT
            ));
        }
        total = total
            .checked_add(*len)
            .ok_or_else(|| frame_err("payload index lengths overflow"))?;
    }
    if total != raw_len {
        return Err(frame_err(
            "the payload index lengths do not sum to the raw length",
        ));
    }
    let raw = codec.decompress(
        "payload",
        &section[index_end..],
        raw_len,
        MAX_PAYLOAD_RAW,
        memory,
    )?;
    let mut at = 0usize;
    let mut out = Vec::with_capacity(index.len());
    for (artifact, len) in index {
        let end = usize::try_from(len)
            .ok()
            .and_then(|n| at.checked_add(n))
            .ok_or_else(|| frame_err("artifact length"))?;
        let bytes = raw
            .get(at..end)
            .ok_or_else(|| frame_err("an artifact extends past the payload"))?;
        out.push(InlineArtifact {
            artifact,
            bytes: bytes.to_vec(),
        });
        at = end;
    }
    Ok(out)
}

/// 2026-10-22: The raw artifacts of a payload `section` written with header `flags`.
pub(super) fn decode_payload(
    flags: u8,
    section: &[u8],
    memory: DecodeMemory,
) -> Result<Vec<InlineArtifact>, Error> {
    if flags & FLAG_SOLID != 0 {
        if flags & FLAG_FULL == 0 {
            return Err(frame_err("a solid payload in a lite file"));
        }
        return decode_solid(section, memory);
    }
    let entries: Vec<PayloadEntry> = decode("payload", section)?;
    entries
        .into_iter()
        .map(|e| {
            Ok(InlineArtifact {
                artifact: e.artifact,
                bytes: e.codec.decompress(
                    "inline artifact",
                    &e.stored,
                    e.raw_len,
                    MAX_ARTIFACT,
                    memory,
                )?,
            })
        })
        .collect()
}
