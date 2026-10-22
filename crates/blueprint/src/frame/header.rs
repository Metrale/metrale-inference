// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The fixed 72-byte `.kbp` header and its caps.
//!
//! ```text
//! [0..4]   magic "MKBP"      [4..6] KBP_FORMAT u16   [6..8] flags u16
//! [8..16]  body_len          [16..48] body_sha256     [48..56] locator_len
//! [56..64] payload_len       [64..72] sig_len         then body, locator, payload, sig
//! ```
//!
//! Owner: blueprint.
//! Invariants:
//! - Little-endian; every length is `u64` and checked against its cap before anything is
//!   sliced or allocated; the sections tile the file exactly (no trailing bytes).
//! - Unknown flag bits and an unknown format are refused.

use metrale_error::{Error, ErrorCode, error};

/// 2026-10-22: The file magic.
pub const MAGIC: &[u8; 4] = b"MKBP";
/// 2026-10-22: The framing version; bumps only on a framing change.
pub const KBP_FORMAT: u16 = 1;
/// 2026-10-22: The header length.
pub const HEADER_LEN: usize = 72;
/// 2026-10-22: Flag: the payload section carries inline artifacts.
pub const FLAG_FULL: u16 = 1;
/// 2026-10-22: Flag: the body section is deflated.
pub const FLAG_COMPRESSED: u16 = 2;
/// 2026-10-22: The flags the signature covers (`FLAG_FULL` is rewritten by lite/full).
pub const SIGNED_FLAGS: u16 = FLAG_COMPRESSED;

/// 2026-10-22: Stored body cap (hard refusal, DESIGN.md §3).
pub const MAX_BODY: u64 = 4 << 20;
/// 2026-10-22: Inflated body cap: a deflate bomb stops here.
pub const MAX_BODY_INFLATED: u64 = 64 << 20;
/// 2026-10-22: Locator section cap.
pub const MAX_LOCATOR: u64 = 16 << 20;
/// 2026-10-22: Payload section cap (the 1 GiB full-file cap).
pub const MAX_PAYLOAD: u64 = 1 << 30;
/// 2026-10-22: Signature block cap.
pub const MAX_SIG: u64 = 256 << 10;
/// 2026-10-22: One inflated artifact's cap.
pub const MAX_ARTIFACT: u64 = 256 << 20;

/// 2026-10-22: The parsed header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// 2026-10-22: Flag bits.
    pub flags: u16,
    /// 2026-10-22: Body section length.
    pub body_len: u64,
    /// 2026-10-22: SHA-256 of the body section.
    pub body_sha256: [u8; 32],
    /// 2026-10-22: Locator section length.
    pub locator_len: u64,
    /// 2026-10-22: Payload section length.
    pub payload_len: u64,
    /// 2026-10-22: Signature block length (0: unsigned).
    pub sig_len: u64,
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

fn cap(section: &'static str, len: u64, max: u64) -> Result<(), Error> {
    if len > max {
        return Err(error!(ErrorCode::BlueprintTooLarge, section, len, max));
    }
    Ok(())
}

/// 2026-10-22: Byte ranges of the four sections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sections {
    /// 2026-10-22: Body.
    pub body: core::ops::Range<usize>,
    /// 2026-10-22: Locator table.
    pub locator: core::ops::Range<usize>,
    /// 2026-10-22: Payload.
    pub payload: core::ops::Range<usize>,
    /// 2026-10-22: Signature block.
    pub sig: core::ops::Range<usize>,
}

impl Header {
    /// 2026-10-22: Parse and bound-check the header of `file`; the sections must tile it.
    pub fn parse(file: &[u8]) -> Result<(Self, Sections), Error> {
        if file.len() < HEADER_LEN {
            return Err(error!(ErrorCode::BlueprintFrame, "truncated header"));
        }
        if &file[0..4] != MAGIC {
            return Err(error!(ErrorCode::BlueprintFrame, "bad magic"));
        }
        let format = u16::from_le_bytes([file[4], file[5]]);
        if format != KBP_FORMAT {
            return Err(error!(
                ErrorCode::BlueprintFormat,
                "KBP_FORMAT", format, KBP_FORMAT
            ));
        }
        let flags = u16::from_le_bytes([file[6], file[7]]);
        if flags & !(FLAG_FULL | FLAG_COMPRESSED) != 0 {
            return Err(error!(ErrorCode::BlueprintFrame, "unknown flag bits"));
        }
        let mut body_sha256 = [0u8; 32];
        body_sha256.copy_from_slice(&file[16..48]);
        let h = Header {
            flags,
            body_len: u64_at(file, 8),
            body_sha256,
            locator_len: u64_at(file, 48),
            payload_len: u64_at(file, 56),
            sig_len: u64_at(file, 64),
        };
        cap("body", h.body_len, MAX_BODY)?;
        cap("locator", h.locator_len, MAX_LOCATOR)?;
        cap("payload", h.payload_len, MAX_PAYLOAD)?;
        cap("signature", h.sig_len, MAX_SIG)?;
        let total = [h.body_len, h.locator_len, h.payload_len, h.sig_len]
            .iter()
            .try_fold(HEADER_LEN as u64, |a, b| a.checked_add(*b))
            .ok_or_else(|| error!(ErrorCode::BlueprintFrame, "section lengths overflow"))?;
        let file_len = u64::try_from(file.len())
            .map_err(|_| error!(ErrorCode::BlueprintFrame, "file length"))?;
        if total != file_len {
            return Err(error!(
                ErrorCode::BlueprintFrame,
                if total > file_len {
                    "truncated: the sections extend past the end"
                } else {
                    "trailing bytes after the sections"
                }
            ));
        }
        let at = |n: u64| {
            usize::try_from(n).map_err(|_| error!(ErrorCode::BlueprintFrame, "section offset"))
        };
        let b0 = HEADER_LEN;
        let b1 = b0 + at(h.body_len)?;
        let l1 = b1 + at(h.locator_len)?;
        let p1 = l1 + at(h.payload_len)?;
        let s1 = p1 + at(h.sig_len)?;
        Ok((
            h,
            Sections {
                body: b0..b1,
                locator: b1..l1,
                payload: l1..p1,
                sig: p1..s1,
            },
        ))
    }

    /// 2026-10-22: The 72 header bytes.
    pub fn write(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..4].copy_from_slice(MAGIC);
        b[4..6].copy_from_slice(&KBP_FORMAT.to_le_bytes());
        b[6..8].copy_from_slice(&self.flags.to_le_bytes());
        b[8..16].copy_from_slice(&self.body_len.to_le_bytes());
        b[16..48].copy_from_slice(&self.body_sha256);
        b[48..56].copy_from_slice(&self.locator_len.to_le_bytes());
        b[56..64].copy_from_slice(&self.payload_len.to_le_bytes());
        b[64..72].copy_from_slice(&self.sig_len.to_le_bytes());
        b
    }

    /// 2026-10-22: The bytes the signature covers:
    /// `"kbp-sig-v1\0" ‖ KBP_FORMAT ‖ signed flags ‖ body_len ‖ body_sha256 ‖ body`.
    pub fn signed_message(&self, body: &[u8]) -> alloc::vec::Vec<u8> {
        let mut m = alloc::vec::Vec::with_capacity(body.len() + 64);
        m.extend_from_slice(b"kbp-sig-v1\0");
        m.extend_from_slice(&KBP_FORMAT.to_le_bytes());
        m.extend_from_slice(&(self.flags & SIGNED_FLAGS).to_le_bytes());
        m.extend_from_slice(&self.body_len.to_le_bytes());
        m.extend_from_slice(&self.body_sha256);
        m.extend_from_slice(body);
        m
    }
}
