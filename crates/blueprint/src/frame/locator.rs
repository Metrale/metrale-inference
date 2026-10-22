// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The unsigned sections: where each module's bytes come from, the host
//! resources, and the inline payload.
//!
//! Owner: blueprint.
//! Invariants:
//! - Unsigned by design: every source is checked the same way (bytes → SHA-256 = the signed
//!   `ModuleSpec.artifact`), so a tampered locator can only fail a fetch, never change a kernel.
//! - `Unsigned.sources[i]` is the source of `KernelSet.modules[i]`.

use alloc::string::String;
use alloc::vec::Vec;

use metrale_error::{Error, ErrorCode, error};
use serde::{Deserialize, Serialize};

use super::MAX_ARTIFACT;
use crate::codec::Codec;
use crate::ids::{ArtifactId, Sha256};
use crate::types::Resources;

/// 2026-10-22: Where one module's bytes come from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelSource {
    /// 2026-10-22: The payload section, entry `entry`.
    Inline {
        /// 2026-10-22: Index into the payload.
        entry: u32,
    },
    /// 2026-10-22: The modules compiled into the loading binary.
    Internal,
    /// 2026-10-22: The local content-addressed cache (Phase 2).
    Disk {
        /// 2026-10-22: Cache path.
        path: String,
    },
    /// 2026-10-22: The registry (Phase 3).
    Remote {
        /// 2026-10-22: Locator URL.
        url: String,
        /// 2026-10-22: Digest of the locator record.
        locator_sha256: Sha256,
    },
}

impl KernelSource {
    /// 2026-10-22: `inline`, `internal`, `disk` or `remote`.
    pub const fn name(&self) -> &'static str {
        match self {
            KernelSource::Inline { .. } => "inline",
            KernelSource::Internal => "internal",
            KernelSource::Disk { .. } => "disk",
            KernelSource::Remote { .. } => "remote",
        }
    }
}

/// 2026-10-22: The locator section.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unsigned {
    /// 2026-10-22: One per module.
    pub sources: Vec<KernelSource>,
    /// 2026-10-22: Host resources (never signed).
    pub resources: Resources,
}

/// 2026-10-22: One inline artifact: its id, raw length, codec and stored bytes. The id is
/// over the raw bytes, so the codec never changes it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineArtifact {
    /// 2026-10-22: SHA-256 of the raw bytes.
    pub artifact: ArtifactId,
    /// 2026-10-22: Raw length.
    pub raw_len: u64,
    /// 2026-10-22: How `stored` encodes the raw bytes.
    pub codec: Codec,
    /// 2026-10-22: Stored bytes.
    pub stored: Vec<u8>,
}

impl InlineArtifact {
    /// 2026-10-22: `raw` stored under `codec`.
    pub fn pack(raw: &[u8], codec: Codec) -> Result<Self, Error> {
        Ok(Self {
            artifact: ArtifactId::of(raw),
            raw_len: u64::try_from(raw.len())
                .map_err(|_| error!(ErrorCode::BlueprintFrame, "artifact length"))?,
            codec,
            stored: codec.compress(raw),
        })
    }

    /// 2026-10-22: The raw bytes, bounded by [`MAX_ARTIFACT`]. Not hash-checked: the caller
    /// checks them against the signed module spec (`resolve`).
    pub fn unpack(&self) -> Result<Vec<u8>, Error> {
        self.codec
            .decompress("inline artifact", &self.stored, self.raw_len, MAX_ARTIFACT)
    }

    /// 2026-10-22: The same raw bytes stored under `codec`.
    pub fn recode(&self, codec: Codec) -> Result<Self, Error> {
        Ok(Self {
            artifact: self.artifact,
            raw_len: self.raw_len,
            codec,
            stored: codec.compress(&self.unpack()?),
        })
    }
}
