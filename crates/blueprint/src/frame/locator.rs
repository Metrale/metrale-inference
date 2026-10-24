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

use serde::{Deserialize, Serialize};

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

/// 2026-10-22: One inline artifact: its id and raw bytes. The id is over the raw bytes; how
/// the payload section stores them (`super::payload`) never changes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InlineArtifact {
    /// 2026-10-22: SHA-256 of the raw bytes, as the payload section states it. Not checked here:
    /// the caller checks the bytes against the signed module spec (`resolve`).
    pub artifact: ArtifactId,
    /// 2026-10-22: The raw bytes.
    pub bytes: Vec<u8>,
}

impl InlineArtifact {
    /// 2026-10-22: `raw` with its id.
    pub fn of(raw: &[u8]) -> Self {
        Self {
            artifact: ArtifactId::of(raw),
            bytes: raw.to_vec(),
        }
    }
}
