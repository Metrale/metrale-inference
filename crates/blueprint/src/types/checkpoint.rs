// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The checkpoint and hardware pins.
//!
//! Owner: blueprint.
//! Invariants: wire types only (see `types/mod.rs`).

use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::ids::Sha256;

/// 2026-10-22: The checkpoint a blueprint serves.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPin {
    /// 2026-10-22: The HF repo id.
    pub repo: String,
    /// 2026-10-22: The git revision (commit sha).
    pub revision: String,
    /// 2026-10-22: SHA-256 of `config.json`.
    pub config_sha256: Sha256,
    /// 2026-10-22: SHA-256 of `hf_quant_config.json`, when the checkpoint has one.
    pub quant_config_sha256: Option<Sha256>,
    /// 2026-10-22: The weights.
    pub weights: WeightsDigest,
}

/// 2026-10-22: The weights pin.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightsDigest {
    /// 2026-10-22: Not pinned: the builder was told to skip it (a declared, reported mode).
    NotPinned,
    /// 2026-10-22: Every shard, by name order, and the digest over them.
    Shards {
        /// 2026-10-22: The shards.
        shards: Vec<ShardDigest>,
        /// 2026-10-22: `sha256` over each shard's `lv(name) ‖ len ‖ sha256`.
        root: Sha256,
    },
}

/// 2026-10-22: One weights shard.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardDigest {
    /// 2026-10-22: File name.
    pub name: String,
    /// 2026-10-22: Length in bytes.
    pub len: u64,
    /// 2026-10-22: SHA-256 of the file.
    pub sha256: Sha256,
}

impl WeightsDigest {
    /// 2026-10-22: The pin over `shards` (sorted by name here, so the caller's order is moot).
    pub fn of_shards(mut shards: Vec<ShardDigest>) -> Self {
        shards.sort_by(|a, b| a.name.cmp(&b.name));
        let mut lv = Vec::new();
        for s in &shards {
            lv.extend(metrale_blueprint_trust::lv_cat(&[
                s.name.as_bytes(),
                &s.len.to_le_bytes(),
                &s.sha256.0,
            ]));
        }
        let root = Sha256::of(&[b"kbp-weights\0", &lv]);
        WeightsDigest::Shards { shards, root }
    }
}

/// 2026-10-22: The GPU vendor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Vendor {
    /// 2026-10-22: NVIDIA (PTX).
    Nvidia,
    /// 2026-10-22: AMD (HIP code objects).
    Amd,
    /// 2026-10-22: Apple (Metal libraries).
    Apple,
}

impl Vendor {
    /// 2026-10-22: The `HARDWARE.toml` spelling.
    pub const fn name(self) -> &'static str {
        match self {
            Vendor::Nvidia => "nvidia",
            Vendor::Amd => "amd",
            Vendor::Apple => "apple",
        }
    }

    /// 2026-10-22: The vendor a `HARDWARE.toml` spelling names.
    pub fn parse(s: &str) -> Option<Self> {
        [Vendor::Nvidia, Vendor::Amd, Vendor::Apple]
            .into_iter()
            .find(|v| v.name() == s)
    }
}

/// 2026-10-22: What hardware the blueprint runs on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareSpec {
    /// 2026-10-22: The class (`kernels/<class>/HARDWARE.toml`), e.g. `gb10`.
    pub class: String,
    /// 2026-10-22: The PTX arch the kernels were compiled for, e.g. `sm_121f`.
    pub ptx_arch: String,
    /// 2026-10-22: The vendor.
    pub vendor: Vendor,
    /// 2026-10-22: SMs per device, as the class declares.
    pub sm_count: u32,
    /// 2026-10-22: Ranks.
    pub world_size: u32,
    /// 2026-10-22: The least device memory a rank must see, MiB.
    pub min_visible_mib: u64,
    /// 2026-10-22: Device ids allowed (the class's devices).
    pub devices: Vec<String>,
}
