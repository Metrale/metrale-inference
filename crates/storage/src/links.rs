// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `StorageLinks`, the transport levers the storage tiers read when they connect
//! a peer or fault a table, projected once from the resolved settings and passed to the
//! constructors (ROTI). The tiers read no environment through it.
//!
//! Owner: storage.
//! Invariants:
//! - Reads only [`InferenceServerSettings`]; a value a tier cannot use is an error, never a
//!   silent default.
//! - Pipeline depths are clamped to `1..=`[`PIPELINE_DEPTH_MAX`], as the tiers always did.

use anyhow::{Context, Result, bail};
use metrale_settings::InferenceServerSettings;

/// 2026-10-17: Most transfers in flight per rail, for the KV backend and the snapshot stripes.
pub const PIPELINE_DEPTH_MAX: usize = 128;
/// 2026-10-17: Smallest snapshot stripe chunk, in bytes.
pub const SSM_CHUNK_BYTES_MIN: usize = 4096;

/// 2026-10-17: One verbs rail: the RDMA device and its GID index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RailLink {
    pub dev: String,
    pub gid: u32,
}

/// 2026-10-17: The KV-paging levers, verbatim; `connect_kv_peer_backend` parses them only
/// when paging is selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvPagingLinks {
    /// 2026-10-17: `METRALE_KV_PAGING`.
    pub selected: Option<String>,
    /// 2026-10-17: `METRALE_KV_PAGING_ARENA_GB`.
    pub arena_gb: Option<String>,
    /// 2026-10-17: `METRALE_KV_PAGING_NS`.
    pub ns: Option<String>,
    /// 2026-10-17: `METRALE_KV_PAGING_SALT`.
    pub salt: Option<String>,
}

/// 2026-10-17: The levers the storage tiers read at connect and fault time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageLinks {
    /// 2026-10-17: Rail 0 of every tier: `METRALE_EXPERT_RDMA_DEV`/`GID`.
    pub rail0: RailLink,
    /// 2026-10-17: KV and snapshot rail 1, present when `METRALE_KV_DUAL_RAIL` is on:
    /// `METRALE_KV_RAIL2_DEV`/`GID`.
    pub kv_rail1: Option<RailLink>,
    /// 2026-10-17: Expert rail 1, present when `METRALE_EXPERT_DUAL_RAIL` is on:
    /// `METRALE_EXPERT_RAIL2_DEV`/`GID`.
    pub expert_rail1: Option<RailLink>,
    /// 2026-10-17: `METRALE_KV_PIPELINE_DEPTH`, clamped.
    pub kv_pipeline_depth: usize,
    /// 2026-10-17: `METRALE_KV_ZERO_COPY`.
    pub kv_zero_copy: bool,
    /// 2026-10-17: `METRALE_SSM_STAGING`.
    pub ssm_staging: bool,
    /// 2026-10-17: `METRALE_SSM_CHUNK_BYTES`, at least [`SSM_CHUNK_BYTES_MIN`].
    pub ssm_chunk_bytes: usize,
    /// 2026-10-17: `METRALE_SSM_PIPELINE_DEPTH`, clamped.
    pub ssm_pipeline_depth: usize,
    /// 2026-10-17: `METRALE_EXPERT_PEER`, the `host:port` of the expert peer.
    pub expert_peer: Option<String>,
    pub kv_paging: KvPagingLinks,
    /// 2026-10-17: `METRALE_PLE_SERIAL_FAULT`.
    pub ple_serial_fault: bool,
}

impl StorageLinks {
    /// 2026-10-17: Project the resolved settings. A negative GID or depth, or a snapshot chunk
    /// below [`SSM_CHUNK_BYTES_MIN`], is an error.
    pub fn from_settings(s: &InferenceServerSettings) -> Result<Self> {
        let rail0 = RailLink {
            dev: s.storage.expert_rdma_dev.clone(),
            gid: gid("METRALE_EXPERT_RDMA_GID", s.storage.expert_rdma_gid)?,
        };
        let kv_rail1 = RailLink {
            dev: s.cache.kv_rail2_dev.clone(),
            gid: gid("METRALE_KV_RAIL2_GID", s.cache.kv_rail2_gid)?,
        };
        let expert_rail1 = RailLink {
            dev: s.kernels.expert_rail2_dev.clone(),
            gid: gid("METRALE_EXPERT_RAIL2_GID", s.kernels.expert_rail2_gid)?,
        };
        let chunk = s.scheduler.ssm_chunk_bytes;
        let ssm_chunk_bytes = usize::try_from(chunk)
            .ok()
            .filter(|&v| v >= SSM_CHUNK_BYTES_MIN)
            .with_context(|| {
                format!("METRALE_SSM_CHUNK_BYTES={chunk} is below {SSM_CHUNK_BYTES_MIN}")
            })?;
        Ok(Self {
            rail0,
            kv_rail1: s.cache.kv_dual_rail.then_some(kv_rail1),
            expert_rail1: s.kernels.expert_dual_rail.then_some(expert_rail1),
            kv_pipeline_depth: depth("METRALE_KV_PIPELINE_DEPTH", s.cache.kv_pipeline_depth)?,
            kv_zero_copy: s.cache.kv_zero_copy,
            ssm_staging: s.kernels.ssm_staging,
            ssm_chunk_bytes,
            ssm_pipeline_depth: depth("METRALE_SSM_PIPELINE_DEPTH", s.kernels.ssm_pipeline_depth)?,
            expert_peer: s.runtime.expert_peer.clone(),
            kv_paging: KvPagingLinks {
                selected: s.runtime.kv_paging.clone(),
                arena_gb: s.runtime.kv_paging_arena_gb.clone(),
                ns: s.cache.kv_paging_ns.clone(),
                salt: s.cache.kv_paging_salt.clone(),
            },
            ple_serial_fault: s.model.ple_serial_fault,
        })
    }

    /// 2026-10-17: The KV and snapshot rails, rail 0 first.
    #[must_use]
    pub fn kv_rails(&self) -> Vec<RailLink> {
        rails(&self.rail0, self.kv_rail1.as_ref())
    }

    /// 2026-10-17: The expert-tier rails, rail 0 first.
    #[must_use]
    pub fn expert_rails(&self) -> Vec<RailLink> {
        rails(&self.rail0, self.expert_rail1.as_ref())
    }
}

fn rails(rail0: &RailLink, rail1: Option<&RailLink>) -> Vec<RailLink> {
    std::iter::once(rail0).chain(rail1).cloned().collect()
}

fn gid(name: &str, v: i64) -> Result<u32> {
    u32::try_from(v).with_context(|| format!("{name}={v} is not a GID index"))
}

fn depth(name: &str, v: i64) -> Result<usize> {
    let Ok(d) = usize::try_from(v) else {
        bail!("{name}={v} is negative");
    };
    Ok(d.clamp(1, PIPELINE_DEPTH_MAX))
}
