// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The storage tiers' old process-environment readers, kept for the callers not
//! yet handed [`StorageLinks`]: `RdmaSnapshotArena::connect`/`connect_paging` and the
//! `NgramRowCache` constructors. Each reproduces its old reader exactly; the built-in values
//! are the declared defaults of the settings rows.
//!
//! Owner: storage.
//! Invariants: nothing outside this module reads these variables from the process.

use std::sync::OnceLock;

use metrale_settings::sources::{CliLayer, RecipeLayer};
use metrale_settings::{Env, InferenceServerSettings};

use crate::links::{
    KvPagingLinks, PIPELINE_DEPTH_MAX, RailLink, SSM_CHUNK_BYTES_MIN, StorageLinks,
};

/// 2026-10-17: An environment with nothing set, for the declared defaults.
struct Unset;

impl Env for Unset {
    fn var(&self, _name: &str) -> metrale_error::Result<Option<String>> {
        Ok(None)
    }
    fn names(&self) -> Vec<String> {
        Vec::new()
    }
}

fn declared() -> InferenceServerSettings {
    metrale_settings::resolve(&Unset, &CliLayer::empty(), &RecipeLayer::empty())
        .expect("the declared defaults resolve")
}

/// 2026-10-17: [`storage_links_from`] over the process environment, read at each call as
/// the old readers did.
#[must_use]
pub fn storage_links() -> StorageLinks {
    storage_links_from(|name| std::env::var(name).ok())
}

/// 2026-10-17: The old readers over `var`: a device name set to anything (empty included)
/// is used verbatim; a GID or KV depth that does not parse as `u32`, untrimmed, falls back to
/// the default; the switches are on only for exactly `1`; a snapshot chunk that does not parse
/// or is below 4096 falls back to the default, and a snapshot depth that does not parse
/// falls back to the default.
#[must_use]
pub fn storage_links_from(var: impl Fn(&str) -> Option<String>) -> StorageLinks {
    let d = declared();
    let text = |name: &str, default: &str| var(name).unwrap_or_else(|| default.to_string());
    let num = |name: &str, default: i64| -> u32 {
        var(name)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| u32::try_from(default).expect("a declared u32 default"))
    };
    let one = |name: &str| var(name).as_deref() == Some("1");
    let usize_or = |name: &str, default: i64| -> usize {
        var(name)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| usize::try_from(default).expect("a declared usize default"))
    };
    let rail = |dev: &str, dev_default: &str, gid: &str, gid_default: i64| RailLink {
        dev: text(dev, dev_default),
        gid: num(gid, gid_default),
    };
    StorageLinks {
        rail0: rail(
            "METRALE_EXPERT_RDMA_DEV",
            &d.storage.expert_rdma_dev,
            "METRALE_EXPERT_RDMA_GID",
            d.storage.expert_rdma_gid,
        ),
        kv_rail1: one("METRALE_KV_DUAL_RAIL").then(|| {
            rail(
                "METRALE_KV_RAIL2_DEV",
                &d.cache.kv_rail2_dev,
                "METRALE_KV_RAIL2_GID",
                d.cache.kv_rail2_gid,
            )
        }),
        expert_rail1: one("METRALE_EXPERT_DUAL_RAIL").then(|| {
            rail(
                "METRALE_EXPERT_RAIL2_DEV",
                &d.kernels.expert_rail2_dev,
                "METRALE_EXPERT_RAIL2_GID",
                d.kernels.expert_rail2_gid,
            )
        }),
        kv_pipeline_depth: (num("METRALE_KV_PIPELINE_DEPTH", d.cache.kv_pipeline_depth) as usize)
            .clamp(1, PIPELINE_DEPTH_MAX),
        kv_zero_copy: one("METRALE_KV_ZERO_COPY"),
        ssm_staging: one("METRALE_SSM_STAGING"),
        ssm_chunk_bytes: var("METRALE_SSM_CHUNK_BYTES")
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v >= SSM_CHUNK_BYTES_MIN)
            .unwrap_or_else(|| {
                usize::try_from(d.scheduler.ssm_chunk_bytes).expect("a declared chunk")
            }),
        ssm_pipeline_depth: usize_or("METRALE_SSM_PIPELINE_DEPTH", d.kernels.ssm_pipeline_depth)
            .clamp(1, PIPELINE_DEPTH_MAX),
        expert_peer: var("METRALE_EXPERT_PEER"),
        kv_paging: KvPagingLinks {
            selected: var("METRALE_KV_PAGING"),
            arena_gb: var("METRALE_KV_PAGING_ARENA_GB"),
            ns: var("METRALE_KV_PAGING_NS"),
            salt: var("METRALE_KV_PAGING_SALT"),
        },
        ple_serial_fault: one("METRALE_PLE_SERIAL_FAULT"),
    }
}

/// 2026-10-17: `METRALE_PLE_SERIAL_FAULT`, exactly `1`, read once per process.
#[must_use]
pub fn ple_serial_fault() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("METRALE_PLE_SERIAL_FAULT").as_deref() == Ok("1"))
}
