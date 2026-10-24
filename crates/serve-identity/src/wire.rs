// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Wire shapes a server answers and the benchmark harness reads back, so one type
//! is the contract on both sides: `GET /forward` ([`LiveForward`]) and `GET /memory`
//! ([`MemoryReport`]). Moved verbatim from metrale-bench (`gate::record_serve` and
//! `benchmarks::default_tier_boot::memory`), which re-export them.
//!
//! Owner: serve identity (wire shapes).
//! Invariants:
//! - A reading the server could not take is `None`, serialised as JSON `null`.
//! - Field names and types are unchanged from the bench originals, so every body a server
//!   ever answered still parses.

use alloc::string::String;

/// 2026-09-28: What a server reports about its forward (`GET /forward`): the server fills it
/// from its model, the harness reads it into the record.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LiveForward {
    /// 2026-09-28: `legacy`, `circuit` or `circuit-reference`.
    pub forward: String,
    /// 2026-09-28: The decode plan's digest; `None` under `legacy`.
    #[serde(default)]
    pub plan_digest: Option<String>,
    /// 2026-10-01: The slot count `--max-batch-size auto` resolved to; `None` for an explicit
    /// count (which the rendered serve already states).
    #[serde(default)]
    pub auto_max_batch_size: Option<usize>,
    /// 2026-10-02: The MoE expert-table decision the serve's memory plan made before load
    /// (`build` or `skip`); `None` when the loader reads none.
    #[serde(default)]
    pub moe_expert_tables: Option<String>,
    /// 2026-10-03: SHA-256 of the kernel tree the serve's memory plan was computed from
    /// (`metrale_kernel_tree::SHA256`); `None` when the serve made no plan.
    #[serde(default)]
    pub kernel_tree: Option<String>,
}

/// 2026-10-01: The body of `GET /memory`. Byte counts unless the name says otherwise.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryReport {
    /// 2026-10-01: Blocks in the model's main paged KV pool.
    pub kv_blocks: usize,
    /// 2026-10-01: Bytes of one block index across every layer, K and V.
    pub kv_block_bytes: u64,
    /// 2026-10-01: The slot count the model was built with.
    pub max_batch_size: usize,
    /// 2026-10-01: Total device memory times `--gpu-memory-utilization`.
    pub budget_bytes: u64,
    /// 2026-10-01: The GPU backend's allocation ledger at request time; `None` on a backend
    /// without one.
    pub ledger_live_bytes: Option<u64>,
    /// 2026-10-01: Host `MemAvailable`, sampled once before the GPU backend initialised.
    pub mem_available_at_start_bytes: Option<u64>,
    /// 2026-10-01: Host `MemAvailable` at request time.
    pub mem_available_now_bytes: Option<u64>,
    /// 2026-10-01: The server process's `RssAnon` at request time.
    pub rss_anon_bytes: Option<u64>,
}
