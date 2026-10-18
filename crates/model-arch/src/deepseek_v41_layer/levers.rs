// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: DeepSeek-V4.1 levers, resolved from the startup settings once per load.
//!
//! Owner: model-arch (DeepSeek-V4.1).
//! Invariants:
//! - `Ds41LoadLevers::from_settings` runs once, in `load_layers`; the `run` half is copied onto
//!   `V41Runtime`, `MoeV41` and `AttnV41`, so a lever is fixed for the model's lifetime, as the
//!   process-wide `OnceLock`s it replaces were, and CUDA-graph captures see one choice.
//! - Each field reads one settings field, with the parse its old reader applied.

use metrale_settings::InferenceServerSettings;

use super::GraphMode;

/// 2026-10-17: A count with a numeric settings default reads a negative value as that default,
/// as the `usize` parse it replaced did. Each constant equals its row's declared default
/// (checked by `count_fallbacks_are_the_declared_defaults`).
pub(crate) const MAX_SEQ: usize = 8192;
pub(crate) const MAX_TOKENS: usize = 2048;
pub(crate) const READER_THREADS: usize = 16;
pub(crate) const STAGING_SLOTS: usize = 16;

fn count(value: i64, negative_reads_as: usize) -> usize {
    usize::try_from(value).unwrap_or(negative_reads_as)
}

fn parsed<T: core::str::FromStr>(value: Option<&str>) -> Option<T> {
    value.and_then(|v| v.parse().ok())
}

/// 2026-10-17: The levers the per-step code reads (layer step, MoE, attention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ds41Levers {
    /// 2026-10-17: `METRALE_DS41_DIAG`: per-layer rms lines, step totals, and a synchronise at
    /// the end of `MoeV41::forward` so `compute_ms` is GPU time.
    pub diag: bool,
    /// 2026-10-17: `METRALE_DS41_TRACE`: the eager step logs FNV-1a checksums of each layer's
    /// attention output, normed input and MoE input and output.
    pub trace: bool,
    /// 2026-10-17: `METRALE_DS41_STEP_GRAPH`: the multi-layer segment graphs (`step_seg.rs`).
    pub step_graph: bool,
    /// 2026-10-17: `METRALE_DS41_STEP_GRAPH_LOG`: running counts when the last segment replays
    /// without a miss, and a line per segment miss.
    pub step_graph_log: bool,
    /// 2026-10-17: `METRALE_DS41_GRAPH_ORACLE` (wins) or `METRALE_DS41_GRAPH`.
    pub graph_mode: GraphMode,
    /// 2026-10-17: `METRALE_DS41_NO_ENGRAM`: skip the engram.
    pub no_engram: bool,
    /// 2026-10-17: `METRALE_DS41_PREFETCH_K`: next-layer experts to start reading while this
    /// layer computes; unset or unparseable is 0 (off).
    pub prefetch_k: usize,
    /// 2026-10-17: `METRALE_DS41_SHARED_SIDE`: the single-token shared expert on a side stream.
    pub shared_side: bool,
    /// 2026-10-17: `METRALE_DS41_ROUTER_STAGED`: the router GEMV writes per-block products.
    pub router_staged: bool,
    /// 2026-10-17: `METRALE_DS41_M1_LOOP`: a single token takes the per-expert group loop.
    pub m1_loop: bool,
    /// 2026-10-17: `METRALE_DS41_PREFILL_GEMV`: groups over eight rows run on the decode GEMV.
    pub prefill_gemv: bool,
    /// 2026-10-17: `METRALE_DS41_DEVICE_ROUTE`: the single-token selection runs on the device.
    pub device_route: bool,
    /// 2026-10-17: `METRALE_DS41_DEVICE_ROUTE_DIAG`: check that selection against the host's.
    pub device_route_diag: bool,
    /// 2026-10-17: `METRALE_DS41_SPARSE_SPLIT`: output-dimension split of the sparse attention,
    /// 1..=8; anything else is 2.
    pub sparse_split: u32,
    /// 2026-10-17: `METRALE_DS41_EXPERT_WARPS`: warps a block of the single-token expert GEMV,
    /// 2, 4 or 8; anything else is 8.
    pub expert_warps: u32,
}

impl Ds41Levers {
    /// 2026-10-17: Resolve from the startup settings.
    pub fn from_settings(s: &InferenceServerSettings) -> Self {
        let (sc, tm, k, st) = (&s.scheduler, &s.telemetry, &s.kernels, &s.storage);
        Self {
            diag: tm.ds41_diag,
            trace: tm.ds41_trace,
            step_graph: sc.ds41_step_graph,
            step_graph_log: tm.ds41_step_graph_log,
            graph_mode: if k.ds41_graph_oracle {
                GraphMode::Oracle
            } else if k.ds41_graph {
                GraphMode::On
            } else {
                GraphMode::Off
            },
            no_engram: s.speculative.ds41_no_engram,
            prefetch_k: parsed(st.ds41_prefetch_k.as_deref()).unwrap_or(0),
            shared_side: s.model.ds41_shared_side,
            router_staged: s.model.ds41_router_staged,
            m1_loop: sc.ds41_m1_loop,
            prefill_gemv: sc.ds41_prefill_gemv,
            device_route: s.runtime.ds41_device_route,
            device_route_diag: tm.ds41_device_route_diag,
            sparse_split: parsed(k.ds41_sparse_split.as_deref())
                .filter(|s| (1..=8).contains(s))
                .unwrap_or(2),
            expert_warps: parsed(k.ds41_expert_warps.as_deref())
                .filter(|w| matches!(w, 2 | 4 | 8))
                .unwrap_or(8),
        }
    }
}

/// 2026-10-17: The levers the loader reads, and the `run` half it hands to the runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct Ds41LoadLevers {
    /// 2026-10-17: `METRALE_DS41_MAX_SEQ`, before the cap at `max_position_embeddings`.
    pub max_seq: usize,
    /// 2026-10-17: `METRALE_DS41_MAX_TOKENS`, before the cap at `max_seq`.
    pub max_tokens: usize,
    /// 2026-10-17: `METRALE_DS41_READER_THREADS`: expert reader threads.
    pub reader_threads: usize,
    /// 2026-10-17: `METRALE_DS41_STAGING_SLOTS`: the device arena's page-locked staging ring.
    pub staging_slots: usize,
    /// 2026-10-17: `METRALE_DS41_CANDIDATE_SOURCE`, used when the config names no candidate
    /// source layer; `None` (unset or negative) leaves the loader its built-in layer.
    pub candidate_source: Option<usize>,
    /// 2026-10-17: `METRALE_DS41_EXPERT_CACHE_GIB`: a finite positive size, else 88.
    pub expert_cache_gib: f64,
    /// 2026-10-17: `METRALE_DS41_ARENA_DEVICE`: device expert slots behind a staging ring
    /// (off: a page-locked arena the GPU reads in place).
    pub arena_device: bool,
    /// 2026-10-17: `METRALE_DS41_READER_POOL`: persistent expert reader threads.
    pub reader_pool: bool,
    /// 2026-10-17: `METRALE_DS41_ENGRAM_Q2K`: the engram `wkv` raw Q2_K on the device.
    pub engram_q2k: bool,
    /// 2026-10-17: `METRALE_DS41_ROUTE_TRACE=<path>`: the expert route trace file.
    pub route_trace: Option<String>,
    /// 2026-10-17: `METRALE_DS41_PREDICT_TRACE=<path>`: the routing prediction trace file.
    pub predict_trace: Option<String>,
    pub run: Ds41Levers,
}

impl Ds41LoadLevers {
    /// 2026-10-17: Resolve from the startup settings; `load_layers` calls it once.
    pub fn from_settings(s: &InferenceServerSettings) -> Self {
        let (sc, st, tm, ca) = (&s.scheduler, &s.storage, &s.telemetry, &s.cache);
        Self {
            max_seq: count(sc.ds41_max_seq, MAX_SEQ),
            max_tokens: count(sc.ds41_max_tokens, MAX_TOKENS),
            reader_threads: count(st.ds41_reader_threads, READER_THREADS),
            staging_slots: count(ca.ds41_staging_slots, STAGING_SLOTS),
            candidate_source: s
                .model
                .ds41_candidate_source
                .and_then(|v| usize::try_from(v).ok()),
            expert_cache_gib: parsed(ca.ds41_expert_cache_gib.as_deref())
                .filter(|g: &f64| g.is_finite() && *g > 0.0)
                .unwrap_or(88.0),
            arena_device: s.runtime.ds41_arena_device,
            reader_pool: s.model.ds41_reader_pool,
            engram_q2k: s.speculative.ds41_engram_q2k,
            route_trace: tm.ds41_route_trace.clone(),
            predict_trace: tm.ds41_predict_trace.clone(),
            run: Ds41Levers::from_settings(s),
        }
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
pub(crate) mod tests;
