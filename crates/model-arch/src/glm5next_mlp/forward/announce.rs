// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The once-per-row-count log lines of the routed W4A16 paths, moved verbatim out of
//! `forward.rs` (which keeps the dispatch) to hold that file under 500 lines. 2026-10-16: also the
//! wide sweep's readback.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: logging only; no launch depends on these.

use super::MOE_ROW_BATCH_MAX_ROWS;

/// 2026-09-25: Log once per row count (counts from 15 up share one bit) whether the routed
/// experts took the row-batched path.
pub(super) fn announce_row_batch(batched: bool, rows: usize) {
    use std::sync::atomic::{AtomicU16, Ordering};
    static SEEN: AtomicU16 = AtomicU16::new(0);
    let bit = 1u16 << rows.min(15);
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
        return;
    }
    if batched {
        tracing::info!("GLM MoE: row-batched expert union ({rows} rows, one sweep each)");
    } else {
        tracing::info!("GLM MoE: per-row expert sweeps ({rows} rows)");
    }
}

/// 2026-09-25: Log once per row count (counts from 31 up share one bit) whether a row group
/// wider than `MOE_ROW_BATCH_MAX_ROWS` took the grouped GEMM; narrower groups log nothing.
pub(super) fn announce_grouped_prefill(on: bool, rows: usize) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    if rows <= MOE_ROW_BATCH_MAX_ROWS {
        return;
    }
    let bit = 1u32 << (rows.min(31));
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
        return;
    }
    if on {
        let t = super::super::forward_prefill_gemm::gemm_tile();
        tracing::info!(
            "GLM MoE prefill: grouped tensor-core W4A16 GEMM, {rows} rows in ONE launch \
             per projection, tile `{}` (M_TILE {}, N_TILE {}, {} threads) \
             (METRALE_GLM_MOE_PREFILL_GEMM=0 to restore the GEMV path; \
             METRALE_GLM_MOE_GEMM_TILE=base for the prior tile)",
            t.name,
            t.m_tile,
            t.n_tile,
            t.threads
        );
    } else {
        tracing::info!(
            "GLM MoE prefill: row-batched GEMV, {rows} rows split into \
             {MOE_ROW_BATCH_MAX_ROWS}-row sweeps"
        );
    }
}

/// 2026-10-16: Log once per row count the routed W4A4 launches past
/// [`MOE_W4A4_MAX_ROWS`](crate::glm5next_mlp::MOE_W4A4_MAX_ROWS) that `on` (the wide sweep lever
/// at the batched decode site, the launch not grouped) puts on the wide union sweep: the
/// lever's readback.
pub(super) fn announce_wide_sweep(on: bool, rows: usize) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEEN: AtomicU64 = AtomicU64::new(0);
    if !on || rows <= crate::glm5next_mlp::MOE_W4A4_MAX_ROWS {
        return;
    }
    let bit = 1u64 << rows.min(63);
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        tracing::info!(
            "GLM MoE: METRALE_GLM_MOE_W4A4_WIDE_SWEEP=1 -- {rows}-row batched decode routed \
             launch on the wide union sweep"
        );
    }
}
