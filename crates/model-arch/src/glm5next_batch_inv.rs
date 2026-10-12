// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `METRALE_GLM_BATCH_INVARIANT=1`: the GLM-5.3 forward on canonical tiers, so a row's
//! bits depend only on its own inputs, never on how many rows share a launch, which rows those are,
//! how a prompt is cut into prefill pieces, or whether a prefix-cache hit moved the cut:
//! - W4A16 dense projections run the tensor-core row tile (`w4a16_tc_rows` order) at every row
//!   count, one row included ([`one_row_warps`]); never the GEMV order, the GEMV tiers or the
//!   dequantize + cuBLASLt path.
//! - BF16 projections (router, DSA indexer `wq_b` / `weights_proj`, `glm_mm`) above
//!   `DENSE_GEMV_BATCHM_MAX_M` rows run as 16-row launches of the row-invariant batched GEMV, not
//!   cuBLASLt (whose algorithm depends on M).
//! - Routed experts run W4A4 at every row count (`METRALE_GLM_MOE_W4A4_GROUPED=1`, whose grouped
//!   GEMM writes the slot GEMV's bytes).
//! - Every tensor-parallel all-reduce takes the one-shot path (FP32 sum in rank order, one BF16
//!   rounding), cut into whole-row slices that fit its buffer ([`ar_slices`]).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Unset or `0`: nothing changes anywhere.
//! - `1` refuses at load a lever that would put a row on another order ([`check_levers`]).
//! - Numerics: decode rows change from the GEMV order to the row tile's at one row only; prefill
//!   pieces above 16 rows change from cuBLASLt to the batched GEMV and from W4A16 to W4A4 experts;
//!   prefill all-reduces above the one-shot bound change from NCCL to the rank-order sum. So the
//!   lever is accuracy-gated (class RO against the w4a16 pin).

use std::sync::OnceLock;

use anyhow::{Result, ensure};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::QuantizedWeight;

/// 2026-10-11: `METRALE_GLM_BATCH_INVARIANT`, read once: unset or `0` off, `1` on; another value
/// is an error.
pub fn enabled() -> Result<bool> {
    static V: OnceLock<Result<bool, String>> = OnceLock::new();
    V.get_or_init(|| parse(std::env::var("METRALE_GLM_BATCH_INVARIANT").ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-11: [`enabled`]'s parse, pure.
pub(crate) fn parse(v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "METRALE_GLM_BATCH_INVARIANT={other:?}: expected 0 or 1"
        )),
    }
}

/// 2026-10-11: The other levers' values the canonical tiers depend on, read by the loader.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Levers {
    /// `METRALE_GLM_W4A16_GEMV_SEG=1`: one row on the GEMV order.
    pub gemv_seg: bool,
    /// `METRALE_GLM_W4A16_GEMV_ROWS`: 2..=r rows on the GEMV tiers (0 = off).
    pub gemv_rows: u32,
    /// `METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS` is set (cuBLASLt over a dequantized weight).
    pub dequant_gemm: bool,
    /// `METRALE_GLM_MOE_W4A4_GROUPED=1` and its kernels resolved (W4A4 experts at every M).
    pub w4a4_grouped_ready: bool,
    /// `METRALE_GLM_DSA_BATCH_QIDX=1` (indexer `wq_b` on cuBLASLt over a whole prefill chunk).
    pub dsa_batch_qidx: bool,
    /// `METRALE_GLM_KDA_CHUNK_PREFILL=1` or `METRALE_GLM_KDA_CHUNK_SCAN` set (chunked scans whose
    /// boundaries follow the prefill piece, not the token).
    pub kda_chunked: bool,
}

/// 2026-10-11: Under the lever, the levers it cannot run with, each named; off, nothing.
pub(crate) fn check_levers(on: bool, l: Levers) -> Result<(), String> {
    if !on {
        return Ok(());
    }
    let mut bad = Vec::new();
    if l.gemv_seg {
        bad.push("METRALE_GLM_W4A16_GEMV_SEG=1 (one row on the GEMV order)");
    }
    if l.gemv_rows > 0 {
        bad.push("METRALE_GLM_W4A16_GEMV_ROWS (rows on the GEMV tiers)");
    }
    if l.dequant_gemm {
        bad.push("METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS (cuBLASLt picks its order by M)");
    }
    if !l.w4a4_grouped_ready {
        bad.push(
            "METRALE_GLM_MOE_W4A4_GROUPED=1 with its kernels is required (else routed experts \
             switch to W4A16 above 16 rows)",
        );
    }
    if l.dsa_batch_qidx {
        bad.push("METRALE_GLM_DSA_BATCH_QIDX=1 (cuBLASLt over a prefill chunk)");
    }
    if l.kda_chunked {
        bad.push(
            "METRALE_GLM_KDA_CHUNK_PREFILL / METRALE_GLM_KDA_CHUNK_SCAN / METRALE_GLM_KDA_CHUNK64 \
             (scan chunks follow the prefill piece)",
        );
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "METRALE_GLM_BATCH_INVARIANT=1 cannot run with: {}",
            bad.join("; ")
        ))
    }
}

/// 2026-10-11: At load (the routed experts' precision plan, once its kernels resolved): under the
/// lever, refuse the levers [`check_levers`] names. Off, nothing.
pub(crate) fn check_at_load(k: &crate::glm5next_mlp::Glm5NextMlpKernels) -> Result<()> {
    if !enabled()? {
        return Ok(());
    }
    let l = Levers {
        gemv_seg: crate::glm5next_w4a16_seg::gemv_seg()?,
        gemv_rows: crate::glm5next_w4a16_seg::gemv_rows()?,
        dequant_gemm: crate::glm5next_w4a16_wide::min_rows()?.is_some(),
        w4a4_grouped_ready: crate::glm5next_mlp::forward::moe_w4a4_grouped()?
            && k.w4a4_expert_rows_grouped() == usize::MAX,
        dsa_batch_qidx: crate::glm5next_layer::dsa_batch_qidx(),
        kda_chunked: crate::glm5next_layer::levers::kda_chunk_prefill()
            || crate::glm5next_kda::chunk16::kda_chunk_scan_min_rows()?.is_some()
            || crate::glm5next_kda::chunk64::kda_chunk64_min_rows()?.is_some(),
    };
    check_levers(true, l).map_err(anyhow::Error::msg)
}

/// 2026-10-11: The warps per block of a one-row W4A16 launch under the lever: 4 is
/// `w4a16_tc_rows_16`, 1 and 2 the narrow points (`ops::w4a16_tc_rows_narrow`); every choice
/// writes the same bytes, so this is speed only.
pub(crate) fn one_row_warps() -> u32 {
    4
}

/// 2026-10-11: `c [1, n] = a [1, k] @ W^T` for a registered W4A16 weight in the row tile's order
/// (the bits that row gets in any launch of 1..=64 rows), on [`one_row_warps`] warps per block.
pub(crate) fn one_row(
    gpu: &dyn GpuBackend,
    w: &QuantizedWeight,
    a: DevicePtr,
    c: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    match one_row_warps() {
        4 => ops::w4a16_tc_rows(gpu, a, w, c, 1, n, k, k, n, stream),
        warps => ops::w4a16_tc_rows_narrow(gpu, a, w, c, 1, n, k, k, n, warps, stream),
    }
}

/// 2026-10-11: The whole-row slices `(first row, rows)` one all-reduce of `rows` rows of
/// `row_bytes` takes: `[(0, rows)]` when the lever is off or at most two ranks reduce (a sum of
/// two partials has one order); else slices of the most rows that fit `oneshot_max_bytes`, so
/// every slice takes the one-shot path. Errors when the lever is on with more than two ranks and
/// no one-shot buffer holds a row.
pub(crate) fn ar_slices(
    on: bool,
    world_size: usize,
    oneshot_max_bytes: usize,
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<(usize, usize)>> {
    if !on || world_size <= 2 || rows == 0 {
        return Ok(vec![(0, rows)]);
    }
    ensure!(
        row_bytes > 0 && oneshot_max_bytes >= row_bytes,
        "METRALE_GLM_BATCH_INVARIANT=1 at {world_size} ranks needs the one-shot all-reduce \
         (--all-reduce-oneshot-max-kb) to hold at least one row ({row_bytes} bytes); it holds \
         {oneshot_max_bytes}"
    );
    let per = oneshot_max_bytes / row_bytes;
    Ok((0..rows)
        .step_by(per)
        .map(|r0| (r0, per.min(rows - r0)))
        .collect())
}

/// 2026-10-11: The tensor-parallel all-reduce of `rows` rows of `row_bytes` at `p`: one
/// collective, or under the lever the slices of [`ar_slices`]; synchronous while a graph is
/// captured, else enqueued behind `stream`.
pub(crate) fn all_reduce_rows(
    comm: &dyn CommBackend,
    p: DevicePtr,
    rows: usize,
    row_bytes: usize,
    graph_capture: bool,
    stream: u64,
) -> Result<()> {
    let slices = ar_slices(
        enabled()?,
        comm.world_size(),
        comm.oneshot_max_bytes(),
        rows,
        row_bytes,
    )?;
    for (r0, n) in slices {
        let (q, bytes) = (p.offset(r0 * row_bytes), n * row_bytes);
        if graph_capture {
            comm.all_reduce(q.0, bytes)?;
        } else {
            comm.all_reduce_async(q.0, bytes, stream)?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm5next_batch_inv_tests.rs"]
mod tests;
