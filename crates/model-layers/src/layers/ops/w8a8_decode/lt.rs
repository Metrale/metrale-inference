// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The wide-row arm of the declared-W8A8 projection. From the class's
//! `[defaults] w8a8_lt_min_rows` rows up (`METRALE_W8A8_LT_MIN_ROWS` overrides; 0: never), and only
//! while every projection family routes `adaptive` (`--activation-quantization`), a per-row-scaled weight's
//! [`super::w8a8_gemv`] runs as one cuBLASLt FP8 GEMM per stacked segment
//! (`cublaslt::fp8_gemm_act_weight_t_rowwise_ldc`): the same E4M3 activation and per-token scale
//! the quantizer left in the scratch, the same E4M3 weight and per-row scale, FP32 accumulation,
//! BF16 out. The declared numerics are kept; the summation order is cuBLASLt's, and its
//! algorithm may change with the row count, so a row's bits can depend on how many rows share
//! the launch: a fixed activation format promises row invariance, so the arm stays off under one
//! (the `adaptive` routing already routes by row count).
//!
//! Why: the skinny GEMV keeps 1-16 token tiles in registers and re-reads the weight once per
//! 128-row launch with little reuse per byte; on the H100 SXM its 128-row launch of the 27B's GDN
//! QKV|Z takes about 185 us against a 25 us weight floor.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`try_w8a8_gemm_lt`] launches only for a `PerRow` weight at `rows >= lt_min_rows() > 0` and
//!   returns `Ok(false)` without launching otherwise; it writes exactly the `rows x n` outputs
//!   the GEMV would.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{W8a8Scale, W8a8Scratch, W8a8Weight};

/// 2026-10-09: The row count the arm starts at, 0 for never: the resolved class default
/// (`target_defaults::resolved`, read once per process, so a graph capture and its replays agree),
/// and 0 whenever a projection family runs a fixed activation format.
pub fn w8a8_lt_min_rows() -> usize {
    if crate::layers::activation_quantization::any_fixed() {
        return 0;
    }
    super::super::target_defaults::resolved()
        .w8a8_lt_min_rows
        .value as usize
}

/// 2026-10-09: Log a cuBLASLt arm that declined a shape: the first time per (arm, rows, n, k),
/// at WARN, so a decline is visible without flooding a long serve.
pub fn lt_decline_log(arm: &str, rows: usize, n: u32, k: u32, ldc: u32, e: &anyhow::Error) {
    static SEEN: std::sync::Mutex<Vec<(String, usize, u32, u32)>> =
        std::sync::Mutex::new(Vec::new());
    let key = (arm.to_string(), rows, n, k);
    let mut seen = SEEN.lock().unwrap_or_else(|p| p.into_inner());
    if !seen.contains(&key) {
        seen.push(key);
        tracing::warn!(
            "cuBLASLt arm {arm} declined rows={rows} n={n} k={k} ldc={ldc}: {e:#}; the in-tree kernel serves it"
        );
    }
}

/// 2026-10-09: Whether the arm takes a launch of `rows` rows of a `scale` weight, under the
/// threshold `min_rows` (0 = off).
fn takes(scale: W8a8Scale, rows: usize, min_rows: usize) -> bool {
    scale == W8a8Scale::PerRow && min_rows > 0 && rows >= min_rows
}

/// 2026-10-09: Run `w` over the quantized activation in `scratch` as cuBLASLt GEMMs when the arm
/// takes the launch: `Ok(true)` when it launched, `Ok(false)` to keep the GEMV.
pub(super) fn try_w8a8_gemm_lt(
    w: &W8a8Weight,
    scratch: &W8a8Scratch,
    rows: usize,
    out: DevicePtr,
    ldc: u32,
    stream: u64,
) -> Result<bool> {
    if !takes(w.scale, rows, w8a8_lt_min_rows()) {
        return Ok(false);
    }
    let mut col = 0u32;
    for seg in &w.segs[..w.count] {
        if let Err(e) = metrale_gpu_runtime::cublaslt::fp8_gemm_act_weight_t_rowwise_ldc(
            scratch.q.0,
            scratch.scale.0,
            seg.weight.0,
            seg.row_scale.0,
            out.offset(col as usize * 2).0,
            rows as u32,
            seg.n,
            w.k,
            ldc,
            stream,
        ) {
            // 2026-10-09: cuBLASLt declines some shapes (an AlgoGetHeuristic NOT_SUPPORTED seen at
            // decode widths on the H100). The GEMV then writes every column, the segments this
            // call already wrote included, so the result is the GEMV's.
            lt_decline_log("w8a8 rowwise", rows, seg.n, w.k, ldc, &e);
            return Ok(false);
        }
        col += seg.n;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_per_row_weights_at_or_above_the_threshold_take_the_arm() {
        assert!(!takes(W8a8Scale::PerRow, 128, 0), "0 is off");
        assert!(!takes(W8a8Scale::PerRow, 63, 64));
        assert!(takes(W8a8Scale::PerRow, 64, 64));
        assert!(takes(W8a8Scale::PerRow, 256, 64));
        assert!(
            !takes(W8a8Scale::Block128, 256, 64),
            "block-scaled weights keep the GEMV (their activation scales are per K group)"
        );
    }
}
