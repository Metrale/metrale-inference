// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The wide-row arm of the declared-W8A8 projection. From
//! `METRALE_W8A8_LT_MIN_ROWS` rows up (unset or 0: never), a per-row-scaled weight's
//! [`super::w8a8_gemv`] runs as one cuBLASLt FP8 GEMM per stacked segment
//! (`cublaslt::fp8_gemm_act_weight_t_rowwise_ldc`): the same E4M3 activation and per-token scale
//! the quantizer left in the scratch, the same E4M3 weight and per-row scale, FP32 accumulation,
//! BF16 out. The declared numerics are kept; the summation order is cuBLASLt's, and its
//! algorithm may change with the row count, so a row's bits can depend on how many rows share
//! the launch (the `adaptive` activation routing already routes by row count).
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

/// 2026-10-09: `METRALE_W8A8_LT_MIN_ROWS`, read once per process (a graph capture and its
/// replays must agree). Unset, empty, unparsable or 0 turns the arm off.
pub fn w8a8_lt_min_rows() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| parse_min_rows(std::env::var("METRALE_W8A8_LT_MIN_ROWS").ok().as_deref()))
}

/// 2026-10-09: The pure parse behind [`w8a8_lt_min_rows`].
fn parse_min_rows(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse().ok()).unwrap_or(0)
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
        metrale_gpu_runtime::cublaslt::fp8_gemm_act_weight_t_rowwise_ldc(
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
        )?;
        col += seg.n;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_threshold_parses_and_zero_or_garbage_is_off() {
        assert_eq!(parse_min_rows(None), 0);
        assert_eq!(parse_min_rows(Some("")), 0);
        assert_eq!(parse_min_rows(Some("abc")), 0);
        assert_eq!(parse_min_rows(Some(" 64 ")), 64);
    }

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
