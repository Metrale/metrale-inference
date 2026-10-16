// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_MOE_W4A4_WIDE_SWEEP=1`: the routed W4A4 experts run the persistent
//! union sweep up to [`WIDE_SWEEP_MAX_ROWS`] rows (`w4a4_gemv_mx{24,32}_moe_union_sweep[_k64]`)
//! instead of handing 17..=32 rows to the grouped W4A4 GEMM. A speculative verify batch (rows =
//! sequences x 2 under MTP) past 16 rows then streams each union expert's weights once, as the
//! narrower sweeps do. Every (row, slot) output is bit-identical to the slot GEMV's
//! (`w4a4_gemv_mx8_moe_slots`), the same bytes the grouped GEMM writes; only the launch changes.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Off unless the lever is `1`; the lever needs `METRALE_GLM_MOE_W4A4_GROUPED=1` (the routed
//!   W4A4 cap past 16 rows and the workspace's every-slot scratch) and the four wide entries,
//!   else the load is refused ([`check_at_load`]).
//! - The union builder covers `rows * top_k <= WIDE_SWEEP_MAX_IDS` ids.
//! - 2026-10-15: Decode-only. The lever is read only at [`MoeCallSite::BatchedDecode`] (the
//!   multi-sequence decode body); every other caller, prefill included, passes
//!   [`MoeCallSite::General`] and keeps `w4a4_grouped::takes_grouped` unchanged.

use anyhow::Result;

use crate::glm5next_mlp::{Glm5NextMlpKernels, MOE_W4A4_MAX_ROWS};

const LEVER: &str = "METRALE_GLM_MOE_W4A4_WIDE_SWEEP";

/// 2026-10-14: The widest routed launch the wide sweep serves (`w4a4_gemv_mx32_moe_union_sweep`,
/// four 8-token MMA tiles).
pub const WIDE_SWEEP_MAX_ROWS: usize = 32;

/// 2026-10-14: Most union ids the wide sweep takes: the sweep's 256-thread item table
/// (`W4A4_WARPS * 32` in `w4a4_gemv_mx_moe.cu`), [`WIDE_SWEEP_MAX_ROWS`] rows at top-8.
pub const WIDE_SWEEP_MAX_IDS: usize = 256;

/// 2026-10-14: `METRALE_GLM_MOE_W4A4_WIDE_SWEEP`, read once. Off when unset or `0`; any other
/// value than `1` is an error.
pub fn moe_w4a4_wide_sweep() -> Result<bool> {
    static V: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| parse_wide_lever(std::env::var(LEVER).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-14: [`moe_w4a4_wide_sweep`]'s parse, pure.
pub(crate) fn parse_wide_lever(v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{LEVER}={other:?}: expected 0 or 1")),
    }
}

/// 2026-10-15: Which caller runs a routed MoE site. The wide sweep is a decode-only lever: it
/// runs only at [`MoeCallSite::BatchedDecode`], so every prefill launch keeps its grouped
/// dispatch whatever the lever says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeCallSite {
    /// 2026-10-15: Every caller but the batched decode body: prefill (spans, pieces, SP), the
    /// single-sequence forward and the drafter. The wide sweep never runs here.
    General,
    /// 2026-10-15: The multi-sequence decode body (`forward_spans`: one row per sequence, or the
    /// draft rows of a batched MTP verify).
    BatchedDecode,
}

/// 2026-10-15: Whether the wide sweep serves a routed launch at `site`: the lever, read only at
/// [`MoeCallSite::BatchedDecode`].
pub fn wide_at(site: MoeCallSite) -> Result<bool> {
    match site {
        MoeCallSite::General => Ok(false),
        MoeCallSite::BatchedDecode => moe_w4a4_wide_sweep(),
    }
}

/// 2026-10-14: The widest routed W4A4 launch the sweeps serve: [`WIDE_SWEEP_MAX_ROWS`] with
/// `wide`, else [`MOE_W4A4_MAX_ROWS`]. Wider launches run the grouped W4A4 GEMM.
pub fn sweep_max_rows(wide: bool) -> usize {
    if wide {
        WIDE_SWEEP_MAX_ROWS
    } else {
        MOE_W4A4_MAX_ROWS
    }
}

/// 2026-10-14: Whether a W4A4 routed launch of `rows` rows runs the grouped W4A4 GEMM: the
/// grouped module's row test, narrowed to past [`WIDE_SWEEP_MAX_ROWS`] when `wide`
/// ([`wide_at`]). With `wide` false it is exactly `w4a4_grouped::takes_grouped`, the test the
/// prefill grouping reads. Pure.
pub(crate) fn takes_grouped_at(rows: usize, wide: bool) -> bool {
    super::w4a4_grouped::takes_grouped(rows) && rows > sweep_max_rows(wide)
}

/// 2026-10-14: Whether the lever can run on this load: `grouped` is
/// `METRALE_GLM_MOE_W4A4_GROUPED`, `ready` whether every wide entry resolved.
pub(crate) fn check_ready(grouped: bool, ready: bool) -> Result<(), String> {
    if !grouped {
        return Err(format!("{LEVER}=1 needs METRALE_GLM_MOE_W4A4_GROUPED=1"));
    }
    if !ready {
        return Err(format!(
            "{LEVER}=1: w4a4_gemv_mx24/mx32_moe_union_sweep[_k64] is absent from this target's PTX"
        ));
    }
    Ok(())
}

/// 2026-10-14: Whether every wide entry (`sweep[2..]` of the plain and `_k64` sets) resolved.
pub(crate) fn wide_ready(k: &Glm5NextMlpKernels) -> bool {
    k.w4a4_moe_sweep[2..]
        .iter()
        .chain(&k.w4a4_moe_k64.sweep[2..])
        .all(|h| h.0 != 0)
}

/// 2026-10-14: The load-time refusal of a lever this load cannot run, from the routed experts'
/// precision plan.
pub fn check_at_load(k: &Glm5NextMlpKernels) -> Result<()> {
    if !moe_w4a4_wide_sweep()? {
        return Ok(());
    }
    check_ready(super::moe_w4a4_grouped()?, wide_ready(k)).map_err(anyhow::Error::msg)
}

#[cfg(test)]
#[path = "w4a4_wide_tests.rs"]
mod tests;
