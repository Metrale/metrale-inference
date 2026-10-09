// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The tensor-parallel routed-expert layout (`--moe-expert-layout tp`): every rank
//! holds a slice of EVERY expert's intermediate width, so all of a token's routed slots are
//! local on every rank and each rank reads the same bytes per token.
//!
//! An expert is three NVFP4 projections, stored as the checkpoint stores them: `gate_proj` and
//! `up_proj` are `[inter, hidden]` and keep the rank's ROWS; `down_proj` is `[hidden, inter]` and
//! keeps the same intermediate indices as COLUMNS. The SwiGLU between them is elementwise, so a
//! rank's down output is its share of the expert's output, and the sum over ranks is the whole
//! expert: the combine and the MLP all-reduce that follows already sum partial outputs (under EP
//! each rank's sum covers its own experts; here it covers its own columns of every expert).
//!
//! The width splits by `metrale_config::tp_split` in [`EXPERT_TP_UNIT`] columns. When the width
//! is not a multiple of the unit it is first padded up to one: the padded columns are zero rows
//! of gate and up (zero codes AND zero block scales) and zero columns of down, so a padded
//! column's gate and up are 0, its clamped SwiGLU is `silu(0) * 0 = 0`, and it adds exactly zero.
//! The per-tensor `weight_scale_2` and `input_scale` are not sliced: every slice of an expert
//! carries the expert's own.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The [`ExpertSlice`]s of ranks `0..tp` partition the padded width into contiguous,
//!   ordered, `EXPERT_TP_UNIT`-aligned ranges, and their `real` parts partition the checkpoint
//!   width `0..full` in the same order.
//! - The slicers are pure functions of host bytes; they never cut a 16-element scale group.

use anyhow::{Result, bail};
use metrale_config::TpSlice;

use super::build_w4a4::{slice_nvfp4_cols, slice_nvfp4_rows};

/// 2026-10-09: The unit every expert's intermediate width splits in under the `tp` layout. The
/// width is N of gate/up and K of down on every routed route, and the strictest reader is the
/// W4A4 down projection (`w4a4_gemv_mx8_moe_slots` / `_moe_union`): K a multiple of 128, whole
/// k128 chunks, 16-byte weight rows and 8-byte scale groups (`build_w4a4::check_w4a4_k`). The
/// W4A16 GEMVs need K % 16 and the grouped prefill GEMM's default tile steps K by 128 (its tails
/// are guarded), so 128 satisfies every kernel the routed experts reach. GLM-5.3's 2048 splits
/// over three ranks as 768 / 640 / 640.
pub const EXPERT_TP_UNIT: usize = 128;

/// 2026-10-09: One rank's slice of every routed expert's intermediate width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlice {
    /// 2026-10-09: The first checkpoint column this rank holds; a multiple of the unit.
    pub start: usize,
    /// 2026-10-09: Checkpoint columns `[start, start + real)` this rank holds, at least 1.
    pub real: usize,
    /// 2026-10-09: The width this rank runs (`Glm5NextMlpConfig::moe_intermediate`), a multiple
    /// of the unit: `real` columns, then `len - real` zero (padded) columns.
    pub len: usize,
    /// 2026-10-09: The checkpoint's `moe_intermediate_size`.
    pub full: usize,
}

impl ExpertSlice {
    /// 2026-10-09: The checkpoint columns this rank holds, as a `TpSlice`.
    pub fn real_cols(&self) -> TpSlice {
        TpSlice {
            start: self.start,
            len: self.real,
        }
    }
    /// 2026-10-09: The zero columns appended after the real ones.
    pub fn pad(&self) -> usize {
        self.len - self.real
    }
}

/// 2026-10-09: How this rank holds the routed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertShard {
    /// 2026-10-09: Whole experts, `local_expert_range()` of them (the `ep` layout).
    Whole,
    /// 2026-10-09: Every expert, each sliced to this rank's columns (the `tp` layout).
    Sliced(ExpertSlice),
}

/// 2026-10-09: Rank `rank`'s [`ExpertSlice`] of a `full`-wide expert over `tp` ranks. Errors
/// when `full` is 0 or not a multiple of 16 (the NVFP4 scale group of down's K), or when the
/// padded width has fewer units than ranks.
pub fn expert_slice(full: usize, tp: usize, rank: usize) -> Result<ExpertSlice> {
    if full == 0 || !full.is_multiple_of(16) {
        bail!(
            "GLM MoE tp layout: moe_intermediate_size {full} must be a positive multiple of 16, \
             the NVFP4 scale group of down_proj's K"
        );
    }
    let padded = full.next_multiple_of(EXPERT_TP_UNIT);
    let s = metrale_config::tp_split(padded, tp, rank, EXPERT_TP_UNIT).map_err(|e| {
        anyhow::anyhow!(
            "GLM MoE tp layout: moe_intermediate_size {full} (padded to {padded}) over \
             {tp} ranks in {EXPERT_TP_UNIT}-column units: {e}"
        )
    })?;
    let real = full.min(s.end()).saturating_sub(s.start);
    if real == 0 {
        bail!(
            "GLM MoE tp layout: rank {rank} of {tp} would hold only padding of a {full}-wide \
             expert"
        );
    }
    Ok(ExpertSlice {
        start: s.start,
        real,
        len: s.len,
        full,
    })
}

/// 2026-10-09: This rank's rows of a gate or up projection (`[full, k]` NVFP4): the real rows,
/// then `pad()` zero rows (zero codes and zero block scales). `(packed, scales)`.
pub fn slice_expert_rows(
    packed: &[u8],
    scales: &[u8],
    k: usize,
    s: &ExpertSlice,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let (mut p, mut sc) = slice_nvfp4_rows(packed, scales, s.full, k, s.real_cols())?;
    p.resize(s.len * k / 2, 0);
    sc.resize(s.len * k / 16, 0);
    Ok((p, sc))
}

/// 2026-10-09: The rows of a gate or up projection already cut to the real rows (a ranged read
/// of `real` rows of `k`), padded with `pad()` zero rows. Errors on a wrong byte count.
pub fn pad_expert_rows(
    mut packed: Vec<u8>,
    mut scales: Vec<u8>,
    k: usize,
    s: &ExpertSlice,
) -> Result<(Vec<u8>, Vec<u8>)> {
    if !k.is_multiple_of(16) || packed.len() != s.real * k / 2 || scales.len() != s.real * k / 16 {
        bail!(
            "GLM MoE tp layout: {} rows of {k}: {} packed and {} scale bytes, expected {} and {}",
            s.real,
            packed.len(),
            scales.len(),
            s.real * k / 2,
            s.real * k / 16
        );
    }
    packed.resize(s.len * k / 2, 0);
    scales.resize(s.len * k / 16, 0);
    Ok((packed, scales))
}

/// 2026-10-09: This rank's columns of a down projection (`[n, full]` NVFP4): per row the real
/// columns, then `pad()` zero columns. `(packed, scales)`.
pub fn slice_expert_cols(
    packed: &[u8],
    scales: &[u8],
    n: usize,
    s: &ExpertSlice,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let (p, sc) = slice_nvfp4_cols(packed, scales, n, s.full, s.real_cols())?;
    let (rp, rs) = (s.real / 2, s.real / 16);
    let (lp, ls) = (s.len / 2, s.len / 16);
    let mut out_p = Vec::with_capacity(n * lp);
    let mut out_s = Vec::with_capacity(n * ls);
    for (row_p, row_s) in p.chunks_exact(rp).zip(sc.chunks_exact(rs)) {
        out_p.extend_from_slice(row_p);
        out_p.resize(out_p.len() + lp - rp, 0);
        out_s.extend_from_slice(row_s);
        out_s.resize(out_s.len() + ls - rs, 0);
    }
    Ok((out_p, out_s))
}

#[cfg(test)]
#[path = "expert_tp_tests.rs"]
mod tests;
