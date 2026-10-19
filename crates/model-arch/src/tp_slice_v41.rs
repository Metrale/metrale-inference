// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: DeepSeek-V4.1 tensor-parallel slicing of a checkpoint tensor's bytes. A tensor is a
//! row-major byte matrix (`row_bytes` per row: packed values, or a scale plane); a rank keeps a
//! row range (w1, w3, wq_b), a column range (w2, wo_b) or both.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants: none beyond the types.

use std::ops::Range;

use anyhow::{Result, ensure};

/// 2026-10-19: Rows `rows` and byte columns `cols` of the `[src.len() / row_bytes, row_bytes]`
/// matrix `src`. Refuses a ragged source and a range outside it.
pub fn slice2d(
    src: &[u8],
    row_bytes: usize,
    rows: Range<usize>,
    cols: Range<usize>,
) -> Result<Vec<u8>> {
    ensure!(
        row_bytes > 0 && src.len().is_multiple_of(row_bytes),
        "{} bytes are not whole rows of {row_bytes}",
        src.len()
    );
    let n_rows = src.len() / row_bytes;
    ensure!(
        rows.start <= rows.end
            && rows.end <= n_rows
            && cols.start <= cols.end
            && cols.end <= row_bytes,
        "slice rows {rows:?} cols {cols:?} of a [{n_rows}, {row_bytes}] byte matrix"
    );
    let mut out = Vec::with_capacity(rows.len() * cols.len());
    for r in rows {
        out.extend_from_slice(&src[r * row_bytes + cols.start..r * row_bytes + cols.end]);
    }
    Ok(out)
}

/// 2026-10-19: Rank `rank`'s share of `total` across `world` ranks in whole `unit`s, the larger
/// shares first (`[24, 24, 16]` heads of 64 over three ranks at a unit of 8). Refuses a total that
/// is not whole units.
pub fn rank_share(total: usize, unit: usize, world: usize, rank: usize) -> Result<Range<usize>> {
    ensure!(
        unit > 0 && world > 0 && rank < world && total.is_multiple_of(unit),
        "share {rank} of {world} of {total} in units of {unit}"
    );
    let units = total / unit;
    let (base, extra) = (units / world, units % world);
    let start = rank * base + rank.min(extra);
    let len = base + usize::from(rank < extra);
    Ok(start * unit..(start + len) * unit)
}

#[cfg(test)]
#[path = "tp_slice_v41_tests.rs"]
mod tests;
