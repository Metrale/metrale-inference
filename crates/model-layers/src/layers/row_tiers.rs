// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: Row-invariant tier policies: whether a row's arithmetic may depend
//! on how many rows share its launch.
//!
//! By default several ops pick a kernel by row count R, and the kernels sum in
//! different orders: the W8A16 projections (scalar-order GEMV up to 16 rows, a
//! tensor-core tile above), the NVFP4 LM head (GEMV below 9 rows, a tile GEMM
//! above), and the MoE router and experts (two- and three-row kernels, the
//! pairwise walk, the grouped paths). A row's logits then depend on R, which
//! depends on the draft count and on which sequences are active, so greedy
//! output varies across `num_drafts` and across runs at a fixed concurrency.
//! Two presence-only levers, off by default, make every row's bits independent
//! of R:
//! - `METRALE_ROW_EXACT_TIERS`: the W8A16 projections keep the scalar GEMV's
//!   order at every R (16-row batch16 chunks above 16 rows).
//! - `METRALE_CANONICAL_TIERS`: the W8A16 projections take the tensor-core tile
//!   family's order at every R (`w8a16_gemm_pipelined` and its 32/64-row twins,
//!   which sum identically).
//!
//! Either one also makes the NVFP4 LM head take its tile GEMM at every R and
//! the FP8 MoE take the grouped kernels with the per-row router at every R of
//! two or more (`GroupedRouting::PerRow`, the bits of `MoeLayer::forward`).
//!
//! Owner: model-layers.
//! Invariants: both levers are read once per process; with both present,
//! `METRALE_CANONICAL_TIERS` wins.

/// 2026-09-27: The projection order a process runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowTiers {
    /// Pick by row count (today's behaviour).
    ByRows,
    /// Scalar-GEMV order at every row count.
    Exact,
    /// Tensor-core tile order at every row count.
    Canonical,
}

/// 2026-09-27: `RowTiers` from the presence of the two levers.
pub fn row_tiers_from(exact: bool, canonical: bool) -> RowTiers {
    if canonical {
        RowTiers::Canonical
    } else if exact {
        RowTiers::Exact
    } else {
        RowTiers::ByRows
    }
}

/// 2026-09-27: This process's `RowTiers`, read once.
pub fn row_tiers() -> RowTiers {
    static T: std::sync::OnceLock<RowTiers> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        let t = row_tiers_from(
            std::env::var_os("METRALE_ROW_EXACT_TIERS").is_some(),
            std::env::var_os("METRALE_CANONICAL_TIERS").is_some(),
        );
        if t != RowTiers::ByRows {
            tracing::info!(
                "row-invariant tiers: {t:?} (one summation order per op at every row count)"
            );
        }
        t
    })
}

/// 2026-09-27: Whether this process runs a row-invariant policy.
pub fn row_invariant() -> bool {
    row_tiers() != RowTiers::ByRows
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-27: Canonical wins over exact; neither lever keeps the by-rows tiers.
    #[test]
    fn canonical_wins_and_absence_keeps_by_rows() {
        assert_eq!(row_tiers_from(false, false), RowTiers::ByRows);
        assert_eq!(row_tiers_from(true, false), RowTiers::Exact);
        assert_eq!(row_tiers_from(false, true), RowTiers::Canonical);
        assert_eq!(row_tiers_from(true, true), RowTiers::Canonical);
    }
}
