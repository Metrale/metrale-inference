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
//! Two policies make every row's bits independent of R:
//! - `METRALE_ROW_EXACT_TIERS` (presence, off by default): the W8A16
//!   projections keep the scalar GEMV's order at every R (16-row batch16
//!   chunks above 16 rows).
//! - Canonical: the W8A16 projections take the tensor-core tile family's order
//!   at every R (`w8a16_gemm_pipelined` and its 32/64-row twins, which sum
//!   identically). 2026-09-27: The default for FP8 MoE checkpoints;
//!   `--no-canonical-tiers` opts out, and `METRALE_CANONICAL_TIERS` (presence)
//!   turns it on for any checkpoint.
//!
//! Either one also makes the NVFP4 LM head take its tile GEMM at every R and
//! the FP8 MoE take the grouped kernels with the per-row router at every R of
//! two or more (`GroupedRouting::PerRow`, the bits of `MoeLayer::forward`).
//!
//! Owner: model-layers.
//! Invariants:
//! - The serve publishes the policy of each model before building it
//!   ([`publish_row_tiers`]); a process that never publishes (a test, an
//!   example) reads the two levers once, and runs by rows without them.
//! - `--no-canonical-tiers` wins over `METRALE_CANONICAL_TIERS`, which wins over
//!   `METRALE_ROW_EXACT_TIERS`.

use std::sync::atomic::{AtomicU8, Ordering};

/// 2026-09-27: The projection order a process runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowTiers {
    /// 2026-09-27: Pick by row count (the order before canonical tiers).
    ByRows,
    /// 2026-09-27: Scalar-GEMV order at every row count.
    Exact,
    /// 2026-09-27: Tensor-core tile order at every row count.
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

/// 2026-09-27: A model's policy: `--no-canonical-tiers` keeps the scalar order
/// if `METRALE_ROW_EXACT_TIERS` is present and the by-rows tiers otherwise; else
/// a present lever decides; else FP8 MoE checkpoints run canonical and the
/// others by rows. The dense mixed FP8/NVFP4 checkpoints (Qwen3.8-27B) keep
/// their tiers: canonical tiers cost them 2-4% of decode throughput.
pub fn resolve_row_tiers(
    no_canonical: bool,
    exact_env: bool,
    canonical_env: bool,
    fp8_moe_checkpoint: bool,
) -> RowTiers {
    if no_canonical {
        row_tiers_from(exact_env, false)
    } else if canonical_env || exact_env {
        row_tiers_from(exact_env, canonical_env)
    } else if fp8_moe_checkpoint {
        RowTiers::Canonical
    } else {
        RowTiers::ByRows
    }
}

const UNPUBLISHED: u8 = 0;

fn encode(t: RowTiers) -> u8 {
    match t {
        RowTiers::ByRows => 1,
        RowTiers::Exact => 2,
        RowTiers::Canonical => 3,
    }
}

/// 2026-09-27: The policy the serve published for the model it builds.
static PUBLISHED: AtomicU8 = AtomicU8::new(UNPUBLISHED);

/// 2026-09-27: Publish the policy of the model about to be built. A model swap
/// publishes again before the next build.
pub fn publish_row_tiers(t: RowTiers) {
    PUBLISHED.store(encode(t), Ordering::Relaxed);
    tracing::info!(
        "row tiers: {t:?}{}",
        if t == RowTiers::ByRows {
            " (kernels picked by row count)"
        } else {
            " (one summation order per op at every row count)"
        }
    );
}

/// 2026-09-27: This process's `RowTiers`: the published policy, else the two
/// levers, read once.
pub fn row_tiers() -> RowTiers {
    match PUBLISHED.load(Ordering::Relaxed) {
        1 => RowTiers::ByRows,
        2 => RowTiers::Exact,
        3 => RowTiers::Canonical,
        _ => {
            static FALLBACK: std::sync::OnceLock<RowTiers> = std::sync::OnceLock::new();
            *FALLBACK.get_or_init(|| {
                row_tiers_from(
                    std::env::var_os("METRALE_ROW_EXACT_TIERS").is_some(),
                    std::env::var_os("METRALE_CANONICAL_TIERS").is_some(),
                )
            })
        }
    }
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

    /// 2026-09-27: FP8 MoE checkpoints default to canonical and the others to by rows;
    /// the opt-out beats the canonical lever but keeps an exact one; a lever
    /// decides for any checkpoint.
    #[test]
    fn fp8_defaults_to_canonical_and_the_opt_out_wins() {
        for fp8 in [false, true] {
            let default = resolve_row_tiers(false, false, false, fp8);
            assert_eq!(
                default,
                if fp8 {
                    RowTiers::Canonical
                } else {
                    RowTiers::ByRows
                }
            );
            assert_eq!(resolve_row_tiers(true, false, false, fp8), RowTiers::ByRows);
            assert_eq!(resolve_row_tiers(true, false, true, fp8), RowTiers::ByRows);
            assert_eq!(resolve_row_tiers(true, true, true, fp8), RowTiers::Exact);
            assert_eq!(resolve_row_tiers(false, true, false, fp8), RowTiers::Exact);
            assert_eq!(
                resolve_row_tiers(false, false, true, fp8),
                RowTiers::Canonical
            );
            assert_eq!(
                resolve_row_tiers(false, true, true, fp8),
                RowTiers::Canonical
            );
        }
    }
}
