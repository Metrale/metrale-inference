// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_MHC_POST_SUM_CTAS`: the launch shape of `glm5next_hc_post_rank_sum`
//! (`glm_hc_post_rank_sum`), the SP-mHC post-mix that a reduce-scatter's next piece waits behind.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Off (unset or `0`): grid `(T, collapse_blocks(H))` of 256 threads, the shape before this
//!   lever.
//! - `N` in `1..=MAX_CTAS`: grid `(min(N, T), 1)` of [`CAPPED_BLOCK`] threads. At the kernel's
//!   44 registers a 1024-thread block takes more than half an SM's register file, so no SM holds
//!   two, and the launch occupies at most `N` SMs. The rest stay empty for a kernel on another
//!   stream that needs a whole SM: the SP comm stream's NCCL send/recv (640 threads, 96
//!   registers, 98 560 B shared memory per block) cannot start beside any post-mix block.
//! - The kernel's grid-stride loops give every element the same arithmetic at any grid, so both
//!   shapes write the same bytes (`examples/glm5next_hc_post_sum_gate.rs` checks both).

use anyhow::Result;

use crate::glm5next_mhc::collapse_blocks;

/// 2026-10-14: The capped launch's block: one block per SM at the kernel's register count.
pub const CAPPED_BLOCK: u32 = 1024;

/// 2026-10-14: The largest accepted cap; anything wider is no cap on a GB10's 48 SMs.
pub const MAX_CTAS: u32 = 4096;

/// 2026-10-14: `METRALE_GLM_MHC_POST_SUM_CTAS`, read once: `None` off, `Some(n)` the cap.
pub fn ctas() -> Result<Option<u32>> {
    static V: std::sync::OnceLock<Result<Option<u32>, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let v = parse(
            std::env::var("METRALE_GLM_MHC_POST_SUM_CTAS")
                .ok()
                .as_deref(),
        )?;
        if let Some(n) = v {
            tracing::warn!(
                "METRALE_GLM_MHC_POST_SUM_CTAS={n}: glm5next_hc_post_rank_sum runs at most {n} \
                 blocks of {CAPPED_BLOCK} (grid-stride, same bytes)"
            );
        }
        Ok(v)
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-14: [`ctas`]'s parse, pure.
pub(crate) fn parse(v: Option<&str>) -> Result<Option<u32>, String> {
    match v {
        None | Some("0") => Ok(None),
        Some(s) => match s.parse::<u32>() {
            Ok(n) if (1..=MAX_CTAS).contains(&n) => Ok(Some(n)),
            _ => Err(format!(
                "METRALE_GLM_MHC_POST_SUM_CTAS={s:?}: expected 0 (off) or a block count in \
                 1..={MAX_CTAS}"
            )),
        },
    }
}

/// 2026-10-14: `(grid, block)` of a `num_tokens`-row launch at `hidden` under `ctas`.
pub fn launch_shape(num_tokens: u32, hidden: u32, ctas: Option<u32>) -> ([u32; 3], [u32; 3]) {
    match ctas {
        None => ([num_tokens, collapse_blocks(hidden), 1], [256, 1, 1]),
        Some(n) => ([n.min(num_tokens).max(1), 1, 1], [CAPPED_BLOCK, 1, 1]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_off_and_counts_and_refuses_the_rest() {
        assert_eq!(parse(None), Ok(None));
        assert_eq!(parse(Some("0")), Ok(None));
        assert_eq!(parse(Some("1")), Ok(Some(1)));
        assert_eq!(parse(Some("44")), Ok(Some(44)));
        assert_eq!(parse(Some("4096")), Ok(Some(4096)));
        for bad in ["4097", "-1", "", "44 ", "on", "1.5"] {
            assert!(parse(Some(bad)).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn off_is_the_one_row_per_block_row_grid() {
        assert_eq!(launch_shape(683, 4096, None), ([683, 16, 1], [256, 1, 1]));
        assert_eq!(launch_shape(1, 200, None), ([1, 1, 1], [256, 1, 1]));
    }

    #[test]
    fn a_cap_bounds_the_blocks_and_never_exceeds_the_rows() {
        assert_eq!(
            launch_shape(683, 4096, Some(44)),
            ([44, 1, 1], [1024, 1, 1])
        );
        assert_eq!(launch_shape(5, 4096, Some(44)), ([5, 1, 1], [1024, 1, 1]));
        assert_eq!(launch_shape(0, 4096, Some(44)), ([1, 1, 1], [1024, 1, 1]));
    }
}
