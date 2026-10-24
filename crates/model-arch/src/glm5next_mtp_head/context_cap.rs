// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The GLM drafter's context cap (`METRALE_GLM_MTP_CONTEXT_CAP`).
//!
//! The drafter's per-sequence state (its KV region, flat indexer cache and graph block table)
//! is reserved up front for `--max-batch-size` sequences, and the model's hidden capture for
//! one prompt, all at the drafter's row count. Uncapped, that count is `--max-seq-len`, so the
//! reserve grows ~32 KiB per token of `--max-seq-len` (16 sequences): 8 GiB per rank at 256k and
//! 16 GiB at 512k, which the KV pool cannot give up. Capped, it is the same at every
//! `--max-seq-len` at or above the cap, and identical to the uncapped serve below it.
//!
//! A sequence whose drafter would write past the cap restarts its drafter at row 0
//! ([`propose_start`]): the drafter then drafts from the rows it writes from there on, as a
//! drafter that never had its prompt context does. Drafts only; the target verifies every one.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - A cap at or above `--max-seq-len`, or 0, leaves every size and every propose unchanged.
//! - [`propose_start`] never returns a start whose drafts would pass `capacity` unless
//!   `num_drafts` alone exceeds it.
//!
//! 2026-10-17: The cap is `MtpLevers::context_cap`, resolved from the startup settings; the
//! loader, the KV pool and the per-sequence reserve take it from there, so they size the same
//! rows.

use anyhow::Result;

/// 2026-10-13: The drafter's context for a serve context of `serve_ctx` rows under `cap`
/// (0 = no cap).
pub fn drafter_max_context(serve_ctx: usize, cap: usize) -> usize {
    if cap == 0 {
        serve_ctx
    } else {
        serve_ctx.min(cap)
    }
}

/// 2026-10-13: The drafter block's DSA config: the text stack's, with `max_context` capped at
/// `cap` (0 = no cap).
pub fn drafter_dsa_config(
    config: &metrale_config::ModelConfig,
    cap: usize,
) -> Result<crate::glm5next_dsa::Glm5NextDsaConfig> {
    let mut c = crate::glm5next_dsa::Glm5NextDsaConfig::from_config(config)?;
    c.max_context = drafter_max_context(c.max_context, cap);
    Ok(c)
}

/// 2026-10-13: The drafter row a propose of `num_drafts` drafts at target position `position`
/// starts at, for a drafter of `drafter_rows` rows and `capacity` row space: `start_row`, or 0
/// when its drafts would not fit (the drafter restarts its row space).
pub(super) fn propose_start(
    drafter_rows: usize,
    position: usize,
    num_drafts: usize,
    capacity: usize,
) -> usize {
    let start = super::batch::start_row(drafter_rows, position);
    if start + num_drafts > capacity {
        0
    } else {
        start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-13: Below or at the cap the drafter's context is the serve's (the uncapped serve,
    /// byte for byte); above it, the cap; cap 0 never clamps.
    #[test]
    fn the_cap_clamps_only_past_itself() {
        assert_eq!(drafter_max_context(40_960, 65_536), 40_960);
        assert_eq!(drafter_max_context(65_536, 65_536), 65_536);
        assert_eq!(drafter_max_context(262_144, 65_536), 65_536);
        assert_eq!(drafter_max_context(1_048_576, 65_536), 65_536);
        assert_eq!(drafter_max_context(1_048_576, 0), 1_048_576);
    }

    /// 2026-10-13: The capped config differs from the text stack's in `max_context` only, and
    /// cap 0 leaves it equal (the negative control: the clamp is what changed it).
    #[test]
    fn the_capped_config_changes_max_context_only() {
        let mut cfg = metrale_config::parse_config(include_str!(
            "../../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json"
        ))
        .expect("real config parses");
        cfg.serve_max_seq_len = 524_288;
        let text = crate::glm5next_dsa::Glm5NextDsaConfig::from_config(&cfg).unwrap();
        let capped = drafter_dsa_config(&cfg, 65_536).unwrap();
        assert_eq!(text.max_context, 524_288);
        assert_eq!(capped.max_context, 65_536);
        let mut back = capped;
        back.max_context = text.max_context;
        assert_eq!(back, text);
        assert_eq!(drafter_dsa_config(&cfg, 0).unwrap(), text);
    }

    /// 2026-10-13: Drafts that fit start where `start_row` says, up to the exact fit.
    #[test]
    fn a_propose_that_fits_starts_at_start_row() {
        assert_eq!(propose_start(1_000, 1_000, 1, 65_536), 1_000);
        assert_eq!(propose_start(1_200, 1_000, 3, 65_536), 1_000);
        assert_eq!(propose_start(65_535, 70_000, 1, 65_536), 65_535);
        assert_eq!(propose_start(65_533, 70_000, 3, 65_536), 65_533);
    }

    /// 2026-10-13: One row past the space restarts the drafter at row 0, whether the drafter is
    /// full or the target's position is the limit.
    #[test]
    fn a_propose_past_the_space_restarts_at_row_zero() {
        assert_eq!(propose_start(65_536, 70_000, 1, 65_536), 0);
        assert_eq!(propose_start(65_534, 70_000, 3, 65_536), 0);
        assert_eq!(propose_start(70_000, 65_536, 1, 65_536), 0);
    }
}
