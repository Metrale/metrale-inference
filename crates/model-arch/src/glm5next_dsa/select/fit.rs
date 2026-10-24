// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The selection scratch's capacity checks, [`DsaSelectScratch::rows_per_pass`] and
//! [`DsaSelectScratch::fits`], moved from `select.rs` (which reached the file-size limit).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - With `METRALE_GLM_DSA_SELECT_FUSED` off both answer exactly as before the move.
//! - A pass the fused selection takes ([`DsaSelectScratch::fused_on`]) writes no scores or
//!   candidacy row, so neither region bounds its rows or is checked for it; every other region
//!   is, as before.

use super::*;

impl DsaSelectScratch {
    /// 2026-10-17: Whether a pass of `geom` takes the fused selection: the lever was on at
    /// allocation and the pass has its shape ([`fused_shape`]). Kernel resolution is checked at
    /// launch, which fails rather than fall back.
    pub fn fused_on(&self, geom: &DsaSelectGeometry) -> bool {
        self.fused && fused_shape(geom)
    }

    /// 2026-10-11: The most query rows one pass over `geom`'s pools fits in the per-row regions
    /// (scores, candidacy, selected pools); `usize::MAX` when `geom` has no pools to score. The
    /// output rows and the pool regions are not counted: `fits` checks those. 2026-10-17: only
    /// the selected pools under the fused selection ([`Self::fused_on`]).
    pub fn rows_per_pass(&self, geom: &DsaSelectGeometry) -> usize {
        let per = |cap: usize, row_bytes: usize| cap.checked_div(row_bytes).unwrap_or(usize::MAX);
        let selected = per(self.capacity[5], geom.select_k * 4);
        if self.fused_on(geom) {
            return selected;
        }
        per(self.capacity[3], geom.n_pools * 4)
            .min(per(self.capacity[4], geom.n_pools))
            .min(selected)
    }

    /// 2026-09-25: Whether `geom` fits what was allocated. `select_tokens` checks it on every
    /// pass, so a pass larger than the allocation is an error rather than an overrun.
    pub fn fits(&self, cfg: &Glm5NextDsaConfig, geom: &DsaSelectGeometry) -> Result<()> {
        self.fits_for(cfg, geom, false)
    }

    /// 2026-10-17: [`Self::fits`] for a pass that does (`fused`) or does not take the fused
    /// selection; `fused` skips the scores and candidacy regions it never writes.
    pub(super) fn fits_for(
        &self,
        cfg: &Glm5NextDsaConfig,
        geom: &DsaSelectGeometry,
        fused: bool,
    ) -> Result<()> {
        let want = geom.scratch_bytes();
        for (i, (w, c)) in want.iter().zip(self.capacity.iter()).enumerate() {
            if fused && (i == 3 || i == 4) {
                continue;
            }
            if w > c {
                bail!(
                    "DSA select: scratch region {i} needs {w} B but only {c} B was \
                     reserved ({} tokens, {} pools, {} query rows)",
                    geom.seq,
                    geom.n_pools,
                    geom.q_rows
                );
            }
        }
        let want_tokens = geom.q_rows * cfg.out_width() * 4;
        if want_tokens > self.tokens_bytes {
            bail!(
                "DSA select: selection output needs {want_tokens} B but only {} B was \
                 reserved",
                self.tokens_bytes
            );
        }
        Ok(())
    }
}
