// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The GLM mHC and SP-mHC launch levers, resolved once from the startup settings
//! ([`MhcLevers::from_settings`]) as part of `GlmLevers`. The loader hands the copy to
//! `Glm5NextMhcKernels` (the launchers read `kernels.levers`) and the layer reads
//! `levers.mhc`; nothing here reads the process environment.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `METRALE_GLM_MHC_POST_PRE_FUSED`, `METRALE_GLM_SP_OVERLAP` and `METRALE_GLM_SP_PIPELINE` on
//!   without `METRALE_GLM_SP_MHC` is an error, and so is the pipeline under the layer dump (whose
//!   taps read the serial order): the resolve fails, not the first prefill. 2026-10-22:
//!   `METRALE_GLM_SP_OVERLAP` left at its default (on) is off without SP-mHC instead
//!   (`glm5next_default_on::under_parent`); set to `1`, it is still refused.
//! - A post-sum cap outside `1..=MAX_CTAS` (other than `0`, off) and a negative pre-mix slice are
//!   errors.

use anyhow::{Result, anyhow, bail};
use metrale_settings::InferenceServerSettings;

use crate::glm5next_hc_post_sum_grid::{CAPPED_BLOCK, MAX_CTAS};

/// 2026-10-17: The mHC and SP-mHC levers. Each field names its variable; the module that runs
/// the lever documents what it changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MhcLevers {
    /// 2026-10-17: `METRALE_GLM_SP_MHC` (`glm5next_sp_mhc`).
    pub sp_mhc: bool,
    /// 2026-10-17: `METRALE_GLM_SP_OVERLAP` (`glm5next_sp_overlap`).
    pub sp_overlap: bool,
    /// 2026-10-17: `METRALE_GLM_SP_PIPELINE` (`glm5next_sp_pipeline`).
    pub sp_pipeline: bool,
    /// 2026-10-17: `METRALE_GLM_MHC_POST_PRE_FUSED` (`glm5next_mhc_post_pre_tok`).
    pub post_pre_fused: bool,
    /// 2026-10-17: `METRALE_GLM_MHC_POST_VEC` (`glm5next_hc_post_vec`).
    pub post_vec: bool,
    /// 2026-10-17: `METRALE_GLM_MHC_POST_SUM_CTAS` (`glm5next_hc_post_sum_grid`): `None` off.
    pub post_sum_ctas: Option<u32>,
    /// 2026-10-12: `METRALE_GLM_MHC_PRE_TOK`: the token-tiled grid from
    /// `HC_PRE_NORM_ROWS_MIN_TOKENS` tokens (`glm5next_mhc_tok`, same bits).
    pub pre_tok: bool,
    /// 2026-10-17: `METRALE_GLM_MHC_PRE_TILE` (`glm5next_mhc_tile`).
    pub pre_tile: bool,
    /// 2026-10-11: `METRALE_GLM_MHC_PRE_PREFETCH`: the per-row pre-norm takes
    /// `glm5next_hc_pre_norm_pf64` where the shape allows (`glm5next_mhc_fused::pre_norm_kernel`):
    /// each thread issues all 64 of its stream loads and its 64 `hc_fn` loads before accumulating
    /// them in the same order, instead of eight at a time. Same bits (`glm5next_hc_fused_gate`
    /// checks it).
    pub pre_prefetch: bool,
    /// 2026-10-21: `METRALE_GLM_MHC_PRE_C1` (on unless `0`): a decode step's pre-norm takes
    /// `glm5next_hc_pre_norm_c1` where `glm5next_mhc_c1::c1_kernel` allows. Same bits.
    pub pre_c1: bool,
    /// 2026-10-11: `METRALE_GLM_MHC_PRE_SPLITK`: below `HC_PRE_NORM_ROWS_MIN_TOKENS` the pre-norm
    /// takes `glm5next_hc_pre_norm_sk64` where the shape allows and the site's `mix` scratch holds
    /// the slice partials (`glm5next_mhc_fused::splitk_kernel`): the mix runs on 64 blocks per
    /// token (k split) instead of one per mix row. Reorders the mix sums (accuracy class RO).
    pub pre_splitk: bool,
    /// 2026-10-17: `METRALE_GLM_MHC_PRE_SLICE` (`glm5next_layer/steps/mhc_slice.rs`): `None` for
    /// one pre-mix launch over all rows, else the most rows one launch takes.
    pub pre_slice: Option<usize>,
}

impl MhcLevers {
    /// 2026-10-17: Resolve from the startup settings, refusing the combinations in the module
    /// invariants.
    pub fn from_settings(s: &InferenceServerSettings) -> Result<Self> {
        let (k, t) = (&s.kernels, &s.telemetry);
        let sp_mhc = k.glm_sp_mhc;
        // 2026-10-22: `METRALE_GLM_SP_OVERLAP` is on by default: without SP-mHC the default is
        // inert (reported once); a set `1` is still refused below.
        let sp_overlap = crate::glm5next_default_on::under_parent(
            s,
            "METRALE_GLM_SP_OVERLAP",
            k.glm_sp_overlap,
            sp_mhc,
            "METRALE_GLM_SP_MHC=1",
        );
        for (env, on) in [
            ("METRALE_GLM_MHC_POST_PRE_FUSED", k.glm_mhc_post_pre_fused),
            ("METRALE_GLM_SP_OVERLAP", sp_overlap),
            ("METRALE_GLM_SP_PIPELINE", k.glm_sp_pipeline),
        ] {
            if on && !sp_mhc {
                bail!("{env}=1 needs METRALE_GLM_SP_MHC=1");
            }
        }
        let dump = crate::glm5next_layer_dump::parse(
            t.glm_layer_dump.as_deref(),
            t.glm_layer_dump_rows.as_deref(),
        )
        .map_err(anyhow::Error::msg)?;
        if k.glm_sp_pipeline && dump.is_some() {
            bail!(
                "METRALE_GLM_SP_PIPELINE=1 with METRALE_GLM_LAYER_DUMP: the dump taps read the \
                 serial order"
            );
        }
        let post_sum_ctas = match k.glm_mhc_post_sum_ctas {
            None | Some(0) => None,
            Some(n) => Some(
                u32::try_from(n)
                    .ok()
                    .filter(|n| (1..=MAX_CTAS).contains(n))
                    .ok_or_else(|| {
                        anyhow!(
                            "METRALE_GLM_MHC_POST_SUM_CTAS={n}: expected 0 (off) or a block \
                             count in 1..={MAX_CTAS}"
                        )
                    })?,
            ),
        };
        let pre_slice = match k.glm_mhc_pre_slice {
            None | Some(0) => None,
            Some(n) => Some(
                usize::try_from(n)
                    .map_err(|_| anyhow!("METRALE_GLM_MHC_PRE_SLICE={n}: expected a row count"))?,
            ),
        };
        Ok(Self {
            sp_mhc,
            sp_overlap,
            sp_pipeline: k.glm_sp_pipeline,
            post_pre_fused: k.glm_mhc_post_pre_fused,
            post_vec: k.glm_mhc_post_vec,
            post_sum_ctas,
            pre_tok: k.glm_mhc_pre_tok,
            pre_tile: k.glm_mhc_pre_tile,
            pre_prefetch: k.glm_mhc_pre_prefetch,
            pre_c1: k.glm_mhc_pre_c1,
            pre_splitk: k.glm_mhc_pre_splitk,
            pre_slice,
        })
    }

    /// 2026-10-17: The log lines these levers used to print at their first read.
    pub(crate) fn announce(&self) {
        if self.post_pre_fused {
            tracing::warn!(
                "METRALE_GLM_MHC_POST_PRE_FUSED=1: SP-mHC runs the attention post-mix and the FFN \
                 pre-norm of each piece as one launch (glm5next_hc_post_pre_norm_tok4, same bits)"
            );
        }
        if self.post_vec {
            let (kernel, block, vec) = {
                use crate::glm5next_hc_post_vec as v;
                (v::KERNEL, v::BLOCK, v::VEC)
            };
            tracing::info!(
                "METRALE_GLM_MHC_POST_VEC=1: the SP-mHC post-mix runs {kernel} ({vec} columns per \
                 thread, {block}-thread blocks, summed row written back only for a dump tap; \
                 same bytes)"
            );
        }
        if let Some(n) = self.post_sum_ctas {
            tracing::warn!(
                "METRALE_GLM_MHC_POST_SUM_CTAS={n}: glm5next_hc_post_rank_sum runs at most {n} \
                 blocks of {CAPPED_BLOCK} (grid-stride, same bytes)"
            );
        }
        if self.pre_tok {
            tracing::warn!(
                "METRALE_GLM_MHC_PRE_TOK=1: wide GLM mHC pre-norms run {} tokens per block \
                 (glm5next_hc_pre_norm_tok4, same bits)",
                crate::glm5next_mhc_tok::HC_PRE_NORM_TOK
            );
        }
        if self.pre_tile {
            use crate::glm5next_mhc_tile as t;
            tracing::info!(
                "METRALE_GLM_MHC_PRE_TILE=1: wide GLM mHC pre-norms (hc 4) run {} tokens per \
                 block through a {}-stage cp.async ring ({}, {} B shared, same bits as tok4)",
                t::HC_PRE_TILE_TOKENS,
                t::HC_PRE_TILE_STAGES,
                t::HC_PRE_TILE_KERNEL,
                t::tile_smem_bytes()
            );
        }
    }
}

/// 2026-10-22: Whether `METRALE_GLM_MHC_FUSED` (default on) runs at this site: the fused kernels
/// are in the build and take the site's shape and BF16 `hc_fn`. Where they do not, the site keeps
/// the three launches of the lever-off build (reported once) instead of failing the forward.
pub fn fused_runs(
    kernels: &crate::glm5next_mhc::Glm5NextMhcKernels,
    site: &crate::glm5next_mhc::Glm5NextMhcSiteWeights,
    hidden: usize,
    hc_mult: usize,
) -> bool {
    let why = if kernels.hc_pre_norm.0 == 0 || kernels.hc_pre_norm_rows.0 == 0 {
        Some("the glm5next_mhc_fused kernels are not in this build".to_string())
    } else if !site.hc_fn_bf16 {
        Some("the site's hc_fn is FP32".to_string())
    } else {
        crate::glm5next_mhc_fused::hc_pre_norm_refusal(hidden as u32, hc_mult as u32)
    };
    match why {
        None => true,
        Some(why) => {
            crate::glm5next_default_on::report_inert("METRALE_GLM_MHC_FUSED", &why);
            false
        }
    }
}

#[cfg(test)]
impl MhcLevers {
    /// 2026-10-17: The levers of the declared defaults, for a test that builds the kernels.
    pub(crate) fn declared() -> Self {
        Self::from_settings(&metrale_model_layers::lever_equivalence::defaults())
            .expect("the declared defaults resolve")
    }
}

#[cfg(test)]
#[path = "glm5next_mhc_levers_tests.rs"]
mod tests;
