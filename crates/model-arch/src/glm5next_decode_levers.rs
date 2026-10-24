// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: The GLM-5.3 decode levers the `met serve` command line selects (R1..R4 of the
//! TP=3 mid-50s campaign), one typed value published once before the model is built. No
//! environment variable reaches them: the serve publishes the flags it parsed, and an
//! unpublished read takes [`GlmDecodeLevers::default`] (every lever off).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The first publication or read wins (`OnceLock`); the serve publishes before the model is
//!   built, so every layer and the drafter read the same value.
//! - The default is today's path: each lever is a kill switch at its default.

use std::str::FromStr;
use std::sync::OnceLock;

/// 2026-10-24: The copy of this rank's vocab shard the MTP drafter's one-row head sweep reads
/// (`--glm-draft-head`). Drafts only: the target verifies every draft with its own head, so the
/// copy moves acceptance, never an emitted token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GlmDraftHead {
    /// 2026-10-24: FP8 E4M3 with a per-row scale (`dense_gemv_fp8w`), 211.7 MB per TP=3 shard.
    #[default]
    Fp8,
    /// 2026-10-24: NVFP4 (E2M1, E4M3 block-16 scales, one FP32 scale per shard;
    /// `w4a16_gemv_sw`), 118.9 MB per TP=3 shard.
    Nvfp4,
}

impl GlmDraftHead {
    /// 2026-10-24: Every value, in the order the flag lists them.
    pub const ALL: [Self; 2] = [Self::Fp8, Self::Nvfp4];

    /// 2026-10-24: The flag value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

impl FromStr for GlmDraftHead {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|h| h.name() == s)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|h| h.name()).collect();
                format!("`{s}`: expected one of {}", names.join(", "))
            })
    }
}

/// 2026-10-24: The command line's GLM decode levers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct GlmDecodeLevers {
    /// 2026-10-24: R1, `--glm-draft-head`.
    pub draft_head: GlmDraftHead,
    /// 2026-10-24: R3, `--glm-mhc-post-pre`: the attention site's reduce + post-mix and the FFN
    /// site's pre-norm as one launch at the decode widths (`crate::glm5next_mhc_post_pre_c1`).
    pub mhc_post_pre: bool,
    /// 2026-10-24: R2, `--glm-router-fused`: the MoE router GEMV and its top-k as one launch at
    /// 2..=8 rows (`crate::glm5next_mlp::forward::router_fused`).
    pub router_fused: bool,
    /// 2026-10-24: R4, `--glm-dsa-topk-seg`: the multi-CTA DSA top-k's five launches (four
    /// histogram passes and the collect) as one persistent-segment launch, `dsa_topk_mc_seg`
    /// (`crate::glm5next_dsa::select::topk`).
    pub topk_seg: bool,
}

static LEVERS: OnceLock<GlmDecodeLevers> = OnceLock::new();

/// 2026-10-24: Publish the command line's levers. Returns the value in force; a caller that gets
/// a different one should warn.
pub fn set_glm_decode_levers_from_cli(levers: GlmDecodeLevers) -> GlmDecodeLevers {
    let _ = LEVERS.set(levers);
    *LEVERS.get().expect("just set")
}

/// 2026-10-24: The levers in force: the default unless the serve published others.
pub fn glm_decode_levers() -> GlmDecodeLevers {
    *LEVERS.get_or_init(GlmDecodeLevers::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-24: Every flag value parses back to itself, an unknown value is refused with the
    /// list, and the default is today's FP8 copy.
    #[test]
    fn draft_head_names_parse_and_the_default_is_fp8() {
        for h in GlmDraftHead::ALL {
            assert_eq!(h.name().parse::<GlmDraftHead>(), Ok(h));
        }
        let e = "int4".parse::<GlmDraftHead>().unwrap_err();
        assert!(e.contains("fp8, nvfp4"), "{e}");
        assert_eq!(GlmDecodeLevers::default().draft_head, GlmDraftHead::Fp8);
    }
}
