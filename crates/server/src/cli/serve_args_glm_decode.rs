// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: The GLM-5.3 decode lever flags (flattened last into `ServeServiceArgs`), published
//! as one typed value (`metrale_model_arch::glm5next_decode_levers`) by `publish_kernel_flags`.
//! No environment variable reaches them.
//!
//! Owner: server CLI.
//! Invariants: the `///` text on the struct's fields is the `--help` output and
//! carries no date.

use clap::Args;
use metrale_model_arch::glm5next_decode_levers::{GlmDecodeLevers, GlmDraftHead};

#[derive(Args, Debug, Clone, PartialEq)]
pub struct ServeGlmDecodeArgs {
    /// Copy of the GLM-5.3 MTP drafter's LM-head shard the one-row draft sweep reads:
    /// `fp8` (the default; E4M3 with a per-row scale) or `nvfp4` (NVFP4, about half the
    /// bytes per draft). Draft-only: the target model verifies every drafted token with
    /// its own head, so only how many drafts are accepted moves. Every rank must use the
    /// same value.
    #[arg(long, value_name = "fp8|nvfp4", default_value = "fp8")]
    pub glm_draft_head: GlmDraftHead,

    /// Run the GLM-5.3 attention site's tensor-parallel reduce + mHC post-mix and the
    /// FFN site's pre-mix + norm as one launch at decode and K=2 verify widths (1-2
    /// rows), where the two launches would otherwise run (default: false). Same bytes.
    /// Refused by `--forward circuit` until a plan rule models it.
    #[arg(long, default_value_t = false)]
    pub glm_mhc_post_pre: bool,

    /// Run the GLM-5.3 MoE router (2-8 rows) and its top-k as one launch, where they would
    /// run as the few-row router GEMV then the rank top-k (default: false). Same logits,
    /// expert ids and weights. Refused by `--forward circuit` until a plan rule models it.
    #[arg(long, default_value_t = false)]
    pub glm_router_fused: bool,

    /// Run the GLM-5.3 DSA multi-CTA top-k (four histogram passes and a collect) as one
    /// persistent-segment launch, where its five launches would otherwise run (default:
    /// false). Same selected pools. Refused by `--forward circuit` until a plan rule
    /// models it.
    #[arg(long, default_value_t = false)]
    pub glm_dsa_topk_seg: bool,
}

impl ServeGlmDecodeArgs {
    /// 2026-10-24: The typed value the serve publishes.
    pub fn levers(&self) -> GlmDecodeLevers {
        GlmDecodeLevers {
            draft_head: self.glm_draft_head,
            mhc_post_pre: self.glm_mhc_post_pre,
            router_fused: self.glm_router_fused,
            topk_seg: self.glm_dsa_topk_seg,
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        glm: super::ServeGlmDecodeArgs,
    }

    /// 2026-10-24: Absent flags give the default levers (every lever off); a given value reaches
    /// the published value; an unknown value fails the parse.
    #[test]
    fn flags_reach_the_typed_levers() {
        use metrale_model_arch::glm5next_decode_levers::{GlmDecodeLevers, GlmDraftHead};
        let none = Cli::try_parse_from(["met"]).unwrap();
        assert_eq!(none.glm.levers(), GlmDecodeLevers::default());
        let nv = Cli::try_parse_from(["met", "--glm-draft-head", "nvfp4"]).unwrap();
        assert_eq!(nv.glm.levers().draft_head, GlmDraftHead::Nvfp4);
        assert!(Cli::try_parse_from(["met", "--glm-draft-head", "int4"]).is_err());
        let pp = Cli::try_parse_from(["met", "--glm-mhc-post-pre"]).unwrap();
        assert_eq!(
            pp.glm.levers(),
            GlmDecodeLevers {
                mhc_post_pre: true,
                ..GlmDecodeLevers::default()
            }
        );
        let rf = Cli::try_parse_from(["met", "--glm-router-fused"]).unwrap();
        assert_eq!(
            rf.glm.levers(),
            GlmDecodeLevers {
                router_fused: true,
                ..GlmDecodeLevers::default()
            }
        );
        let ts = Cli::try_parse_from(["met", "--glm-dsa-topk-seg"]).unwrap();
        assert_eq!(
            ts.glm.levers(),
            GlmDecodeLevers {
                topk_seg: true,
                ..GlmDecodeLevers::default()
            }
        );
    }
}
