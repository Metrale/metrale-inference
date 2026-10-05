// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The FP8 MoE recipes draft twice only at one or two active sequences.
//! Two drafts lower J/tok there and raise it from C4 up (measured on dgx2), so a recipe
//! value or default that widens the two-draft rungs must fail here, not in a campaign.

use super::*;

const FP8_MOE: [&str; 5] = [
    "qwen3.6-35b-a3b-fp8-bf16head",
    "qwen3.6-35b-a3b-fp8-mtp",
    "qwen3.6-35b-a3b-fp8-nvfp4head",
    "qwen3.6-35b-a3b-fp8-nvfp4head-experts-nvfp4",
    "qwen3.6-35b-a3b-fp8-nvfp4head-experts-nvfp4-gate-up",
];

fn in_tree(stem: &str) -> Recipe {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../recipes/qwen3.6")
        .join(format!("{stem}.yaml"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Recipe::parse(format!("qwen3.6/{stem}"), &text).unwrap_or_else(|e| panic!("{stem}: {e:#}"))
}

#[test]
fn fp8_moe_recipes_draft_twice_only_at_one_or_two_sequences() {
    use metrale_model_layers::speculative::{ladder_drafts_from_steps, parse_mtp_k_ladder};
    for stem in FP8_MOE {
        let r = in_tree(stem);
        let num_drafts: usize = r.defaults["num_drafts"].parse().expect("num_drafts");
        let ladder = parse_mtp_k_ladder(&r.defaults["mtp_k_ladder"]).expect("mtp_k_ladder");
        for width in 1..=128 {
            let want = if width <= 2 { 2 } else { 1 };
            assert_eq!(
                ladder_drafts_from_steps(&ladder, width, num_drafts),
                want,
                "{stem}: {width} active sequences"
            );
        }
        assert_eq!(r.defaults["mtp_dcut_ratio"], "1.0", "{stem}: D-Cut off");
    }
}
