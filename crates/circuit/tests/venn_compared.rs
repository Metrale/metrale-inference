// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: A compared model that is not golden: `met circuit venn --target
//! glm-5.3/glm-5.3-flash-nvfp4 --against nemotron-3.5/nemotron-3.5-lightning-30b-a3b-nvfp4` used to
//! fail on the first Lightning node no FUSIONS.toml rule covers (`l0.mamba.norm`, rms_norm, decode
//! at 1 row): the Venn fused every compared model strictly, and Nemotron-H has almost no rules.
//! Now a non-golden compared model is planned the way `met circuit plan` plans any checkpoint
//! (a placeholder for each node no rule covers), its placeholder nodes count as no usage, and the
//! report lists them; a golden compared model is still fused strictly.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;
mod venn_common;

use metrale_circuit::Mode;
use metrale_circuit::venn::{self, VennArgs};
use venn_common::*;

const GLM: &str = "glm-5.3/glm-5.3-flash-nvfp4";

fn glm_against_lightning() -> VennArgs {
    VennArgs {
        target: GLM.into(),
        against: vec![LIGHTNING.into()],
        modes: Mode::ALL.to_vec(),
        rows: vec![1, 16, 128],
        verify_rows: vec![2],
        out: "kernels/circuits/venn/glm-5.3-flash-vs-nemotron-3.5-lightning.md".into(),
    }
}

#[test]
fn a_non_golden_compared_model_is_planned_with_placeholders_and_its_gaps_are_listed() {
    let r = report_of(&glm_against_lightning());
    let gaps = r
        .uncovered
        .iter()
        .find(|u| u.recipe == LIGHTNING && u.run.mode == Mode::Decode && u.run.rows == 1)
        .expect("Lightning's uncovered sites at decode n=1");
    assert!(gaps.sites.iter().any(|s| s == "mamba.norm"), "{gaps:?}");
    assert!(gaps.sites.iter().any(|s| s == "mamba.ssm"), "{gaps:?}");
    // 2026-10-10: Sites a rule does cover are not listed: Lightning's NVFP4 head is planned on
    // the W4A16 tensor-core rows kernel.
    assert!(!gaps.sites.iter().any(|s| s == "head.lm_head"), "{gaps:?}");
    let text = venn::render(&r);
    assert!(
        text.contains("## Compared models without full rules"),
        "the report must disclose the placeholder sites"
    );
    assert!(text.contains("`mamba.norm`"));
}

#[test]
fn the_covered_nodes_of_a_non_golden_compared_model_still_count_as_usages() {
    let r = report_of(&glm_against_lightning());
    let head = row(&r, Mode::Decode, 1, "head.lm_head");
    let named = head
        .primary
        .iter()
        .chain(&head.also)
        .any(|f| format!("{f:?}").contains(LIGHTNING));
    assert!(
        named,
        "Lightning's planned head kernel is a compared usage: {head:?}"
    );
}

#[test]
fn a_golden_compared_model_is_still_fused_strictly() {
    // 2026-10-10: The Qwen3.8 dense instance is golden: its report lists no placeholder sites.
    let r = report_of(&VennArgs {
        against: vec![DENSE.into()],
        ..glm_against_lightning()
    });
    assert!(r.uncovered.is_empty(), "{:?}", r.uncovered);
}
