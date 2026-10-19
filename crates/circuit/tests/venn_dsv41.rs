// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The DeepSeek-V4.1-Flash kernel Venn against the Qwen MoE and dense pair: the
//! checked-in report is what the tool produces, the routed experts are a policy variant of the
//! grouped tensor-core MoE family, the hyper-connection, clamp and indexer ops reuse the
//! GLM-5.3 families, and the latent attention, compressor and Engram ops are novel.
//! GLM-5.3-Flash is not a compared model: its fusion rules cover no routed-expert node, so the
//! tool refuses to plan it; its families still appear here through the kernel manifest.
//! Regenerate the report after an intended change with
//! `cargo test -p metrale-circuit --test venn_dsv41 -- --ignored regenerate`.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;
mod venn_common;

use metrale_circuit::Mode;
use metrale_circuit::venn::{self, Class, ParamKind, VennArgs};
use venn_common::*;

const DSV41: &str = "deepseek-v4.1/deepseek-v4.1-flash-mxfp4";
const QWEN_MOE: &str = "qwen3.6/qwen3.6-35b-a3b-nvfp4-declared";
const QWEN_DENSE: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth-declared";
const REPORT: &str = "kernels/circuits/venn/deepseek-v4.1-flash-vs-qwen3.6-35b-a3b.md";

fn dsv41_args() -> VennArgs {
    VennArgs {
        target: DSV41.into(),
        against: vec![QWEN_MOE.into(), QWEN_DENSE.into()],
        modes: vec![Mode::Decode, Mode::MultiSeq, Mode::Verify],
        rows: vec![1, 16, 128],
        verify_rows: vec![4],
        out: REPORT.into(),
    }
}

#[test]
fn the_checked_in_report_is_current() {
    let text = venn::report_text(&Tree, &dsv41_args(), None).expect("report");
    assert!(
        text == common::read(REPORT),
        "{REPORT} is stale; regenerate with `cargo test -p metrale-circuit --test venn_dsv41 \
         -- --ignored regenerate` and review the diff"
    );
}

/// 2026-10-18: The largest decode share is the routed experts; the grouped kernel needs the
/// MXFP4 weight, the FP8 activation and the clamped SwiGLU epilogue as policies, not a rewrite.
#[test]
fn the_routed_experts_are_a_policy_variant_of_the_grouped_moe_kernel() {
    let r = report_of(&dsv41_args());
    let table = r
        .tables
        .iter()
        .find(|t| t.run.mode == Mode::Decode && t.run.rows == 1)
        .expect("decode n=1");
    assert_eq!(table.rows[0].site, "moe.experts_gate_up");
    let up = finding(&table.rows[0], "moe_grouped_tc");
    assert_eq!(up.class, Class::PolicyVariant);
    for (param, target, other) in [
        ("weight", "mxfp4/g32", "nvfp4/g16"),
        ("activation", "fp8/g32", "bf16"),
        ("epilogue", "swiglu_clamp", "silu_mul"),
    ] {
        assert_eq!(
            diff(up, param),
            (target.into(), other.into(), ParamKind::Policy),
            "{param}"
        );
    }
    let down = finding(
        row(&r, Mode::Decode, 1, "moe.experts_down"),
        "moe_grouped_tc",
    );
    assert_eq!(
        diff(down, "weight"),
        ("mxfp4/g32".into(), "nvfp4/g16".into(), ParamKind::Policy)
    );
}

#[test]
fn the_shared_glm_ops_classify_against_the_glm_families() {
    let r = report_of(&dsv41_args());
    for (site, family) in [
        ("moe.pre", "glm_mhc"),
        ("moe.post", "glm_mhc"),
        ("embed.expand", "glm_mhc"),
        ("head.contract", "glm_mhc"),
        ("csa_src.select", "dsa_indexer"),
        ("moe.experts_clamp", "glm_swiglu_clamp"),
        ("csa_src.ckv_write", "glm_mla_latent_write"),
    ] {
        let f = row(&r, Mode::Decode, 1, site)
            .primary
            .as_ref()
            .unwrap_or_else(|| panic!("{site} unclassified"));
        assert_eq!(f.family, family, "{site}");
    }
}

/// 2026-10-18: No family attends a shared key-value latent over a sliding window plus
/// compressed rows, compresses rows, or looks up and gates Engram n-grams.
#[test]
fn the_v4_only_ops_are_novel() {
    let r = report_of(&dsv41_args());
    for site in [
        "swa.attend",
        "csa_src.attend",
        "hca_reuse.attend",
        "csa_src.compress",
        "engram.lookup",
        "engram.gate",
    ] {
        assert!(
            row(&r, Mode::Decode, 1, site).primary.is_none(),
            "{site} is classified"
        );
    }
}

#[test]
#[ignore = "writes the checked-in report; run explicitly to regenerate"]
fn regenerate() {
    let text = venn::report_text(&Tree, &dsv41_args(), None).expect("report");
    std::fs::write(common::root().join(REPORT), text).expect("write report");
}
