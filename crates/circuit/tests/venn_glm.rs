// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The GLM-5.3-Flash kernel Venn against the Qwen3.6-35B-A3B NVFP4 MoE and the
//! Qwen3.8-27B declared W4A4 dense model: the checked-in report is what the tool produces, KDA
//! reads as a per-channel policy of the gated delta rule, the hyper-connection and the sparse
//! latent attention are GLM's own families, and the multi-sequence veto is the top flag.
//! Regenerate the report after an intended change with
//! `cargo test -p metrale-circuit --test venn_glm -- --ignored regenerate`.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;
mod venn_common;

use metrale_circuit::Mode;
use metrale_circuit::venn::{self, Class, ParamKind, VennArgs};
use venn_common::*;

const GLM: &str = "glm-5.3/glm-5.3-flash-nvfp4";
const QWEN_MOE: &str = "qwen3.6/qwen3.6-35b-a3b-nvfp4-declared";
const QWEN_DENSE: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth-declared";
const GLM_REPORT: &str = "kernels/circuits/venn/glm-5.3-flash-vs-qwen3.6-35b-a3b.md";

fn glm_args() -> VennArgs {
    VennArgs {
        target: GLM.into(),
        against: vec![QWEN_MOE.into(), QWEN_DENSE.into()],
        modes: vec![Mode::Decode, Mode::MultiSeq, Mode::Verify],
        rows: vec![1, 16, 128],
        verify_rows: vec![4],
        out: GLM_REPORT.into(),
    }
}

#[test]
fn the_checked_in_glm_report_is_current() {
    let text = venn::report_text(&Tree, &glm_args(), None).expect("report");
    assert!(
        text == common::read(GLM_REPORT),
        "{GLM_REPORT} is stale; regenerate with `cargo test -p metrale-circuit --test venn_glm \
         -- --ignored regenerate` and review the diff"
    );
}

#[test]
fn kda_is_a_per_channel_policy_of_the_gated_delta_rule() {
    let r = report_of(&glm_args());
    let recur = row(&r, Mode::Decode, 1, "kda.recur");
    let p = recur.primary.as_ref().expect("classified");
    assert_eq!(
        (p.family.as_str(), p.class),
        ("kda_recurrent", Class::SharedUnmeasured)
    );
    let gdn = finding(recur, "gdn_recurrence");
    assert_eq!(gdn.class, Class::PolicyVariant);
    assert_eq!(
        diff(gdn, "decay"),
        ("channel".into(), "head".into(), ParamKind::Policy)
    );
    // 2026-10-08: No KDA kernel takes more than one row, so a 16-row step is the strided GDN
    // kernel's policy variant, not a KDA kernel.
    let wide = row(&r, Mode::MultiSeq, 16, "kda.recur")
        .primary
        .as_ref()
        .expect("classified");
    assert_eq!(
        (wide.family.as_str(), wide.class),
        ("gdn_recurrence_strided", Class::PolicyVariant)
    );
    let norm = finding(row(&r, Mode::Decode, 1, "kda.out_norm"), "gated_rms_norm");
    assert_eq!(
        diff(norm, "gate_act"),
        ("sigmoid".into(), "silu".into(), ParamKind::Policy)
    );
}

#[test]
fn the_glm_only_ops_classify_against_their_own_families() {
    let r = report_of(&glm_args());
    for (site, family) in [
        ("kda.pre", "glm_mhc"),
        ("kda.post", "glm_mhc"),
        ("dsa.attend", "glm_mla_decode"),
        ("dsa.select", "dsa_indexer"),
        ("dsa.pool", "dsa_indexer"),
        ("dsa.idx_k_norm", "layer_norm"),
        ("moe.experts_clamp", "glm_swiglu_clamp"),
    ] {
        let f = row(&r, Mode::Decode, 1, site)
            .primary
            .as_ref()
            .unwrap_or_else(|| panic!("{site} unclassified"));
        assert_eq!(f.family, family, "{site}");
    }
}

#[test]
fn the_multi_sequence_veto_is_the_top_flag() {
    let r = report_of(&glm_args());
    let top = &r.flags[0];
    assert_eq!(
        top.source,
        "legacy: crates/model-arch/src/glm5next_layer/mod.rs:261"
    );
    assert!(top.run.mode == Mode::MultiSeq && top.run.rows == 128);
}

#[test]
#[ignore = "writes the checked-in report; run explicitly to regenerate"]
fn regenerate() {
    let text = venn::report_text(&Tree, &glm_args(), None).expect("report");
    std::fs::write(common::root().join(GLM_REPORT), text).expect("write report");
}
