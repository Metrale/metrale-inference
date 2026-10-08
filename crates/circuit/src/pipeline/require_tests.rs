// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The required pipeline on the toy circuit (NVFP4 W4A16 linears, a BF16 head): the
//! reference requirement, the `activation_quantization` policy (fixed, ladder, family override,
//! adaptive), rule-stated steps and hand-off formats, keyed state precisions, and every missing
//! input refused.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::{Need, keyed_state, required};
use crate::pipeline::PipelineError;
use crate::state::StateDtype;
use crate::test_toy;

fn settings(aq: &str) -> BTreeMap<String, String> {
    BTreeMap::from([("activation_quantization".to_string(), aq.to_string())])
}

fn need_line(aq: &str, rows: u64, node: &str) -> Result<String, PipelineError> {
    let c = test_toy::circuit(1);
    let s = settings(aq);
    let need = Need::base(&c, &s, rows);
    let n = c.node(node).expect("node");
    required(&need, n).map(|p| p.to_string())
}

const W4A16: &str = "bf16 -> [act bf16 | weight nvfp4/g16->bf16 | mma bf16*bf16 | accumulate f32 | scale f32] -> bf16";

#[test]
fn the_reference_requirement_follows_the_circuit_formats() {
    assert_eq!(need_line("adaptive", 1, "l0.ffn.up").unwrap(), W4A16);
    assert_eq!(
        need_line("adaptive", 1, "head.lm_head").unwrap(),
        "bf16 -> [act bf16 | weight bf16->bf16 | mma bf16*bf16 | accumulate f32 | scale none] -> bf16"
    );
    assert_eq!(
        need_line("adaptive", 1, "l0.ffn.add").unwrap(),
        "bf16,bf16 -> [compute f32] -> bf16"
    );
    assert_eq!(
        need_line("adaptive", 1, "embed.embed").unwrap(),
        "- -> [gather bf16] -> bf16"
    );
    // 2026-10-02: `declared` is the instantiated input format, as `adaptive` without a rule.
    assert_eq!(need_line("declared", 1, "l0.ffn.up").unwrap(), W4A16);
}

// 2026-10-02: A fixed rung quantizes the activation, and the weight is then consumed natively
// (exactly converted to E4M3 under FP8). Mutation: ignoring the ladder's rows, or the family
// override, picks the wrong rung below.
#[test]
fn a_fixed_activation_policy_sets_the_act_weight_and_multiply_requirement() {
    assert_eq!(
        need_line("fp8", 1, "l0.ffn.up").unwrap(),
        "bf16 -> [act fp8/token | weight nvfp4/g16->e4m3 | mma e4m3*e4m3 | accumulate f32 | scale f32] -> bf16"
    );
    let a4 = "bf16 -> [act nvfp4/g16 | weight nvfp4/g16->e2m1 | mma e2m1*e2m1 | accumulate f32 | scale f32] -> bf16";
    assert_eq!(need_line("nvfp4", 1, "l0.ffn.up").unwrap(), a4);
    let ladder = "1=bf16;2-=nvfp4";
    assert_eq!(need_line(ladder, 1, "l0.ffn.up").unwrap(), W4A16);
    assert_eq!(need_line(ladder, 4, "l0.ffn.up").unwrap(), a4);
    assert_eq!(need_line("bf16,ffn:nvfp4", 1, "l0.ffn.up").unwrap(), a4);
    assert!(
        need_line("bf16,ffn:nvfp4", 1, "head.lm_head")
            .unwrap()
            .contains("act bf16")
    );
    // 2026-10-02: A BF16 weight has no FP8 multiply: refused, not widened.
    let e = need_line("fp8", 1, "head.lm_head").unwrap_err().to_string();
    assert!(
        e.contains("no multiply takes a bf16 weight under a fp8/token"),
        "{e}"
    );
}

#[test]
fn a_policy_without_the_setting_or_with_a_bad_one_has_no_requirement() {
    let c = test_toy::circuit(1);
    let empty = BTreeMap::new();
    let need = Need::base(&c, &empty, 1);
    let e = required(&need, c.node("l0.ffn.up").unwrap()).unwrap_err();
    assert!(
        matches!(&e, PipelineError::Required { detail, .. } if detail.contains("activation_quantization")),
        "{e}"
    );
    // 2026-10-02: An element-wise op needs no setting.
    assert!(required(&need, c.node("l0.ffn.act").unwrap()).is_ok());
    assert!(need_line("fp6", 1, "l0.ffn.up").is_err());
}

fn act_down_rules(down_steps: &str, act_holds: &str) -> Vec<crate::rules::Rule> {
    test_toy::rules(&test_toy::fused(
        "act_down",
        &format!(
            r#"{{ op = "silu_mul"{act_holds} }}, {{ op = "linear", role = "down"{down_steps} }}"#
        ),
        50,
    ))
}

// 2026-10-02: A rule's `holds` changes the fused edge's format on both sides, and its `steps`
// replace the reference value of the steps it names (the weight and multiply following a stated
// or held activation). Mutation: reading the stored format for an in-group input breaks both.
#[test]
fn rule_stated_hand_offs_and_steps_enter_the_requirement() {
    let c = test_toy::circuit(1);
    let rules = act_down_rules(r#", steps = { scale = "bf16" }"#, r#", holds = "f32""#);
    let plan = test_toy::plan(&c, &rules, &test_toy::policy(), 1);
    let s = settings("adaptive");
    let need = Need::of_plan(&c, &plan, &rules, &s);
    let line = |id: &str| required(&need, c.node(id).unwrap()).unwrap().to_string();
    assert_eq!(line("l0.ffn.act"), "bf16 -> [compute f32] -> f32");
    assert_eq!(
        line("l0.ffn.down"),
        "f32 -> [act f32 | weight nvfp4/g16->f32 | mma f32*f32 | accumulate f32 | scale bf16] -> bf16"
    );
    // 2026-10-02: Outside the group nothing changed.
    assert_eq!(line("l0.ffn.up"), W4A16);
}

// 2026-10-02: A rule that runs the activation at a format a fixed policy does not route there is
// an error naming both, in either direction.
#[test]
fn a_stated_activation_against_a_fixed_policy_is_refused() {
    let c = test_toy::circuit(1);
    let rules = act_down_rules(r#", steps = { act = "nvfp4/g16" }"#, "");
    let plan = test_toy::plan(&c, &rules, &test_toy::policy(), 1);
    let down = c.node("l0.ffn.down").unwrap();
    let s = settings("bf16");
    let e = required(&Need::of_plan(&c, &plan, &rules, &s), down).unwrap_err();
    assert!(
        e.to_string()
            .contains("the rule runs the activation at nvfp4/g16"),
        "{e}"
    );
    let s = settings("adaptive");
    let p = required(&Need::of_plan(&c, &plan, &rules, &s), down).unwrap();
    assert!(p.to_string().contains("mma e2m1*e2m1"), "{p}");
}

#[test]
fn keyed_state_formats_read_their_setting() {
    let s = |k: &str, v: &str| BTreeMap::from([(k.to_string(), v.to_string())]);
    assert_eq!(
        keyed_state("ssm_h_storage", &s("ssm_h_dtype", "f32")),
        Ok(StateDtype::F32)
    );
    assert_eq!(
        keyed_state("ssm_h_storage", &s("ssm_h_dtype", "f16-pool")),
        Ok(StateDtype::F16)
    );
    assert!(keyed_state("ssm_h_storage", &s("ssm_h_dtype", "bf8")).is_err());
    assert!(keyed_state("ssm_h_storage", &BTreeMap::new()).is_err());
    assert_eq!(
        keyed_state("conv_dtype", &s("conv_dtype", "bf16")),
        Ok(StateDtype::Bf16)
    );
}

#[test]
fn mxfp4_pipeline_cannot_borrow_nvfp4_scale_semantics() {
    use super::{Format, StepKind};
    for act in [Format::Bf16, Format::F32, Format::Nvfp4 { group: 16 }] {
        let err = super::derived(StepKind::Weight, act, Format::Mxfp4).unwrap_err();
        assert!(err.contains("E8M0/group32"), "{err}");
    }
}
