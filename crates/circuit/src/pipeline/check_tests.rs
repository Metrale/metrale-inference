// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The plan check on the toy circuit: kernels that declare exactly what the plan
//! requires pass; a kernel declaring another accumulator, a kernel in no family, and a fused
//! group whose intermediate changes format without the rule stating it are each refused with
//! the node, rule and difference named.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::check_plan;
use crate::pipeline::{Mismatch, PipelineError};
use crate::rules::Rule;
use crate::test_toy;
use crate::venn::families::{Families, parse_families};

/// 2026-10-02: One family per toy kernel, each declaring the toy pipeline of its op, `extra`
/// families appended; `swap` rewrites the text first (the mutations).
fn families(extra: &str, swap: (&str, &str)) -> Families {
    let mut s = String::from(
        "schema = 1\nhardware = \"toy\"\n[roofline]\ndram_gbps = 1.0\nbf16_tflops = 1.0\nfp8_tflops = 1.0\nnvfp4_tflops = 1.0\ncontext_tokens = 1\n",
    );
    for (kernel, op) in [
        ("embed", "embed"),
        ("norm", "rms_norm"),
        ("up", "linear"),
        ("act", "silu_mul"),
        ("down", "linear"),
        ("add", "residual_add"),
        ("final_norm", "final_norm"),
        ("lm_head", "lm_head"),
    ] {
        s.push_str(&format!(
            "[[family]]\nid = \"{kernel}\"\ndescription = \"toy\"\ncompute = \"cuda_core\"\nkernels = [\"m::{kernel}\"]\nrows = [1, 128]\n{}\nop = [{{ op = \"{op}\" }}]\n[[family.point]]\nvalues = {{}}\nhow = \"instantiation\"\nfiles = [\"t.cu\"]\n",
            test_toy::pipelines(&[op])
        ));
    }
    s.push_str(extra);
    parse_families(&s.replace(swap.0, swap.1)).unwrap_or_else(|e| panic!("toy families: {e}"))
}

fn settings() -> BTreeMap<String, String> {
    BTreeMap::from([(
        "activation_quantization".to_string(),
        "adaptive".to_string(),
    )])
}

fn check(rules: &[Rule], fams: &Families) -> Result<super::PlanPipelines, PipelineError> {
    let c = test_toy::circuit(2);
    let plan = test_toy::plan(&c, rules, &test_toy::policy(), 1);
    check_plan(&c, &plan, rules, fams, &settings())
}

fn mismatches(e: PipelineError) -> Vec<Mismatch> {
    match e {
        PipelineError::Mismatch(m) => m,
        other => panic!("not a mismatch: {other}"),
    }
}

#[test]
fn kernels_that_declare_the_requirement_pass_and_every_node_gets_its_pipeline() {
    let rules = test_toy::rules("");
    let p = check(&rules, &families("", ("", ""))).unwrap();
    let c = test_toy::circuit(2);
    assert_eq!(p.nodes.iter().flatten().count(), c.nodes.len());
    assert_eq!(
        p.line(c.node("l1.ffn.down").unwrap()).unwrap(),
        "bf16 -> [act bf16 | weight nvfp4/g16->bf16 | mma bf16*bf16 | accumulate f32 | scale f32] -> bf16"
    );
}

// 2026-10-02: Mutation: a declared accumulator that is not the required one. Comparing only
// formats, or only the inputs and outputs, would let it plan.
#[test]
fn a_kernel_with_another_accumulator_is_refused() {
    let lin = test_toy::pipelines(&["linear"]);
    let wrong = lin.replace("accumulate = \"f32\"", "accumulate = \"bf16\"");
    let fams = families(
        "",
        (
            &format!("kernels = [\"m::down\"]\nrows = [1, 128]\n{lin}"),
            &format!("kernels = [\"m::down\"]\nrows = [1, 128]\n{wrong}"),
        ),
    );
    let m = mismatches(check(&test_toy::rules(""), &fams).unwrap_err());
    let nodes: Vec<&str> = m.iter().map(|m| m.node.as_str()).collect();
    assert_eq!(nodes, ["l0.ffn.down", "l1.ffn.down"]);
    assert_eq!(m[0].rule, "down");
    assert_eq!(m[0].family, "down");
    assert_eq!(m[0].diffs, ["accumulate: required f32, declared bf16"]);
}

#[test]
fn a_kernel_no_family_declares_is_refused() {
    let fams = families("", ("kernels = [\"m::act\"]", "kernels = [\"m::act_old\"]"));
    let m = mismatches(check(&test_toy::rules(""), &fams).unwrap_err());
    assert_eq!(m.len(), 2);
    assert!(
        m[0].diffs[0].contains("no kernel family declares a pipeline"),
        "{:?}",
        m[0]
    );
}

/// 2026-10-02: A rule fusing the SiLU into the down projection, and a family whose kernel keeps
/// the SiLU product in FP32 registers (`out_act` is what it declares handing on).
fn fused_case(holds: &str, out_act: &str) -> Result<super::PlanPipelines, PipelineError> {
    let rules = test_toy::rules(&test_toy::fused(
        "act_down",
        &format!(r#"{{ op = "silu_mul"{holds} }}, {{ op = "linear", role = "down" }}"#),
        50,
    ));
    let family = format!(
        r#"[[family]]
id = "act_down"
description = "toy"
compute = "cuda_core"
kernels = ["m::act_down"]
rows = [1, 128]
pipeline.silu_mul = {{ in = ["bf16"], compute = "f32", out = ["{out_act}"] }}
pipeline.linear = {{ in = ["f32"], act = "f32", weight = "nvfp4/g16->f32", mma = "f32*f32", accumulate = "f32", scale = "f32", out = ["bf16"] }}
op = [{{ op = "silu_mul" }}, {{ op = "linear", roles = ["down"] }}]
[[family.point]]
values = {{}}
how = "instantiation"
files = ["t.cu"]
"#
    );
    check(&rules, &families(&family, ("", "")))
}

// 2026-10-02: Fusion composes the members' pipelines at the elided edge: the kernel's hand-off
// format must be the stored one, or the one the rule states it holds. Mutation: skipping the
// fused outputs (or reading the stored format for an in-group input) lets the silent FP32
// hand-off plan.
#[test]
fn a_fused_edge_that_changes_format_must_be_stated_and_declared_alike() {
    let m = mismatches(fused_case("", "f32").unwrap_err());
    let act = m
        .iter()
        .find(|m| m.node == "l0.ffn.act")
        .expect("act refused");
    assert_eq!(act.diffs, ["out[0]: required bf16, declared f32"]);
    let down = m
        .iter()
        .find(|m| m.node == "l0.ffn.down")
        .expect("down refused");
    assert!(
        down.diffs
            .contains(&"in[0]: required bf16, declared f32".to_string()),
        "{down:?}"
    );
    let ok = fused_case(r#", holds = "f32""#, "f32").unwrap();
    let c = test_toy::circuit(2);
    assert_eq!(
        ok.line(c.node("l0.ffn.act").unwrap()).unwrap(),
        "bf16 -> [compute f32] -> f32 (rule)"
    );
    // 2026-10-02: Stated by the rule, but the kernel hands on BF16: refused the other way.
    let m = mismatches(fused_case(r#", holds = "f32""#, "bf16").unwrap_err());
    assert!(
        m.iter()
            .any(|m| m.node == "l0.ffn.act" && m.diffs == ["out[0]: required f32, declared bf16"])
    );
}
