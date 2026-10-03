// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Declared pipelines: a manifest entry that does not cover its family's ops, names
//! a step, op, role, kernel or parameter it does not have, or fails to parse at a point is
//! refused at load; resolution picks the most specific declaration (kernel, then point, then
//! family; a narrowed key before a plain one) and refuses an ambiguous or uninstantiated one.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::{Site, declared_for};
use crate::rules::KernelId;
use crate::test_toy;
use crate::venn::families::{Families, parse_families};

const W4A16: &str = r#"{ in = ["bf16"], act = "bf16", weight = "{weight}->bf16", mma = "bf16*bf16", accumulate = "f32", scale = "f32", out = ["bf16"] }"#;

fn manifest(pipes: &str, extra_points: &str) -> String {
    format!(
        r#"
schema = 1
hardware = "toy"
[roofline]
dram_gbps = 1.0
bf16_tflops = 1.0
fp8_tflops = 1.0
nvfp4_tflops = 1.0
context_tokens = 1
[[family]]
id = "f"
description = "toy"
compute = "cuda_core"
kernels = ["m::a", "m::b"]
rows = [1, 128]
{pipes}
op = [{{ op = "linear", roles = ["gate_up", "down"] }}, {{ op = "silu_mul" }}, {{ op = "relu2" }}]
[[family.param]]
name = "weight"
kind = "policy"
from = {{ linear = "weight" }}
[[family.point]]
values = {{ weight = "nvfp4/g16" }}
how = "instantiation"
files = ["f.cu"]
{extra_points}
"#
    )
}

fn base() -> String {
    [
        format!("pipeline.linear = {W4A16}"),
        r#"pipeline."linear:down after silu_mul" = { in = ["f32"], act = "f32", weight = "{weight}->f32", mma = "f32*f32", accumulate = "f32", scale = "f32", out = ["bf16"] }"#.into(),
        r#"pipeline.silu_mul = { in = ["bf16"], compute = "f32", out = ["bf16"] }"#.into(),
        r#"pipeline."silu_mul feeds linear:down" = { in = ["bf16"], compute = "f32", out = ["f32"] }"#.into(),
        r#"pipeline.relu2 = "uninstantiated""#.into(),
        r#"kernel_pipeline."m::b".silu_mul = { in = ["bf16"], compute = "bf16", out = ["bf16"] }"#.into(),
    ]
    .join("\n")
}

const BF16_POINT: &str = r#"[[family.point]]
values = { weight = "bf16" }
how = "copy"
files = ["g.cu"]
pipeline.linear = { in = ["bf16"], act = "bf16", weight = "bf16->bf16", mma = "bf16*bf16", accumulate = "f32", scale = "none", out = ["bf16"] }
"#;

fn fams(pipes: &str, points: &str) -> Result<Families, String> {
    parse_families(&manifest(pipes, points)).map_err(|e| e.to_string())
}

#[test]
fn a_well_formed_declaration_loads() {
    let f = fams(&base(), BF16_POINT).unwrap();
    let fam = &f.families[0];
    assert_eq!(fam.pipeline.family.len(), 5);
    assert_eq!(fam.pipeline.kernels.len(), 1);
    assert_eq!(fam.points[1].pipeline.len(), 1);
}

#[test]
fn malformed_declarations_are_refused_at_load() {
    let cases: [(String, &str); 9] = [
        (
            base().replace("pipeline.relu2 = \"uninstantiated\"", ""),
            "no `pipeline.relu2`",
        ),
        (
            base().replace(", accumulate = \"f32\", scale = \"f32\", out = [\"bf16\"] }\npipeline.\"linear:down", ", scale = \"f32\", out = [\"bf16\"] }\npipeline.\"linear:down"),
            "states no `accumulate`",
        ),
        (
            base().replace("pipeline.silu_mul = { in = [\"bf16\"], compute", "pipeline.silu_mul = { in = [\"bf16\"], epilogue = \"f32\", compute"),
            "`epilogue` is not a step of `silu_mul`",
        ),
        (format!("{}\npipeline.rope = {{ in = [\"bf16\", \"bf16\"], compute = \"f32\", out = [\"bf16\", \"bf16\"] }}", base()), "the family does not implement"),
        (
            format!("{}\npipeline.\"linear:q\" = {W4A16}", base()),
            "`linear:q`, which the family does not implement",
        ),
        (
            base().replace("\"silu_mul feeds linear:down\"", "\"silu_mul feeds mamba3\""),
            "`mamba3` is no op",
        ),
        (
            base().replace("kernel_pipeline.\"m::b\"", "kernel_pipeline.\"m::c\""),
            "`m::c`, which is not its kernel",
        ),
        (
            base().replace("{weight}->bf16", "{wieght}->bf16"),
            "`{wieght}` is no parameter of the point",
        ),
        (
            base().replace("pipeline.relu2 = \"uninstantiated\"", "pipeline.relu2 = \"todo\""),
            "neither a table nor",
        ),
    ];
    for (pipes, want) in cases {
        let e = fams(&pipes, "").unwrap_err();
        assert!(e.contains(want), "want `{want}`, got: {e}");
    }
    // 2026-10-02: A value that parses at one point and not another is refused with the point.
    let e = fams(
        &base(),
        &BF16_POINT.replace("weight = \"bf16\" }", "weight = \"fp6\" }"),
    )
    .unwrap_err();
    assert!(e.contains("fp6"), "{e}");
}

fn resolve(
    fams: &Families,
    kernels: &[&str],
    node: &str,
    point: &[(&str, &str)],
) -> Result<String, String> {
    let c = test_toy::circuit(1);
    let fam = &fams.families[0];
    let group: Vec<KernelId> = kernels
        .iter()
        .map(|k| KernelId {
            module: "m".into(),
            func: (*k).into(),
        })
        .collect();
    let values: BTreeMap<String, String> = point
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let site = Site::of(&c, c.node(node).expect("node"));
    declared_for(
        &fam.pipeline,
        &fam.point_pipelines(),
        &group,
        &site,
        &values,
    )
    .map(|p| p.to_string())
}

// 2026-10-02: Mutation: dropping the kernel level, the point level or the narrowed keys from
// `lookup` changes one of the answers below.
#[test]
fn the_most_specific_declaration_wins() {
    let f = fams(&base(), BF16_POINT).unwrap();
    let nv = [("weight", "nvfp4/g16")];
    assert_eq!(
        resolve(&f, &["a"], "l0.ffn.up", &nv).unwrap(),
        "bf16 -> [act bf16 | weight nvfp4/g16->bf16 | mma bf16*bf16 | accumulate f32 | scale f32] -> bf16"
    );
    // 2026-10-02: The point's own declaration over the family's.
    assert!(
        resolve(&f, &["a"], "l0.ffn.up", &[("weight", "bf16")])
            .unwrap()
            .contains("scale none")
    );
    // 2026-10-02: Narrowed by what the node reads from, and by what it feeds.
    assert!(
        resolve(&f, &["a"], "l0.ffn.down", &nv)
            .unwrap()
            .starts_with("f32 -> [act f32")
    );
    assert_eq!(
        resolve(&f, &["a"], "l0.ffn.act", &[]).unwrap(),
        "bf16 -> [compute f32] -> f32"
    );
    // 2026-10-02: A kernel's declaration over everything else.
    assert_eq!(
        resolve(&f, &["a", "b"], "l0.ffn.act", &[]).unwrap(),
        "bf16 -> [compute bf16] -> bf16"
    );
}

#[test]
fn an_ambiguous_uninstantiated_or_unplaced_node_is_refused() {
    let two = format!(
        "{}\nkernel_pipeline.\"m::a\".silu_mul = {{ in = [\"bf16\"], compute = \"f16\", out = [\"bf16\"] }}",
        base()
    );
    let f = fams(&two, "").unwrap();
    let e = resolve(&f, &["a", "b"], "l0.ffn.act", &[]).unwrap_err();
    assert!(
        e.contains("the group's kernels declare different pipelines"),
        "{e}"
    );
    let f = fams(&base(), "").unwrap();
    let e = resolve(&f, &["a"], "l0.ffn.up", &[("weight", "fp8/channel")]).unwrap_err();
    assert!(e.contains("not one the family instantiates"), "{e}");
    let mut c = test_toy::circuit(1);
    let act = c.node("l0.ffn.act").unwrap();
    c.nodes[act].op = crate::ir::OpKind::Relu2;
    let site = Site::of(&c, act);
    let fam = &f.families[0];
    let e = declared_for(
        &fam.pipeline,
        &fam.point_pipelines(),
        &[],
        &site,
        &BTreeMap::new(),
    )
    .unwrap_err();
    assert!(e.contains("parameterization target"), "{e}");
}
