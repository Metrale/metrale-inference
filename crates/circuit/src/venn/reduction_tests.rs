// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Reduction trees in KERNEL_FAMILIES.toml: a declared tree is parsed and named by
//! its kernels, every malformed or dangling name is refused, a `numerics` parameter takes tree
//! ids, the parity verdict follows the tree, and gb10's RMSNorm kernels all run one declared,
//! byte-comparable tree.
//!
//! Owner: metrale-circuit tests.
//! Invariants: every test fails when the behaviour it names is removed (the mutation notes say
//! which change each one catches).

use super::super::{ParamKind, parse_families};
use super::{SplitSource, Verdict, verdict};
use crate::rules::KernelId;

const HEAD: &str = r#"
schema = 1
hardware = "toy"
[roofline]
dram_gbps = 249.0
bf16_tflops = 123.7
fp8_tflops = 243.6
nvfp4_tflops = 490.8
context_tokens = 4096
"#;

const TREE: &str = r#"
[[reduction]]
id = "rows"
dim = "H"
levels = [{ level = "thread", width = "H / 32", order = "sequential" }, { level = "warp", width = "32", order = "butterfly" }]
split = "none"
round = ["f32_acc", "bf16_out"]
fma = false
mma = "none"
"#;

fn manifest(trees: &str, table: &str, param: &str, value: &str) -> String {
    format!(
        r#"{HEAD}{trees}
[[family]]
id = "norm"
description = "toy"
compute = "memory"
kernels = ["m::norm"]
rows = [1, 64]
pipeline.rms_norm = {{ in = ["bf16"], compute = "f32", out = ["bf16"] }}
{table}
op = [{{ op = "rms_norm" }}]
{param}
[[family.point]]
values = {{ {value} }}
how = "instantiation"
files = ["a.cu"]
"#
    )
}

const NUMERICS: &str = "[[family.param]]\nname = \"tree\"\nkind = \"numerics\"";

// 2026-10-05: Mutation: not reading `[[reduction]]` or the family's table leaves them empty.
#[test]
fn a_declared_tree_is_named_by_its_kernel() {
    let f = parse_families(&manifest(
        TREE,
        r#"reduction = { "m::norm" = "rows" }"#,
        "",
        "",
    ))
    .unwrap();
    let tree = &f.reductions["rows"];
    assert_eq!(tree.levels.len(), 2);
    assert_eq!(tree.split, SplitSource::None);
    let k = KernelId {
        module: "m".into(),
        func: "norm".into(),
    };
    assert_eq!(
        f.families[0].reduction.get(&k).map(String::as_str),
        Some("rows")
    );
}

// 2026-10-05: Mutation: dropping any one of these checks lets a dangling or malformed name load.
#[test]
fn dangling_and_malformed_trees_are_refused() {
    let named = r#"reduction = { "m::norm" = "rows" }"#;
    let cases = [
        (
            manifest("", named, "", ""),
            "which no [[reduction]] declares",
        ),
        (
            manifest(TREE, r#"reduction = { "m::other" = "rows" }"#, "", ""),
            "which is not its kernel",
        ),
        (
            manifest(&TREE.replace("butterfly", "zigzag"), named, "", ""),
            "order `zigzag`",
        ),
        (
            manifest(&TREE.replace("\"warp\"", "\"grid\""), named, "", ""),
            "level `grid`",
        ),
        (
            manifest(
                &TREE.replace("split = \"none\"", "split = \"sm\""),
                named,
                "",
                "",
            ),
            "split `sm`",
        ),
        (
            manifest(&format!("{TREE}{TREE}"), named, "", ""),
            "declared twice",
        ),
        (
            manifest(TREE, "", NUMERICS, "tree = \"cols\""),
            "names no [[reduction]]",
        ),
        (
            manifest(
                TREE,
                "",
                &format!("{NUMERICS}\nfrom = \"op\""),
                "tree = \"rows\"",
            ),
            "reads nothing: drop `from`",
        ),
    ];
    for (text, want) in cases {
        let e = parse_families(&text).unwrap_err().to_string();
        assert!(e.contains(want), "want `{want}` in: {e}");
    }
}

// 2026-10-05: Mutation: refusing the `numerics` kind, or refusing `undeclared` as its value.
#[test]
fn a_numerics_parameter_takes_tree_ids_or_undeclared() {
    for value in ["tree = \"rows\"", "tree = \"undeclared\""] {
        let f = parse_families(&manifest(TREE, "", NUMERICS, value)).unwrap();
        assert_eq!(
            f.families[0].param("tree").unwrap().kind,
            ParamKind::Numerics
        );
    }
    assert_eq!(ParamKind::Numerics.name(), "numerics");
}

// 2026-10-05: Mutation: reading an atomic level or a device-derived split as bytes, or an
// undeclared tree as anything but unknown.
#[test]
fn the_verdict_follows_the_tree() {
    let f = parse_families(&manifest(TREE, "", "", "")).unwrap();
    let mut t = f.reductions["rows"].clone();
    assert_eq!(verdict(Some(&t)), Verdict::Bytes);
    assert_eq!(verdict(None), Verdict::Unknown);
    t.split = SplitSource::Device;
    assert!(matches!(verdict(Some(&t)), Verdict::Tolerance(_)));
    let mut a = f.reductions["rows"].clone();
    a.levels[1].order = super::Order::Atomic;
    assert!(!a.deterministic());
    assert!(matches!(verdict(Some(&a)), Verdict::Tolerance(_)));
}

// 2026-10-05: Mutation: a gb10 RMSNorm kernel left out of the family's table, or the tree
// edited to an atomic or device-derived one, changes its verdict.
#[test]
fn gb10_rms_norm_kernels_run_one_byte_comparable_tree() {
    let text = include_str!("../../../../kernels/gb10/common/KERNEL_FAMILIES.toml");
    let f = parse_families(text).unwrap();
    let norm = f.families.iter().find(|x| x.id == "rms_norm").unwrap();
    assert_eq!(
        norm.reduction.len(),
        norm.kernels.len(),
        "every kernel declared"
    );
    for k in &norm.kernels {
        let id = &norm.reduction[k];
        assert_eq!(verdict(f.reductions.get(id)), Verdict::Bytes, "{k}");
    }
}
