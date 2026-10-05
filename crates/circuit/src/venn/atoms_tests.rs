// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Atom bundles in KERNEL_FAMILIES.toml: a well-formed bundle parses and a point
//! names it through a policy parameter of `atom_bundle`; every malformed field or dangling name
//! is refused.
//!
//! Owner: metrale-circuit tests.
//! Invariants: every test fails when the behaviour it names is removed (the mutation notes say
//! which change each one catches).

use super::super::parse_families;

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

const BUNDLE: &str = r#"
[[atom_bundle]]
id = "b"
mma = { inst = "mma.sync.m16n8k16", m = 16, n = 8, k = 16, scope = "warp" }
copy = [{ operand = "a", path = "global_to_shared", inst = "cp.async", bytes = 16 }]
schedule = { kind = "multistage", stages = 3 }
swizzle = { a = "2,4,3" }
accumulator = "registers"
classes = ["toy"]
"#;

fn manifest(bundle: &str, param: &str, value: &str) -> String {
    format!(
        r#"{HEAD}{bundle}
[[family]]
id = "g"
description = "toy"
compute = "tensor_core"
mma = "mma.sync.m16n8k16.bf16"
kernels = ["m::g"]
rows = [1, 64]
pipeline.argmax = {{ in = ["bf16"], compare = "bf16", out = ["i32"] }}
op = [{{ op = "argmax" }}]
{param}
[[family.point]]
values = {{ {value} }}
how = "instantiation"
files = ["a.cu"]
"#
    )
}

const ATOMS: &str = "[[family.param]]\nname = \"atoms\"\nkind = \"policy\"\nof = \"atom_bundle\"";

// 2026-10-05: Mutation: not reading `[[atom_bundle]]`, or refusing a policy parameter without
// `from` when it has a domain.
#[test]
fn a_point_names_a_declared_bundle() {
    let f = parse_families(&manifest(BUNDLE, ATOMS, "atoms = \"b\"")).unwrap();
    let b = &f.bundles["b"];
    assert_eq!(b.shape, (16, 8, 16));
    assert!(b.single_class());
    assert_eq!(
        f.families[0].param("atoms").unwrap().of.as_deref(),
        Some("atom_bundle")
    );
}

// 2026-10-05: Mutation: dropping any one check lets a malformed bundle or a dangling name load.
#[test]
fn malformed_bundles_and_dangling_names_are_refused() {
    let ok = "atoms = \"b\"";
    let cases = [
        (
            manifest(&BUNDLE.replace("\"warp\"", "\"lane\""), ATOMS, ok),
            "scope `lane`",
        ),
        (
            manifest(&BUNDLE.replace("global_to_shared", "teleport"), ATOMS, ok),
            "copy path `teleport`",
        ),
        (
            manifest(&BUNDLE.replace("multistage", "eager"), ATOMS, ok),
            "schedule `eager`",
        ),
        (
            manifest(&BUNDLE.replace("\"registers\"", "\"l2\""), ATOMS, ok),
            "accumulator `l2`",
        ),
        (
            manifest(&BUNDLE.replace("[\"toy\"]", "[]"), ATOMS, ok),
            "no class",
        ),
        (
            manifest(&BUNDLE.replace("2,4,3", "2,4"), ATOMS, ok),
            "swizzle of `a`",
        ),
        (
            manifest(&format!("{BUNDLE}{BUNDLE}"), ATOMS, ok),
            "declared twice",
        ),
        (
            manifest(BUNDLE, ATOMS, "atoms = \"c\""),
            "names no [[atom_bundle]]",
        ),
        (
            manifest(BUNDLE, &ATOMS.replace("atom_bundle", "tile"), ok),
            "only a policy parameter of `atom_bundle` has a domain",
        ),
        (
            manifest(BUNDLE, &ATOMS.replace("\nof = \"atom_bundle\"", ""), ok),
            "needs `from`",
        ),
    ];
    for (text, want) in cases {
        let e = parse_families(&text).unwrap_err().to_string();
        assert!(e.contains(want), "want `{want}` in: {e}");
    }
}
