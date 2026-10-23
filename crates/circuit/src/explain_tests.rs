// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The kernel-math table: entries come from the manifest and the format semantics,
//! once each, and a family the manifest lacks refuses.

use super::*;
use crate::venn::parse_families;

const MANIFEST: &str = r#"
schema = 1
hardware = "toy"
[roofline]
dram_gbps = 1.0
bf16_tflops = 1.0
fp8_tflops = 1.0
nvfp4_tflops = 1.0
context_tokens = 1
[[family]]
id = "norm"
description = "RMSNorm"
explain.about = "One block per row."
explain.numerics = "FP32 sum of squares."
explain.math = ['y = x / \mathrm{rms}(x)']
compute = "memory"
kernels = ["m::norm"]
rows = [1, 8]
pipeline.rms_norm = { in = ["bf16"], compute = "f32", out = ["bf16"] }
op = [{ op = "rms_norm" }]
[[family.point]]
values = {}
how = "instantiation"
files = ["n.cu"]
"#;

#[test]
fn the_table_holds_each_named_family_and_format_once() {
    let m = parse_families(MANIFEST).unwrap();
    let nv = Format::parse("nvfp4/g16").unwrap();
    let t = kernel_math(&m, ["norm", "norm"], [Format::Bf16, nv, Format::Bf16]).unwrap();
    assert_eq!(t.families.len(), 1);
    let n = &t.families["norm"];
    assert_eq!(n.description, "RMSNorm");
    assert_eq!(n.math, ["y = x / \\mathrm{rms}(x)"]);
    assert_eq!(n.numerics, "FP32 sum of squares.");
    assert_eq!(t.formats.keys().collect::<Vec<_>>(), ["bf16", "nvfp4/g16"]);
    assert_eq!(t.formats["nvfp4/g16"], nv.explain());
}

#[test]
fn a_family_without_an_entry_refuses() {
    let m = parse_families(MANIFEST).unwrap();
    let e = kernel_math(&m, ["norm", "gemv"], []).unwrap_err();
    assert!(e.contains("`gemv`") && e.contains("no math"), "{e}");
}

#[test]
fn a_requant_is_a_change_into_a_quantized_format() {
    let nv = Format::parse("nvfp4/g16").unwrap();
    let tok = Format::parse("fp8/token").unwrap();
    assert_eq!(requant(Some(Format::Bf16), nv), Some((Format::Bf16, nv)));
    assert_eq!(requant(Some(nv), tok), Some((nv, tok)));
    assert_eq!(requant(Some(nv), nv), None, "same format: no requant");
    assert_eq!(
        requant(Some(nv), Format::Bf16),
        None,
        "a dequant is not a requant"
    );
    assert_eq!(requant(Some(Format::F32), Format::Bf16), None);
    assert_eq!(
        requant(None, nv),
        None,
        "no producer input: nothing to state"
    );
}
