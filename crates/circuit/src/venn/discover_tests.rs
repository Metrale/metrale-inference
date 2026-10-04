// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Discovery reads macro invocations and file copies, and drift is caught both
//! ways: a point in the sources the manifest lacks, and a declared point the sources lack.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use super::{KernelSources, drift, invocations};
use crate::venn::families::parse_families;

const MANIFEST: &str = r#"
schema = 1
hardware = "toy"
[roofline]
dram_gbps = 249.0
bf16_tflops = 123.7
fp8_tflops = 243.6
nvfp4_tflops = 490.8
context_tokens = 4096

[[family]]
id = "gemv"
description = "toy"
compute = "cuda_core"
kernels = ["m::gemv"]
rows = [1, 128]
pipeline.linear = { in = ["fp8/token"], act = "fp8/token", weight = "{weight}->e4m3", mma = "e4m3*e4m3", accumulate = "f32", scale = "f32", out = ["bf16"] }
op = [{ op = "linear" }]
[[family.param]]
name = "weight"
kind = "policy"
from = "weight"
[[family.discover]]
kind = "macro"
file = "k/gemv.cu"
name = "TPL"
args = { weight = 1 }
map = { weight = { "false" = "fp8/channel", "true" = "fp8/block128x128" } }
[[family.point]]
values = { weight = "fp8/channel" }
how = "instantiation"
files = ["k/gemv.cu"]
[[family.point]]
values = { weight = "fp8/block128x128" }
how = "instantiation"
files = ["k/gemv.cu"]

[[family]]
id = "attn"
description = "toy"
compute = "cuda_core"
kernels = ["m::attn"]
rows = [1, 128]
pipeline.paged_attention = { in = ["bf16", "bf16", "bf16"], cache = "bf16", scores = "f32", softmax = "f32", accumulate = "f32", out = ["bf16"] }
op = [{ op = "paged_attention" }]
[[family.param]]
name = "head_dim"
kind = "compile"
from = "dim:head_dim"
[[family.discover]]
kind = "file"
glob = "k/attn_*_128.cu"
values = { head_dim = "128" }
[[family.point]]
values = { head_dim = "256" }
how = "instantiation"
files = ["k/attn.cu"]
[[family.point]]
values = { head_dim = "128" }
how = "copy"
files = ["k/attn_fp8_128.cu"]
"#;

const GEMV: &str =
    "TPL(gemv_row_mb1, false, 1)\n  TPL(gemv_blk_mb1, true, 1)\n#define TPL(NAME, BLK, MB) x\n";

fn sources(paths: &[&str], gemv: &str) -> KernelSources {
    KernelSources {
        paths: paths.iter().map(|p| p.to_string()).collect(),
        texts: [("k/gemv.cu".to_string(), gemv.to_string())].into(),
    }
}

const PATHS: [&str; 3] = ["k/gemv.cu", "k/attn.cu", "k/attn_fp8_128.cu"];

#[test]
fn invocations_split_top_level_arguments_only() {
    let calls = invocations("  X(a, f(b, c), T<d, e>)\nY(z)\nX (no)\n", "X");
    assert_eq!(
        calls,
        vec![vec!["a".to_string(), "f(b, c)".into(), "T<d, e>".into()]]
    );
}

#[test]
fn a_manifest_that_matches_its_sources_has_no_drift() {
    let fams = parse_families(MANIFEST).unwrap();
    assert_eq!(drift(&fams, &sources(&PATHS, GEMV)), Vec::<String>::new());
}

#[test]
fn a_new_copy_in_the_sources_is_drift() {
    let fams = parse_families(MANIFEST).unwrap();
    let mut paths = PATHS.to_vec();
    paths.push("k/attn_turbo5_128.cu");
    let d = drift(&fams, &sources(&paths, GEMV));
    assert_eq!(d.len(), 1, "{d:?}");
    assert!(d[0].contains("k/attn_turbo5_128.cu"), "{d:?}");
}

#[test]
fn a_declared_point_the_sources_lack_is_drift() {
    let fams = parse_families(MANIFEST).unwrap();
    // 2026-09-29: The copy file is gone: its point names a missing file and no rule finds it.
    let d = drift(&fams, &sources(&PATHS[..2], GEMV));
    assert!(d.iter().any(|p| p.contains("does not exist")), "{d:?}");
    assert!(d.iter().any(|p| p.contains("no rule finds it")), "{d:?}");
    // 2026-09-29: The block-scaled instantiation line is gone from the macro file.
    let d = drift(&fams, &sources(&PATHS, "TPL(gemv_row_mb1, false, 1)\n"));
    assert_eq!(d.len(), 1, "{d:?}");
    assert!(d[0].contains("fp8/block128x128"), "{d:?}");
}

#[test]
fn an_unmapped_macro_argument_or_a_missing_macro_file_is_drift() {
    let fams = parse_families(MANIFEST).unwrap();
    let d = drift(
        &fams,
        &sources(&PATHS, &format!("{GEMV}TPL(gemv_x, maybe, 1)\n")),
    );
    assert!(d.iter().any(|p| p.contains("`maybe`")), "{d:?}");
    let mut src = sources(&PATHS, GEMV);
    src.texts.clear();
    let d = drift(&fams, &src);
    assert!(d.iter().any(|p| p.contains("was not supplied")), "{d:?}");
}
