// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: KERNEL_FAMILIES.toml validation: every malformed entry is refused with the
//! typed error that names it.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use super::{FamilyError, ParamKind, parse_families};

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

fn attn(extra: &str) -> String {
    format!(
        r#"{HEAD}
[[family]]
id = "attn"
description = "toy"
compute = "cuda_core"
kernels = ["m::attn"]
rows = [1, 64]
op = [{{ op = "paged_attention" }}]
[[family.param]]
name = "head_dim"
kind = "compile"
from = "dim:head_dim"
[[family.param]]
name = "ctx"
kind = "runtime"
from = "in_dim"
[[family.point]]
values = {{ head_dim = "256" }}
how = "instantiation"
files = ["a.cu"]
{extra}"#
    )
}

#[test]
fn a_well_formed_family_parses_with_its_kinds() {
    let f = parse_families(&attn("")).unwrap();
    let a = &f.families[0];
    assert_eq!(a.param("head_dim").unwrap().kind, ParamKind::Compile);
    assert_eq!(a.param("ctx").unwrap().kind, ParamKind::Runtime);
    assert_eq!(a.rows, (1, 64));
    assert!(a.multi_row());
}

#[test]
fn an_unknown_family_parameter_is_a_typed_error_wherever_it_appears() {
    for extra in [
        "[[family.point]]\nvalues = { head_dim = \"128\", tile = \"64\" }\nhow = \"copy\"\nfiles = [\"b.cu\"]\n",
        "[[family.evidence]]\npoint = { hdim = \"256\" }\nrows = [1]\nmicrobench = \"x\"\n",
        "[[family.discover]]\nkind = \"file\"\nglob = \"*_128.cu\"\nvalues = { hdim = \"128\" }\n",
        // 2026-09-29: A runtime parameter never defines a point.
        "[[family.point]]\nvalues = { head_dim = \"128\", ctx = \"4096\" }\nhow = \"copy\"\nfiles = [\"b.cu\"]\n",
    ] {
        let e = parse_families(&attn(extra)).unwrap_err();
        assert!(
            matches!(e, FamilyError::UnknownParam { ref family, .. } if family == "attn"),
            "{extra}: {e}"
        );
    }
}

#[test]
fn an_unknown_op_is_a_typed_error() {
    for bad in [
        "op = [{ op = \"mamba3_scan\" }]",
        "op = [{ op = \"linear\", roles = [\"qq\"] }]",
        "op = [{ op = \"rms_norm\", feeds = [\"nope\"] }]",
    ] {
        let text = attn("").replace("op = [{ op = \"paged_attention\" }]", bad);
        let e = parse_families(&text).unwrap_err();
        assert!(matches!(e, FamilyError::UnknownOp { .. }), "{bad}: {e}");
    }
}

#[test]
fn malformed_entries_are_refused() {
    let cases = [
        // 2026-09-29: Evidence off every instantiated point.
        (
            "[[family.evidence]]\npoint = { head_dim = \"128\" }\nrows = [1]\nmicrobench = \"x\"\n",
            "not an instantiated point",
        ),
        (
            "[[family.evidence]]\npoint = { head_dim = \"256\" }\nrows = [1]\n",
            "exactly one",
        ),
        (
            "[[family.evidence]]\npoint = { head_dim = \"256\" }\nrows = [0]\nmicrobench = \"x\"\n",
            "at least 1",
        ),
        (
            "[[family.point]]\nvalues = {}\nhow = \"copy\"\nfiles = [\"b.cu\"]\n",
            "every compile-time",
        ),
        (
            "[[family.point]]\nvalues = { head_dim = \"256\" }\nhow = \"copy\"\nfiles = [\"b.cu\"]\n",
            "listed twice",
        ),
        (
            "[[family.point]]\nvalues = { head_dim = \"128\" }\nhow = \"clone\"\nfiles = [\"b.cu\"]\n",
            "how `clone`",
        ),
    ];
    for (extra, want) in cases {
        let e = parse_families(&attn(extra)).unwrap_err().to_string();
        assert!(e.contains(want), "{extra}: {e}");
    }
    let unknown_extractor = attn("").replace("from = \"in_dim\"", "from = \"stride\"");
    assert!(
        parse_families(&unknown_extractor)
            .unwrap_err()
            .to_string()
            .contains("unknown extractor")
    );
    let bad_kind = attn("").replace("kind = \"runtime\"", "kind = \"dynamic\"");
    assert!(
        parse_families(&bad_kind)
            .unwrap_err()
            .to_string()
            .contains("parameter kind")
    );
    let twice = format!(
        "{}\n{}",
        attn(""),
        attn("")
            .replace(HEAD, "")
            .replace("id = \"attn\"", "id = \"attn2\"")
    );
    assert!(
        parse_families(&twice)
            .unwrap_err()
            .to_string()
            .contains("m::attn")
    );
    let no_rows = attn("").replace("rows = [1, 64]", "rows = [0, 64]");
    assert!(
        parse_families(&no_rows)
            .unwrap_err()
            .to_string()
            .contains("rows")
    );
}
