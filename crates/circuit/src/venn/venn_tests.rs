// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Classification rules on the toy circuit (crate::test_toy): a runtime-only
//! difference stays shared, evidence counts only at its own point and row count, a copy is not
//! sharing, the ranking is deterministic, and every failure is typed.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::classify::{classify_node, usages};
use super::families::parse_families;
use super::report::{Side, VennInputs};
use super::{Class, Compared, Finding, ParamKind, Run, Subject, VennError, build, render};
use crate::circuit_toml::CircuitError;
use crate::instances::{Instance, PrecisionSpec};
use crate::ir::{Circuit, OpParseError};
use crate::precision::PrecisionTable;
use crate::rules::Mode;
use crate::test_toy;

/// 2026-09-29: A manifest for the toy rules: `gemv` varies `hidden` at compile time, `norm` a
/// runtime `width`. `extra_gemv` is appended to the gemv family (points, evidence).
fn manifest(gemv_points: &str) -> String {
    format!(
        r#"
schema = 1
hardware = "toy"
[roofline]
dram_gbps = 249.0
bf16_tflops = 123.7
fp8_tflops = 243.6
nvfp4_tflops = 490.8
context_tokens = 4096

[[family]]
id = "embed"
description = "toy"
compute = "cuda_core"
kernels = ["m::embed"]
rows = [1, 128]
op = [{{ op = "embed" }}]
[[family.point]]
values = {{}}
how = "instantiation"
files = ["toy.cu"]

[[family]]
id = "norm"
description = "toy"
compute = "cuda_core"
kernels = ["m::norm", "m::final_norm"]
rows = [1, 128]
op = [{{ op = "rms_norm" }}, {{ op = "final_norm" }}]
[[family.param]]
name = "width"
kind = "runtime"
from = "in_dim"
[[family.point]]
values = {{}}
how = "instantiation"
files = ["toy.cu"]
[[family.evidence]]
point = {{}}
rows = [1]
microbench = "toy"

[[family]]
id = "gemv"
description = "toy"
compute = "cuda_core"
kernels = ["m::up", "m::down"]
rows = [1, 128]
op = [{{ op = "linear", weight = ["nvfp4/g16"] }}]
[[family.param]]
name = "hidden"
kind = "compile"
from = "dim:hidden"
{gemv_points}

[[family]]
id = "act"
description = "toy"
compute = "cuda_core"
kernels = ["m::act"]
rows = [1, 128]
op = [{{ op = "silu_mul" }}]
[[family.point]]
values = {{}}
how = "instantiation"
files = ["toy.cu"]

[[family]]
id = "add"
description = "toy"
compute = "cuda_core"
kernels = ["m::add"]
rows = [1, 128]
op = [{{ op = "residual_add" }}]
[[family.point]]
values = {{}}
how = "instantiation"
files = ["toy.cu"]

[[family]]
id = "head"
description = "toy"
compute = "cuda_core"
kernels = ["m::lm_head"]
rows = [1, 128]
op = [{{ op = "lm_head" }}]
[[family.point]]
values = {{}}
how = "instantiation"
files = ["toy.cu"]
"#
    )
}

const BOTH_POINTS: &str = r#"
[[family.point]]
values = { hidden = "64" }
how = "instantiation"
files = ["toy.cu"]
[[family.point]]
values = { hidden = "32" }
how = "instantiation"
files = ["toy.cu"]
[[family.evidence]]
point = { hidden = "64" }
rows = [1]
microbench = "toy"
"#;

fn toy(hidden: u64) -> Circuit {
    let mut shape = test_toy::shape(2);
    shape.dims.insert("hidden".into(), hidden);
    let table = PrecisionTable::parse(test_toy::PRECISION).unwrap();
    crate::instantiate(test_toy::CIRCUIT, &[], &shape, &table).unwrap()
}

/// 2026-09-29: The primary finding of the target node `local` in layer 0.
fn primary(
    target: &Circuit,
    against: &Circuit,
    fams: &str,
    local: &str,
    rows: u64,
) -> Result<Option<Finding>, VennError> {
    let fams = parse_families(fams).unwrap();
    let rules = test_toy::rules("");
    let plan = test_toy::plan(against, &rules, &test_toy::policy(), rows);
    let settings = BTreeMap::new();
    let a = [Subject {
        recipe: "toy/against",
        circuit: against,
        settings: &settings,
        plan: Some(&plan),
    }];
    let used = usages(&a, &fams)?;
    let t = Subject {
        recipe: "toy/target",
        circuit: target,
        settings: &settings,
        plan: None,
    };
    let idx = target.node(&format!("l0.ffn.{local}")).expect("node");
    Ok(classify_node(&t, idx, rows, &used, &fams)?.0)
}

#[test]
fn a_runtime_only_difference_is_shared_not_an_opportunity() {
    // 2026-09-29: The norm's width differs (32 vs 64) and is a runtime argument; the record at
    // one row makes it Shared, and the runtime difference is still shown.
    let f = primary(&toy(32), &toy(64), &manifest(BOTH_POINTS), "norm", 1)
        .unwrap()
        .unwrap();
    assert_eq!((f.family.as_str(), f.class), ("norm", Class::Shared));
    assert_eq!(f.diffs.len(), 1);
    assert_eq!(
        (f.diffs[0].param.as_str(), f.diffs[0].kind),
        ("width", ParamKind::Runtime)
    );
    assert!(matches!(f.compared, Compared::Model { .. }));
    // 2026-09-29: Without a record at that row count it is shared but unmeasured, never an
    // opportunity.
    let f = primary(&toy(32), &toy(64), &manifest(BOTH_POINTS), "norm", 2)
        .unwrap()
        .unwrap();
    assert_eq!(f.class, Class::SharedUnmeasured);
    // 2026-09-29: Also when the point both run exists only as a file copy: the copy is what the
    // compared model runs, so sharing it is sharing.
    let copied = manifest(BOTH_POINTS).replacen(
        "from = \"in_dim\"\n[[family.point]]\nvalues = {}\nhow = \"instantiation\"",
        "from = \"in_dim\"\n[[family.point]]\nvalues = {}\nhow = \"copy\"",
        1,
    );
    assert_ne!(copied, manifest(BOTH_POINTS), "the anchor must match");
    let f = primary(&toy(32), &toy(64), &copied, "norm", 1)
        .unwrap()
        .unwrap();
    assert_eq!(f.class, Class::Shared);
}

#[test]
fn evidence_at_one_point_does_not_count_at_another() {
    let fams = manifest(BOTH_POINTS);
    // 2026-09-29: Control: the same point as the compared model, measured at one row.
    let same = primary(&toy(64), &toy(64), &fams, "down", 1)
        .unwrap()
        .unwrap();
    assert_eq!(same.class, Class::Shared);
    assert!(same.diffs.is_empty());
    // 2026-09-29: hidden 32 is instantiated but the record is at 64: unmeasured, and the
    // compile-time difference from the compared model is listed.
    let other = primary(&toy(32), &toy(64), &fams, "down", 1)
        .unwrap()
        .unwrap();
    assert_eq!(other.class, Class::SharedUnmeasured);
    assert!(other.evidence.is_empty());
    assert_eq!(other.diffs[0].param, "hidden");
    assert_eq!(
        (
            other.diffs[0].target.as_str(),
            other.diffs[0].other.as_str()
        ),
        ("32", "64")
    );
    // 2026-09-29: A record at one row says nothing about two.
    let rows2 = primary(&toy(64), &toy(64), &fams, "down", 2)
        .unwrap()
        .unwrap();
    assert_eq!(rows2.class, Class::SharedUnmeasured);
}

#[test]
fn a_point_that_exists_only_as_a_copy_or_not_at_all_is_an_opportunity() {
    let copy = BOTH_POINTS.replacen("how = \"instantiation\"\nfiles = [\"toy.cu\"]\n[[family.point]]\nvalues = { hidden = \"32\" }\nhow = \"instantiation\"", "how = \"instantiation\"\nfiles = [\"toy.cu\"]\n[[family.point]]\nvalues = { hidden = \"32\" }\nhow = \"copy\"", 1);
    assert_ne!(copy, BOTH_POINTS, "the anchor must match");
    let f = primary(&toy(32), &toy(64), &manifest(&copy), "down", 1)
        .unwrap()
        .unwrap();
    assert_eq!(f.class, Class::ParameterizationOpportunity);
    assert_eq!(f.instantiated, Some(super::families::How::Copy));
    let only_64 = "[[family.point]]\nvalues = { hidden = \"64\" }\nhow = \"instantiation\"\nfiles = [\"toy.cu\"]\n";
    let f = primary(&toy(32), &toy(64), &manifest(only_64), "down", 1)
        .unwrap()
        .unwrap();
    assert_eq!(f.class, Class::ParameterizationOpportunity);
    assert_eq!(f.instantiated, None);
    assert_eq!(f.diffs[0].kind, ParamKind::Compile);
}

#[test]
fn a_missing_parameter_value_and_an_unmapped_kernel_are_typed_errors() {
    let needs_param = manifest(BOTH_POINTS).replace(
        "name = \"width\"\nkind = \"runtime\"\nfrom = \"in_dim\"",
        "name = \"width\"\nkind = \"runtime\"\nfrom = \"param:width\"",
    );
    let e = primary(&toy(32), &toy(64), &needs_param, "norm", 1).unwrap_err();
    assert!(
        matches!(e, VennError::MissingValue { ref family, ref param, .. } if family == "norm" && param == "width"),
        "{e}"
    );
    let no_act =
        manifest(BOTH_POINTS).replace("kernels = [\"m::act\"]", "kernels = [\"m::act_gone\"]");
    let e = primary(&toy(32), &toy(64), &no_act, "norm", 1).unwrap_err();
    assert!(
        matches!(e, VennError::UnmappedKernel { ref op, .. } if op == "silu_mul"),
        "{e}"
    );
}

#[test]
fn an_unknown_op_in_the_target_circuit_is_refused_by_name() {
    let text = test_toy::CIRCUIT.replace("op = \"silu_mul\"", "op = \"mamba3_scan\"");
    let table = PrecisionTable::parse(test_toy::PRECISION).unwrap();
    let e = crate::instantiate(&text, &[], &test_toy::shape(1), &table).unwrap_err();
    assert!(
        matches!(&e, CircuitError::Op { source: OpParseError::UnknownOp(op), .. } if op == "mamba3_scan"),
        "{e}"
    );
}

fn instance(recipe: &str, hidden: u64) -> Instance {
    let mut shape = test_toy::shape(2);
    shape.dims.insert("hidden".into(), hidden);
    Instance {
        recipe: recipe.into(),
        checkpoint: recipe.into(),
        arch: "toy".into(),
        variant: None,
        precision: PrecisionSpec::Table("toy".into()),
        target: "toy/toy/nvfp4".into(),
        golden: false,
        shape,
        policy: test_toy::policy(),
        plans: BTreeMap::new(),
        verify_batch: Vec::new(),
    }
}

#[test]
fn the_report_is_deterministic_and_ranked_by_time_then_site() {
    let fams = parse_families(&manifest(BOTH_POINTS)).unwrap();
    let rules = test_toy::rules("");
    let loaded = |h| crate::Loaded {
        circuit: toy(h),
        rules: rules.clone(),
        runtime: Vec::new(),
        rules_digest: String::new(),
    };
    let (t, a) = (instance("toy/target", 32), instance("toy/against", 64));
    let (lt, la) = (loaded(32), loaded(64));
    let meas = BTreeMap::new();
    let runs = vec![
        Run {
            mode: Mode::Decode,
            rows: 1,
        },
        Run {
            mode: Mode::MultiSeq,
            rows: 16,
        },
    ];
    let inputs = VennInputs {
        target: Side {
            instance: &t,
            loaded: &lt,
        },
        against: vec![Side {
            instance: &a,
            loaded: &la,
        }],
        families: &fams,
        measurements: &meas,
        runs,
        command: "toy".into(),
    };
    let (r1, r2) = (build(&inputs).unwrap(), build(&inputs).unwrap());
    assert_eq!(render(&r1), render(&r2));
    for t in &r1.tables {
        for w in t.rows.windows(2) {
            let ordered = w[0].time_us > w[1].time_us
                || (w[0].time_us == w[1].time_us && w[0].site < w[1].site);
            assert!(ordered, "{} then {}", w[0].site, w[1].site);
        }
        let total: f64 = t.rows.iter().map(|r| r.share).sum();
        assert!((total - 1.0).abs() < 1e-9, "shares sum to {total}");
    }
    // 2026-09-29: The two layers of one template node are one site.
    let down = r1.tables[0]
        .rows
        .iter()
        .find(|r| r.site == "ffn.down")
        .unwrap();
    assert_eq!(down.count, 2);
}
