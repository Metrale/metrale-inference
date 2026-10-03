// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The tensor-core policy on the toy circuit (`test_toy`): enforced (a covered op on
//! a CUDA-core kernel is a violation, one on a tensor-core kernel is not), an exemption allows
//! exactly what it lists (op, rows, every kernel), a violation is refused, and the policy file's
//! malformed entries are refused. Mutation: moving the toy `up` kernel's family from tensor
//! cores to CUDA cores turns a compliant plan into a violation.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use super::*;
use crate::test_toy;
use crate::venn::parse_families;

/// 2026-10-02: Families for the toy rule kernels `m::<id>`, `up` on `up_unit`.
fn families(up_unit: &str) -> Families {
    let mut s = String::from(
        "schema = 1\nhardware = \"toy\"\n[roofline]\ndram_gbps = 1.0\nbf16_tflops = 1.0\nfp8_tflops = 1.0\nnvfp4_tflops = 1.0\ncontext_tokens = 1\n",
    );
    for (id, op, unit, ops) in [
        (
            "embed",
            r#"{ op = "embed" }"#,
            "compute = \"memory\"",
            &["embed"][..],
        ),
        (
            "norm",
            r#"{ op = "rms_norm" }, { op = "final_norm" }"#,
            "compute = \"memory\"",
            &["rms_norm", "final_norm"][..],
        ),
        (
            "up",
            r#"{ op = "linear", roles = ["gate_up"] }"#,
            up_unit,
            &["linear"][..],
        ),
        (
            "act",
            r#"{ op = "silu_mul" }"#,
            "compute = \"memory\"",
            &["silu_mul"][..],
        ),
        (
            "down",
            r#"{ op = "linear", roles = ["down"] }"#,
            "compute = \"cuda_core\"",
            &["linear"][..],
        ),
        (
            "add",
            r#"{ op = "residual_add" }"#,
            "compute = \"memory\"",
            &["residual_add"][..],
        ),
        (
            "lm_head",
            r#"{ op = "lm_head" }"#,
            "compute = \"cuda_core\"",
            &["lm_head"][..],
        ),
    ] {
        let extra = if id == "norm" {
            r#", "m::final_norm""#
        } else {
            ""
        };
        let pipes = test_toy::pipelines(ops);
        s.push_str(&format!(
            "[[family]]\nid = \"{id}\"\ndescription = \"toy\"\n{unit}\nkernels = [\"m::{id}\"{extra}]\nrows = [1, 128]\n{pipes}\nop = [{op}]\n[[family.point]]\nvalues = {{}}\nhow = \"instantiation\"\nfiles = [\"k.cu\"]\n"
        ));
    }
    parse_families(&s).unwrap_or_else(|e| panic!("toy families: {e}"))
}

const TC: &str = "compute = \"tensor_core\"\nmma = \"mma.sync.m16n8k16.bf16\"";

fn policy(extra: &str) -> TcPolicy {
    let text = format!(
        "[hardware]\narch = \"sm\"\n[tensor_core_policy]\n[[tensor_core_policy.require]]\nops = [\"linear\", \"lm_head\"]\nmodes = [\"decode\"]\nmin_rows = 1\nweights = [\"any\"]\n{extra}"
    );
    let t: toml::Table = toml::from_str(&text).unwrap();
    parse_policy(&t).unwrap().unwrap()
}

fn run(families: &Families, policy: &TcPolicy, rows: u64) -> TcAudit {
    let c = test_toy::circuit(2);
    let rules = test_toy::rules("");
    let plan = test_toy::plan(&c, &rules, &test_toy::policy(), rows);
    audit(&c, &plan, families, policy, "novel").unwrap()
}

fn sites(v: &[Finding]) -> Vec<&str> {
    v.iter().map(|f| f.site.as_str()).collect()
}

#[test]
fn covered_ops_off_tensor_cores_are_violations_and_on_them_are_not() {
    let a = run(&families(TC), &policy(""), 4);
    // 2026-10-02: Two layers' up and down, and the head: five covered nodes, two on TC.
    assert_eq!((a.covered, a.on_tensor_cores), (5, 2));
    assert_eq!(sites(&a.violations), ["ffn.down", "head.lm_head"]);
    assert!(a.exempted.is_empty() && a.gaps.is_empty());
    assert!(a.violations.iter().all(|f| f.unit == "cuda_core"));
}

#[test]
fn below_the_minimum_rows_or_outside_the_formats_nothing_is_covered() {
    let p = policy("").require[0].clone();
    let high = TcPolicy {
        require: vec![Require {
            min_rows: 8,
            ..p.clone()
        }],
        exempt: vec![],
    };
    assert_eq!(run(&families(TC), &high, 4).covered, 0);
    assert_eq!(run(&families(TC), &high, 8).covered, 5);
    // 2026-10-02: The toy's linears are NVFP4 and its head BF16: a BF16-only rule covers the head.
    let bf16 = TcPolicy {
        require: vec![Require {
            weights: Weights::Only([Format::Bf16].into()),
            ..p
        }],
        exempt: vec![],
    };
    assert_eq!(
        sites(&run(&families(TC), &bf16, 4).violations),
        ["head.lm_head"]
    );
}

#[test]
fn an_exemption_allows_exactly_its_op_rows_and_kernels() {
    let ex = "[[tensor_core_policy.exempt]]\nops = [\"linear:down\"]\nmodes = [\"decode\"]\nrows = [1, 8]\nkernels = [\"m::down\"]\nkind = \"measured\"\nreason = \"toy GEMV beats the tile\"\nevidence = \"toy @ decode\"\n";
    let p = policy(ex);
    let a = run(&families(TC), &p, 4);
    assert_eq!(sites(&a.violations), ["head.lm_head"]);
    assert_eq!(a.exempted.len(), 1);
    assert_eq!(
        (a.exempted[0].0.site.as_str(), a.exempted[0].1),
        ("ffn.down", 0)
    );
    // 2026-10-02: Past its rows the exemption no longer applies.
    assert_eq!(
        sites(&run(&families(TC), &p, 16).violations),
        ["ffn.down", "head.lm_head"]
    );
    // 2026-10-02: An exemption that does not list the group's kernel allows nothing.
    let other = policy(&ex.replace("m::down", "m::other"));
    assert_eq!(
        sites(&run(&families(TC), &other, 4).violations),
        ["ffn.down", "head.lm_head"]
    );
    // 2026-10-02: An exemption for another op allows nothing either.
    let wrong_op = policy(&ex.replace("linear:down", "linear:gate_up"));
    assert_eq!(
        sites(&run(&families(TC), &wrong_op, 4).violations),
        ["ffn.down", "head.lm_head"]
    );
}

#[test]
fn a_kernel_moved_off_tensor_cores_becomes_a_violation() {
    let compliant = run(&families(TC), &policy(""), 4);
    assert!(!sites(&compliant.violations).contains(&"ffn.up"));
    let mutated = run(&families("compute = \"cuda_core\""), &policy(""), 4);
    assert_eq!(
        sites(&mutated.violations),
        ["ffn.up", "ffn.down", "head.lm_head"]
    );
}

#[test]
fn a_kernel_in_no_family_is_refused_not_guessed() {
    let mut f = families(TC);
    f.families.retain(|x| x.id != "down");
    let c = test_toy::circuit(1);
    let rules = test_toy::rules("");
    let plan = test_toy::plan(&c, &rules, &test_toy::policy(), 4);
    let e = audit(&c, &plan, &f, &policy(""), "novel").unwrap_err();
    assert!(e.contains("m::down"), "{e}");
}

#[test]
fn malformed_policies_are_refused() {
    let head = "[hardware]\narch = \"sm\"\n[tensor_core_policy]\n";
    let req = |ops: &str, modes: &str, min: u64, w: &str| {
        format!(
            "[[tensor_core_policy.require]]\nops = {ops}\nmodes = {modes}\nmin_rows = {min}\nweights = {w}\n"
        )
    };
    let good = req("[\"linear\"]", "[\"decode\"]", 1, "[\"any\"]");
    let exempt = |kind: &str, evidence: &str| {
        format!(
            "[[tensor_core_policy.exempt]]\nops = [\"linear\"]\nmodes = [\"decode\"]\nrows = [1, 4]\nkernels = [\"m::x\"]\nkind = \"{kind}\"\nreason = \"r\"\n{evidence}"
        )
    };
    for (what, body) in [
        (
            "unknown op",
            req("[\"matmul\"]", "[\"decode\"]", 1, "[\"any\"]"),
        ),
        (
            "unknown role",
            req("[\"linear:qq\"]", "[\"decode\"]", 1, "[\"any\"]"),
        ),
        ("no modes", req("[\"linear\"]", "[]", 1, "[\"any\"]")),
        (
            "unknown mode",
            req("[\"linear\"]", "[\"decoding\"]", 1, "[\"any\"]"),
        ),
        (
            "zero rows",
            req("[\"linear\"]", "[\"decode\"]", 0, "[\"any\"]"),
        ),
        ("no weights", req("[\"linear\"]", "[\"decode\"]", 1, "[]")),
        (
            "bad weight",
            req("[\"linear\"]", "[\"decode\"]", 1, "[\"fp7\"]"),
        ),
        (
            "measured without evidence",
            format!("{good}{}", exempt("measured", "")),
        ),
        ("unknown kind", format!("{good}{}", exempt("someday", ""))),
        ("unknown key", format!("{good}speed = 1\n")),
        ("no requirement", String::new()),
    ] {
        let t: toml::Table = toml::from_str(&format!("{head}{body}")).unwrap();
        assert!(parse_policy(&t).is_err(), "{what}: accepted");
    }
    let t: toml::Table = toml::from_str("[hardware]\narch = \"sm\"\n").unwrap();
    assert_eq!(parse_policy(&t), Ok(None));
    let t: toml::Table = toml::from_str(&format!(
        "{head}{good}{}",
        exempt("measured", "evidence = \"m @ r\"\n")
    ))
    .unwrap();
    assert!(parse_policy(&t).unwrap().is_some());
}

// 2026-10-02: The rule-set lint: the toy's CUDA-core `down` and `lm_head` rules (rows 1-128,
// decode/multi_seq/verify) against a decode-only policy. Compliant only when an exemption covers
// each rule's op, modes, the whole row range and every kernel; the tensor-core `up` rule never
// needs one.
#[test]
fn every_rule_off_tensor_cores_needs_an_exemption_for_its_whole_range() {
    let rules = test_toy::rules("");
    let fams = families(TC);
    let bare = lint_rules(&rules, &fams, &policy(""));
    assert_eq!(bare.len(), 2, "{bare:?}");
    assert!(
        bare.iter().any(|m| m.contains("`down`")) && bare.iter().any(|m| m.contains("`lm_head`"))
    );
    let ex = |ops: &str, rows: &str| {
        format!(
            "[[tensor_core_policy.exempt]]\nops = [{ops}]\nmodes = [\"decode\"]\nrows = [{rows}]\nkernels = [\"m::down\", \"m::lm_head\"]\nkind = \"backlog\"\nreason = \"toy\"\n"
        )
    };
    assert!(
        lint_rules(
            &rules,
            &fams,
            &policy(&ex("\"linear:down\", \"lm_head\"", "1, 128"))
        )
        .is_empty()
    );
    // 2026-10-02: An exemption short of the rule's rows leaves it a violation.
    assert_eq!(
        lint_rules(
            &rules,
            &fams,
            &policy(&ex("\"linear:down\", \"lm_head\"", "1, 64"))
        )
        .len(),
        2
    );
    // 2026-10-02: One for another role leaves `down` a violation.
    assert_eq!(
        lint_rules(
            &rules,
            &fams,
            &policy(&ex("\"linear:gate_up\", \"lm_head\"", "1, 128"))
        )
        .len(),
        1
    );
    // 2026-10-02: Mutation: the `up` family moved to CUDA cores adds its rule.
    assert_eq!(
        lint_rules(&rules, &families("compute = \"cuda_core\""), &policy("")).len(),
        3
    );
}
