// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: FUSIONS.toml loading: one valid rule, and every load error.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use super::*;

const OK: &str = r#"
schema = 1

[[rule]]
id = "add_norm"
pattern = [
  { op = "residual_add", local = "add", keep = true },
  { op = "rms_norm", local = "post_norm" },
]
kernels = [{ module = "norm", func = "residual_add_rms_norm" }]
repeat = "chunk64"
emitter = "residual_add_rms_norm"
rows = [1, 128]
modes = ["decode", "verify"]
requires = ["w8a8_decode"]
when = { kv_cache_dtype = "bf16" }
numerics = "reference"
priority = 100
cite = "k/rms_norm.cu:382"
"#;

fn err_of(edit: impl Fn(&str) -> String) -> RuleError {
    parse_rules(&edit(OK)).expect_err("edited rule should not load")
}

#[test]
fn a_valid_rule_loads_every_field() {
    let rules = parse_rules(OK).unwrap();
    assert_eq!(rules.len(), 1);
    let r = &rules[0];
    assert_eq!(r.id, "add_norm");
    assert_eq!(r.pattern.len(), 2);
    assert_eq!(r.pattern[0].op, OpKind::ResidualAdd);
    assert!(r.pattern[0].keep && !r.pattern[1].keep);
    assert_eq!(r.pattern[1].local.as_deref(), Some("post_norm"));
    assert_eq!(r.kernels[0].to_string(), "norm::residual_add_rms_norm");
    assert_eq!(r.repeat, Repeat::Chunk(64));
    assert_eq!(r.repeat.count(128), Some(2));
    assert_eq!(r.repeat.count(65), Some(2));
    assert_eq!(r.rows, (1, 128));
    assert_eq!(
        r.modes.iter().copied().collect::<Vec<_>>(),
        [Mode::Decode, Mode::Verify]
    );
    assert!(r.requires.contains("w8a8_decode"));
    assert_eq!(
        r.when.get("kv_cache_dtype").map(String::as_str),
        Some("bf16")
    );
    assert_eq!(r.numerics, Numerics::Reference);
}

#[test]
fn a_role_list_and_a_layer_kind_load() {
    let text = OK.replace(
        r#"{ op = "rms_norm", local = "post_norm" },"#,
        r#"{ op = "linear", roles = ["o", "down"], layer_kind = "linear_attention" },"#,
    );
    let r = &parse_rules(&text).unwrap()[0];
    assert_eq!(r.pattern[1].roles.len(), 2);
    assert_eq!(r.pattern[1].layer_kind, Some(LayerKind::LinearAttention));
}

#[test]
fn bit_identical_needs_a_microtest_and_differs_a_lever() {
    assert_eq!(
        err_of(|t| t.replace("\"reference\"", "\"bit_identical\"")),
        RuleError::MissingMicrotest("add_norm".into())
    );
    assert_eq!(
        err_of(|t| t.replace("\"reference\"", "\"bit_identical\"\nmicrotest = \" \"")),
        RuleError::MissingMicrotest("add_norm".into())
    );
    assert_eq!(
        err_of(|t| t.replace("\"reference\"", "\"differs\"")),
        RuleError::MissingLever("add_norm".into())
    );
    let ok = parse_rules(&OK.replace("\"reference\"", "\"differs\"\nlever = \"x\"")).unwrap();
    assert_eq!(ok[0].numerics, Numerics::Differs { lever: "x".into() });
}

#[test]
fn a_lever_or_microtest_on_the_wrong_class_is_refused() {
    for edit in [
        "\"reference\"\nlever = \"x\"",
        "\"reference\"\nmicrotest = \"m\"",
        "\"bit_identical\"\nmicrotest = \"m\"\nlever = \"x\"",
        "\"differs\"\nlever = \"x\"\nmicrotest = \"m\"",
        "\"approx\"",
    ] {
        assert!(
            matches!(
                err_of(|t| t.replace("\"reference\"", edit)),
                RuleError::Numerics { .. }
            ),
            "{edit}"
        );
    }
}

#[test]
fn shape_errors_are_refused() {
    let cases: [(&str, &str); 7] = [
        ("rows = [1, 128]", "rows = [0, 128]"),
        ("rows = [1, 128]", "rows = [9, 8]"),
        ("modes = [\"decode\", \"verify\"]", "modes = []"),
        ("modes = [\"decode\", \"verify\"]", "modes = [\"prefill\"]"),
        ("cite = \"k/rms_norm.cu:382\"", "cite = \"  \""),
        ("repeat = \"chunk64\"", "repeat = \"chunk0\""),
        ("keep = true },", "keep = true, sibling = true },"),
    ];
    for (from, to) in cases {
        assert!(
            matches!(err_of(|t| t.replace(from, to)), RuleError::Shape { .. }),
            "{to}"
        );
    }
    let empty = OK.replace(
        "pattern = [\n  { op = \"residual_add\", local = \"add\", keep = true },\n  { op = \"rms_norm\", local = \"post_norm\" },\n]",
        "pattern = []",
    );
    assert!(matches!(parse_rules(&empty), Err(RuleError::Shape { .. })));
}

#[test]
fn unknown_ops_roles_and_formats_are_refused() {
    for (from, to) in [
        ("op = \"rms_norm\"", "op = \"group_norm\""),
        (
            "op = \"rms_norm\", local",
            "op = \"linear\", role = \"up\", local",
        ),
        ("op = \"rms_norm\", local", "op = \"linear\", local"),
        (
            "op = \"rms_norm\", local",
            "op = \"rms_norm\", role = \"o\", local",
        ),
        (
            "op = \"rms_norm\", local",
            "op = \"rms_norm\", weight = \"fp9\", local",
        ),
        (
            "op = \"rms_norm\", local",
            "op = \"linear\", role = \"o\", roles = [\"o\"], local",
        ),
        (
            "op = \"rms_norm\", local",
            "op = \"rms_norm\", layer_kind = \"sliding\", local",
        ),
    ] {
        assert!(
            matches!(err_of(|t| t.replace(from, to)), RuleError::Op { .. }),
            "{to}"
        );
    }
}

#[test]
fn duplicate_ids_bad_toml_and_bad_schema_are_refused() {
    let twice = format!("{OK}\n{}", OK.replace("schema = 1", ""));
    assert_eq!(
        parse_rules(&twice),
        Err(RuleError::DuplicateId("add_norm".into()))
    );
    assert!(matches!(parse_rules("schema = "), Err(RuleError::Parse(_))));
    assert!(matches!(
        parse_rules(&OK.replace("schema = 1", "schema = 3")),
        Err(RuleError::Parse(_))
    ));
    assert!(matches!(
        parse_rules(&OK.replace("priority = 100", "priority = 100\nweight = 3")),
        Err(RuleError::Parse(_))
    ));
}

/// 2026-10-23: `ranks` loads on a collective element as a group-size predicate, and is refused on
/// any other op (PCND: never read as "any size") and when malformed.
#[test]
fn ranks_loads_on_a_collective_only() {
    let collective = OK.replace(
        r#"  { op = "residual_add", local = "add", keep = true },
  { op = "rms_norm", local = "post_norm" },"#,
        r#"  { op = "all_reduce", ranks = { min = 3 } },"#,
    );
    let r = &parse_rules(&collective).unwrap()[0];
    let p = r.pattern[0].ranks.expect("the predicate loads");
    assert!(p.holds(3) && p.holds(8) && !p.holds(2));
    let on_norm = OK.replace(
        r#"{ op = "rms_norm", local = "post_norm" }"#,
        r#"{ op = "rms_norm", local = "post_norm", ranks = { max = 2 } }"#,
    );
    let e = parse_rules(&on_norm).expect_err("ranks on a norm");
    assert!(e.to_string().contains("`ranks` on `rms_norm`, which is no collective"), "{e}");
    let empty = collective.replace("ranks = { min = 3 }", "ranks = {}");
    let e = parse_rules(&empty).expect_err("a predicate with no test");
    assert!(e.to_string().contains("ranks: states no test"), "{e}");
}

/// 2026-10-23: `path` loads on a one-element collective rule with the launches it issues (the
/// exchange and a summing kernel; the ring all-reduce and no kernel for `nccl`), and is a load
/// error anywhere else: on a norm, an unknown name, a `nccl` rule with a kernel, a `one_shot`
/// one without the exchange.
#[test]
fn path_loads_on_a_collective_with_its_launches_only() {
    use crate::collective::CollectivePath;
    let one_shot = OK.replace(
        r#"  { op = "residual_add", local = "add", keep = true },
  { op = "rms_norm", local = "post_norm" },"#,
        r#"  { op = "all_reduce" },"#,
    )
    .replace(
        "emitter = \"residual_add_rms_norm\"",
        "emitter = \"tp_all_reduce\"\npath = \"one_shot\"\ncomm = [{ api = \"exchange\", kernel = \"k\", at = 0 }]",
    );
    assert_eq!(parse_rules(&one_shot).unwrap()[0].path, Some(CollectivePath::OneShot));
    let nccl = one_shot
        .replace("path = \"one_shot\"", "path = \"nccl\"")
        .replace("api = \"exchange\"", "api = \"all_reduce_ring\"")
        .replace(r#"kernels = [{ module = "norm", func = "residual_add_rms_norm" }]"#, "kernels = []");
    assert_eq!(parse_rules(&nccl).unwrap()[0].path, Some(CollectivePath::Nccl));
    for (bad, want) in [
        (
            OK.replace("emitter = \"residual_add_rms_norm\"", "emitter = \"e\"\npath = \"nccl\""),
            "not headed by an all_reduce or all_gather",
        ),
        (one_shot.replace("\"one_shot\"", "\"ring\""), "unknown path `ring`"),
        (
            nccl.replace("kernels = []", r#"kernels = [{ module = "m", func = "f" }]"#),
            "path `nccl` issues one `all_reduce_ring` launch and no kernel",
        ),
        (
            one_shot.replace("api = \"exchange\"", "api = \"all_reduce_ring\""),
            "path `one_shot` issues one `exchange` launch",
        ),
    ] {
        let e = parse_rules(&bad).expect_err(want);
        assert!(e.to_string().contains(want), "{e}");
    }
}

/// 2026-10-24: `sm_budget` loads onto a rule of one kernel and no exchange; absent it is `None`;
/// zero, and a rule of two kernels, are refused.
#[test]
fn an_sm_budget_loads_on_a_one_kernel_rule_only() {
    assert_eq!(parse_rules(OK).unwrap()[0].sm_budget, None);
    let with = |t: &str, v: &str| {
        t.replace("priority = 100", &format!("priority = 100\nsm_budget = {v}"))
    };
    let r = &parse_rules(&with(OK, "32")).unwrap()[0];
    assert_eq!(r.sm_budget.map(|k| k.get()), Some(32));
    assert!(parse_rules(&with(OK, "0")).is_err());
    let two = with(OK, "32").replace(
        r#"kernels = [{ module = "norm", func = "residual_add_rms_norm" }]"#,
        r#"kernels = [{ module = "norm", func = "a" }, { module = "norm", func = "b" }]"#,
    );
    match parse_rules(&two) {
        Err(RuleError::Shape { detail, .. }) => {
            assert!(detail.contains("sm_budget = 32"), "{detail}")
        }
        other => panic!("{other:?}"),
    }
}
