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
        ("modes = [\"decode\", \"verify\"]", "modes = [\"warmup\"]"),
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
        ("op = \"rms_norm\"", "op = \"layer_norm\""),
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

// 2026-10-03: `departs` is a reference rule's statement that its `act` step departs from the
// policy's activation format: refused on a rule of another class, and without a stated `act`.
#[test]
fn a_departure_needs_a_reference_rule_and_its_stated_act() {
    let rule = |numerics: &str, steps: &str| {
        format!(
            "schema = 1\n[[rule]]\nid = \"head\"\npattern = [{{ op = \"lm_head\", weight = \"nvfp4/g16\", steps = {{ {steps} }}, departs = true }}]\nkernels = [{{ module = \"w4a16\", func = \"w4a16_gemm_t\" }}]\nrepeat = \"once\"\nemitter = \"lm_head_nvfp4_tile\"\nrows = [1, 8]\nmodes = [\"decode\"]\nnumerics = \"{numerics}\"\n{}priority = 20\ncite = \"x\"\n",
            if numerics == "differs" {
                "lever = \"l\"\n"
            } else {
                ""
            }
        )
    };
    let act = "act = \"fp8/tensor\", mma = \"e4m3*e4m3\"";
    let ok = crate::parse_rules(&rule("reference", act)).unwrap();
    assert!(ok[0].pattern[0].departs);
    assert!(crate::parse_rules(&rule("differs", act)).is_err());
    assert!(crate::parse_rules(&rule("reference", "mma = \"e4m3*e4m3\"")).is_err());
}
