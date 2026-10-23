// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Precision tables and their glob.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use super::*;

const TABLE: &str = r#"
schema = 1
checkpoint = "org/model"
tier = "nvfp4"

[[linear]]
match = "*.in_proj_b"
weight = "bf16"
activation = "bf16"

[[linear]]
match = "model.layers.*.mlp.*"
weight = "nvfp4/g16"
activation = "bf16"

[[linear]]
match = "*"
weight = "fp8/block128x128"
activation = "bf16"
"#;

#[test]
fn first_match_wins_and_the_catch_all_answers_the_rest() {
    let t = PrecisionTable::parse(TABLE).unwrap();
    assert_eq!(t.checkpoint, "org/model");
    assert_eq!(t.linear("model.layers.3.in_proj_b").weight, Format::Bf16);
    assert_eq!(
        t.linear("model.layers.3.mlp.down_proj").weight,
        Format::Nvfp4 { group: 16 }
    );
    assert_eq!(
        t.linear("model.layers.3.self_attn.q_proj").weight,
        Format::parse("fp8/block128x128").unwrap()
    );
}

#[test]
fn glob_stars_cross_dots_and_anchor_both_ends() {
    assert!(glob("a.*.c", "a.b.x.c"));
    assert!(glob("*", ""));
    assert!(glob("a*b*c", "abc"));
    assert!(!glob("a*b*c", "acb"));
    assert!(!glob("*.in_proj_b", "x.in_proj_ba"));
    assert!(!glob("lm_head", "mtp.lm_head"));
    assert!(glob("*lm_head", "mtp.lm_head"));
    assert!(!glob("ab*ba", "aba"));
}

#[test]
fn a_table_without_a_trailing_catch_all_is_refused() {
    let no_star = TABLE.replace("match = \"*\"", "match = \"model.*\"");
    assert_eq!(
        PrecisionTable::parse(&no_star),
        Err(PrecisionError::NoCatchAll)
    );
    let empty = "schema = 1\ncheckpoint = \"x\"\ntier = \"t\"\nlinear = []\n";
    assert_eq!(
        PrecisionTable::parse(empty),
        Err(PrecisionError::NoCatchAll)
    );
}

#[test]
fn a_bad_format_or_schema_is_refused() {
    let bad = TABLE.replace("weight = \"bf16\"", "weight = \"bf17\"");
    assert!(matches!(
        PrecisionTable::parse(&bad),
        Err(PrecisionError::Format { pattern, .. }) if pattern == "*.in_proj_b"
    ));
    let schema = TABLE.replace("schema = 1", "schema = 2");
    assert!(matches!(
        PrecisionTable::parse(&schema),
        Err(PrecisionError::Parse(_))
    ));
    let unknown = TABLE.replace("tier = \"nvfp4\"", "tier = \"nvfp4\"\nextra = 1");
    assert!(matches!(
        PrecisionTable::parse(&unknown),
        Err(PrecisionError::Parse(_))
    ));
}

/// 2026-10-23: An `unquantized` module takes only a format an entry naming it states (a recipe's
/// load-time quantization), never the catch-all: without the named entry it stays BF16 under a
/// table whose catch-all is NVFP4; with it, it takes the entry's FP8 and its input is quantized.
#[test]
fn an_unquantized_module_takes_a_named_entry_and_never_the_catch_all() {
    use crate::format::Scale;
    use crate::test_toy::{CIRCUIT, PRECISION, shape};
    let text = CIRCUIT.replace("binding = [\"{L}.up\"]", "binding = [\"{L}.up\"]\nunquantized = true");
    assert_ne!(text, CIRCUIT);
    let named = PRECISION.replace(
        "[[linear]]\nmatch = \"*\"",
        "[[linear]]\nmatch = \"layers.*.up\"\nweight = \"fp8/channel\"\nactivation = \"fp8/token\"\n\n[[linear]]\nmatch = \"*\"",
    );
    assert_ne!(named, PRECISION);
    let up = |table: &str| {
        let t = PrecisionTable::parse(table).unwrap();
        let c = crate::instantiate(&text, &[], &shape(1), &t).unwrap();
        let n = &c.nodes[c.node("l0.ffn.up").expect("up")];
        (n.weight, c.edges[n.inputs[0]].format)
    };
    assert_eq!(up(PRECISION), (Some(Format::Bf16), Format::Bf16), "the catch-all never reaches it");
    let fp8 = |scale| Format::Fp8E4m3 { scale };
    assert_eq!(up(&named), (Some(fp8(Scale::PerChannel)), fp8(Scale::PerToken)));
    let t = PrecisionTable::parse(&named).unwrap();
    assert!(t.named("layers.0.up").is_some() && t.named("layers.0.down").is_none());
}
