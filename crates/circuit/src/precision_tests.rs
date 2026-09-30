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
