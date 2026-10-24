// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Typed values are lossless, and the recipe projection refuses what it cannot
//! type. (The every-recipe argv identity runs in the server, where recipes are parsed.)

use std::collections::BTreeMap;

use metrale_blueprint::recipe_view::{RecipeParts, project, recipe_parts};
use metrale_blueprint::types::{Fixed, FlagValue, LeverValue};
use metrale_error::ErrorCode;

#[test]
fn fixed_parses_only_what_it_renders_back() {
    for s in ["0.82", "1.0", "-0.5", "12.250", "0.001"] {
        assert_eq!(Fixed::parse(s).unwrap().render(), s);
    }
    assert_eq!(
        Fixed::parse("0.82"),
        Some(Fixed {
            mantissa: 82,
            exp: -2
        })
    );
    for s in [
        "00.5", "-0.0", ".5", "5.", "1e-3", "nan", "inf", "1", "1.2.3", "+1.5",
    ] {
        assert_eq!(Fixed::parse(s), None, "{s}");
    }
}

#[test]
fn flag_values_round_trip_every_spelling() {
    for s in [
        "true",
        "false",
        "0",
        "524288",
        "-3",
        "0.82",
        "0.820",
        "fp8",
        "{\"a\":1}",
        "007",
        "",
        "1e9",
    ] {
        assert_eq!(FlagValue::parse(s).render(), s, "{s}");
    }
    assert_eq!(
        FlagValue::parse("0.82"),
        FlagValue::Fixed(Fixed {
            mantissa: 82,
            exp: -2
        })
    );
    assert_eq!(
        FlagValue::parse("007"),
        FlagValue::Text("007".into()),
        "not canonical, kept as text"
    );
}

fn parts(env: &[(&str, &str)]) -> RecipeParts {
    RecipeParts {
        id: "fam/r".into(),
        version: "2".into(),
        model: "acme/m".into(),
        runtime: Some("metrale".into()),
        container: "c".into(),
        min_nodes: 3,
        defaults: [
            ("port", "8890"),
            ("max_model_len", "4096"),
            ("gpu_memory_utilization", "0.82"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect(),
        env: env
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<BTreeMap<_, _>>(),
    }
}

#[test]
fn levers_are_typed_by_the_lever_table() {
    let p = project(&parts(&[
        ("METRALE_GLM_ROW_GROUP", "128"),
        ("NCCL_NET_GDR_LEVEL", "LOC"),
    ]))
    .unwrap();
    assert_eq!(
        p.settings.levers["METRALE_GLM_ROW_GROUP"],
        LeverValue::Int(128)
    );
    assert_eq!(p.settings.passthrough["NCCL_NET_GDR_LEVEL"], "LOC");
    assert_eq!(
        p.resources.flags["port"], "8890",
        "the port is a host resource"
    );
    assert!(!p.policy.flags.contains_key("port"));
    assert_eq!(p.world_size, 3);
}

#[test]
fn an_undeclared_lever_and_a_non_canonical_value_are_refused() {
    let undeclared = project(&parts(&[("METRALE_NO_SUCH_LEVER_EVER", "1")])).unwrap_err();
    assert_eq!(undeclared.code(), ErrorCode::BlueprintRecipe);
    let bad_int = project(&parts(&[("METRALE_GLM_ROW_GROUP", "0128")])).unwrap_err();
    assert_eq!(bad_int.code(), ErrorCode::BlueprintRecipe);
}

#[test]
fn the_projection_renders_back_to_the_same_parts() {
    let original = parts(&[
        ("METRALE_GLM_ROW_GROUP", "128"),
        ("NCCL_NET_GDR_LEVEL", "LOC"),
    ]);
    let p = project(&original).unwrap();
    let mut bp = common_blueprint();
    bp.checkpoint.repo = p.model.clone();
    bp.hardware.world_size = p.world_size;
    bp.policy = p.policy.clone();
    bp.settings = p.settings.clone();
    assert_eq!(recipe_parts(&bp, &p.resources), original);
}

#[path = "common/mod.rs"]
mod common;

fn common_blueprint() -> metrale_blueprint::Blueprint {
    common::blueprint()
}
