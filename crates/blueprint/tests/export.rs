// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The store exports: the committed JSON Schemas equal the ones generated from the
//! types (regenerate with `met blueprint export --schema card|graph -o schemas/...`), and a
//! card and a graph of the fixture validate against them.

mod common;

use common::*;
use metrale_blueprint::export::{ExportContext, card, card_schema, graph, graph_schema};
use metrale_blueprint::frame;
use metrale_blueprint::types::Mode;
use serde_json::Value;

const CARD_SCHEMA_FILE: &str = include_str!("../../../schemas/kbp-card.v1.json");
const GRAPH_SCHEMA_FILE: &str = include_str!("../../../schemas/kbp-graph.v1.json");

/// 2026-10-22: The rendering `met blueprint export --schema` writes.
fn render(v: &Value) -> String {
    let mut s = serde_json::to_string_pretty(v).unwrap();
    s.push('\n');
    s
}

#[test]
fn the_committed_schemas_are_the_generated_ones() {
    assert_eq!(
        CARD_SCHEMA_FILE,
        render(&card_schema()),
        "schemas/kbp-card.v1.json is stale"
    );
    assert_eq!(
        GRAPH_SCHEMA_FILE,
        render(&graph_schema()),
        "schemas/kbp-graph.v1.json is stale"
    );
}

/// 2026-10-22: A validator for the subset of JSON Schema the generator emits.
fn validate(schema: &Value, root: &Value, v: &Value, path: &str) -> Result<(), String> {
    if let Some(r) = schema.get("$ref").and_then(Value::as_str) {
        let name = r
            .strip_prefix("#/$defs/")
            .ok_or(format!("{path}: bad $ref {r}"))?;
        return validate(&root["$defs"][name], root, v, path);
    }
    if let Some(any) = schema.get("anyOf").and_then(Value::as_array) {
        return any
            .iter()
            .any(|s| validate(s, root, v, path).is_ok())
            .then_some(())
            .ok_or(format!("{path}: matches no anyOf branch"));
    }
    if let Some(e) = schema.get("enum").and_then(Value::as_array) {
        return e
            .contains(v)
            .then_some(())
            .ok_or(format!("{path}: {v} not in enum"));
    }
    let ty = schema["type"]
        .as_str()
        .ok_or(format!("{path}: schema without type"))?;
    let ok = match ty {
        "string" => v.is_string(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        "number" => v.is_number(),
        "integer" => v.is_i64() || v.is_u64(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        _ => false,
    };
    if !ok {
        return Err(format!("{path}: {v} is not {ty}"));
    }
    if schema.get("minimum").is_some() && v.as_u64().is_none() {
        return Err(format!("{path}: negative"));
    }
    if let Some(items) = schema.get("items") {
        for (i, x) in v.as_array().unwrap().iter().enumerate() {
            validate(items, root, x, &format!("{path}[{i}]"))?;
        }
    }
    if ty == "object" {
        let obj = v.as_object().unwrap();
        if let Some(props) = schema.get("properties").and_then(Value::as_object) {
            for req in schema["required"].as_array().unwrap() {
                let k = req.as_str().unwrap();
                obj.get(k).ok_or(format!("{path}: missing {k}"))?;
            }
            for (k, x) in obj {
                let s = props.get(k).ok_or(format!("{path}: unexpected {k}"))?;
                validate(s, root, x, &format!("{path}.{k}"))?;
            }
        } else if let Some(ap) = schema.get("additionalProperties").filter(|a| a.is_object()) {
            for (k, x) in obj {
                validate(ap, root, x, &format!("{path}.{k}"))?;
            }
        }
    }
    Ok(())
}

fn ctx(len: usize) -> ExportContext {
    ExportContext {
        exported_by: "test".into(),
        verified_at_ms: NOW,
        file_len: len as u64,
    }
}

#[test]
fn a_card_and_a_graph_validate_and_state_the_blueprint() {
    let w = world();
    let f = w.signed();
    let o = frame::open(&f, &w.policy(), MEM).unwrap();
    let c = serde_json::to_value(card::card(&o, &ctx(f.len()))).unwrap();
    let cs = card_schema();
    validate(&cs, &cs, &c, "card").unwrap();
    assert_eq!(c["blueprint_id"], format!("{}", o.id));
    assert_eq!(c["kernels"]["count"], 2);
    assert_eq!(c["kernels"]["sources"]["internal"], 2);
    assert_eq!(c["precision"]["linear_by_format"]["nvfp4/g16/bf16"], 1);
    assert_eq!(c["model"]["max_seq_len"], 4096);
    assert_eq!(c["sizes"]["lite_file"], f.len());
    assert_eq!(c["sizes"]["full_file"], Value::Null);
    assert_eq!(c["signature"]["status"], "verified");
    assert_eq!(c["execution"], "legacy:toy");
    assert_eq!(
        c["sealed"],
        Value::Null,
        "a plain file has no sealed object"
    );

    let g =
        serde_json::to_value(graph::graph(&o, (Mode::Decode, 1), &ctx(f.len())).unwrap()).unwrap();
    let gs = graph_schema();
    validate(&gs, &gs, &g, "graph").unwrap();
    assert_eq!(g["nodes"].as_array().unwrap().len(), 2);
    assert_eq!(g["edges"][0]["from"], 0);
    assert_eq!(g["edges"][0]["to"], 1);
    assert_eq!(g["edges"][0]["fused"], true);
    assert_eq!(g["nodes"][0]["family"], "elementwise");
    assert_eq!(g["collapsed"]["layout"][0], "toy");
    assert!(
        graph::graph(&o, (Mode::Verify, 4), &ctx(f.len())).is_err(),
        "an arm the blueprint lacks"
    );
}

#[test]
fn the_validator_rejects_what_the_schema_forbids() {
    let cs = card_schema();
    let w = world();
    let f = w.signed();
    let o = frame::open(&f, &w.policy(), MEM).unwrap();
    let mut c = serde_json::to_value(card::card(&o, &ctx(f.len()))).unwrap();
    c["surprise"] = Value::from(1);
    assert!(validate(&cs, &cs, &c, "card").is_err());
    c.as_object_mut().unwrap().remove("surprise");
    c["signature"]["status"] = Value::from("trusted-ish");
    assert!(validate(&cs, &cs, &c, "card").is_err());
}

#[test]
fn a_sealed_card_validates_and_states_only_public_facts() {
    let w = world();
    let f = sealed_full(&w);
    let o = frame::open(&f, &w.policy(), MEM).unwrap();
    let c = serde_json::to_value(card::card(&o, &ctx(f.len()))).unwrap();
    let cs = card_schema();
    validate(&cs, &cs, &c, "card").unwrap();
    let s = &c["sealed"];
    assert_eq!(s["suite"], "hybrid-x25519-mlkem768-hkdf-a256gcm");
    assert_eq!(s["codec"], "kzx");
    assert_eq!(
        s["recipients"][0]["fingerprint"],
        format!("{}", recipient(7).public().fingerprint())
    );
    assert_eq!(
        s["customer_tag"],
        format!("{}", frame::customer_tag("acme"))
    );
    assert_eq!(s["modules_sealed"], 2);
    let m = &o.sealed.as_ref().unwrap().manifest;
    assert_eq!(
        s["ciphertext_bytes"],
        m.artifacts.iter().map(|a| a.ct_len).sum::<u64>()
    );
    assert_eq!(s["issued_at"], NOW);
    assert_eq!(s["signature"]["status"], "verified");
    assert_eq!(s["signature"]["kid"], "test-blueprints");
    assert_eq!(
        c["blueprint_id"],
        format!("{}", o.id),
        "the same id as the plain card"
    );
    assert!(!c.to_string().contains("\"acme\""), "never the customer id");
}
