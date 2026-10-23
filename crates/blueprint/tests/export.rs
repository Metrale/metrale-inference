// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The store exports: the committed JSON Schemas equal the ones generated from the
//! types (regenerate with `met blueprint export --schema card|graph -o schemas/...`), a card and
//! a graph of the fixture validate against them, and the graph carries the math of every family
//! and format it names (or refuses).

mod common;

use common::*;
use metrale_blueprint::export::{ExportContext, card, card_schema, graph, graph_schema};
use metrale_blueprint::frame;
use metrale_blueprint::types::Mode;
use metrale_circuit::Format;
use metrale_circuit::venn::{Families, parse_families};
use serde_json::Value;

const CARD_SCHEMA_FILE: &str = include_str!("../../../schemas/kbp-card.v1.json");
const GRAPH_SCHEMA_FILE: &str = include_str!("../../../schemas/kbp-graph.v2.json");
/// 2026-10-22: v1 as published: kept, never regenerated (a new version is a new file).
const GRAPH_V1_FILE: &str = include_str!("../../../schemas/kbp-graph.v1.json");

/// 2026-10-22: The fixture's one family, with its math.
const FAMILIES: &str = r#"
schema = 1
hardware = "gb10"
[roofline]
dram_gbps = 1.0
bf16_tflops = 1.0
fp8_tflops = 1.0
nvfp4_tflops = 1.0
context_tokens = 1
[[family]]
id = "elementwise"
description = "Residual add"
explain.about = "One thread per element."
explain.numerics = "FP32 add, RNE to BF16."
explain.math = ['y = \mathrm{rn}_{\mathrm{bf16}}(a + b)']
compute = "memory"
kernels = ["add::add_one"]
rows = [1, 8]
pipeline.residual_add = { in = ["bf16", "bf16"], compute = "f32", out = ["bf16"] }
op = [{ op = "residual_add" }]
[[family.point]]
values = {}
how = "instantiation"
files = ["a.cu"]
"#;

fn families() -> Families {
    parse_families(FAMILIES).unwrap()
}

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
        "schemas/kbp-graph.v2.json is stale"
    );
    assert!(GRAPH_V1_FILE.contains("\"kbp-graph/v1\""));
    assert!(GRAPH_V1_FILE.contains("kbp-graph.v1.json"));
    assert!(!GRAPH_V1_FILE.contains("explain"), "v1 is the published v1");
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

    let g = serde_json::to_value(
        graph::graph(&o, (Mode::Decode, 1), &ctx(f.len()), &families()).unwrap(),
    )
    .unwrap();
    let gs = graph_schema();
    validate(&gs, &gs, &g, "graph").unwrap();
    assert_eq!(g["nodes"].as_array().unwrap().len(), 2);
    assert_eq!(g["edges"][0]["from"], 0);
    assert_eq!(g["edges"][0]["to"], 1);
    assert_eq!(g["edges"][0]["fused"], true);
    assert_eq!(g["nodes"][0]["family"], "elementwise");
    assert_eq!(g["collapsed"]["layout"][0], "toy");
    assert_eq!(g["schema"], "kbp-graph/v2");
    assert_eq!(g["edges"][0]["dim_value"], 64);
    assert_eq!(g["edges"][0]["bytes_per_token"], 128, "64 BF16 values");
    assert_eq!(g["edges"][0]["requant"], Value::Null, "bf16 to bf16");
    let fam = &g["explain"]["families"]["elementwise"];
    assert_eq!(fam["math"][0], "y = \\mathrm{rn}_{\\mathrm{bf16}}(a + b)");
    assert_eq!(fam["numerics"], "FP32 add, RNE to BF16.");
    assert_eq!(g["explain"]["families"].as_object().unwrap().len(), 1);
    assert_eq!(
        g["explain"]["formats"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        ["bf16"]
    );
    assert_eq!(
        g["explain"]["formats"]["bf16"]["title"],
        Format::Bf16.explain().title
    );
    assert!(
        graph::graph(&o, (Mode::Verify, 4), &ctx(f.len()), &families()).is_err(),
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

/// 2026-10-22: A graph naming a family the manifest has no math for is not exported.
#[test]
fn a_family_without_math_refuses_the_graph() {
    let w = world();
    let f = w.signed();
    let o = frame::open(&f, &w.policy(), MEM).unwrap();
    let other = parse_families(&FAMILIES.replace("id = \"elementwise\"", "id = \"gemv\"")).unwrap();
    let e = graph::graph(&o, (Mode::Decode, 1), &ctx(f.len()), &other).unwrap_err();
    assert!(e.to_string().contains("`elementwise`"), "{e}");
}

/// 2026-10-22: An edge stored quantized after a BF16 read is a requantisation, with the
/// quantized format's bytes (64 E2M1 values = 32 B, 4 E4M3 scales, one FP32 row global).
#[test]
fn a_quantized_edge_states_its_requant_and_bytes() {
    let w = world();
    let mut bp = blueprint();
    let nv = Format::Nvfp4 { group: 16 };
    bp.plans[0].edge_formats[1] = nv;
    let file = frame::encode_unsigned(&bp, &unsigned(&bp), &[]).unwrap();
    let f = frame::sign(&file, &w.leaf, vec![w.cert()], MEM).unwrap();
    let o = frame::open(&f, &w.policy(), MEM).unwrap();
    let g = serde_json::to_value(
        graph::graph(&o, (Mode::Decode, 1), &ctx(f.len()), &families()).unwrap(),
    )
    .unwrap();
    let e = &g["edges"][0];
    assert_eq!(e["format"], "nvfp4/g16");
    assert_eq!(e["requant"]["from"], "bf16");
    assert_eq!(e["requant"]["to"], "nvfp4/g16");
    assert_eq!(e["bytes_per_token"], 40);
    assert!(g["explain"]["formats"]["nvfp4/g16"]["encode"].is_string());
    let gs = graph_schema();
    validate(&gs, &gs, &g, "graph").unwrap();
}
