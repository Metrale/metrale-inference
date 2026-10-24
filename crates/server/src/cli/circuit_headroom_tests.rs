// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `met circuit headroom` on the checked-out tree: the GLM-5.3-Flash TP=3 GB10
//! headroom reports under kernels/circuits/plans/headroom/ are current (Markdown, JSON and both
//! SVGs), the JSON validates against schemas/headroom.v1.json (and a validator that accepts a
//! broken document would fail here), and a flag that contradicts the capture is refused.
//! Regenerate the reports after an intended change with
//! `cargo test -p met circuit_headroom -- --ignored regenerate_headroom`.
//!
//! Owner: server CLI tests.
//! Invariants: the tests read the repository at the workspace root; nothing is written except by
//! the ignored `regenerate_headroom`.

use std::path::PathBuf;

use serde_json::Value;

use super::{CircuitHeadroomArgs, HeadroomFormat, command, compute, render};
use crate::cli::CircuitMode;
use metrale_circuit::Mode;
use metrale_circuit::venn::Run;

const DIR: &str = "kernels/circuits/plans/headroom";
const SCHEMA: &str = "schemas/headroom.v1.json";
/// 2026-10-22: The captures with a checked-in report: (trace, report stem). 2026-10-23: t4c is
/// the first `--forward circuit` serve, so its reports carry a non-zero plan-executed share.
const CAPTURES: [(&str, &str); 4] = [
    (
        "docs/kernel-perf/traces/glm-5.3-flash-nvfp4-tp3--gb10--f13-p1k.toml",
        "glm-5.3-flash-nvfp4-tp3--gb10--f13-p1k",
    ),
    (
        "docs/kernel-perf/traces/glm-5.3-flash-nvfp4-tp3--gb10--f13-p128k.toml",
        "glm-5.3-flash-nvfp4-tp3--gb10--f13-p128k",
    ),
    (
        "docs/kernel-perf/traces/glm-5.3-flash-nvfp4-tp3--gb10--t4c-p1k.toml",
        "glm-5.3-flash-nvfp4-tp3--gb10--t4c-p1k",
    ),
    (
        "docs/kernel-perf/traces/glm-5.3-flash-nvfp4-tp3--gb10--t4c-p128k.toml",
        "glm-5.3-flash-nvfp4-tp3--gb10--t4c-p128k",
    ),
];
const FORMATS: [(HeadroomFormat, &str); 4] = [
    (HeadroomFormat::Md, "md"),
    (HeadroomFormat::Json, "json"),
    (HeadroomFormat::SvgRoofline, "roofline.svg"),
    (HeadroomFormat::SvgWaterfall, "waterfall.svg"),
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn args(trace: Option<&str>) -> CircuitHeadroomArgs {
    CircuitHeadroomArgs {
        recipe: "glm-5.3/glm-5.3-flash-nvfp4-tp3-512k".into(),
        hardware: "gb10".into(),
        mode: CircuitMode::Verify,
        rows: Some(2),
        also: vec![Run {
            mode: Mode::Draft,
            rows: 1,
        }],
        trace: trace.map(PathBuf::from).into_iter().collect(),
        step_kernel: None,
        attribution: None,
        kv_tokens: None,
        tokens_per_step: None,
        launch_floor_us: None,
        target_pct: 80.0,
        format: HeadroomFormat::Md,
        out: None,
        check: false,
        root: None,
    }
}

fn outputs() -> Vec<(String, String)> {
    let root = root();
    let mut out = Vec::new();
    for (trace, stem) in CAPTURES {
        let a = args(Some(trace));
        let h = compute(&root, &a).unwrap_or_else(|e| panic!("{trace}: {e:#}"));
        for (f, ext) in FORMATS {
            let text = render(&h, f, &command(&a)).expect("render");
            out.push((format!("{DIR}/{stem}.{ext}"), text));
        }
    }
    out
}

#[test]
fn the_headroom_reports_are_current() {
    let root = root();
    for (rel, text) in outputs() {
        let on_disk = std::fs::read_to_string(root.join(&rel)).unwrap_or_else(|e| {
            panic!("{rel}: {e}; regenerate with --ignored regenerate_headroom")
        });
        assert!(
            on_disk == text,
            "{rel} is stale; regenerate with `cargo test -p met circuit_headroom -- --ignored regenerate_headroom`"
        );
    }
}

#[test]
#[ignore = "writes the headroom reports"]
fn regenerate_headroom() {
    let root = root();
    for (rel, text) in outputs() {
        std::fs::write(root.join(&rel), text).expect("write");
    }
}

/// 2026-10-22: The subset of JSON Schema schemas/headroom.v1.json uses.
fn validate(schema: &Value, root: &Value, v: &Value, path: &str) -> Result<(), String> {
    if let Some(r) = schema.get("$ref").and_then(Value::as_str) {
        let name = r
            .strip_prefix("#/$defs/")
            .ok_or(format!("{path}: bad $ref"))?;
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
    let ok = match schema["type"]
        .as_str()
        .ok_or(format!("{path}: schema has no type"))?
    {
        "object" => {
            let o = v.as_object().ok_or(format!("{path}: not an object"))?;
            if let Some(props) = schema.get("properties").and_then(Value::as_object) {
                for req in schema["required"].as_array().into_iter().flatten() {
                    let k = req.as_str().unwrap_or_default();
                    if !o.contains_key(k) {
                        return Err(format!("{path}: missing `{k}`"));
                    }
                }
                for (k, x) in o {
                    match props.get(k) {
                        Some(s) => validate(s, root, x, &format!("{path}.{k}"))?,
                        None => return Err(format!("{path}: unknown field `{k}`")),
                    }
                }
            } else if let Some(each) = schema.get("additionalProperties") {
                for (k, x) in o {
                    validate(each, root, x, &format!("{path}.{k}"))?;
                }
            }
            true
        }
        "array" => {
            let a = v.as_array().ok_or(format!("{path}: not an array"))?;
            for (i, x) in a.iter().enumerate() {
                validate(&schema["items"], root, x, &format!("{path}[{i}]"))?;
            }
            true
        }
        "string" => v.is_string(),
        "number" => v.is_number(),
        "integer" => v.as_u64().is_some() || v.as_i64().is_some(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        t => return Err(format!("{path}: type {t} not handled")),
    };
    ok.then_some(())
        .ok_or(format!("{path}: {v} is not {}", schema["type"]))
}

#[test]
fn the_json_report_validates_against_the_committed_schema() {
    let root = root();
    let schema: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join(SCHEMA)).expect(SCHEMA)).unwrap();
    let a = args(Some(CAPTURES[0].0));
    let h = compute(&root, &a).expect("headroom");
    let doc: Value = serde_json::to_value(&h).unwrap();
    validate(&schema, &schema, &doc, "$").expect("the report validates");
    // The plan-only report (every measured field null) validates too.
    let bare: Value = serde_json::to_value(compute(&root, &args(None)).expect("bare")).unwrap();
    validate(&schema, &schema, &bare, "$").expect("the plan-only report validates");
    // Negative controls: the validator refuses what the schema forbids.
    let mut extra = doc.clone();
    extra["sites"][0]["guess"] = Value::from(1);
    assert!(
        validate(&schema, &schema, &extra, "$")
            .unwrap_err()
            .contains("unknown field `guess`")
    );
    let mut missing = doc.clone();
    missing["totals"]
        .as_object_mut()
        .unwrap()
        .remove("residual_us");
    assert!(
        validate(&schema, &schema, &missing, "$")
            .unwrap_err()
            .contains("missing `residual_us`")
    );
    let mut bound = doc;
    bound["sites"][0]["bound"] = Value::from("guessed");
    assert!(
        validate(&schema, &schema, &bound, "$")
            .unwrap_err()
            .contains("not in enum")
    );
}

#[test]
fn the_glm_capture_closes_its_step_budget() {
    let h = compute(&root(), &args(Some(CAPTURES[0].0))).expect("headroom");
    let t = &h.totals;
    let (step, sites, out, res) = (
        t.step_us.unwrap(),
        t.sites_exclusive_us.unwrap(),
        t.outside_exclusive_us.unwrap(),
        t.residual_us.unwrap(),
    );
    assert!((sites + out + res - step).abs() < 1e-6 * step);
    assert!(res >= 0.0, "exclusive time exceeds the step: {res}");
    // Nothing captured is dropped: every kernel row is in a site or outside the plan.
    let trace = metrale_circuit::hardware::headroom_trace::parse_trace(
        &std::fs::read_to_string(root().join(CAPTURES[0].0)).unwrap(),
    )
    .unwrap();
    let captured: f64 =
        trace.kernels.iter().map(|k| k.exclusive_us).sum::<f64>() / trace.steps as f64;
    assert!((captured - sites - out).abs() < 1e-6 * step);
}

#[test]
fn a_flag_that_contradicts_the_capture_is_refused() {
    let mut a = args(Some(CAPTURES[0].0));
    a.kv_tokens = Some(999);
    let e = compute(&root(), &a).unwrap_err().to_string();
    assert!(
        e.contains("--kv-tokens 999 contradicts the capture's 1646"),
        "{e}"
    );
    // The same value agrees.
    a.kv_tokens = Some(1646);
    assert!(compute(&root(), &a).is_ok());
}

#[test]
fn also_takes_a_mode_and_a_row_count() {
    assert_eq!(
        super::parse_run("draft:1"),
        Ok(Run {
            mode: Mode::Draft,
            rows: 1
        })
    );
    for bad in ["draft", "nope:1", "verify:0", "verify:x"] {
        assert!(super::parse_run(bad).is_err(), "{bad}");
    }
}
