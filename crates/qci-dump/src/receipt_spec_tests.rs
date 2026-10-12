// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Run spec → receipt: pins come from the recipe, digests from the caller.

use serde_json::{Value, json};

use super::*;
use crate::contract::receipt::Verdict;

const FINGERPRINT: &str = r#"{
  "version": 1, "hardware_class": "strix-halo",
  "accelerators": [{"vendor": "amd", "product": "Radeon 8060S", "architecture": "gfx1151",
                    "memory_bytes": {"measured": 103079215104}, "count": 1}],
  "host": {"cpu_arch": "x86_64", "cpu_model": {"measured": "Ryzen AI Max+ 395"},
           "memory_bytes": {"measured": 137438953472}, "unified_memory": true,
           "os_kernel": {"measured": "6.17.0"}},
  "driver": {"measured": {"name": "amdgpu", "version": "6.17.0"}},
  "toolkit": {"measured": {"name": "rocm", "version": "7.0"}}
}"#;

fn spec() -> Value {
    json!({
        "runtime_commit": "99b2613072a69a8708048347f1b74d9cc7880ed7",
        "recipe_digest": "11".repeat(32),
        "quantization": {"weights": "int4", "activations": "bf16", "kv_cache": "bf16"},
        "topology": {"nodes": 1, "accelerators_per_node": 1, "tensor_parallel": 1,
                     "pipeline_parallel": 1, "expert_parallel": 1},
        "hardware_fingerprint": "fp.json",
        "started_at_unix_ms": 10, "finished_at_unix_ms": 20, "synthetic": false,
        "checks": [
            {"name": "coherence", "kind": "coherence", "passed": true, "failure": null,
             "parameters": {"prompts": "8"}, "artifact": "coh.log", "metrics": []},
            {"name": "decode", "kind": "benchmark", "passed": false, "failure": "below_threshold",
             "parameters": {"concurrency": "1"}, "artifact": "bench.json",
             "metrics": [{"name": "decode", "unit": "milli_tokens_per_second", "value": 9000}]}
        ]
    })
}

fn digests() -> BTreeMap<String, Sha256Digest> {
    BTreeMap::from([
        (
            "coh.log".to_string(),
            canonical::sha256(b"coherence output"),
        ),
        ("bench.json".to_string(), canonical::sha256(b"bench output")),
    ])
}

fn runner() -> SoftwareVersion {
    SoftwareVersion {
        name: "qci-dump".into(),
        version: "test".into(),
    }
}

fn run(spec: Value, fp: &str) -> Result<CompatibilityReceipt, String> {
    let parsed = parse(spec.to_string().as_bytes())?;
    build(
        "gemma-4-26b-a4b",
        parsed,
        fp.as_bytes(),
        &digests(),
        runner(),
    )
}

#[test]
fn the_model_key_comes_from_the_recipe_pins_and_digests_from_the_caller() {
    let r = run(spec(), FINGERPRINT).unwrap();
    assert_eq!(r.key.model.repository, "unsloth/gemma-4-26B-A4B-it-GGUF");
    assert_eq!(
        r.key.model.revision.to_hex(),
        "c099eb48e663fd284577b04978a94ffccb261841"
    );
    assert_eq!(
        r.checks[0].artifact.sha256,
        canonical::sha256(b"coherence output")
    );
    assert_eq!(r.checks[1].artifact.uri, "bench.json");
    assert!(matches!(r.verdict, Verdict::Fail { .. }));
}

#[test]
fn the_fingerprint_digest_is_canonical_not_byte_level() {
    let reordered: Value = serde_json::from_str(FINGERPRINT).unwrap();
    let mut m = serde_json::Map::new();
    for key in [
        "toolkit",
        "driver",
        "host",
        "accelerators",
        "hardware_class",
        "version",
    ] {
        m.insert(key.to_string(), reordered[key].clone());
    }
    let a = run(spec(), FINGERPRINT).unwrap().key.hardware_fingerprint;
    let b = run(spec(), &Value::Object(m).to_string())
        .unwrap()
        .key
        .hardware_fingerprint;
    assert_eq!(a, b);
    let other = FINGERPRINT.replace("7.0", "7.1");
    assert_ne!(a, run(spec(), &other).unwrap().key.hardware_fingerprint);
}

#[test]
fn an_unread_artifact_an_unknown_case_or_a_bad_fingerprint_is_an_error() {
    let mut s = spec();
    s["checks"][1]["artifact"] = json!("never-read.json");
    assert!(run(s, FINGERPRINT).unwrap_err().contains("never-read.json"));
    let parsed = parse(spec().to_string().as_bytes()).unwrap();
    assert!(
        build(
            "no-such-case",
            parsed,
            FINGERPRINT.as_bytes(),
            &digests(),
            runner()
        )
        .is_err()
    );
    let upper = FINGERPRINT.replace("strix-halo", "Strix-Halo");
    assert!(run(spec(), &upper).unwrap_err().contains("/hardware_class"));
}

#[test]
fn a_check_must_write_its_failure_field_even_when_null() {
    let mut s = spec();
    s["checks"][0].as_object_mut().unwrap().remove("failure");
    assert!(run(s, FINGERPRINT).unwrap_err().contains("failure"));
}

#[test]
fn a_refused_receipt_says_why() {
    let mut s = spec();
    s["checks"].as_array_mut().unwrap().remove(0);
    assert!(run(s, FINGERPRINT).unwrap_err().contains("coherence"));
}
