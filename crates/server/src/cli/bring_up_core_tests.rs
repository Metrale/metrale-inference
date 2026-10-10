// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Tests for the bring-up core: plan order, model resolution, engine identity,
//! prefix-cache counters from real exposition text, verdicts and the record schema.
//!
//! Owner: server CLI (`met ml-utils`).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use serde_json::json;

use super::*;

#[test]
fn the_plan_is_cold_then_warm_per_size_then_high_isl_cold_and_warm() {
    let p = plan(&[256, 1024], 32768);
    let got: Vec<(Bench, usize)> = p.iter().map(|s| (s.bench, s.tokens)).collect();
    assert_eq!(
        got,
        [
            (Bench::Cold, 256),
            (Bench::Cold, 1024),
            (Bench::Warm, 256),
            (Bench::Warm, 1024),
            (Bench::HighIslCold, 32768),
            (Bench::HighIslWarm, 32768),
        ]
    );
    assert_eq!(Bench::HighIslWarm.gate(), "high-isl-ttft-warm");
    assert_eq!(Bench::Cold.size_param(), "prompt_lengths");
    assert_eq!(Bench::HighIslCold.size_param(), "min_prompt_tokens");
}

fn models(ids: &[&str]) -> serde_json::Value {
    json!({"object": "list", "data": ids.iter().map(|i| json!({"id": i, "owned_by": "vllm"})).collect::<Vec<_>>()})
}

#[test]
fn the_model_is_the_named_one_if_served_else_the_only_one() {
    assert_eq!(
        resolve_model(&models(&["GLM-5.3-Flash"]), None).unwrap(),
        "GLM-5.3-Flash"
    );
    assert_eq!(resolve_model(&models(&["a", "b"]), Some("b")).unwrap(), "b");
    let e = resolve_model(&models(&["a", "b"]), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("pass --model") && e.contains("\"a\""), "{e}");
    let e = resolve_model(&models(&["a"]), Some("c"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("not c"), "{e}");
    let e = resolve_model(&models(&[]), None).unwrap_err().to_string();
    assert!(e.contains("lists no model"), "{e}");
    assert!(resolve_model(&json!({"error": "x"}), None).is_err());
}

#[test]
fn the_engine_is_named_by_its_own_reports() {
    assert_eq!(engine_kind(Some("vllm"), false), EngineKind::Vllm);
    assert_eq!(engine_kind(Some("metrale"), false), EngineKind::Metrale);
    assert_eq!(engine_kind(Some("vllm"), true), EngineKind::Metrale);
    assert_eq!(engine_kind(Some("sglang"), false), EngineKind::Unknown);
    assert_eq!(engine_kind(None, false), EngineKind::Unknown);
}

const VLLM: &str = "\
# HELP vllm:prefix_cache_hits_total Prefix cache hits, in terms of number of cached tokens.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine=\"0\",model_name=\"GLM-5.3-Flash\"} 1200.0
vllm:prefix_cache_hits_total{engine=\"1\",model_name=\"GLM-5.3-Flash\"} 34.0
vllm:prefix_cache_hits_created{engine=\"0\",model_name=\"GLM-5.3-Flash\"} 1.79e+09
vllm:cache_config_info{block_size=\"64\",cache_dtype=\"fp8\",enable_prefix_caching=\"True\",engine=\"0\"} 1.0
";

const METRALE: &str = "\
# TYPE metrale_prefix_cache_hits_total counter
metrale_prefix_cache_hits_total 3
# TYPE metrale_prefix_cache_hit_tokens_total counter
metrale_prefix_cache_hit_tokens_total 4096
";

#[test]
fn hit_tokens_are_summed_over_label_sets_and_named() {
    assert_eq!(
        hit_tokens(VLLM),
        Some(("vllm:prefix_cache_hits_total", 1234.0))
    );
    assert_eq!(
        hit_tokens(METRALE),
        Some(("metrale_prefix_cache_hit_tokens_total", 4096.0))
    );
    assert_eq!(hit_tokens("# nothing\nother_total 5\n"), None);
}

#[test]
fn the_cache_config_labels_are_read_verbatim() {
    let c = cache_config(VLLM);
    assert_eq!(c["enable_prefix_caching"], "True");
    assert_eq!(c["block_size"], "64");
    assert_eq!(c["cache_dtype"], "fp8");
    assert!(cache_config(METRALE).is_empty());
}

#[test]
fn warm_must_hit_most_of_every_sample_and_cold_almost_nothing() {
    // 2026-10-10: 3 samples of 1000 tokens: warm needs >= 2250 hit tokens.
    assert_eq!(
        cache_verdict(Bench::Warm, Some(2250.0), 3, 7, 1000),
        CacheVerdict::Verified
    );
    assert_eq!(
        cache_verdict(Bench::Warm, Some(2249.0), 3, 7, 1000),
        CacheVerdict::Violated
    );
    // 2026-10-10: 4 cold requests may hit 4 x 64 template tokens, no more.
    assert_eq!(
        cache_verdict(Bench::Cold, Some(256.0), 3, 4, 1000),
        CacheVerdict::Verified
    );
    assert_eq!(
        cache_verdict(Bench::HighIslCold, Some(257.0), 3, 4, 32768),
        CacheVerdict::Violated
    );
    assert_eq!(
        cache_verdict(Bench::Warm, None, 3, 7, 1000),
        CacheVerdict::NotExposed
    );
    assert_eq!(requests_of(Bench::Warm, 3), 7);
    assert_eq!(requests_of(Bench::HighIslCold, 1), 2);
}

fn gate_metrics() -> BTreeMap<String, f64> {
    [
        ("samples", 2.0),
        ("median_ms", 50.0),
        ("p90_ms", 60.0),
        ("min_ms", 45.0),
        ("prompt_tokens", 1030.0),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

#[test]
fn a_row_quotes_the_gates_metrics_and_judges_the_counter_delta() {
    let step = Step {
        bench: Bench::Warm,
        tokens: 1024,
    };
    let name = "vllm:prefix_cache_hits_total";
    let r = row(
        &step,
        2,
        &gate_metrics(),
        "completed".into(),
        Some((name, 100.0)),
        Some((name, 4100.0)),
        json!({}),
    );
    assert_eq!(
        (r.p50_ms, r.p90_ms, r.min_ms),
        (Some(50.0), Some(60.0), Some(45.0))
    );
    assert_eq!((r.samples, r.reps, r.prompt_tokens), (2, 2, Some(1030.0)));
    assert_eq!(r.cache.hit_tokens, Some(4000.0));
    assert_eq!(r.cache.verdict, CacheVerdict::Verified);
    let none = row(
        &step,
        2,
        &gate_metrics(),
        "completed".into(),
        None,
        Some((name, 4100.0)),
        json!({}),
    );
    assert_eq!(none.cache.verdict, CacheVerdict::NotExposed);
    let other = "metrale_prefix_cache_hit_tokens_total";
    let mixed = row(
        &step,
        2,
        &gate_metrics(),
        "completed".into(),
        Some((other, 0.0)),
        Some((name, 4100.0)),
        json!({}),
    );
    assert_eq!(
        mixed.cache.hit_tokens, None,
        "two different counters are not a delta"
    );
}

#[test]
fn a_record_of_another_schema_is_refused() {
    let rec = Record {
        schema: SCHEMA.to_string(),
        url: "http://x".into(),
        model: "m".into(),
        tokenizer: None,
        tokenizer_sha256: None,
        engine: Engine {
            kind: EngineKind::Vllm,
            label: "vllm".into(),
            owned_by: None,
            version: None,
            max_model_len: None,
            serve_identity: None,
            cache_config: BTreeMap::new(),
        },
        started_at: 1,
        metrale_version: "0".into(),
        ttft: Vec::new(),
        concurrency: None,
    };
    let text = serde_json::to_string(&rec).unwrap();
    assert_eq!(Record::parse(&text).unwrap(), rec);
    let other = text.replace(SCHEMA, "metrale-bring-up-bench/0");
    assert!(
        Record::parse(&other)
            .unwrap_err()
            .to_string()
            .contains("expected")
    );
}
