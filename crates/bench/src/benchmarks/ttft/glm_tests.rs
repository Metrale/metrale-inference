// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Tests for the GLM-5.3 TTFT gates: each is its driver under its own id, in the
//! mode its id names, and the synthetic ones report one median per prompt length. Driven
//! through the production client against the loopback endpoint of `long_prompt_tests.rs`.
//!
//! Owner: bench, ttft.
//! Invariants: the requests are real HTTP to a loopback socket; only the replies are scripted.

use super::*;
use crate::benchmarks::baseline;
use crate::benchmarks::ttft::long_prompt::{endpoint, gate_with, run};
use crate::params::ParamValue;
use crate::result::RunStatus;

const SYNTHETIC: [&BenchmarkDescriptor; 6] = [
    &GLM_COLD_64K,
    &GLM_COLD_256K,
    &GLM_COLD_512K,
    &GLM_WARM_64K,
    &GLM_WARM_256K,
    &GLM_WARM_512K,
];
const HIGH_ISL: [&BenchmarkDescriptor; 6] = [
    &GLM_HIGH_ISL_COLD_64K,
    &GLM_HIGH_ISL_COLD_256K,
    &GLM_HIGH_ISL_COLD_512K,
    &GLM_HIGH_ISL_WARM_64K,
    &GLM_HIGH_ISL_WARM_256K,
    &GLM_HIGH_ISL_WARM_512K,
];

/// 2026-10-15: Requests one sample costs: one cold request, or a prime and a re-send warm.
fn per_sample(d: &BenchmarkDescriptor) -> usize {
    if d.id.contains("warm") { 2 } else { 1 }
}

/// 2026-10-15: A synthetic GLM gate asked for 16, 32 and 16 tokens reports `median_ms_16` and
/// `median_ms_32`, sends one or two requests per sample as its id's mode requires (the
/// synthetic driver sends no warm-up), and stores its baseline under its own id. Controls: a
/// ctor pointing at the shared descriptor stores under `ttft-cold-gate`/`ttft-warm-gate`; a
/// cold/warm swap moves the count.
#[tokio::test]
async fn a_synthetic_glm_gate_runs_under_its_own_id_with_one_median_per_length() {
    for d in SYNTHETIC {
        let (target, bodies) = endpoint(Some(16)).await;
        let (mut g, store) = gate_with(d, target, d.id, |values| {
            values.set("prompt_lengths", ParamValue::IntList(vec![16, 32, 16]));
            values.set("repeats", ParamValue::Int(1));
        })
        .await;
        assert_eq!(g.descriptor().id, d.id);
        let done = run(&mut g).await.expect("the run completes");
        assert_eq!(done.status, RunStatus::Completed, "{}", d.id);
        assert_eq!(done.metrics["samples"], 3.0, "{}", d.id);
        assert!(
            done.metrics.contains_key("median_ms_16"),
            "{}: {:?}",
            d.id,
            done.metrics
        );
        assert!(
            done.metrics.contains_key("median_ms_32"),
            "{}: {:?}",
            d.id,
            done.metrics
        );
        let sent = bodies.lock().expect("bodies").len();
        assert_eq!(sent, 3 * per_sample(d), "{}", d.id);
        assert!(
            baseline::load_for(&store, d.id, Some("test-model")).is_some(),
            "{} stored no baseline under its own id",
            d.id
        );
    }
}

/// 2026-10-15: A high-ISL GLM gate sends the committed long prompt under its own id and adds no
/// per-length metric: its one length is the fixture's, already `median_ms`.
#[tokio::test]
async fn a_high_isl_glm_gate_runs_under_its_own_id_without_per_length_metrics() {
    for d in HIGH_ISL {
        let (target, bodies) = endpoint(Some(32_768)).await;
        let (mut g, store) = gate_with(d, target, d.id, |_| {}).await;
        assert_eq!(g.descriptor().id, d.id);
        let done = run(&mut g).await.expect("the run completes");
        assert_eq!(done.status, RunStatus::Completed, "{}", d.id);
        assert_eq!(done.metrics["prompt_tokens"], 32_768.0, "{}", d.id);
        assert!(
            !done.metrics.keys().any(|k| k.starts_with("median_ms_")),
            "{}: {:?}",
            d.id,
            done.metrics
        );
        let samples = done.metrics["samples"] as usize;
        let sent = bodies.lock().expect("bodies").len();
        assert_eq!(sent, 1 + samples * per_sample(d), "{}", d.id);
        assert!(
            baseline::load_for(&store, d.id, Some("test-model")).is_some(),
            "{}",
            d.id
        );
    }
}

/// 2026-10-15: Every committed GLM TTFT entry serves the context length its id names, and each
/// GLM TTFT gate has exactly one default entry: an id and a `max_model_len` pin that disagree
/// would bound one context with another's numbers.
#[test]
fn each_committed_glm_entry_serves_the_context_its_id_names() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let entries = crate::gate::bench::load_all(&root).expect("the BENCH.toml files load");
    for d in SYNTHETIC.iter().chain(HIGH_ISL.iter()) {
        let mine: Vec<_> = entries
            .iter()
            .filter(|(_, e)| e.gate == d.id && e.default)
            .collect();
        assert_eq!(mine.len(), 1, "{}: one default entry", d.id);
        let (target, entry) = mine[0];
        assert_eq!(target.model, "glm-5.3-flash", "{}", d.id);
        let context = d.id.rsplit('-').next().expect("a suffix");
        let tokens = match context {
            "64k" => "65536",
            "256k" => "262144",
            "512k" => "524288",
            other => panic!("{}: unknown context {other}", d.id),
        };
        assert_eq!(
            entry
                .serve_overrides
                .get("max_model_len")
                .map(String::as_str),
            Some(tokens),
            "{}",
            d.id
        );
    }
}
