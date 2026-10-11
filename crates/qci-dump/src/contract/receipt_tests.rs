// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Receipt assembly: verdict derivation and every refusal.

use super::*;

fn check(name: &str, kind: CheckKind, passed: bool, failure: Option<FailureKind>) -> CheckResult {
    CheckResult {
        check: Check {
            name: name.to_string(),
            kind,
            passed,
            parameters: BTreeMap::from([("concurrency".to_string(), "1".to_string())]),
            artifact: ArtifactRef {
                uri: format!("artifacts/{name}.json"),
                sha256: Sha256Digest([7; 32]),
            },
            metrics: vec![Metric {
                name: "decode".to_string(),
                unit: MetricUnit::MilliTokensPerSecond,
                value: 41_500,
            }],
        },
        failure,
    }
}

fn input(checks: Vec<CheckResult>) -> ReceiptInput {
    ReceiptInput {
        key: ReceiptKey {
            runtime_commit: GitCommit([0xaa; 20]),
            model: ModelRevision {
                repository: "unsloth/gemma-4-26B-A4B-it-GGUF".to_string(),
                revision: GitCommit::from_hex("c099eb48e663fd284577b04978a94ffccb261841").unwrap(),
            },
            quantization: Quantization {
                weights: Precision::Int4,
                activations: Precision::Bf16,
                kv_cache: Precision::Bf16,
            },
            hardware_fingerprint: Sha256Digest([3; 32]),
            topology: Topology {
                nodes: 1,
                accelerators_per_node: 1,
                tensor_parallel: 1,
                pipeline_parallel: 1,
                expert_parallel: 1,
            },
        },
        recipe: Sha256Digest([5; 32]),
        checks,
        started_at_unix_ms: 1_000,
        finished_at_unix_ms: 2_000,
        runner: SoftwareVersion {
            name: "qci-dump".to_string(),
            version: "1".to_string(),
        },
        synthetic: false,
    }
}

fn both_pass() -> Vec<CheckResult> {
    vec![
        check("coherence", CheckKind::Coherence, true, None),
        check("decode", CheckKind::Benchmark, true, None),
    ]
}

#[test]
fn every_check_passing_is_a_pass_and_round_trips() {
    let r = assemble(input(both_pass())).unwrap();
    assert_eq!(r.verdict, Verdict::Pass);
    assert_eq!(r.version, 1);
    let v = serde_json::to_value(&r).unwrap();
    assert_eq!(v["verdict"], "pass");
    assert_eq!(
        v["key"]["model"]["revision"],
        "c099eb48e663fd284577b04978a94ffccb261841"
    );
    assert_eq!(v["key"]["quantization"]["weights"], "int4");
    assert_eq!(
        v["checks"][1]["metrics"][0]["unit"],
        "milli_tokens_per_second"
    );
    assert_eq!(
        serde_json::from_value::<CompatibilityReceipt>(v).unwrap(),
        r
    );
}

#[test]
fn any_failed_check_fails_the_receipt_and_is_named_in_order() {
    let mut checks = both_pass();
    checks[0] = check(
        "coherence",
        CheckKind::Coherence,
        false,
        Some(FailureKind::IncoherentOutput),
    );
    checks.push(check(
        "tools",
        CheckKind::ToolCall,
        false,
        Some(FailureKind::InvalidToolCall),
    ));
    let r = assemble(input(checks)).unwrap();
    let Verdict::Fail { failures } = &r.verdict else {
        panic!("expected fail, got {:?}", r.verdict);
    };
    let named: Vec<(&str, FailureKind)> = failures
        .iter()
        .map(|f| (f.check.as_str(), f.kind))
        .collect();
    assert_eq!(
        named,
        [
            ("coherence", FailureKind::IncoherentOutput),
            ("tools", FailureKind::InvalidToolCall)
        ]
    );
    let v = serde_json::to_value(&r).unwrap();
    assert_eq!(
        v["verdict"]["fail"]["failures"][1]["kind"],
        "invalid_tool_call"
    );
}

#[test]
fn coherence_and_benchmark_evidence_are_both_required() {
    let only_bench = vec![check("decode", CheckKind::Benchmark, true, None)];
    assert_eq!(
        assemble(input(only_bench)),
        Err(Refusal::NoCheckOfKind(CheckKind::Coherence))
    );
    let only_coherence = vec![check("coherence", CheckKind::Coherence, true, None)];
    assert_eq!(
        assemble(input(only_coherence)),
        Err(Refusal::NoCheckOfKind(CheckKind::Benchmark))
    );
    assert_eq!(
        assemble(input(vec![])),
        Err(Refusal::NoCheckOfKind(CheckKind::Coherence))
    );
}

#[test]
fn contradictory_checks_are_refused() {
    let mut c = both_pass();
    c[1] = check("decode", CheckKind::Benchmark, false, None);
    assert_eq!(
        assemble(input(c)),
        Err(Refusal::FailedWithoutKind("decode".into()))
    );
    let mut c = both_pass();
    c[1] = check(
        "decode",
        CheckKind::Benchmark,
        true,
        Some(FailureKind::BelowThreshold),
    );
    assert_eq!(
        assemble(input(c)),
        Err(Refusal::PassedWithFailureKind("decode".into()))
    );
    let mut c = both_pass();
    c.push(check("decode", CheckKind::Accuracy, true, None));
    assert_eq!(
        assemble(input(c)),
        Err(Refusal::DuplicateCheck("decode".into()))
    );
}

#[test]
fn keys_that_name_no_real_combination_are_refused() {
    let mut i = input(both_pass());
    i.key.quantization.weights = Precision::Adaptive;
    assert_eq!(
        assemble(i),
        Err(Refusal::InvalidField("/key/quantization/weights".into()))
    );
    let mut i = input(both_pass());
    i.key.topology.tensor_parallel = 0;
    assert_eq!(
        assemble(i),
        Err(Refusal::InvalidField("/key/topology".into()))
    );
    let mut i = input(both_pass());
    i.key.model.repository = "gemma".into();
    assert_eq!(
        assemble(i),
        Err(Refusal::InvalidField("/key/model/repository".into()))
    );
    let mut i = input(both_pass());
    i.checks[0]
        .check
        .parameters
        .insert("max-tokens".into(), "1".into());
    assert_eq!(
        assemble(i),
        Err(Refusal::InvalidField("/checks/0/parameters".into()))
    );
    let mut i = input(both_pass());
    i.finished_at_unix_ms = 999;
    assert_eq!(assemble(i), Err(Refusal::TimeOrder));
}

#[test]
fn a_read_receipt_whose_verdict_contradicts_its_checks_is_invalid() {
    let mut r = assemble(input(both_pass())).unwrap();
    assert_eq!(r.validate(), Ok(()));
    r.checks[1].passed = false;
    assert_eq!(r.validate(), Err(Violation("/verdict".into())));
}

#[test]
fn unknown_fields_are_refused_on_read() {
    let mut v = serde_json::to_value(assemble(input(both_pass())).unwrap()).unwrap();
    v["support_label"] = serde_json::json!("supported");
    assert!(serde_json::from_value::<CompatibilityReceipt>(v).is_err());
}
