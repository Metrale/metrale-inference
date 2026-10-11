// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `InferenceUsage` derivation and wire shape.

use std::time::Duration;

use super::*;

fn span(queue_ms: u64, ttft_ms: f64, decode_ms: f64) -> SchedulerSpan {
    SchedulerSpan {
        queue: Some(Duration::from_millis(queue_ms)),
        ttft_ms,
        decode_ms,
    }
}

#[test]
fn queue_is_measured_and_the_received_anchored_spans_are_derived() {
    let t = RuntimeTimings::derive(1_700_000_000_000, Some(span(250, 40.4, 1000.0)), 11);
    assert_eq!(t.received_at_unix_ms, 1_700_000_000_000);
    assert_eq!(t.queue_ms, Field::Measured(250));
    assert_eq!(t.ttft_ms, Field::Derived(290));
    assert_eq!(t.total_ms, Field::Derived(1290));
    assert_eq!(t.tpot_us, Field::Derived(100_000));
}

#[test]
fn tpot_is_not_applicable_below_two_tokens() {
    for n in [0, 1] {
        let t = RuntimeTimings::derive(0, Some(span(1, 1.0, 5.0)), n);
        assert_eq!(
            t.tpot_us,
            Field::Absent(AbsentReason::NotApplicable),
            "n={n}"
        );
    }
}

#[test]
fn no_finish_frame_leaves_every_span_absent_not_zero() {
    let t = RuntimeTimings::derive(9, None, 0);
    for f in [t.queue_ms, t.ttft_ms, t.total_ms, t.tpot_us] {
        assert_eq!(f, Field::Absent(AbsentReason::Unavailable));
    }
}

#[test]
fn disagreeing_clocks_drop_the_anchored_spans_but_keep_tpot() {
    let s = SchedulerSpan {
        queue: None,
        ttft_ms: 3.0,
        decode_ms: 9.0,
    };
    let t = RuntimeTimings::derive(0, Some(s), 4);
    assert_eq!(t.queue_ms, Field::Absent(AbsentReason::Unavailable));
    assert_eq!(t.ttft_ms, Field::Absent(AbsentReason::Unavailable));
    assert_eq!(t.total_ms, Field::Absent(AbsentReason::Unavailable));
    assert_eq!(t.tpot_us, Field::Derived(3000));
}

#[test]
fn non_finite_scheduler_times_are_unavailable() {
    let t = RuntimeTimings::derive(0, Some(span(2, f64::NAN, f64::INFINITY)), 5);
    assert_eq!(t.queue_ms, Field::Measured(2));
    assert_eq!(t.ttft_ms, Field::Absent(AbsentReason::Unavailable));
    assert_eq!(t.tpot_us, Field::Absent(AbsentReason::Unavailable));
}

#[test]
fn reasoning_without_a_channel_is_unsupported_and_unfinished_keeps_only_the_prompt() {
    let t = Tokens::finished(12, 3, 4, None);
    assert_eq!(t.prompt, Field::Measured(12));
    assert_eq!(t.completion, Field::Measured(3));
    assert_eq!(t.cached_prompt, Field::Measured(4));
    assert_eq!(t.reasoning, Field::Absent(AbsentReason::Unsupported));
    assert_eq!(
        Tokens::finished(1, 1, 0, Some(0)).reasoning,
        Field::Measured(0)
    );
    let u = Tokens::unfinished(7);
    assert_eq!(u.prompt, Field::Measured(7));
    assert_eq!(u.completion, Field::Absent(AbsentReason::Unavailable));
}

#[test]
fn error_statuses_map_to_closed_kinds() {
    assert_eq!(EngineErrorKind::of_status(429), EngineErrorKind::Overloaded);
    assert_eq!(
        EngineErrorKind::of_status(503),
        EngineErrorKind::Unavailable
    );
    assert_eq!(EngineErrorKind::of_status(400), EngineErrorKind::BadRequest);
    assert_eq!(EngineErrorKind::of_status(500), EngineErrorKind::Internal);
    assert_eq!(EngineErrorKind::of_status(502), EngineErrorKind::Internal);
}

#[test]
fn finish_reasons_map_to_outcomes() {
    assert_eq!(
        Outcome::of_finish_reason("timeout", "timeout"),
        Outcome::Timeout
    );
    assert_eq!(
        Outcome::of_finish_reason("length", "timeout"),
        Outcome::Completed
    );
    assert_eq!(
        Outcome::of_finish_reason("stop", "timeout").label(),
        "completed"
    );
}

fn record(outcome: Outcome) -> InferenceUsage {
    InferenceUsage::new(
        RuntimeIdentity {
            runtime_commit: GitCommit([0xab; 20]),
            recipe: None,
        },
        (RequestId([1; 16]), TraceId([2; 16])),
        "served/model",
        outcome,
        Tokens::finished(10, 2, 0, None),
        RuntimeTimings::derive(5, Some(span(1, 2.0, 3.0)), 2),
        6,
    )
}

#[test]
fn the_wire_record_has_every_field_and_writes_null_for_no_recipe() {
    let v = serde_json::to_value(record(Outcome::Completed)).unwrap();
    let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        [
            "version",
            "request_id",
            "trace_id",
            "served_model",
            "recipe",
            "runtime_commit",
            "outcome",
            "tokens",
            "timings",
            "energy",
            "finished_at_unix_ms"
        ]
    );
    assert_eq!(v["version"], 1);
    assert_eq!(v["request_id"], "01".repeat(16));
    assert_eq!(v["trace_id"], "02".repeat(16));
    assert!(v["recipe"].is_null());
    assert_eq!(v["runtime_commit"], "ab".repeat(20));
    assert_eq!(v["outcome"], "completed");
    assert_eq!(v["tokens"]["reasoning"]["absent"], "unsupported");
    assert_eq!(v["timings"]["queue_ms"]["measured"], 1);
    // 2026-10-11: An absent figure names no rail (mfs DECISIONS PC-S).
    assert_eq!(v["energy"], serde_json::json!({"absent": "unsupported"}));
    let back: InferenceUsage = serde_json::from_value(v).unwrap();
    assert_eq!(back, record(Outcome::Completed));
}

#[test]
fn an_engine_error_is_tagged_with_its_closed_kind() {
    let r = record(Outcome::EngineError {
        error: EngineErrorKind::Internal,
    });
    let v = serde_json::to_value(r).unwrap();
    assert_eq!(v["outcome"]["engine_error"]["error"], "internal");
}

#[test]
fn a_figure_carries_its_rail_and_window_and_a_rail_without_one_is_refused() {
    let mut r = record(Outcome::Completed);
    r.energy = Field::Derived(RequestEnergy {
        millijoules: 190,
        rail: EnergyRail::GpuRail,
        window_mono_ns: Field::Derived((100, 300)),
    });
    let v = serde_json::to_value(&r).unwrap();
    assert_eq!(
        v["energy"],
        serde_json::json!({"derived": {"millijoules": 190, "rail": "gpu_rail",
                                       "window_mono_ns": {"derived": [100, 300]}}})
    );
    assert_eq!(serde_json::from_value::<InferenceUsage>(v).unwrap(), r);
    let mut v = serde_json::to_value(record(Outcome::Completed)).unwrap();
    v["energy"]["rail"] = serde_json::json!("gpu_rail");
    assert!(serde_json::from_value::<InferenceUsage>(v).is_err());
}

#[test]
fn the_record_rules_refuse_a_zero_trace_and_an_empty_model() {
    assert_eq!(record(Outcome::Completed).validate(), Ok(()));
    let mut r = record(Outcome::Completed);
    r.trace_id = TraceId([0; 16]);
    assert_eq!(r.validate(), Err(Violation("/trace_id".into())));
    let mut r = record(Outcome::Completed);
    r.served_model.clear();
    assert_eq!(r.validate(), Err(Violation("/served_model".into())));
    // 2026-10-11: A served model name is a route token, not free text (PC-R).
    let mut r = record(Outcome::Completed);
    r.served_model = "my model".into();
    assert_eq!(r.validate(), Err(Violation("/served_model".into())));
}

#[test]
fn a_missing_recipe_or_an_engine_error_detail_is_refused() {
    let mut v = serde_json::to_value(record(Outcome::Completed)).unwrap();
    v.as_object_mut().unwrap().remove("recipe");
    assert!(serde_json::from_value::<InferenceUsage>(v).is_err());
    let mut v = serde_json::to_value(record(Outcome::EngineError {
        error: EngineErrorKind::Internal,
    }))
    .unwrap();
    v["outcome"]["engine_error"]["detail"] = serde_json::json!("engine text");
    assert!(serde_json::from_value::<InferenceUsage>(v).is_err());
}

#[test]
fn unknown_fields_are_refused() {
    let mut v = serde_json::to_value(record(Outcome::Cancelled)).unwrap();
    v["prompt_text"] = serde_json::json!("hello");
    assert!(serde_json::from_value::<InferenceUsage>(v).is_err());
}
