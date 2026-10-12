// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `RuntimeHealth` assembly: the state rule, absent-not-zero below
//! `--telemetry basic`, the device unit conversions, and that every document validates.

use super::*;

fn facts() -> HealthFacts<'static> {
    HealthFacts {
        commit: GitCommit([0xf2; 20]),
        version: "1.0.0",
        boot_id: BootId([0x31; 16]),
        fault: false,
        swapping: false,
        model: Some(("openai/gpt-oss-20b", Some(Sha256Digest([0x17; 32])))),
        now_unix_ms: 1_791_000_000_000,
        start_unix_ms: 1_790_999_940_000,
        active_requests: 3,
        telemetry_on: true,
        sched: Some(SchedShape {
            pending: 2,
            kv_blocks_free: 588,
            kv_blocks_total: 1000,
            ticks: 9,
            ..SchedShape::default()
        }),
        device: Some(DeviceFacts {
            ordinal: 0,
            reading: DeviceReading {
                energy_counter_mj: Some(912_345_678),
                power_mw: Some(61_000),
                temperature_c: Some(54),
                gpu_util_pct: Some(93),
                mem_used_bytes: Some(101),
                mem_total_bytes: Some(128),
                ..DeviceReading::default()
            },
            mono_ns: 7_200,
        }),
    }
}

#[test]
fn a_loaded_model_with_telemetry_is_ready_and_fully_measured() {
    let h = assemble(&facts());
    assert_eq!(h.validate(), Ok(()));
    let v = serde_json::to_value(&h).unwrap();
    assert_eq!(v["state"], "ready");
    assert!(v["reason"].is_null());
    assert_eq!(v["runtime"]["git_commit"], "f2".repeat(20));
    assert_eq!(v["boot_id"], "31".repeat(16));
    assert_eq!(
        v["models"],
        serde_json::json!([{"served_model": "openai/gpt-oss-20b", "recipe": "17".repeat(32)}])
    );
    assert_eq!(v["uptime_ms"], 60_000);
    assert_eq!(v["queue_depth"], serde_json::json!({"measured": 2}));
    assert_eq!(v["active_requests"], serde_json::json!({"measured": 3}));
    assert_eq!(
        v["kv_occupancy_permille"],
        serde_json::json!({"derived": 412})
    );
    let a = &v["accelerators"][0];
    assert_eq!(a["device"], "nvml:0");
    assert_eq!(a["product"], serde_json::json!({"absent": "unsupported"}));
    assert_eq!(a["temperature_mc"], serde_json::json!({"measured": 54_000}));
    assert_eq!(a["gpu_util_permille"], serde_json::json!({"measured": 930}));
    assert_eq!(
        a["energy_cumulative_mj"],
        serde_json::json!({"measured": 912_345_678})
    );
    assert_eq!(a["mono_ns"], 7_200);
    // 2026-10-11: The served document is a valid `runtime-health` v1 document.
    let back: RuntimeHealth = serde_json::from_value(v).unwrap();
    assert_eq!(back, h);
}

#[test]
fn below_telemetry_basic_the_scheduler_and_device_figures_are_absent_not_zero() {
    let mut f = facts();
    f.telemetry_on = false;
    f.sched = None;
    f.device = None;
    let v = serde_json::to_value(assemble(&f)).unwrap();
    for key in ["queue_depth", "kv_occupancy_permille"] {
        assert_eq!(
            v[key],
            serde_json::json!({"absent": "unsupported"}),
            "{key}"
        );
    }
    assert_eq!(v["accelerators"], serde_json::json!([]));
    assert_eq!(v["active_requests"], serde_json::json!({"measured": 3}));
    let mut f = facts();
    f.sched = None;
    let v = serde_json::to_value(assemble(&f)).unwrap();
    assert_eq!(v["queue_depth"], serde_json::json!({"absent": "not_yet"}));
    let mut f = facts();
    if let Some(d) = f.device.as_mut() {
        d.reading.gpu_util_pct = None;
        d.reading.temperature_c = Some(0);
    }
    let v = serde_json::to_value(assemble(&f)).unwrap();
    assert_eq!(
        v["accelerators"][0]["gpu_util_permille"],
        serde_json::json!({"absent": "unavailable"})
    );
    assert_eq!(
        v["accelerators"][0]["temperature_mc"],
        serde_json::json!({"measured": 0}),
        "a measured zero stays a measurement"
    );
}

#[test]
fn the_state_follows_fault_then_model_then_swap() {
    let state = |fault: bool, loaded: bool, swapping: bool| {
        let mut f = facts();
        f.fault = fault;
        f.swapping = swapping;
        if !loaded {
            f.model = None;
        }
        let h = assemble(&f);
        assert_eq!(h.validate(), Ok(()), "{fault} {loaded} {swapping}");
        (h.state, h.reason, h.models.len())
    };
    assert_eq!(state(false, true, false), (HealthState::Ready, None, 1));
    assert_eq!(
        state(true, true, false),
        (
            HealthState::Degraded,
            Some(DegradedReason::AcceleratorError),
            1
        )
    );
    assert_eq!(
        state(false, false, true),
        (HealthState::Degraded, Some(DegradedReason::ModelLoading), 0)
    );
    assert_eq!(state(false, false, false), (HealthState::Starting, None, 0));
}

/// 2026-10-11: Through the handler with an empty host: an unstamped build (any local test
/// build) answers 503 `unstamped_build`; a stamped one answers a valid `starting` document.
#[tokio::test]
async fn the_handler_serves_a_valid_document_or_says_the_build_is_unstamped() {
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    let _ = crate::metrics::PROCESS_START_UNIX_S.set(1_790_000_000.0);
    let app = axum::Router::new()
        .route("/health/runtime", axum::routing::get(runtime_health))
        .with_state(Arc::new(ModelHost::empty()));
    let resp = app
        .oneshot(Request::get("/health/runtime").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    if crate::build_stamp::build_commit().is_none() {
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v["error"]["code"], "unstamped_build");
    } else {
        assert_eq!(status, StatusCode::OK);
        let doc: RuntimeHealth = serde_json::from_value(v).unwrap();
        assert_eq!(doc.validate(), Ok(()));
        assert_eq!(doc.state, HealthState::Starting);
    }
}
