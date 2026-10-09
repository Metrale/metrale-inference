// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The `--mock` store holds exactly the planned tensors the skip predicate keeps,
//! with the bytes `synthesize` produces (the bytes `mockify` writes), uploaded to the device.
//!
//! Owner: model-weights.
//! Invariants: the mock backend is the engine's test backend (no mocking of this code).

use std::collections::BTreeMap;

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_ml_utils::{MockInputs, plan_mock, synthesize, testkit};

use super::*;

#[test]
fn the_store_holds_the_synthesized_bytes_minus_what_the_loader_skips() {
    let (config, side, index) = testkit::moe_nvfp4();
    let spec = testkit::spec("1", "mode = \"uniform\"");
    let plan = plan_mock(&MockInputs {
        source_id: "toy/model",
        revision: None,
        config_json: &config,
        hf_quant_config: Some(&side),
        index: &index,
        spec: &spec,
        routing: None,
        calibration: None,
        stats: None,
    })
    .unwrap();
    let gpu = MockGpuBackend::new();
    let skip = |n: &str| n.starts_with("mtp.") || n.ends_with(".input_scale");
    let store = load_synthetic(&plan, &gpu, &skip, 3).unwrap();
    let mut want = BTreeMap::new();
    for u in 0..plan.units.len() {
        for (t, b) in synthesize(&plan, u).unwrap() {
            if !skip(&plan.tensors[t].name) {
                want.insert(plan.tensors[t].name.clone(), b);
            }
        }
    }
    assert_eq!(store.len(), want.len());
    assert!(want.keys().any(|n| n.ends_with(".weight_scale_2")));
    for (name, bytes) in &want {
        let t = store.get(name).unwrap();
        assert_eq!(gpu.read_alloc(t.ptr).unwrap(), *bytes, "{name}");
        assert_eq!(t.byte_size(), bytes.len(), "{name}");
    }
    assert!(!store.contains("mtp.fc.weight"));
}

#[test]
fn every_planned_dtype_has_a_store_dtype() {
    for d in [
        Dtype::Bf16,
        Dtype::F32,
        Dtype::F8E4m3,
        Dtype::U8,
        Dtype::I8,
        Dtype::I64,
    ] {
        assert!(store_dtype(d).is_ok(), "{d:?}");
    }
    assert!(store_dtype(Dtype::F16).is_err());
}
