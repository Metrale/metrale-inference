// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The directory sink and source round-trip a mock and its skeleton through the real
//! filesystem: headers read back, the skeleton is sparse and the same length, nothing is written
//! over an existing directory.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use metrale_ml_utils::io::mem::MemCheckpoint;
use metrale_ml_utils::{MockInputs, plan_mock, read_source, testkit, write_mock, write_skeleton};

use super::*;

fn plan() -> (metrale_ml_utils::MockPlan, MemCheckpoint) {
    let (config, index) = testkit::moe_fp8();
    let spec = testkit::spec("1", "mode = \"uniform\"");
    let plan = plan_mock(&MockInputs {
        source_id: "toy/model",
        revision: None,
        config_json: &config,
        hf_quant_config: None,
        index: &index,
        spec: &spec,
        routing: None,
    })
    .unwrap();
    let src = MemCheckpoint {
        files: BTreeMap::from([("config.json".into(), config.into_bytes())]),
    };
    (plan, src)
}

#[test]
fn a_mock_and_its_skeleton_round_trip_through_the_filesystem() {
    let tmp = tempfile::tempdir().unwrap();
    let (plan, src) = plan();
    let full = tmp.path().join("full");
    let mut sink = FsSink::create(&full).unwrap();
    write_mock(&plan, &src, &mut sink, 2).unwrap();
    assert!(!full.exists(), "nothing is published before commit");
    sink.commit().unwrap();
    let skel = tmp.path().join("skel");
    let mut sink = FsSink::create(&skel).unwrap();
    write_skeleton(&plan, &src, &mut sink).unwrap();
    sink.commit().unwrap();

    let a = read_source(&FsCheckpoint::new(&full)).unwrap();
    let b = read_source(&FsCheckpoint::new(&skel)).unwrap();
    assert_eq!(a.index, b.index);
    assert_eq!(a.index.len(), plan.tensors.len());
    let shards: Vec<_> = FsCheckpoint::new(&full)
        .files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".safetensors"))
        .collect();
    for s in &shards {
        let (fa, fb) = (
            std::fs::metadata(full.join(s)).unwrap(),
            std::fs::metadata(skel.join(s)).unwrap(),
        );
        assert_eq!(fa.len(), fb.len(), "{s}");
    }
    assert!(
        FsSink::create(&full).is_err(),
        "an existing directory is never written over"
    );
}
