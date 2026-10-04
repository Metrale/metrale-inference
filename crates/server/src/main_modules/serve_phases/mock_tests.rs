// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: A mock directory is disclosed by its resolved spec and by its shard metadata, a
//! real directory is not, and a resolved spec that disagrees with the shards is refused.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use metrale_ml_utils::io::mem::MemCheckpoint;
use metrale_ml_utils::{MockInputs, plan_mock, testkit, write_skeleton};

use super::*;

fn skeleton(dir: &Path) -> String {
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
    let mut sink = FsSink::create(dir).unwrap();
    write_skeleton(&plan, &src, &mut sink).unwrap();
    sink.commit().unwrap();
    plan.digest
}

#[test]
fn a_mock_is_disclosed_by_its_spec_and_by_its_shards() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("mock");
    let digest = skeleton(&dir);
    assert_eq!(disclosure(&dir).unwrap().as_deref(), Some(digest.as_str()));
    std::fs::remove_file(dir.join(RESOLVED_FILE)).unwrap();
    assert_eq!(
        disclosure(&dir).unwrap().as_deref(),
        Some(digest.as_str()),
        "a mock without its resolved spec is still named by its shards"
    );
    std::fs::write(dir.join(RESOLVED_FILE), "edited").unwrap();
    assert!(
        disclosure(&dir).is_err(),
        "an edited resolved spec disagrees with the shards"
    );
}

#[test]
fn a_real_directory_discloses_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("config.json"), "{}").unwrap();
    assert_eq!(disclosure(tmp.path()).unwrap(), None);
}
