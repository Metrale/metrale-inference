// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: A written mock is a checkpoint this engine reads back: its headers give exactly
//! the planned tensors, every shard names the digest, the index maps every tensor, the bytes are
//! the synthesized ones whatever the thread count, and its config resolves to a circuit at the
//! declared precision (Path A, through the in-memory sink and source).
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::*;
use crate::io::mem::{MemCheckpoint, MemSink};
use crate::plan::{MockInputs, plan_mock};
use crate::testkit;

fn source(config: &str, side: Option<&str>) -> MemCheckpoint {
    let mut files = BTreeMap::from([
        ("config.json".to_string(), config.as_bytes().to_vec()),
        ("tokenizer.json".to_string(), b"{\"tok\": 1}".to_vec()),
        ("README.md".to_string(), b"the source's card".to_vec()),
    ]);
    if let Some(s) = side {
        files.insert("hf_quant_config.json".into(), s.as_bytes().to_vec());
    }
    MemCheckpoint { files }
}

fn write(
    config: &str,
    side: Option<&str>,
    index: &TensorIndex,
    threads: usize,
) -> (MockPlan, MemCheckpoint) {
    let spec = testkit::spec("1", "mode = \"uniform\"");
    let plan = plan_mock(&MockInputs {
        source_id: "toy/model",
        revision: None,
        config_json: config,
        hf_quant_config: side,
        index,
        spec: &spec,
        routing: None,
    })
    .unwrap();
    let mut sink = MemSink::default();
    let report = write_mock(&plan, &source(config, side), &mut sink, threads).unwrap();
    assert_eq!(report.copied, vec!["tokenizer.json".to_string()]);
    (plan, sink.out)
}

#[test]
fn a_written_mock_reads_back_as_planned() {
    let (config, side, index) = testkit::moe_nvfp4();
    let (plan, out) = write(&config, Some(&side), &index, 4);
    assert!(!out.files.contains_key("README.md"));
    assert_eq!(out.files[RESOLVED_FILE], plan.resolved.as_bytes());
    let back = read_source(&out).unwrap();
    assert_eq!(back.index.len(), plan.tensors.len());
    for t in &plan.tensors {
        let e = back
            .index
            .get(&t.name)
            .unwrap_or_else(|| panic!("{}", t.name));
        assert_eq!((e.dtype, &e.shape), (t.dtype, &t.shape));
    }
    for (name, header) in out.shard_headers().unwrap() {
        let v: serde_json::Value = serde_json::from_slice(&header).unwrap();
        assert_eq!(
            v["__metadata__"][MOCK_METADATA_KEY],
            plan.digest.as_str(),
            "{name}"
        );
    }
    let idx: serde_json::Value =
        serde_json::from_slice(&out.files["model.safetensors.index.json"]).unwrap();
    assert_eq!(
        idx["weight_map"].as_object().unwrap().len(),
        plan.tensors.len()
    );
    let mock_side = String::from_utf8(out.files["hf_quant_config.json"].clone()).unwrap();
    let mock_config = String::from_utf8(out.files["config.json"].clone()).unwrap();
    metrale_circuit::resolve_checkpoint(
        &mock_config,
        metrale_circuit::QuantMetadata {
            hf_quant_config: Some(&mock_side),
        },
        &metrale_circuit::ServePrecision::Declared,
    )
    .expect("the mock config resolves to a circuit");
}

#[test]
fn the_bytes_do_not_depend_on_the_thread_count() {
    let (config, index) = testkit::dense_ct();
    let (_, one) = write(&config, None, &index, 1);
    let (_, many) = write(&config, None, &index, 7);
    assert_eq!(one.files, many.files);
}

#[test]
fn a_mock_of_a_mock_keeps_every_layer_and_tensor() {
    let (config, index) = testkit::moe_fp8();
    let (plan, out) = write(&config, None, &index, 2);
    let back = read_source(&out).unwrap();
    let spec = testkit::spec("1", "mode = \"uniform\"");
    let again = plan_mock(&MockInputs {
        source_id: "toy/model",
        revision: None,
        config_json: &back.config,
        hf_quant_config: None,
        index: &back.index,
        spec: &spec,
        routing: None,
    })
    .unwrap();
    assert_eq!(again.selection.kept_layers(), plan.selection.kept_layers());
    let names = |p: &MockPlan| {
        p.tensors
            .iter()
            .map(|t| t.name.clone())
            .collect::<std::collections::BTreeSet<_>>()
    };
    assert_eq!(names(&again), names(&plan));
}

#[test]
fn a_skeleton_has_the_mock_headers_and_no_data() {
    let (config, index) = testkit::moe_fp8();
    let (plan, full) = write(&config, None, &index, 3);
    let mut sink = MemSink::default();
    write_skeleton(&plan, &source(&config, None), &mut sink).unwrap();
    let skel = sink.out;
    assert_eq!(skel.shard_headers().unwrap(), full.shard_headers().unwrap());
    for (name, bytes) in &full.files {
        assert_eq!(skel.files[name].len(), bytes.len(), "{name}");
        if !name.ends_with(".safetensors") {
            assert_eq!(&skel.files[name], bytes, "{name}");
        }
    }
    let data_start = |b: &[u8]| 8 + u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let (_, shard) = skel
        .files
        .iter()
        .find(|(n, _)| n.ends_with(".safetensors"))
        .unwrap();
    assert!(shard[data_start(shard)..].iter().all(|&b| b == 0));
}
