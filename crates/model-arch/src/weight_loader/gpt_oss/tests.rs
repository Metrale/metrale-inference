// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Actual pinned tensor-header tests. Inert address placeholders prove
//! binding and rejection, never checkpoint values, GPU execution or model parity.

use super::*;
use metrale_config::{LayerType, parse_config};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};
use std::collections::HashMap;

fn config() -> ModelConfig {
    parse_config(include_str!(
        "../../../../circuit/tests/fixtures/checkpoints/openai--gpt-oss-20b/config.json"
    ))
    .unwrap()
}

fn headers() -> HashMap<String, WeightTensor> {
    let value: serde_json::Value =
        serde_json::from_str(include_str!("checkpoint-headers.json")).unwrap();
    assert_eq!(
        value["revision"],
        "6cee5e81ee83917806bbde320786a8fb61efebee"
    );
    value["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, t)| {
            (
                t["name"].as_str().unwrap().to_owned(),
                WeightTensor {
                    // 2026-10-07: Widely separated inert addresses; no allocation or GPU reads.
                    ptr: DevicePtr((i as u64 + 1) * (1 << 32)),
                    shape: t["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|n| n.as_u64().unwrap() as usize)
                        .collect(),
                    dtype: match t["dtype"].as_str().unwrap() {
                        "BF16" => WeightDtype::BF16,
                        "U8" => WeightDtype::UInt8,
                        x => panic!("unexpected fixture dtype {x}"),
                    },
                },
            )
        })
        .collect()
}

#[test]
fn binds_all_pinned_headers_preserving_projection_and_expert_layout() {
    let config = config();
    let store = WeightStore::from_map(headers());
    assert_eq!(store.len(), 459);
    let checkpoint = GptOssCheckpoint::bind(&store, &config).unwrap();
    assert_eq!(checkpoint.layers.len(), 24);
    assert_eq!(checkpoint.policy, config.gpt_oss.unwrap());
    assert_eq!(checkpoint.embedding.shape(), [201088, 2880]);
    assert_eq!(checkpoint.head.shape(), [201088, 2880]);
    assert_ne!(checkpoint.embedding.ptr(), checkpoint.head.ptr());
    for (i, layer) in checkpoint.layers.iter().enumerate() {
        assert_eq!(
            layer.kind,
            if i % 2 == 0 {
                LayerType::SlidingAttention
            } else {
                LayerType::FullAttention
            }
        );
        assert_eq!(layer.q.weight.shape(), [4096, 2880]);
        assert_eq!(layer.k.weight.shape(), [512, 2880]);
        assert_eq!(layer.v.bias.shape(), [512]);
        assert_eq!(layer.o.weight.shape(), [2880, 4096]);
        assert_eq!(layer.sinks.shape(), [64]);
        assert_eq!(layer.router.weight.shape(), [32, 2880]);
        assert_eq!(layer.gate_up_bias.shape(), [32, 5760]);
        assert_eq!(layer.down_bias.shape(), [32, 2880]);
        let expert = layer.gate_up.expert(31).unwrap();
        assert_eq!((expert.rows(), expert.cols()), (5760, 2880));
        assert_eq!(layer.down.expert(31).unwrap().rows(), 2880);
        assert_eq!(layer.input_norm.dtype(), WeightDtype::BF16);
        assert_eq!(
            layer.input_norm.ptr(),
            store
                .get(&format!("model.layers.{i}.input_layernorm.weight"))
                .unwrap()
                .ptr
        );
    }
}

#[test]
fn every_missing_header_is_rejected() {
    let config = config();
    for name in headers().keys() {
        let mut map = headers();
        map.remove(name);
        let store = WeightStore::from_map(map);
        let error = GptOssCheckpoint::bind(&store, &config).err().unwrap();
        assert!(
            format!("{error:#}").contains(name),
            "missing {name}: {error:#}"
        );
    }
}

#[test]
fn every_tensor_rejects_wrong_shape_and_dtype() {
    let config = config();
    for name in headers().keys() {
        for change_dtype in [false, true] {
            let mut map = headers();
            let tensor = map.get_mut(name).unwrap();
            if change_dtype {
                tensor.dtype = WeightDtype::FP32;
            } else {
                tensor.shape[0] += 1;
            }
            let store = WeightStore::from_map(map);
            let error = GptOssCheckpoint::bind(&store, &config).err().unwrap();
            assert!(
                format!("{error:#}").contains(name),
                "wrong metadata {name}: {error:#}"
            );
        }
    }
}

#[test]
fn rejects_unknown_tensor_null_addresses_and_mutated_family_contract() {
    let config = config();
    let mut map = headers();
    map.insert(
        "unexpected.weight".into(),
        WeightTensor {
            ptr: DevicePtr(1),
            shape: vec![1],
            dtype: WeightDtype::BF16,
        },
    );
    assert!(GptOssCheckpoint::bind(&WeightStore::from_map(map), &config).is_err());
    for ptr in [DevicePtr::NULL, DevicePtr(u64::MAX - 1)] {
        let mut map = headers();
        map.get_mut("model.norm.weight").unwrap().ptr = ptr;
        assert!(GptOssCheckpoint::bind(&WeightStore::from_map(map), &config).is_err());
    }
    let store = WeightStore::from_map(headers());
    let mut changed = config.clone();
    changed.gpt_oss = None;
    assert!(GptOssCheckpoint::bind(&store, &changed).is_err());
    changed = config.clone();
    changed.tie_word_embeddings = true;
    assert!(GptOssCheckpoint::bind(&store, &changed).is_err());
    changed = config.clone();
    changed.layer_types[0] = LayerType::LinearAttention;
    assert!(GptOssCheckpoint::bind(&store, &changed).is_err());
    changed = config;
    changed.num_attention_heads = usize::MAX;
    assert!(GptOssCheckpoint::bind(&store, &changed).is_err());
}
