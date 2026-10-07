// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Launch/lifetime controls only. Mock kernels execute no arithmetic.
use super::*;
use crate::weight_loader::{
    ModelWeightLoader,
    gpt_oss::{GptOssCheckpoint, loader::GptOssWeightLoader},
};
use metrale_cache::kv_cache::KvCacheConfig;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};
fn fixture() -> (ModelConfig, WeightStore) {
    let config = metrale_config::parse_config(include_str!(
        "../../../../../circuit/tests/fixtures/checkpoints/openai--gpt-oss-20b/config.json"
    ))
    .unwrap();
    let json: serde_json::Value =
        serde_json::from_str(include_str!("../checkpoint-headers.json")).unwrap();
    let map = json["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, t)| {
            (
                t["name"].as_str().unwrap().to_owned(),
                WeightTensor {
                    ptr: DevicePtr((i as u64 + 1) * (1 << 32)),
                    shape: t["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect(),
                    dtype: if t["dtype"] == "BF16" {
                        WeightDtype::BF16
                    } else {
                        WeightDtype::UInt8
                    },
                },
            )
        })
        .collect();
    (config, WeightStore::from_map(map))
}
fn cache(gpu: &dyn GpuBackend, dtype: KvCacheDtype) -> PagedKvCache {
    PagedKvCache::new(
        KvCacheConfig {
            block_size: 16,
            num_kv_heads: 8,
            head_dim: 64,
            num_layers: 24,
            dtype,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        },
        2,
        gpu,
    )
    .unwrap()
}
#[test]
fn one_token_composes_stages_and_uses_its_actual_position() {
    let (config, store) = fixture();
    let gpu = MockGpuBackend::new();
    let bound = GptOssCheckpoint::bind(&store, &config).unwrap();
    let layer = GptOssLayer::new(&bound.layers[0], &config, 0, &gpu).unwrap();
    let mut state = layer.alloc_state(&gpu).unwrap();
    // 2026-10-07: Mock router does nothing: supply distinct IDs for ABI tracing.
    let ids = [1u32, 5, 7, 9]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    let ids_ptr = state.as_any().downcast_ref::<State>().unwrap().ids;
    gpu.copy_h2d(&ids, ids_ptr).unwrap();
    let hidden = gpu.alloc(5760).unwrap();
    let mut cache = cache(&gpu, KvCacheDtype::Bf16);
    let mut blocks = vec![];
    layer
        .forward_token(hidden, state.as_mut(), &mut cache, 0, &mut blocks, &gpu, 0)
        .unwrap();
    assert!(
        layer
            .forward_token(hidden, state.as_mut(), &mut cache, 17, &mut blocks, &gpu, 0)
            .is_err()
    );
    assert!(
        layer
            .forward_token(hidden, state.as_mut(), &mut cache, 0, &mut blocks, &gpu, 0)
            .is_err()
    );
    // An appended alias preserves the old prefix yet would overwrite it at 16.
    blocks.push(blocks[0]);
    assert!(
        layer
            .forward_token(hidden, state.as_mut(), &mut cache, 1, &mut blocks, &gpu, 0)
            .is_err()
    );
    assert_eq!(
        state
            .as_any()
            .downcast_ref::<State>()
            .unwrap()
            .next_position,
        1
    );
    blocks.pop();
    for position in 1..=17 {
        layer
            .forward_token(
                hidden,
                state.as_mut(),
                &mut cache,
                position,
                &mut blocks,
                &gpu,
                0,
            )
            .unwrap();
    }
    let original = blocks[0];
    blocks[0] = blocks[1];
    assert!(
        layer
            .forward_token(hidden, state.as_mut(), &mut cache, 18, &mut blocks, &gpu, 0)
            .is_err()
    );
    blocks[0] = original;
    let mut other_pool = self::cache(&gpu, KvCacheDtype::Bf16);
    assert!(
        layer
            .forward_token(
                hidden,
                state.as_mut(),
                &mut other_pool,
                18,
                &mut blocks,
                &gpu,
                0
            )
            .is_err()
    );
    let s = state.as_any().downcast_ref::<State>().unwrap();
    let mut bytes = [0u8; 4];
    gpu.copy_d2h(s.position, &mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), 17);
    gpu.copy_d2h(s.length, &mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), 18);
    assert_eq!(blocks.len(), 2);
    let mut slot = [0u8; 8];
    gpu.copy_d2h(s.slot, &mut slot).unwrap();
    assert_eq!(i64::from_le_bytes(slot), i64::from(blocks[1]) * 16 + 1);
    assert!(layer.decode_graph_unsupported());
    assert!(layer.decode_multi_seq_unsupported());
    let before = gpu.alloc_count();
    layer.release_state(state.as_mut(), &gpu).unwrap();
    layer.release_state(state.as_mut(), &gpu).unwrap();
    assert_eq!(gpu.alloc_count() + 1, before);
    assert!(
        layer
            .forward_token(hidden, state.as_mut(), &mut cache, 18, &mut blocks, &gpu, 0)
            .is_err()
    );
    gpu.free(hidden).unwrap();
}
#[test]
fn constructor_and_runtime_fail_closed() {
    let (config, store) = fixture();
    let gpu = MockGpuBackend::new();
    let loader = GptOssWeightLoader;
    assert!(
        loader
            .load_layers(&store, &config, &gpu, &[KvCacheDtype::Fp8; 24])
            .is_err()
    );
    let bound = GptOssCheckpoint::bind(&store, &config).unwrap();
    let layer = GptOssLayer::new(&bound.layers[0], &config, 0, &gpu).unwrap();
    let mut state = layer.alloc_state(&gpu).unwrap();
    let hidden = gpu.alloc(5760).unwrap();
    let mut wrong_cache = cache(&gpu, KvCacheDtype::Fp8);
    assert!(
        layer
            .forward_token(
                hidden,
                state.as_mut(),
                &mut wrong_cache,
                0,
                &mut vec![],
                &gpu,
                0
            )
            .is_err()
    );
    gpu.deny_kernel("gpt_oss_rope", "gpt_oss_rope_bf16");
    assert!(GptOssLayer::new(&bound.layers[0], &config, 0, &gpu).is_err());
    layer.release_state(state.as_mut(), &gpu).unwrap();
    gpu.free(hidden).unwrap();
}
