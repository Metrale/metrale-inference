// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Which O weight the fixed `nvfp4` arm (`pinned_o`) sees. A layer built for NVFP4
//! holds O in `attn.o_proj` and leaves `o_weight` unset, so a check on `o_weight` alone never
//! admitted it: O then ran W4A16 at 2 and 3 rows (`w4a16_gemv_batch2/3`) and W4A4 at every
//! other row count, and a row's bits depended on the batch width.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: none beyond the types.

use crate::layers::{FfnComponent, qwen3_attention::Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Fp8Weight, QuantWeight, QuantizedWeight, WeightQuantFormat,
};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// 2026-10-03: A layer built as the NVFP4 loaders build it, O in `attn.o_proj` when `o_nvfp4`.
fn layer(gpu: &MockGpuBackend, o_nvfp4: bool) -> Qwen3AttentionLayer {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = 128;
    let dense = DenseWeight {
        weight: gpu.alloc(128 * 128 * 2).unwrap(),
    };
    let mut o_proj = QuantizedWeight::null();
    if o_nvfp4 {
        o_proj.weight = gpu.alloc(128 * 64).unwrap();
        o_proj.weight_scale = gpu.alloc(128 * 8).unwrap();
    }
    let attn = AttentionWeights {
        q_proj: dense,
        k_proj: dense,
        v_proj: dense,
        o_proj,
        q_norm: dense,
        k_norm: dense,
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    Qwen3AttentionLayer::new(
        dense,
        attn,
        dense,
        FfnComponent::None,
        0,
        None,
        None,
        None,
        gpu,
        KvCacheDtype::Bf16,
        0,
        &config,
    )
    .unwrap()
}

#[test]
fn an_nvfp4_layer_with_o_only_in_attn_o_proj_is_admitted() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu, true);
    assert!(
        l.o_weight.is_none(),
        "the NVFP4 constructor leaves o_weight unset"
    );
    assert!(l.o_is_nvfp4());
}

#[test]
fn an_fp8_o_is_not_admitted_even_with_an_nvfp4_copy() {
    let gpu = MockGpuBackend::new();
    let mut l = layer(&gpu, true);
    l.o_weight = Some(QuantWeight::Fp8(Fp8Weight {
        weight: gpu.alloc(128 * 128).unwrap(),
        row_scale: gpu.alloc(4).unwrap(),
        n: 128,
        k: 128,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    }));
    assert!(!l.o_is_nvfp4());
}

#[test]
fn no_o_weight_is_not_admitted() {
    let gpu = MockGpuBackend::new();
    assert!(!layer(&gpu, false).o_is_nvfp4());
}
