// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The shared expert's E4M3 prefill copies exist only under
//! `--nemotron-shared-expert-e4m3`.
//!
//! Owner: model-arch (Nemotron-H).
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_layers::weight_map::{
    DenseWeight, NemotronExpertWeight, NemotronMoeWeights, QuantizedWeight,
};

use crate::nemotron_moe::NemotronMoeLayer;

const H: usize = 64;
const INTER: usize = 32;

fn nvfp4(gpu: &MockGpuBackend, n: usize, k: usize) -> QuantizedWeight {
    QuantizedWeight {
        weight: gpu.alloc(n * k / 2).unwrap(),
        weight_scale: gpu.alloc(n * k / 16).unwrap(),
        weight_scale_2: 1.0,
        ..QuantizedWeight::null()
    }
}

/// 2026-10-02: A two-expert layer on the mock; `gate_f32` stores the router FP32 (as
/// Nemotron-3-Nano does) instead of BF16.
pub(in crate::nemotron_moe) fn layer(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    gate_f32: bool,
) -> NemotronMoeLayer {
    let dense = |bytes| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    let weights = NemotronMoeWeights {
        gate: dense(2 * H * if gate_f32 { 4 } else { 2 }),
        gate_f32,
        e_score_correction_bias: dense(2 * 4),
        experts: (0..2)
            .map(|_| NemotronExpertWeight {
                up_proj: nvfp4(gpu, INTER, H),
                down_proj: nvfp4(gpu, H, INTER),
            })
            .collect(),
        shared_up: nvfp4(gpu, INTER, H),
        shared_up_fp8: None,
        shared_down: nvfp4(gpu, H, INTER),
        shared_down_fp8: None,
        fc1_latent_proj: None,
        fc2_latent_proj: None,
    };
    NemotronMoeLayer::new(weights, dense(H * 2), config, gpu, INTER, 1).unwrap()
}

pub(in crate::nemotron_moe) fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = H;
    c.num_experts = 2;
    c.num_experts_per_tok = 1;
    c.moe_intermediate_size = INTER;
    c.shared_expert_intermediate_size = INTER;
    c.moe_latent_size = 0;
    c
}

#[test]
fn the_shared_expert_keeps_bf16_mma_by_default() {
    let gpu = MockGpuBackend::new();
    let config = config();
    let mut l = layer(&gpu, &config, false);
    l.prepare_prefill_weights(&gpu, &config, false);
    assert!(l.shared_up_t.is_none() && l.shared_down_t.is_none());
}

/// 2026-09-29: Path B: the opt-in builds both transposed copies the E4M3 GEMM reads.
#[test]
fn the_opt_in_builds_the_e4m3_copies() {
    let gpu = MockGpuBackend::new();
    let config = config();
    let mut l = layer(&gpu, &config, false);
    l.prepare_prefill_weights(&gpu, &config, true);
    assert!(l.shared_up_t.is_some() && l.shared_down_t.is_some());
}
