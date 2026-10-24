// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: `takes`: the fused router replaces exactly the few-row GEMV + rank top-k pair.
//! The bytes are the GPU gate's job (`examples/glm5next_router_fused_gate.rs`).
//!
//! Owner: model-arch (GLM-5.3 MLP).

use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::takes;
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-24: The served levers' pair: the batched router rows and the rank top-k on, every
/// kernel present (the mock resolves them all).
fn served() -> (Glm5NextMlpKernels, Glm5NextMlpConfig) {
    let gpu = MockGpuBackend::new();
    let mut levers = crate::glm5next_layer::GlmLevers::declared();
    levers.router_rows = true;
    levers.router_topk_rank = true;
    let k = Glm5NextMlpKernels::resolve(&gpu, levers).expect("mock resolves every kernel");
    let cfg = crate::glm5next_plan::fixture::mlp_config_at(0, 3);
    (k, cfg)
}

/// 2026-10-24: One piece of 2..=8 rows on the served levers takes the fused launch (positive).
#[test]
fn the_served_pair_at_two_to_eight_rows_is_fused() {
    let (k, cfg) = served();
    assert!(
        k.levers.dense.batchm_few,
        "the declared levers take the few-row GEMV"
    );
    for rows in 2..=8 {
        assert!(takes(&k, &cfg, &[rows]), "{rows} rows");
    }
}

/// 2026-10-24: Every other pair keeps its two launches: one row (the M = 1 GEMV), nine rows (the
/// wide entry), several pieces, the runtime-M GEMV (batchm_few off), the base top-k, the
/// one-output-per-block A/B entry, a missing rank kernel, more experts or slots than the kernel
/// ranks (negative).
#[test]
fn any_other_pair_keeps_its_two_launches() {
    let (k, cfg) = served();
    assert!(!takes(&k, &cfg, &[1]));
    assert!(!takes(&k, &cfg, &[9]));
    assert!(!takes(&k, &cfg, &[1, 1]));
    let mut off = k;
    off.levers.dense.batchm_few = false;
    assert!(!takes(&off, &cfg, &[2]));
    let mut base = k;
    base.levers.router_topk_rank = false;
    assert!(!takes(&base, &cfg, &[2]));
    let mut rows = k;
    rows.levers.router_rows = false;
    assert!(!takes(&rows, &cfg, &[2]));
    let mut npb1 = k;
    npb1.gemv_batchm_few = npb1.gemv_batchm_few.with_router_npb1().unwrap();
    assert!(!takes(&npb1, &cfg, &[2]));
    let mut no_rank = k;
    no_rank.router_rank = KernelHandle(0);
    assert!(!takes(&no_rank, &cfg, &[2]));
    let wide = Glm5NextMlpConfig {
        num_experts: 513,
        ..cfg.clone()
    };
    assert!(!takes(&k, &wide, &[2]));
    let many = Glm5NextMlpConfig { top_k: 17, ..cfg };
    assert!(!takes(&k, &many, &[2]));
}
