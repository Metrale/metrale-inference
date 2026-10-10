// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Part 3 of `glm5next_moe_narrow_bench`: `forward_moe` at one row under the
//! fixture's router (its weight and correction bias, per layer) on the fixture's router inputs,
//! with the own-slots sweep against the slot GEMV (the same kernel table with
//! `w4a4_moe_slots_sweep` unresolved): every token's output bytes, then both timed.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on the first byte mismatch.

use std::cell::Cell;

use anyhow::{Result, ensure};
use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_mlp::build::build_moe;
use metrale_model_arch::glm5next_mlp::forward::{Glm5NextMlpWorkspace, forward_moe};
use metrale_model_arch::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use metrale_model_arch::glm5next_mlp::weights::Glm5NextExpertWeights;
use metrale_model_arch::glm5next_mlp::{ExpertShard, Glm5NextMlpConfig, Glm5NextMlpKernels};

use crate::device::*;
use crate::{Fixture, Setup};

/// 2026-10-09: Rank 0 of TP=3 / EP=3 (the serve's shapes).
fn config() -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        hidden: HIDDEN,
        local_dense_intermediate: 4096,
        dense_start: 0,
        moe_intermediate: MI,
        local_shared_intermediate: 688,
        shared_start: 0,
        num_experts: EXPERTS,
        local_experts: LOCAL,
        ep_rank: 0,
        top_k: TOP_K,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 3,
        ep_world_size: 3,
        expert_shard: ExpertShard::Whole,
    }
}

pub fn run(gpu: &MetraleCudaBackend, s: &Setup, f: &Fixture, st: u64) -> Result<()> {
    let g: &dyn GpuBackend = gpu;
    let c = config();
    let new = Glm5NextMlpKernels::resolve(g)?;
    ensure!(
        new.w4a4_moe_slots_sweep.0 != 0,
        "w4a4_gemv_mx8_moe_slots_sweep did not resolve"
    );
    let old = Glm5NextMlpKernels {
        w4a4_moe_slots_sweep: KernelHandle(0),
        ..new
    };
    let mut rng = Rng(0x6a09_e667_f3bc_c908);
    let shared: Vec<f32> = (0..HIDDEN * 688 * 3).map(|_| rng.unit() * 0.02).collect();
    let ws = Glm5NextMlpWorkspace::new(g, &c, MAX_ROWS)?;
    let (o_new, o_old) = (s.out[0], s.out[1]);
    let row_bytes = HIDDEN * 2;
    let n = env_or("GLM_BENCH_ITERS", ITERS);
    for l in 0..f.layers {
        let router = f.router[l * EXPERTS * HIDDEN..(l + 1) * EXPERTS * HIDDEN].to_vec();
        let bias = f.bias[l * EXPERTS..(l + 1) * EXPERTS].to_vec();
        let load = |name: &str| -> Result<Vec<f32>> {
            Ok(match name {
                "mlp.gate.weight" => router.clone(),
                "mlp.gate.e_score_correction_bias" => bias.clone(),
                _ => shared.clone(),
            })
        };
        let expert = |id: usize| -> Result<Glm5NextExpertWeights> {
            Ok(Glm5NextExpertWeights {
                gate_proj: s.gate.projs[id],
                up_proj: s.upt.projs[id],
                down_proj: s.down.projs[id],
            })
        };
        let w = build_moe(
            g,
            &c,
            688 * 3,
            &load,
            &expert,
            &|has| {
                GroupPrecision::resolve(
                    MlpGroup::RoutedExperts,
                    ActivationQuantization::default()
                        .ladder(ProjFamily::Moe)
                        .clone(),
                    Nvfp4Act::A4,
                    new.w4a4_expert_rows(),
                    has,
                )
            },
            MAX_ROWS,
        )?;
        let hid = &f.hidden[l * f.tokens * HIDDEN..(l + 1) * f.tokens * HIDDEN];
        let xd = up_bf16(g, hid)?;
        let mut h_new = 0xcbf2_9ce4_8422_2325u64;
        for t in 0..f.tokens {
            let x = xd.offset(t * row_bytes);
            forward_moe(g, &new, &c, &w, x, o_new, 1, &ws, false, 0)?;
            let a = read(g, o_new, row_bytes)?;
            forward_moe(g, &old, &c, &w, x, o_old, 1, &ws, false, 0)?;
            ensure!(
                a == read(g, o_old, row_bytes)?,
                "layer {l} token {t}: forward_moe differs between the own-slots sweep and the \
                 slot GEMV"
            );
            h_new = a.iter().fold(h_new, |h, &b| {
                (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
            });
        }
        println!(
            "forward_moe 1 row, fixture layer {l}: {} tokens identical (own-slots sweep vs slot \
             GEMV), fnv1a {h_new:016x}",
            f.tokens
        );
        let i = Cell::new(0usize);
        let step = |k: &Glm5NextMlpKernels, o| -> Result<()> {
            let t = (i.get() * 7) % f.tokens;
            i.set(i.get() + 1);
            forward_moe(g, k, &c, &w, xd.offset(t * row_bytes), o, 1, &ws, true, st)
        };
        let t = time_set(g, st, &[&|| step(&old, o_old), &|| step(&new, o_new)])?;
        println!(
            "  forward_moe 1 row ({n} tokens per replay): slot GEMV {}  own-slots sweep {}",
            fmt(t[0]),
            fmt(t[1])
        );
    }
    Ok(())
}
