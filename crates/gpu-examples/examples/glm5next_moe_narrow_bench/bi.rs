// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Part 5 of `glm5next_moe_narrow_bench` (batch invariance), no fixture needed:
//! `forward_moe` over rows 0..M as one launch against each row alone (M = 1), for M up to 128,
//! on a seeded router and shared expert (BF16 here), under two expert precision plans: today's
//! (W4A4 up to 16 rows, the W4A16 paths above) and the canonical one (W4A4 at every row count
//! through the grouped W4A4 GEMM). Prints, per plan, the row counts at which some row's bytes
//! differ from that row alone; the router and the BF16 shared expert are part of the site.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Reports, never fails on a variant row count (finding them is the point); errors only on a
//!   setup failure, or when the canonical plan's grouped kernels did not resolve.

use anyhow::{Result, ensure};
use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_mlp::build::build_moe;
use metrale_model_arch::glm5next_mlp::forward::{Glm5NextMlpWorkspace, forward_moe};
use metrale_model_arch::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use metrale_model_arch::glm5next_mlp::weights::{Glm5NextExpertWeights, Glm5NextMoeWeights};
use metrale_model_arch::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

use crate::Setup;
use crate::device::*;
use crate::forward::config;

const ROWS: [usize; 12] = [1, 2, 3, 4, 8, 15, 16, 17, 24, 32, 64, 128];
const MAXR: usize = 128;

fn site(
    g: &dyn GpuBackend,
    s: &Setup,
    c: &Glm5NextMlpConfig,
    w4a4_rows: usize,
) -> Result<Glm5NextMoeWeights> {
    let mut rr = Rng(0x510e_527f_ade6_82d1);
    let router: Vec<f32> = (0..EXPERTS * HIDDEN).map(|_| rr.unit() * 0.05).collect();
    let bias = vec![0.0f32; EXPERTS];
    let mut rs = Rng(0x1f83_d9ab_fb41_bd6b);
    let shared: Vec<f32> = (0..HIDDEN * 688 * 3).map(|_| rs.unit() * 0.02).collect();
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
    build_moe(
        g,
        c,
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
                w4a4_rows,
                has,
            )
        },
        MAXR,
    )
}

pub fn run(gpu: &MetraleCudaBackend, s: &Setup, st: u64) -> Result<()> {
    let g: &dyn GpuBackend = gpu;
    let c = config();
    let k = Glm5NextMlpKernels::resolve(g, metrale_gpu_examples::glm_levers()?)?;
    ensure!(
        k.w4a4_expert_rows_grouped() == usize::MAX,
        "the grouped W4A4 GEMM, the sort or the indexed combine did not resolve"
    );
    let mut rng = Rng(0x5be0_cd19_137e_2179);
    let x: Vec<f32> = (0..MAXR * HIDDEN).map(|_| rng.unit() * 3.0).collect();
    let xd = up_bf16(g, &x)?;
    let row_bytes = HIDDEN * 2;
    let out = g.alloc(MAXR * row_bytes)?;
    for (plan, cap, grouped) in [
        ("today (W4A4 <= 16 rows)", k.w4a4_expert_rows(), false),
        (
            "canonical (W4A4 grouped)",
            k.w4a4_expert_rows_grouped(),
            true,
        ),
    ] {
        let w = site(g, s, &c, cap)?;
        let ws = Glm5NextMlpWorkspace::new_forked(g, &c, MAXR, grouped, 0)?;
        let mut alone = Vec::with_capacity(MAXR);
        for r in 0..MAXR {
            forward_moe(
                g,
                &k,
                &c,
                &w,
                xd.offset(r * row_bytes),
                out,
                1,
                &ws,
                false,
                st,
            )?;
            g.synchronize(st)?;
            alone.push(read(g, out, row_bytes)?);
        }
        let mut variant = Vec::new();
        for m in ROWS {
            forward_moe(g, &k, &c, &w, xd, out, m, &ws, false, st)?;
            g.synchronize(st)?;
            let got = read(g, out, m * row_bytes)?;
            let bad = (0..m)
                .filter(|&r| got[r * row_bytes..(r + 1) * row_bytes] != alone[r][..])
                .count();
            if bad > 0 {
                variant.push(format!("M={m}: {bad}/{m} rows differ"));
            }
        }
        if variant.is_empty() {
            println!(
                "batch invariance, forward_moe, {plan}: every row equals its launch alone at M {ROWS:?}"
            );
        } else {
            println!(
                "batch invariance, forward_moe, {plan}: VARIANT {}",
                variant.join("; ")
            );
        }
    }
    Ok(())
}
