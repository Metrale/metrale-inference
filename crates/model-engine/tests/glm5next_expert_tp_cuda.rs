// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: GLM-5.3's tp routed-expert layout (`--moe-expert-layout tp`) on the GPU: three
//! simulated TP ranks, each holding its slice of every expert (cut by the production slicers)
//! and its slice of the shared expert, run `forward_moe` on the same rows; the sum of their
//! outputs (what the MLP all-reduce computes) is compared with one rank holding every expert
//! whole. Covered per tier: W4A16 at 1 row (slot GEMV), 4 and 16 rows (row-batched union) and
//! 128 rows (grouped prefill GEMM); declared W4A4 at 1 row (slot GEMV) and 4 and 16 rows
//! (union GEMV). Two widths: 640 (five 128-column units: 256 / 256 / 128, no padding) and 336
//! (padded to 384: the last rank holds 80 real columns and 48 zero ones). 2026-10-10: In
//! 64-column units: 640 splits 256 / 192 / 192 and GLM-5.3's 2048 (added) 704 / 704 / 640, so a
//! rank's W4A4 down runs the `_k64` twins (K % 128 == 64); 336 is 128 / 128 / 128 with 80 real
//! columns on the last rank.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! Run on a GB10 whose GPU is free, with an external timeout:
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! cargo test -p metrale-model-engine --test glm5next_expert_tp_cuda --no-run
//! GLM_EXPERT_TP_GPU_ORDINAL=0 timeout 300s cargo test -p metrale-model-engine \
//! --test glm5next_expert_tp_cuda -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use anyhow::{Context, Result, ensure};
use half::bf16;
use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily, tp_split};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mlp::build::build_moe;
use metrale_model_arch::glm5next_mlp::expert_tp::{
    expert_slice, slice_expert_cols, slice_expert_rows,
};
use metrale_model_arch::glm5next_mlp::forward::{Glm5NextMlpWorkspace, forward_moe};
use metrale_model_arch::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use metrale_model_arch::glm5next_mlp::weights::{Glm5NextExpertWeights, Nvfp4Proj, W4a4ActScales};
use metrale_model_arch::glm5next_mlp::{ExpertShard, Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-09: The tp sum against the whole expert at the SAME tier: the same dequantized
/// weights and, at W4A4, the same activation blocks (the slices are 128-aligned, so every
/// 16-value block quantizes identically); the difference is the FP32 summation order and the
/// BF16 rounding of each rank's partial before the sum. 2026-10-10: Measured on GB10 over the
/// 640, 336 and 2048 widths, both tiers and every row count: cosine >= 0.999994, rel L2 0.0032
/// to 0.0036 (one BF16 rounding of three partials); the bounds sit ~1.7x outside.
const SAME_TIER_MIN_COSINE: f64 = 0.99997;
const SAME_TIER_MAX_REL_L2: f64 = 0.006;
/// 2026-10-09: The padded width's W4A4 against the whole expert at W4A16 (a 336-wide whole
/// expert has no W4A4 down: K is not a multiple of 128): the W4A4-vs-W4A16 bound of
/// `glm5next_w4a4_cuda.rs`.
const CROSS_TIER_MIN_COSINE: f64 = 0.97;
const CROSS_TIER_MAX_REL_L2: f64 = 0.25;

const TP: usize = 3;
const HIDDEN: usize = 512;
const SHARED: usize = 256;
const EXPERTS: usize = 16;
const TOP_K: usize = 4;
/// 2026-10-09: 128 is the grouped prefill GEMM's default row floor.
const MAX_ROWS: usize = 128;

fn gpu() -> Result<MetraleCudaBackend> {
    let ordinal = std::env::var("GLM_EXPERT_TP_GPU_ORDINAL")?.parse()?;
    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    MetraleCudaBackend::new(ordinal, &target.modules)
}

/// 2026-10-09: The whole-expert reference: one rank, every expert whole.
fn whole_cfg(full: usize) -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        hidden: HIDDEN,
        local_dense_intermediate: SHARED,
        dense_start: 0,
        moe_intermediate: full,
        local_shared_intermediate: SHARED,
        shared_start: 0,
        num_experts: EXPERTS,
        local_experts: EXPERTS,
        ep_rank: 0,
        top_k: TOP_K,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 1,
        ep_world_size: 1,
        expert_shard: ExpertShard::Whole,
    }
}

/// 2026-10-09: Rank `r` of the tp layout: its expert slice and its shared-expert columns.
fn rank_cfg(full: usize, r: usize) -> Result<Glm5NextMlpConfig> {
    let s = expert_slice(full, TP, r)?;
    let sh = tp_split(SHARED, TP, r, 8)?;
    let c = Glm5NextMlpConfig {
        moe_intermediate: s.len,
        local_shared_intermediate: sh.len,
        shared_start: sh.start,
        tp_world_size: TP,
        expert_shard: ExpertShard::Sliced(s),
        ..whole_cfg(full)
    };
    c.validate()?;
    Ok(c)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// 2026-10-09: A random NVFP4 `[n, k]`: `(packed, scales)`, block scales 0.5 to 1.0.
fn host_nvfp4(rng: &mut Rng, n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let p = (0..n * k / 2).map(|_| rng.next() as u8).collect();
    let s = (0..n * k / 16)
        .map(|_| 0x30 + (rng.next() % 9) as u8)
        .collect();
    (p, s)
}

/// 2026-10-09: One host expert, whole.
struct HostExpert {
    gate: (Vec<u8>, Vec<u8>),
    up: (Vec<u8>, Vec<u8>),
    down: (Vec<u8>, Vec<u8>),
}

fn upload(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len())?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect()
}

fn read_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect())
}

fn proj(gpu: &dyn GpuBackend, w: (Vec<u8>, Vec<u8>), act: f32) -> Result<Nvfp4Proj> {
    Ok(Nvfp4Proj {
        packed: upload(gpu, &w.0)?,
        scale: upload(gpu, &w.1)?,
        scale_2: 0.01,
        input_scale: Some(act),
    })
}

/// 2026-10-09: Expert `e` as `cfg`'s rank holds it: whole, or cut by the production slicers.
fn bind(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextMlpConfig,
    e: &HostExpert,
    sc: W4a4ActScales,
) -> Result<Glm5NextExpertWeights> {
    let (g, u, d) = match cfg.expert_shard {
        ExpertShard::Whole => (e.gate.clone(), e.up.clone(), e.down.clone()),
        ExpertShard::Sliced(s) => (
            slice_expert_rows(&e.gate.0, &e.gate.1, HIDDEN, &s)?,
            slice_expert_rows(&e.up.0, &e.up.1, HIDDEN, &s)?,
            slice_expert_cols(&e.down.0, &e.down.1, HIDDEN, &s)?,
        ),
    };
    Ok(Glm5NextExpertWeights {
        gate_proj: proj(gpu, g, sc.gate_up)?,
        up_proj: proj(gpu, u, sc.gate_up)?,
        down_proj: proj(gpu, d, sc.down)?,
    })
}

fn plan(stamp: Nvfp4Act, rows: usize) -> Result<GroupPrecision> {
    GroupPrecision::resolve(
        MlpGroup::RoutedExperts,
        ActivationQuantization::default()
            .ladder(ProjFamily::Moe)
            .clone(),
        stamp,
        rows,
        true,
    )
}

fn agreement(got: &[f32], want: &[f32]) -> (f64, f64) {
    let (mut dot, mut gg, mut ww, mut dd) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (g as f64, w as f64);
        dot += g * w;
        gg += g * g;
        ww += w * w;
        dd += (g - w) * (g - w);
    }
    (
        dot / (gg.sqrt() * ww.sqrt()).max(1e-30),
        (dd / ww.max(1e-30)).sqrt(),
    )
}

/// 2026-10-09: The routed site of `cfg` at tier `stamp` over `rows` rows of `x`: output rows.
struct Site<'a> {
    gpu: &'a MetraleCudaBackend,
    k: Glm5NextMlpKernels,
    stream: u64,
    x: DevicePtr,
    out: DevicePtr,
}

impl Site<'_> {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        cfg: &Glm5NextMlpConfig,
        experts: &[HostExpert],
        sc: W4a4ActScales,
        stamp: Nvfp4Act,
        router: &[f32],
        shared: &[Vec<f32>; 3],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let gpu = self.gpu;
        let bound: Vec<Glm5NextExpertWeights> = experts
            .iter()
            .map(|e| bind(gpu, cfg, e, sc))
            .collect::<Result<_>>()?;
        let load = |n: &str| -> Result<Vec<f32>> {
            Ok(match n {
                "mlp.gate.weight" => router.to_vec(),
                "mlp.gate.e_score_correction_bias" => vec![0.0; EXPERTS],
                n if n.ends_with("gate_proj.weight") => shared[0].clone(),
                n if n.ends_with("up_proj.weight") => shared[1].clone(),
                _ => shared[2].clone(),
            })
        };
        let expert = |id: usize| -> Result<Glm5NextExpertWeights> { Ok(bound[id]) };
        let w = build_moe(
            gpu,
            cfg,
            SHARED,
            &load,
            &expert,
            &|_| plan(stamp, self.k.w4a4_expert_rows()),
            MAX_ROWS,
        )?;
        let ws = Glm5NextMlpWorkspace::new(gpu, cfg, MAX_ROWS)?;
        forward_moe(
            gpu,
            &self.k,
            cfg,
            &w,
            self.x,
            self.out,
            rows,
            &ws,
            false,
            self.stream,
        )?;
        gpu.synchronize(self.stream)?;
        read_bf16(gpu, self.out, rows * HIDDEN)
    }
}

/// 2026-10-09: For each width, tier and row count: the sum over the three rank slices against
/// the whole experts, within the bounds above.
#[test]
#[ignore = "requires an explicitly selected idle CUDA device and the glm-5.3-flash nvfp4 kernels"]
fn tp_sliced_experts_sum_to_the_whole() -> Result<()> {
    let gpu = gpu()?;
    let k = Glm5NextMlpKernels::resolve(&gpu)?;
    ensure!(
        k.w4a4_expert_rows() >= 16,
        "the W4A4 slot kernels did not resolve"
    );
    let mut rng = Rng(0x5851_f42d_4c95_7f2d);
    let x: Vec<f32> = (0..MAX_ROWS * HIDDEN).map(|_| rng.unit() * 3.0).collect();
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let sc = W4a4ActScales {
        gate_up: amax / (6.0 * 448.0),
        down: 100.0 / (6.0 * 448.0),
    };
    let router: Vec<f32> = (0..EXPERTS * HIDDEN).map(|_| rng.unit() * 0.05).collect();
    let shared: [Vec<f32>; 3] = std::array::from_fn(|_| {
        (0..HIDDEN * SHARED)
            .map(|_| rng.unit() * 0.02)
            .collect::<Vec<f32>>()
    });
    let site = Site {
        gpu: &gpu,
        k,
        stream: gpu.create_stream()?,
        x: upload(&gpu, &bf16_bytes(&x))?,
        out: gpu.alloc(MAX_ROWS * HIDDEN * 2)?,
    };
    for full in [640usize, 336, 2048] {
        let experts: Vec<HostExpert> = (0..EXPERTS)
            .map(|_| HostExpert {
                gate: host_nvfp4(&mut rng, full, HIDDEN),
                up: host_nvfp4(&mut rng, full, HIDDEN),
                down: host_nvfp4(&mut rng, HIDDEN, full),
            })
            .collect();
        let whole = whole_cfg(full);
        let ranks: Vec<Glm5NextMlpConfig> =
            (0..TP).map(|r| rank_cfg(full, r)).collect::<Result<_>>()?;
        let whole_w4a4 = full.is_multiple_of(128);
        for (stamp, tier, row_set) in [
            (Nvfp4Act::Unstamped, "w4a16", &[1usize, 4, 16, MAX_ROWS][..]),
            (Nvfp4Act::A4, "w4a4", &[1usize, 4, 16][..]),
        ] {
            for &rows in row_set {
                let mut sum = vec![0f32; rows * HIDDEN];
                for c in &ranks {
                    let part = site.run(c, &experts, sc, stamp, &router, &shared, rows)?;
                    for (a, v) in sum.iter_mut().zip(part) {
                        *a += v;
                    }
                }
                let same = stamp == Nvfp4Act::Unstamped || whole_w4a4;
                let ref_stamp = if same { stamp } else { Nvfp4Act::Unstamped };
                let want = site.run(&whole, &experts, sc, ref_stamp, &router, &shared, rows)?;
                let (cos, rel) = agreement(&sum, &want);
                let (min_cos, max_rel) = if same {
                    (SAME_TIER_MIN_COSINE, SAME_TIER_MAX_REL_L2)
                } else {
                    (CROSS_TIER_MIN_COSINE, CROSS_TIER_MAX_REL_L2)
                };
                println!(
                    "width {full} {tier} rows={rows} (vs whole {}): cosine {cos:.6}, rel L2 {rel:.5}",
                    if same { tier } else { "w4a16" }
                );
                ensure!(
                    cos >= min_cos && rel <= max_rel,
                    "width {full} {tier} rows={rows}: cosine {cos}, rel L2 {rel}"
                );
            }
        }
    }
    Ok(())
}
