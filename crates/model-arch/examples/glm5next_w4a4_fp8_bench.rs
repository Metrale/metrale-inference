// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Microbench of GLM-5.3's declared W4A4 routed experts against the W4A16 path,
//! and of the `--dense-quantization fp8` W8A8 projections against the BF16 GEMVs, at the shapes
//! one rank of the three-box TP=3 / EP=3 serve runs. Timing only (the numerics are the
//! `glm5next_w4a4_cuda` / `glm5next_fp8_dense_cuda` tests).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Part 1, routed site (`forward_moe`, router + experts + shared expert + combine), hidden 4096,
//! expert width 2048, 288 experts with this rank's 96 bound, top 8, at 1, 8 and 16 rows, for
//! two routings: `spread` (independent rows) and `same` (every row the same token, the union is
//! 8 experts: the shape of a speculative verify whose drafts route alike). Each line gives the
//! W4A16 and W4A4 microseconds per call and the expert bytes the union reads.
//!
//! Part 2, dense projections at the KDA shapes of rank 0 (q: 4096 -> 2816, o: 2816 -> 4096) and
//! the DSA absorbed o (11264 -> 4096), at 1 and 8 rows: the BF16 GEMV (`dense_mm_bf16`) against
//! the FP8 W8A8 projection (quantize + GEMV), microseconds per call and effective GB/s.
//!
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! GLM_BENCH_GPU_ORDINAL=0 cargo run -p metrale-model-arch --release \
//!     --example glm5next_w4a4_fp8_bench --features cuda,gpu-examples
//! ```

use anyhow::{Context, Result};
use half::bf16;
use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_fp8_dense;
use metrale_model_arch::glm5next_mlp::build::build_moe;
use metrale_model_arch::glm5next_mlp::forward::{Glm5NextMlpWorkspace, forward_moe};
use metrale_model_arch::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use metrale_model_arch::glm5next_mlp::weights::{Glm5NextExpertWeights, Nvfp4Proj};
use metrale_model_arch::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use metrale_model_layers::layers::ops::{DenseMmKernels, dense_mm_bf16};
use metrale_model_layers::layers::{DenseQuantization, set_dense_quantization_from_cli};

const ITERS: usize = 200;

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

fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn up_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(
        gpu,
        &v.iter()
            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

/// 2026-10-09: Mean microseconds of `f` over `ITERS` calls after 20 warm-up calls.
fn time(gpu: &dyn GpuBackend, stream: u64, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..20 {
        f()?;
    }
    gpu.synchronize(stream)?;
    let t = std::time::Instant::now();
    for _ in 0..ITERS {
        f()?;
    }
    gpu.synchronize(stream)?;
    Ok(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
}

/// 2026-10-09: One rank of GLM-5.3-Flash at TP=3 / EP=3 (rank 0: experts 0..96, shared 768 under
/// the FP8 split; 688 otherwise, which does not change the routed cost).
fn cfg() -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        hidden: 4096,
        local_dense_intermediate: 4096,
        dense_start: 0,
        moe_intermediate: 2048,
        local_shared_intermediate: 688,
        shared_start: 0,
        num_experts: 288,
        local_experts: 96,
        ep_rank: 0,
        top_k: 8,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 3,
        ep_world_size: 3,
    }
}

fn expert_proj(
    gpu: &dyn GpuBackend,
    rng: &mut Rng,
    n: usize,
    k: usize,
    gs: f32,
) -> Result<Nvfp4Proj> {
    let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
    let scales: Vec<u8> = (0..n * k / 16)
        .map(|_| 0x30 + (rng.next() % 9) as u8)
        .collect();
    Ok(Nvfp4Proj {
        packed: up(gpu, &packed)?,
        scale: up(gpu, &scales)?,
        scale_2: 0.01,
        input_scale: Some(gs),
    })
}

fn routed(gpu: &MetraleCudaBackend, stream: u64) -> Result<()> {
    let (c, k) = (cfg(), Glm5NextMlpKernels::resolve(gpu)?);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let experts: Vec<Glm5NextExpertWeights> = (0..c.local_experts)
        .map(|_| -> Result<_> {
            Ok(Glm5NextExpertWeights {
                gate_proj: expert_proj(gpu, &mut rng, 2048, 4096, 3.0 / 2688.0)?,
                up_proj: expert_proj(gpu, &mut rng, 2048, 4096, 3.0 / 2688.0)?,
                down_proj: expert_proj(gpu, &mut rng, 4096, 2048, 100.0 / 2688.0)?,
            })
        })
        .collect::<Result<_>>()?;
    let router: Vec<f32> = (0..c.num_experts * c.hidden)
        .map(|_| rng.unit() * 0.05)
        .collect();
    let full_shared = c.local_shared_intermediate * 3;
    let shared: Vec<f32> = (0..c.hidden * full_shared)
        .map(|_| rng.unit() * 0.02)
        .collect();
    let load = |n: &str| -> Result<Vec<f32>> {
        Ok(match n {
            "mlp.gate.weight" => router.clone(),
            "mlp.gate.e_score_correction_bias" => vec![0.0; c.num_experts],
            _ => shared.clone(),
        })
    };
    let expert = |id: usize| -> Result<Glm5NextExpertWeights> { Ok(experts[id]) };
    let build = |stamp: Nvfp4Act| {
        build_moe(
            gpu,
            &c,
            full_shared,
            &load,
            &expert,
            &|has| {
                GroupPrecision::resolve(
                    MlpGroup::RoutedExperts,
                    ActivationQuantization::default()
                        .ladder(ProjFamily::Moe)
                        .clone(),
                    stamp,
                    k.w4a4_expert_rows(),
                    has,
                )
            },
            16,
        )
    };
    let (w4a4, w4a16) = (build(Nvfp4Act::A4)?, build(Nvfp4Act::Unstamped)?);
    let ws = Glm5NextMlpWorkspace::new(gpu, &c, 16)?;
    let out = gpu.alloc(16 * c.hidden * 2)?;
    let one: Vec<f32> = (0..c.hidden).map(|_| rng.unit() * 3.0).collect();
    let spread: Vec<f32> = (0..16 * c.hidden).map(|_| rng.unit() * 3.0).collect();
    let same: Vec<f32> = (0..16).flat_map(|_| one.clone()).collect();
    println!("routed site, one rank (96 of 288 experts), us per forward_moe call:");
    for (name, x) in [("spread", &spread), ("same", &same)] {
        let xd = up_bf16(gpu, x)?;
        for rows in [1usize, 8, 16] {
            let a = time(gpu, stream, || {
                forward_moe(gpu, &k, &c, &w4a16, xd, out, rows, &ws, false, stream)
            })?;
            let b = time(gpu, stream, || {
                forward_moe(gpu, &k, &c, &w4a4, xd, out, rows, &ws, false, stream)
            })?;
            println!(
                "  {name:6} rows={rows:2}: W4A16 {a:8.1} us   W4A4 {b:8.1} us   ratio {:.2}",
                b / a
            );
        }
    }
    Ok(())
}

fn dense(gpu: &MetraleCudaBackend, stream: u64) -> Result<()> {
    let bf = DenseMmKernels {
        gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
        gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
        batchm: gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?,
    };
    let quantize = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
    glm5next_fp8_dense::prepare(gpu, 11264)?;
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    println!("dense projections, us per call (GB/s of weight bytes):");
    for (n, k, what) in [
        (2816usize, 4096usize, "kda q 4096->2816"),
        (4096, 2816, "kda o 2816->4096"),
        (4096, 11264, "dsa o_absorb 11264->4096"),
    ] {
        let w = up_bf16(
            gpu,
            &(0..n * k).map(|_| rng.unit() * 0.02).collect::<Vec<_>>(),
        )?;
        glm5next_fp8_dense::register(gpu, quantize, w, n, k, what, stream)?;
        let x = up_bf16(gpu, &(0..16 * k).map(|_| rng.unit()).collect::<Vec<_>>())?;
        let out = gpu.alloc(16 * n * 2)?;
        for m in [1usize, 8] {
            let b = time(gpu, stream, || {
                dense_mm_bf16(gpu, &bf, x, w, out, m, n, k, stream)
            })?;
            let f = time(gpu, stream, || {
                glm5next_fp8_dense::proj(gpu, w, x, out, m, n, k, stream).map(|_| ())
            })?;
            let gbs = |us: f64, bytes: usize| bytes as f64 / us / 1e3;
            println!(
                "  {what:26} m={m}: BF16 {b:7.1} us ({:5.0} GB/s)   FP8 {f:7.1} us ({:5.0} GB/s)",
                gbs(b, n * k * 2),
                gbs(f, n * k)
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    set_dense_quantization_from_cli(DenseQuantization::Fp8);
    let ordinal = std::env::var("GLM_BENCH_GPU_ORDINAL")
        .context("GLM_BENCH_GPU_ORDINAL names the idle GPU to run on")?
        .parse()?;
    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(ordinal, &target.modules)?;
    let stream = gpu.create_stream()?;
    routed(&gpu, stream)?;
    dense(&gpu, stream)
}
