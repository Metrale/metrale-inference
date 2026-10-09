// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: GLM-5.3's declared W4A4 MLP on the GPU: the static-scale quantizer against the
//! dynamic one, each MLP group at W4A4 against its 16-bit path within the NVFP4 W4A4
//! tolerance, and a row's W4A4 output bit-identical at every row count.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! Run on a GB10 whose GPU is free, with an external timeout:
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! cargo test -p metrale-model-engine --test glm5next_w4a4_cuda --no-run
//! GLM_W4A4_GPU_ORDINAL=0 timeout 300s cargo test -p metrale-model-engine \
//! --test glm5next_w4a4_cuda -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use anyhow::{Context, Result, ensure};
use half::bf16;
use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mlp::build::build_moe;
use metrale_model_arch::glm5next_mlp::forward::{
    Glm5NextMlpWorkspace, forward_dense_site, forward_moe,
};
use metrale_model_arch::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use metrale_model_arch::glm5next_mlp::weights::{
    Glm5NextDenseMlpWeights, Glm5NextDenseNvfp4Weights, Glm5NextDenseSite, Glm5NextExpertWeights,
    Glm5NextMoeWeights, Nvfp4Proj, W4a4ActScales,
};
use metrale_model_arch::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-08: The tolerance of a W4A4 output against the W4A16 / BF16 output of the same
/// weights: NVFP4 activations carry a per-element error of up to a quarter of an E2M1 step, so
/// the outputs agree in direction, not in bits. Unmeasured bounds; tighten after the first run.
const MIN_COSINE: f64 = 0.97;
const MAX_REL_L2: f64 = 0.25;

/// 2026-10-08: ModelOpt's static scale for activations of amplitude `amax`.
fn modelopt_scale(amax: f32) -> f32 {
    amax / (6.0 * 448.0)
}

fn gpu() -> Result<MetraleCudaBackend> {
    let ordinal = std::env::var("GLM_W4A4_GPU_ORDINAL")?.parse()?;
    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    MetraleCudaBackend::new(ordinal, &target.modules)
}

/// 2026-10-08: A small GLM-shaped MLP: K multiples of 128, 16 experts, top 4, one rank.
fn cfg() -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        hidden: 512,
        local_dense_intermediate: 384,
        dense_start: 0,
        moe_intermediate: 256,
        local_shared_intermediate: 256,
        shared_start: 0,
        num_experts: 16,
        local_experts: 16,
        ep_rank: 0,
        top_k: 4,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 1,
        ep_world_size: 1,
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// 2026-10-08: Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn e4m3_to_f32(b: u8) -> f32 {
    let (s, e, m) = ((b >> 7) & 1, ((b >> 3) & 0xF) as i32, (b & 7) as f32);
    let v = if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if s == 1 { -v } else { v }
}

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// 2026-10-08: A random NVFP4 `[n, k]` weight (block scales 0.5 to 1.0) and its dequantized
/// `f32` values, the BF16 reference's weights.
struct HostNvfp4 {
    packed: Vec<u8>,
    scales: Vec<u8>,
    scale_2: f32,
    dequant: Vec<f32>,
}

fn host_nvfp4(rng: &mut Rng, n: usize, k: usize, scale_2: f32) -> HostNvfp4 {
    let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
    let scales: Vec<u8> = (0..n * k / 16)
        .map(|_| 0x30 + (rng.next() % 9) as u8)
        .collect();
    let dequant = (0..n * k)
        .map(|i| {
            let (r, c) = (i / k, i % k);
            let byte = packed[r * k / 2 + c / 2];
            let code = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
            E2M1[code as usize] * e4m3_to_f32(scales[r * k / 16 + c / 16]) * scale_2
        })
        .collect();
    HostNvfp4 {
        packed,
        scales,
        scale_2,
        dequant,
    }
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

fn proj(gpu: &dyn GpuBackend, w: &HostNvfp4, input_scale: f32) -> Result<Nvfp4Proj> {
    Ok(Nvfp4Proj {
        packed: upload(gpu, &w.packed)?,
        scale: upload(gpu, &w.scales)?,
        scale_2: w.scale_2,
        input_scale: Some(input_scale),
    })
}

/// 2026-10-08: Cosine similarity and relative L2 error of `got` against `want`.
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

fn plan(group: MlpGroup, stamp: Nvfp4Act, rows: usize) -> Result<GroupPrecision> {
    let family = match group {
        MlpGroup::RoutedExperts => ProjFamily::Moe,
        MlpGroup::DenseMlp => ProjFamily::Ffn,
    };
    GroupPrecision::resolve(
        group,
        ActivationQuantization::default().ladder(family).clone(),
        stamp,
        rows,
        true,
    )
}

/// 2026-10-08: Routed site, W4A4 against W4A16 on the same weights and the same router, at 1,
/// 3, 8 and 16 rows; then every row of the 16-row W4A4 run bit-identical to that row alone.
#[test]
#[ignore = "requires an explicitly selected idle CUDA device and the glm-5.3-flash nvfp4 kernels"]
fn routed_w4a4_tracks_w4a16_and_is_row_invariant() -> Result<()> {
    let gpu = gpu()?;
    let (c, k) = (cfg(), Glm5NextMlpKernels::resolve(&gpu)?);
    ensure!(
        k.w4a4_expert_rows() >= 16,
        "the W4A4 slot kernels did not resolve"
    );
    let stream = gpu.create_stream()?;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let x: Vec<f32> = (0..16 * c.hidden).map(|_| rng.unit() * 3.0).collect();
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    // 2026-10-08: The down input's scale is the checkpoint's: the SwiGLU clamp bound 10 x 10.
    let scales = W4a4ActScales {
        gate_up: modelopt_scale(amax),
        down: modelopt_scale(100.0),
    };
    let experts: Vec<Glm5NextExpertWeights> = (0..c.num_experts)
        .map(|_| -> Result<_> {
            let gu = |r: &mut Rng| host_nvfp4(r, c.moe_intermediate, c.hidden, 0.01);
            Ok(Glm5NextExpertWeights {
                gate_proj: proj(&gpu, &gu(&mut rng), scales.gate_up)?,
                up_proj: proj(&gpu, &gu(&mut rng), scales.gate_up)?,
                down_proj: proj(
                    &gpu,
                    &host_nvfp4(&mut rng, c.hidden, c.moe_intermediate, 0.01),
                    scales.down,
                )?,
            })
        })
        .collect::<Result<_>>()?;
    let router: Vec<f32> = (0..c.num_experts * c.hidden)
        .map(|_| rng.unit() * 0.05)
        .collect();
    let shared: Vec<f32> = (0..c.hidden * c.local_shared_intermediate)
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
            &gpu,
            &c,
            c.local_shared_intermediate,
            &load,
            &expert,
            &|_| plan(MlpGroup::RoutedExperts, stamp, k.w4a4_expert_rows()),
            16,
        )
    };
    let (w4a4, w4a16) = (build(Nvfp4Act::A4)?, build(Nvfp4Act::Unstamped)?);
    let ws = Glm5NextMlpWorkspace::new(&gpu, &c, 16)?;
    let xd = upload(&gpu, &bf16_bytes(&x))?;
    let out = gpu.alloc(16 * c.hidden * 2)?;
    let run = |w: &Glm5NextMoeWeights, rows: usize| -> Result<Vec<f32>> {
        forward_moe(&gpu, &k, &c, w, xd, out, rows, &ws, stream)?;
        gpu.synchronize(stream)?;
        read_bf16(&gpu, out, rows * c.hidden)
    };
    for rows in [1usize, 3, 8, 16] {
        let (a, b) = (run(&w4a4, rows)?, run(&w4a16, rows)?);
        let (cos, rel) = agreement(&a, &b);
        println!("routed rows={rows}: cosine {cos:.5}, rel L2 {rel:.4}");
        ensure!(
            cos >= MIN_COSINE && rel <= MAX_REL_L2,
            "routed rows={rows}: {cos} / {rel}"
        );
    }
    let wide = run(&w4a4, 16)?;
    for r in 0..16 {
        let xr = xd.offset(r * c.hidden * 2);
        forward_moe(&gpu, &k, &c, &w4a4, xr, out, 1, &ws, stream)?;
        gpu.synchronize(stream)?;
        let one = read_bf16(&gpu, out, c.hidden)?;
        ensure!(
            one.iter()
                .map(|v| v.to_bits())
                .eq(wide[r * c.hidden..(r + 1) * c.hidden]
                    .iter()
                    .map(|v| v.to_bits())),
            "routed row {r}: bits differ between 1 and 16 rows"
        );
    }
    Ok(())
}

/// 2026-10-08: Dense site, W4A4 against BF16 weights dequantized from the same NVFP4 bytes, at
/// 1, 5 and 40 rows (40 is a 32-row and an 8-row chunk); row 37 bit-identical alone.
#[test]
#[ignore = "requires an explicitly selected idle CUDA device and the glm-5.3-flash nvfp4 kernels"]
fn dense_w4a4_tracks_bf16_and_is_row_invariant() -> Result<()> {
    let gpu = gpu()?;
    let (c, k) = (cfg(), Glm5NextMlpKernels::resolve(&gpu)?);
    let (h, i) = (c.hidden, c.local_dense_intermediate);
    let stream = gpu.create_stream()?;
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let x: Vec<f32> = (0..40 * h).map(|_| rng.unit() * 3.0).collect();
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let scales = W4a4ActScales {
        gate_up: modelopt_scale(amax),
        down: modelopt_scale(100.0),
    };
    let (g, u, d) = (
        host_nvfp4(&mut rng, i, h, 0.01),
        host_nvfp4(&mut rng, i, h, 0.01),
        host_nvfp4(&mut rng, h, i, 0.01),
    );
    let nvfp4 = Glm5NextDenseNvfp4Weights {
        gate_proj: proj(&gpu, &g, scales.gate_up)?,
        up_proj: proj(&gpu, &u, scales.gate_up)?,
        down_proj: proj(&gpu, &d, scales.down)?,
    };
    let bf16w = Glm5NextDenseMlpWeights {
        gate_proj: upload(&gpu, &bf16_bytes(&g.dequant))?,
        up_proj: upload(&gpu, &bf16_bytes(&u.dequant))?,
        down_proj: upload(&gpu, &bf16_bytes(&d.dequant))?,
    };
    let site = |stamp: Nvfp4Act| -> Result<Glm5NextDenseSite> {
        Ok(Glm5NextDenseSite {
            bf16: Some(bf16w),
            nvfp4: Some((nvfp4, scales)),
            precision: plan(MlpGroup::DenseMlp, stamp, k.w4a4_dense_rows())?,
        })
    };
    let (w4a4, wide16) = (site(Nvfp4Act::A4)?, site(Nvfp4Act::Unstamped)?);
    let ws = Glm5NextMlpWorkspace::new(&gpu, &c, 40)?;
    let xd = upload(&gpu, &bf16_bytes(&x))?;
    let out = gpu.alloc(40 * h * 2)?;
    let run = |s: &Glm5NextDenseSite, x: DevicePtr, rows: usize| -> Result<Vec<f32>> {
        forward_dense_site(&gpu, &k, &c, s, i, x, out, rows, &ws, stream)?;
        gpu.synchronize(stream)?;
        read_bf16(&gpu, out, rows * h)
    };
    for rows in [1usize, 5, 40] {
        let (a, b) = (run(&w4a4, xd, rows)?, run(&wide16, xd, rows)?);
        let (cos, rel) = agreement(&a, &b);
        println!("dense rows={rows}: cosine {cos:.5}, rel L2 {rel:.4}");
        ensure!(
            cos >= MIN_COSINE && rel <= MAX_REL_L2,
            "dense rows={rows}: {cos} / {rel}"
        );
    }
    let all = run(&w4a4, xd, 40)?;
    let one = run(&w4a4, xd.offset(37 * h * 2), 1)?;
    ensure!(
        one.iter()
            .map(|v| v.to_bits())
            .eq(all[37 * h..38 * h].iter().map(|v| v.to_bits())),
        "dense row 37: bits differ between 1 and 40 rows"
    );
    Ok(())
}

/// 2026-10-08: `w4a4_quant_rows_static` given the scale `w4a4_quant_rows` computes for a row,
/// amax(row) / (6 * 448), writes the same codes, block scales and global: the static entry is
/// the dynamic one with only the global's source changed.
#[test]
#[ignore = "requires an explicitly selected idle CUDA device and the glm-5.3-flash nvfp4 kernels"]
fn static_quantizer_matches_the_dynamic_one_at_the_same_scale() -> Result<()> {
    let gpu = gpu()?;
    let dynamic = gpu.kernel("w4a4_gemv_mx", "w4a4_quant_rows")?;
    let fixed = gpu.kernel("w4a4_gemv_mx_moe", "w4a4_quant_rows_static")?;
    let stream = gpu.create_stream()?;
    let k = 4096usize;
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    for row in 0..8 {
        // 2026-10-08: Rows of very different amplitude, one with a single outlier.
        let amp = [1e-3f32, 0.05, 1.0, 7.0, 300.0, 1.0, 0.2, 40.0][row];
        let mut x: Vec<f32> = (0..k).map(|_| rng.unit() * amp).collect();
        if row == 5 {
            x[777] = 90.0;
        }
        let x: Vec<f32> = x.iter().map(|v| bf16::from_f32(*v).to_f32()).collect();
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let xd = upload(&gpu, &bf16_bytes(&x))?;
        let mut outs = Vec::new();
        for (kern, gs) in [(dynamic, None), (fixed, Some(modelopt_scale(amax)))] {
            let (aq, as_, ag) = (gpu.alloc(k / 2)?, gpu.alloc(k / 16)?, gpu.alloc(4)?);
            let launch = KernelLaunch::new(&gpu, kern)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(xd)
                .arg_ptr(aq)
                .arg_ptr(as_)
                .arg_ptr(ag)
                .arg_u32(k as u32);
            match gs {
                Some(gs) => launch.arg_f32(gs).launch(stream)?,
                None => launch.launch(stream)?,
            }
            gpu.synchronize(stream)?;
            let mut b = vec![0u8; k / 2 + k / 16 + 4];
            gpu.copy_d2h(aq, &mut b[..k / 2])?;
            gpu.copy_d2h(as_, &mut b[k / 2..k / 2 + k / 16])?;
            gpu.copy_d2h(ag, &mut b[k / 2 + k / 16..])?;
            outs.push(b);
        }
        ensure!(
            outs[0] == outs[1],
            "row {row} (amplitude {amp}): static and dynamic differ"
        );
    }
    Ok(())
}
