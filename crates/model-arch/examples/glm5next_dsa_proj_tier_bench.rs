// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The DSA absorbed projections of one TP=3 rank (22 heads) on real GLM-5.3-Flash
//! weights, three ways, against the BF16 product they approximate:
//!
//! * FP8 W8A8 (`glm5next_fp8_dense`, today's `--dense-quantization w4a16` DSA tier),
//! * NVFP4 W4A16 (`glm5next_w4a16_dense`, the `METRALE_GLM_DSA_W4A16=1` choice),
//! * for `o` only, the un-absorbed decode (V up-projection per head, then `o_proj`), timed as
//!   `o_proj` alone on FP8 and W4A16 (the per-head V up-projection reads another
//!   22 x 256 x 512 weights and is not timed).
//!
//! For each: microseconds per call at 1 and 16 rows (median and p10-p90 of 41 samples; calls
//! rotate over [`COPIES`] copies of the weight so it is not L2-resident), and the
//! output error against the BF16 GEMV of the BF16 absorbed weight (RMS relative to the output
//! RMS, and the worst element relative to the output RMS), on random unit-scale inputs.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: none beyond the types.
//!
//!   GLM_CKPT=<checkpoint dir> METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash \
//!   METRALE_TARGET_QUANT=nvfp4 cargo run -p metrale-model-arch --release \
//!       --example glm5next_dsa_proj_tier_bench --features cuda,gpu-examples

use std::io::{Read, Seek, SeekFrom};

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::Glm5NextDsaConfig;
use metrale_model_arch::glm5next_dsa::build::{absorb_o, absorb_q};
use metrale_model_arch::{glm5next_fp8_dense as fp8, glm5next_w4a16_dense as w4};
use metrale_model_layers::layers::ops::{DenseMmKernels, dense_mm_bf16};
use metrale_model_layers::layers::{DenseQuantization, set_dense_quantization_from_cli};

const LAYER: usize = 3;
const HEADS: usize = 22;
const FULL_HEADS: usize = 64;
const SAMPLES: usize = 41;
const PER_SAMPLE: usize = 5;
/// 2026-10-09: Copies of each weight per tier, rotated per call.
const COPIES: usize = 3;

/// 2026-10-09: One BF16 tensor of the checkpoint as f32, read by its header offsets.
fn tensor(dir: &str, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(format!(
        "{dir}/model.safetensors.index.json"
    ))?)?;
    let file = index["weight_map"][name]
        .as_str()
        .with_context(|| format!("{name} not in the index"))?;
    let mut f = std::fs::File::open(format!("{dir}/{file}"))?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let hn = u64::from_le_bytes(n);
    let mut hdr = vec![0u8; hn as usize];
    f.read_exact(&mut hdr)?;
    let hdr: serde_json::Value = serde_json::from_slice(&hdr)?;
    let t = &hdr[name];
    if t["dtype"] != "BF16" {
        bail!("{name}: {} not BF16", t["dtype"]);
    }
    let shape: Vec<usize> = t["shape"]
        .as_array()
        .context("shape")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect();
    let (a, b) = (
        t["data_offsets"][0].as_u64().context("off")?,
        t["data_offsets"][1].as_u64().context("off")?,
    );
    f.seek(SeekFrom::Start(8 + hn + a))?;
    let mut raw = vec![0u8; (b - a) as usize];
    f.read_exact(&mut raw)?;
    Ok((
        shape,
        raw.chunks(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
    ))
}

fn up_bf16(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect())
}

/// 2026-10-09: Microseconds per call, (median, p10, p90) over the samples; `f` gets the call
/// index, so callers rotate weight copies.
fn time(g: &dyn GpuBackend, mut f: impl FnMut(usize) -> Result<()>) -> Result<(f64, f64, f64)> {
    let mut it = 0usize;
    for _ in 0..3 {
        f(it)?;
        it += 1;
    }
    let mut v = Vec::new();
    for _ in 0..SAMPLES {
        g.synchronize(0)?;
        let t = std::time::Instant::now();
        for _ in 0..PER_SAMPLE {
            f(it)?;
            it += 1;
        }
        g.synchronize(0)?;
        v.push(t.elapsed().as_secs_f64() * 1e6 / PER_SAMPLE as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((
        v[SAMPLES / 2],
        v[SAMPLES / 10],
        v[SAMPLES - 1 - SAMPLES / 10],
    ))
}

/// 2026-10-09: (RMS error, worst error), both relative to the reference's RMS.
fn err(out: &[f32], reference: &[f32]) -> (f64, f64) {
    let rms = (reference.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / reference.len() as f64)
        .sqrt();
    let (mut s, mut w) = (0f64, 0f64);
    for (a, b) in out.iter().zip(reference) {
        let d = (*a as f64 - *b as f64).abs();
        s += d * d;
        w = w.max(d);
    }
    ((s / out.len() as f64).sqrt() / rms, w / rms)
}

fn main() -> Result<()> {
    set_dense_quantization_from_cli(DenseQuantization::W4a16);
    let dir = std::env::var("GLM_CKPT").context("GLM_CKPT names the checkpoint directory")?;
    let p = |s: &str| format!("model.language_model.layers.{LAYER}.self_attn.{s}.weight");
    let cfg = Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: HEADS,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 8192,
    };
    let (kvl, ql, vd, hid) = (512usize, 1536usize, 256usize, 4096usize);
    let (_, q_b) = tensor(&dir, &p("q_b_proj"))?;
    let (_, kv_b) = tensor(&dir, &p("kv_b_proj"))?;
    let (_, o_full) = tensor(&dir, &p("o_proj"))?;
    let (_, q_a) = tensor(&dir, &p("q_a_proj"))?;
    // 2026-10-09: Rank 0's heads: the first 22 heads' rows of q_b and kv_b, columns of o_proj.
    let q_b = q_b[..HEADS * 256 * ql].to_vec();
    let kv_b = kv_b[..HEADS * 512 * kvl].to_vec();
    let o_local: Vec<f32> = (0..hid)
        .flat_map(|r| o_full[r * FULL_HEADS * vd..][..HEADS * vd].to_vec())
        .collect();
    let q_abs = absorb_q(&cfg, &q_b, &kv_b, HEADS)?;
    let o_abs = absorb_o(&cfg, &o_local, &kv_b, HEADS)?;

    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm target")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let bf = DenseMmKernels {
        gemm: g.kernel("gemm", "dense_gemm_bf16")?,
        gemv: g.kernel("gemv", "dense_gemv_bf16")?,
        batchm: g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?,
    };
    let quant8 = g.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
    let quant4 = w4::Nvfp4QuantKernels::load(g)?;
    fp8::prepare(g, HEADS * kvl)?;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut unit = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    };
    println!(
        "DSA projections, rank 0 (22 heads), layer {LAYER}: us median (p10-p90) | err rms / worst vs BF16"
    );
    for (what, w, n, k) in [
        ("q_a_proj", &q_a, ql, hid),
        ("q_absorb", &q_abs, HEADS * kvl, ql),
        ("o_absorb", &o_abs, hid, HEADS * kvl),
        ("o_proj (un-absorbed)", &o_local, hid, HEADS * vd),
    ] {
        let x: Vec<f32> = (0..16 * k).map(|_| unit()).collect();
        let xd = up_bf16(g, &x)?;
        // 2026-10-09: COPIES distinct copies of each weight, rotated per call, so a call does
        // not find its weight in L2 from the previous one (a serve runs other layers between).
        let (mut w_ref, mut w8, mut w4key) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..COPIES {
            w_ref.push(up_bf16(g, w)?);
            let w8c = up_bf16(g, w)?;
            fp8::register(g, quant8, w8c, n, k, what, 0)?;
            w8.push(w8c);
            let w4src = up_bf16(g, w)?;
            w4key.push(w4::register(g, &quant4, w4src, n, k, what, 0)?);
            g.free(w4src)?;
        }
        let out = g.alloc(16 * n * 2)?;
        for m in [1usize, 16] {
            dense_mm_bf16(g, &bf, xd, w_ref[0], out, m, n, k, 0)?;
            let r = down_bf16(g, out, m * n)?;
            let tb = time(g, |i| {
                dense_mm_bf16(g, &bf, xd, w_ref[i % COPIES], out, m, n, k, 0)
            })?;
            let t8 = time(g, |i| {
                fp8::proj(g, w8[i % COPIES], xd, out, m, n, k, 0).map(|_| ())
            })?;
            let e8 = err(&down_bf16(g, out, m * n)?, &r);
            let t4 = time(g, |i| {
                w4::proj(g, w4key[i % COPIES], xd, out, m, n, k, 0).map(|_| ())
            })?;
            let e4 = err(&down_bf16(g, out, m * n)?, &r);
            println!(
                "  {what:21} m={m:2} BF16 {:6.1} | FP8 {:6.1} ({:.1}-{:.1}) {:.4}/{:.4} | W4A16 {:6.1} ({:.1}-{:.1}) {:.4}/{:.4}",
                tb.0, t8.0, t8.1, t8.2, e8.0, e8.1, t4.0, t4.1, t4.2, e4.0, e4.1
            );
        }
        for p in w_ref {
            g.free(p)?;
        }
    }
    Ok(())
}
