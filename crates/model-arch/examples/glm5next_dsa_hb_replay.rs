// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Replays one row dumped by `METRALE_GLM_DSA_DECODE_HB_CHECK` (a `case_<n>`
//! directory): rebuilds its query, selection and selected latents in a fresh paged cache, runs
//! the per-head and the head-batched DSA decode, and prints each head's difference from an f64
//! reference and between the kernels, next to the outputs the serve recorded.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: none beyond the types.
//!
//!   HB_CASE=<dir>/case_0 [HB_SPLITS=8] METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash \
//!   METRALE_TARGET_QUANT=nvfp4 cargo run -p metrale-model-arch --release \
//!       --example glm5next_dsa_hb_replay --features cuda,gpu-examples

use anyhow::{Context, Result};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const BLOCK: usize = 64;

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}
fn e4m3(b: u8) -> f64 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f64;
    s * if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

fn main() -> Result<()> {
    let case = std::env::var("HB_CASE").context("HB_CASE names a case_<n> directory")?;
    let splits: u32 = std::env::var("HB_SPLITS").map_or(Ok(8), |v| v.parse())?;
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{case}/meta.json"))?)?;
    let get = |k: &str| meta[k].as_u64().with_context(|| format!("meta {k}"));
    let (heads, kvl, w) = (
        get("heads")? as usize,
        get("kv_lora")? as usize,
        get("out_width")? as usize,
    );
    let seq_len = meta["seq_len"].as_i64().context("seq_len")? as usize;
    let (ks, vs) = (
        meta["k_scale"].as_f64().context("k_scale")?,
        meta["v_scale"].as_f64().context("v_scale")?,
    );
    let q = std::fs::read(format!("{case}/q.bin"))?;
    let sel: Vec<i32> = std::fs::read(format!("{case}/sel.bin"))?
        .chunks(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let tokens = std::fs::read(format!("{case}/tokens.bin"))?;
    let rec_hb = bf16s(&std::fs::read(format!("{case}/out_hb.bin"))?);
    let rec_ref = bf16s(&std::fs::read(format!("{case}/out_ref.bin"))?);

    // 2026-10-09: Token t at slot t of an identity-mapped paged cache.
    let blocks = seq_len.div_ceil(BLOCK).max(1);
    let mut cache = vec![0u8; blocks * BLOCK * kvl];
    for (j, &t) in sel.iter().enumerate() {
        if t >= 0 && (t as usize) < seq_len {
            cache[t as usize * kvl..][..kvl].copy_from_slice(&tokens[j * kvl..][..kvl]);
        }
    }
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm target")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let i32b = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let (qd, cd) = (up(g, &q)?, up(g, &cache)?);
    let bt = up(g, &i32b(&(0..blocks as i32).collect::<Vec<_>>()))?;
    let sl = up(g, &i32b(&[seq_len as i32]))?;
    let sd = up(g, &i32b(&sel))?;
    let n = heads * kvl * 2;
    let (o_old, o_hb) = (g.alloc(n)?, g.alloc(n)?);
    let inv = (kvl as f32).powf(-0.5);
    KernelLaunch::new(
        g,
        g.kernel("glm5next_dsa_mla_decode", "glm5next_dsa_mla_decode_fp8")?,
    )
    .grid([heads as u32, 1, 1])
    .block([512, 1, 1])
    .arg_ptr(qd)
    .arg_ptr(cd)
    .arg_ptr(cd)
    .arg_ptr(o_old)
    .arg_ptr(bt)
    .arg_ptr(sl)
    .arg_ptr(sd)
    .arg_u32(w as u32)
    .arg_u32(blocks as u32)
    .arg_u32(heads as u32)
    .arg_u32(1)
    .arg_u32(kvl as u32)
    .arg_u32(BLOCK as u32)
    .arg_f32(inv)
    .arg_f32(ks as f32)
    .arg_f32(vs as f32)
    .arg_u64((BLOCK * kvl) as u64)
    .launch(0)?;
    let ws = (
        g.alloc(splits as usize * heads * kvl * 4)?,
        g.alloc(splits as usize * heads * 8)?,
    );
    KernelLaunch::new(
        g,
        g.kernel(
            "glm5next_dsa_mla_decode_hb",
            "glm5next_dsa_mla_decode_hb_fp8",
        )?,
    )
    .grid([splits, 1, heads.div_ceil(24) as u32])
    .block([256, 1, 1])
    .arg_ptr(qd)
    .arg_ptr(cd)
    .arg_ptr(o_hb)
    .arg_ptr(bt)
    .arg_ptr(sl)
    .arg_ptr(sd)
    .arg_ptr(ws.0)
    .arg_ptr(ws.1)
    .arg_u32(w as u32)
    .arg_u32(blocks as u32)
    .arg_u32(heads as u32)
    .arg_u32(BLOCK as u32)
    .arg_f32(inv)
    .arg_f32(ks as f32)
    .arg_f32(vs as f32)
    .arg_u64((BLOCK * kvl) as u64)
    .arg_u32(splits)
    .launch(0)?;
    if splits > 1 {
        KernelLaunch::new(
            g,
            g.kernel("glm5next_dsa_mla_decode_hb", "glm5next_dsa_mla_merge")?,
        )
        .grid([heads as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(ws.0)
        .arg_ptr(ws.1)
        .arg_ptr(sl)
        .arg_ptr(o_hb)
        .arg_u32(heads as u32)
        .arg_u32(splits)
        .arg_f32(vs as f32)
        .launch(0)?;
    }
    let old = bf16s(&read(g, o_old, n)?);
    let hb = bf16s(&read(g, o_hb, n)?);
    let qf = bf16s(&q);
    let valid: Vec<usize> = (0..w)
        .filter(|&j| sel[j] >= 0 && (sel[j] as usize) < seq_len)
        .collect();
    println!(
        "{case}: seq_len {seq_len}, {} valid of {w} entries, k_scale {ks}, v_scale {vs}",
        valid.len()
    );
    println!("head | err vs f64 (head-scale ulps): old  hb  rec_ref rec_hb | max|O| | top p");
    for h in 0..heads {
        let qh = &qf[h * kvl..][..kvl];
        let sc: Vec<f64> = valid
            .iter()
            .map(|&j| {
                (0..kvl)
                    .map(|i| qh[i] as f64 * e4m3(tokens[j * kvl + i]) * ks)
                    .sum::<f64>()
                    * (kvl as f64).powf(-0.5)
            })
            .collect();
        let m = sc.iter().fold(f64::MIN, |a, b| a.max(*b));
        let p: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
        let l: f64 = p.iter().sum();
        let o: Vec<f64> = (0..kvl)
            .map(|i| {
                valid
                    .iter()
                    .zip(&p)
                    .map(|(&j, p)| p * e4m3(tokens[j * kvl + i]) * vs)
                    .sum::<f64>()
                    / l
            })
            .collect();
        let ulp = o.iter().fold(0f64, |a, x| a.max(x.abs())).max(1e-30) / 256.0;
        let e = |v: &[f32]| {
            (0..kvl).fold(0f64, |a, i| a.max((v[h * kvl + i] as f64 - o[i]).abs())) / ulp
        };
        let top = p.iter().fold(0f64, |a, b| a.max(*b)) / l;
        println!(
            "{h:4} | {:8.2} {:8.2} {:8.2} {:8.2} | {:9.3e} | {top:.3}",
            e(&old),
            e(&hb),
            e(&rec_ref),
            e(&rec_hb),
            ulp * 256.0
        );
    }
    Ok(())
}
