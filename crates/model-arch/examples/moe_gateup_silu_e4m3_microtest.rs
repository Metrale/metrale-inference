// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `moe_w8a8_gateup_silu_e4m3_w1` / `_w2` (gate and up grouped GEMMs + SiLU(gate)
//! * up + E4M3 group quant in one kernel) against the chain they replace: two
//! `moe_w8a8_grouped_gemm_e4m3_gu` launches, then `silu_mul_quant_fp8`, at Qwen3.6-35B-A3B
//! shapes (256 experts, top-8, hidden 2048, expert intermediate 512).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for both entries at every token count, the E4M3 bytes and the
//!   FP32 scales equal the chain's byte for byte.
//!
//! Operands use every non-NaN E4M3 code. Mean times over 10 launches are printed.
//!
//! Run (GB10): cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!   --example moe_gateup_silu_e4m3_microtest

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use std::time::Instant;

const E: usize = 256;
const TOP_K: usize = 8;
const H: usize = 2048;
const INTER: usize = 512;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn e4m3(&mut self) -> u8 {
        loop {
            let c = (self.next() >> 8) as u8;
            if c & 0x7f != 0x7f {
                return c;
            }
        }
    }
    fn scales(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|_| (((self.next() % 64 + 1) as f32) / 4096.0).to_le_bytes())
            .collect()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    gpu.copy_d2h(p, &mut v)?;
    Ok(v)
}

/// 2026-09-28: Per-expert `[INTER, H]` weights and scales as device pointer tables.
fn table(gpu: &dyn GpuBackend, rng: &mut Rng) -> Result<(DevicePtr, DevicePtr)> {
    let (mut wp, mut sp) = (Vec::new(), Vec::new());
    for _ in 0..E {
        let w: Vec<u8> = (0..INTER * H).map(|_| rng.e4m3()).collect();
        wp.extend_from_slice(&upload(gpu, &w)?.0.to_le_bytes());
        let s = rng.scales((INTER / 128) * (H / 128));
        sp.extend_from_slice(&upload(gpu, &s)?.0.to_le_bytes());
    }
    Ok((upload(gpu, &wp)?, upload(gpu, &sp)?))
}

fn main() -> Result<()> {
    // 2026-09-28: The Qwen3.6-35B-A3B target: `ptx_modules()` aliases the first compiled target,
    // whose `moe_silu_mul` shadow may lack `silu_mul_quant_fp8`.
    let sets = metrale_kernels::all_ptx_sets();
    let set = sets
        .iter()
        .find(|s| s.target.model == "qwen3.6-35b-a3b")
        .context("qwen3.6-35b-a3b kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &set.modules)?;
    let module = "moe_w8a8_grouped_gemm_e4m3";
    let gu = gpu.kernel(module, "moe_w8a8_grouped_gemm_e4m3_gu")?;
    let fused = [
        gpu.kernel(module, "moe_w8a8_gateup_silu_e4m3_w1")?,
        gpu.kernel(module, "moe_w8a8_gateup_silu_e4m3_w2")?,
    ];
    let build = gpu.kernel("moe", "moe_build_tile_worklist")?;
    let silu = gpu.kernel("moe_silu_mul", "silu_mul_quant_fp8")?;
    let sms = gpu.sm_count()?;
    let mut rng = Rng(0x6775_7369_2026_0928);
    let (gate, up) = (table(&gpu, &mut rng)?, table(&gpu, &mut rng)?);
    let tile = ops::MOE_E4M3_GU;
    for tokens in [300usize, 4100] {
        let te = tokens * TOP_K;
        let mut rows: Vec<Vec<i32>> = vec![Vec::new(); E];
        for t in 0..tokens {
            let mut chosen: Vec<usize> = Vec::new();
            while chosen.len() < TOP_K {
                let e = rng.next() as usize % E;
                if !chosen.contains(&e) {
                    chosen.push(e);
                }
            }
            chosen.iter().for_each(|&e| rows[e].push(t as i32));
        }
        let i32s = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let mut offsets = vec![0i32];
        rows.iter()
            .for_each(|r| offsets.push(offsets.last().copied().unwrap_or(0) + r.len() as i32));
        let sorted: Vec<i32> = rows.concat();
        let (off_d, sorted_d) = (
            upload(&gpu, &i32s(&offsets))?,
            upload(&gpu, &i32s(&sorted))?,
        );
        let a: Vec<u8> = (0..tokens * H).map(|_| rng.e4m3()).collect();
        let (a_d, as_d) = (
            upload(&gpu, &a)?,
            upload(&gpu, &rng.scales(tokens * H / 128))?,
        );
        let items = (te.div_ceil(tile.m_tile as usize) + E + 1) * (INTER / 128);
        let (wl, tt) = (gpu.alloc(items * 8)?, gpu.alloc(16)?);
        let (go, uo) = (gpu.alloc(te * INTER * 2)?, gpu.alloc(te * INTER * 2)?);
        let (q_ref, s_ref) = (gpu.alloc(te * INTER)?, gpu.alloc(te * INTER / 128 * 4)?);
        let (q_new, s_new) = (gpu.alloc(te * INTER)?, gpu.alloc(te * INTER / 128 * 4)?);
        let worklist = || {
            ops::moe_build_tile_worklist(
                &gpu,
                build,
                off_d,
                gate.0,
                wl,
                tt,
                E as u32,
                (INTER / 128) as u32,
                tile.m_tile,
                0,
            )
        };
        let chain = || -> Result<()> {
            worklist()?;
            for (t, out) in [(gate, go), (up, uo)] {
                ops::moe_w8a8_grouped_gemm_e4m3(
                    &gpu,
                    gu,
                    tile,
                    a_d,
                    as_d,
                    t.0,
                    t.1,
                    out,
                    off_d,
                    sorted_d,
                    INTER as u32,
                    H as u32,
                    wl,
                    tt,
                    sms * 2,
                    0,
                )?;
            }
            ops::silu_mul_quant_fp8(
                &gpu,
                silu,
                go,
                uo,
                q_ref,
                s_ref,
                DevicePtr::NULL,
                te as u32,
                INTER as u32,
                0,
            )
        };
        chain()?;
        gpu.synchronize(0)?;
        let (qr, sr) = (
            read(&gpu, q_ref, te * INTER)?,
            read(&gpu, s_ref, te * INTER / 32)?,
        );
        let t_chain = {
            let t = Instant::now();
            for _ in 0..10 {
                chain()?;
            }
            gpu.synchronize(0)?;
            t.elapsed().as_secs_f64() * 1e2
        };
        for (i, &k) in fused.iter().enumerate() {
            gpu.memset(q_new, 0x5a, te * INTER)?;
            gpu.memset(s_new, 0x5a, te * INTER / 32)?;
            let run = || -> Result<()> {
                worklist()?;
                ops::moe_w8a8_gateup_silu_e4m3(
                    &gpu,
                    k,
                    a_d,
                    as_d,
                    gate,
                    up,
                    q_new,
                    s_new,
                    off_d,
                    sorted_d,
                    INTER as u32,
                    H as u32,
                    wl,
                    tt,
                    sms * (i as u32 + 1),
                    0,
                )
            };
            run()?;
            gpu.synchronize(0)?;
            ensure!(
                read(&gpu, q_new, te * INTER)? == qr,
                "tokens {tokens} w{}: E4M3 bytes differ",
                i + 1
            );
            ensure!(
                read(&gpu, s_new, te * INTER / 32)? == sr,
                "tokens {tokens} w{}: scales differ",
                i + 1
            );
            let t = Instant::now();
            for _ in 0..10 {
                run()?;
            }
            gpu.synchronize(0)?;
            let t_new = t.elapsed().as_secs_f64() * 1e2;
            println!(
                "tokens {tokens} w{}: bit-identical; chain {t_chain:.3} ms, fused {t_new:.3} ms ({:.2}x)",
                i + 1,
                t_chain / t_new
            );
        }
        for p in [
            off_d, sorted_d, a_d, as_d, wl, tt, go, uo, q_ref, s_ref, q_new, s_new,
        ] {
            gpu.free(p)?;
        }
    }
    println!("PASS: fused gate/up + SiLU + quant equals the unfused chain");
    Ok(())
}
