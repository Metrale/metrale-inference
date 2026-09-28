// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: The routed-expert prefill GEMMs at Qwen3.6-35B-A3B shapes (256 experts,
//! top-8, hidden 2048, expert intermediate 512): `moe_w8a8_grouped_gemm_e4m3_gu` / `_dn`
//! against `moe_w8a8_grouped_gemm_pm4`, over one routing and one set of operands.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every token count and routing, each e4m3 output equals
//!   PM4's byte for byte on every live row, a second e4m3 launch on a 7-CTA grid writes the
//!   same bytes (the result does not depend on how items map to CTAs), and the guard band
//!   past the live rows still holds the sentinel.
//!
//! Operands use every non-NaN E4M3 code (the full exponent range, subnormals included),
//! which is where a different in-MMA summation grouping would show. Gate/up gathers rows
//! through `sorted_token_ids`; down reads sorted rows directly (NULL ids), as in the
//! prefill. Mean times over 10 launches are printed.
//!
//! Run (GB10): cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!   --example moe_w8a8_e4m3_grouped_microtest

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::{self, MoeE4m3Tile};
use std::time::Instant;

const E: usize = 256;
const TOP_K: usize = 8;
const H: usize = 2048;
const INTER: usize = 512;
const PM4_M_TILE: u32 = 128;
const PM4_N_TILE: u32 = 64;
const GUARD_ROWS: usize = 2;
const SENTINEL: u8 = 0x5a;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// 2026-09-27: A uniformly drawn non-NaN E4M3 code (NaN is 0x7F / 0xFF).
    fn e4m3(&mut self) -> u8 {
        loop {
            let c = (self.next() >> 8) as u8;
            if c & 0x7f != 0x7f {
                return c;
            }
        }
    }
    fn scale(&mut self) -> f32 {
        ((self.next() % 64 + 1) as f32) / 4096.0
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 2026-09-27: Per-expert `[n, k]` FP8 weights and `[n/128, k/128]` scales, as device
/// tables of per-expert pointers.
fn expert_table(
    gpu: &dyn GpuBackend,
    rng: &mut Rng,
    n: usize,
    k: usize,
) -> Result<(DevicePtr, DevicePtr)> {
    let (mut wp, mut sp) = (Vec::new(), Vec::new());
    for _ in 0..E {
        let w: Vec<u8> = (0..n * k).map(|_| rng.e4m3()).collect();
        let s: Vec<f32> = (0..(n / 128) * (k / 128)).map(|_| rng.scale()).collect();
        wp.extend_from_slice(&upload(gpu, &w)?.0.to_le_bytes());
        sp.extend_from_slice(&upload(gpu, &f32s(&s))?.0.to_le_bytes());
    }
    Ok((upload(gpu, &wp)?, upload(gpu, &sp)?))
}

struct Case<'a> {
    name: &'a str,
    a: DevicePtr,
    a_scale: DevicePtr,
    table: (DevicePtr, DevicePtr),
    sorted: DevicePtr,
    n: u32,
    k: u32,
    kernel: KernelHandle,
    tile: MoeE4m3Tile,
}

/// 2026-09-27: One PM4 launch and two e4m3 launches (full and 7-CTA grids) into
/// sentinel-filled outputs; returns an error on any byte difference.
#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &dyn GpuBackend,
    c: &Case,
    build: KernelHandle,
    pm4: KernelHandle,
    offsets: DevicePtr,
    te: usize,
    wl: DevicePtr,
    tt: DevicePtr,
    sms: u32,
) -> Result<()> {
    let out_bytes = (te + GUARD_ROWS) * c.n as usize * 2;
    let outs: Vec<DevicePtr> = (0..3)
        .map(|_| gpu.alloc(out_bytes))
        .collect::<Result<_>>()?;
    for &o in &outs {
        gpu.memset(o, SENTINEL, out_bytes)?;
    }
    let pm4_launch = |out: DevicePtr| -> Result<()> {
        ops::moe_build_tile_worklist(
            gpu,
            build,
            offsets,
            c.table.0,
            wl,
            tt,
            E as u32,
            c.n / PM4_N_TILE,
            PM4_M_TILE,
            0,
        )?;
        ops::moe_w8a8_grouped_gemm_pm4(
            gpu, pm4, c.a, c.a_scale, c.table.0, c.table.1, out, offsets, c.sorted, E as u32, c.n,
            c.k, wl, tt, 16384, 0,
        )
    };
    let e4m3_launch = |out: DevicePtr, grid: u32| -> Result<()> {
        ops::moe_build_tile_worklist(
            gpu,
            build,
            offsets,
            c.table.0,
            wl,
            tt,
            E as u32,
            c.n / c.tile.n_tile,
            c.tile.m_tile,
            0,
        )?;
        ops::moe_w8a8_grouped_gemm_e4m3(
            gpu, c.kernel, c.tile, c.a, c.a_scale, c.table.0, c.table.1, out, offsets, c.sorted,
            c.n, c.k, wl, tt, grid, 0,
        )
    };
    pm4_launch(outs[0])?;
    e4m3_launch(outs[1], sms * c.tile.ctas_per_sm)?;
    e4m3_launch(outs[2], 7)?;
    gpu.synchronize(0)?;
    let host: Vec<Vec<u8>> = outs
        .iter()
        .map(|&o| {
            let mut v = vec![0u8; out_bytes];
            gpu.copy_d2h(o, &mut v).map(|_| v)
        })
        .collect::<Result<_>>()?;
    let live = te * c.n as usize * 2;
    let diff = host[0][..live]
        .chunks_exact(2)
        .zip(host[1][..live].chunks_exact(2))
        .filter(|(x, y)| x != y)
        .count();
    ensure!(
        diff == 0,
        "{}: {diff} of {} outputs differ from PM4",
        c.name,
        live / 2
    );
    ensure!(
        host[1] == host[2],
        "{}: e4m3 output depends on the grid",
        c.name
    );
    ensure!(
        host.iter()
            .all(|h| h[live..].iter().all(|&b| b == SENTINEL)),
        "{}: write past the live rows",
        c.name
    );
    let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        gpu.synchronize(0)?;
        let t = Instant::now();
        for _ in 0..10 {
            f()?;
        }
        gpu.synchronize(0)?;
        Ok(t.elapsed().as_secs_f64() * 1e2)
    };
    let t_pm4 = time(&|| pm4_launch(outs[0]))?;
    let t_new = time(&|| e4m3_launch(outs[1], sms * c.tile.ctas_per_sm))?;
    println!(
        "  {:<8} bit-identical ({} values); PM4 {t_pm4:.3} ms, e4m3 {t_new:.3} ms ({:.2}x)",
        c.name,
        live / 2,
        t_pm4 / t_new
    );
    for o in outs {
        gpu.free(o)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let build = gpu.kernel("moe", "moe_build_tile_worklist")?;
    let pm4 = gpu.kernel("moe_w8a8_grouped_gemm", "moe_w8a8_grouped_gemm_pm4")?;
    let gu = gpu.kernel(
        "moe_w8a8_grouped_gemm_e4m3",
        "moe_w8a8_grouped_gemm_e4m3_gu",
    )?;
    let dn = gpu.kernel(
        "moe_w8a8_grouped_gemm_e4m3",
        "moe_w8a8_grouped_gemm_e4m3_dn",
    )?;
    let sms = gpu.sm_count()?;
    let mut rng = Rng(0x6534_6d33_2026_0927);
    let gate = expert_table(&gpu, &mut rng, INTER, H)?;
    let down = expert_table(&gpu, &mut rng, H, INTER)?;
    for (tokens, skewed) in [(300usize, false), (1100, true), (8200, false)] {
        // 2026-09-27: Routing: top-8 distinct experts per token, uniform or with every third
        // expert drawn 4x as often; rows sorted by expert as moe_sort_by_expert leaves them.
        let mut rows: Vec<Vec<i32>> = vec![Vec::new(); E];
        for t in 0..tokens {
            let mut chosen: Vec<usize> = Vec::new();
            while chosen.len() < TOP_K {
                let r = rng.next() as usize % (E + if skewed { 3 * (E / 3) } else { 0 });
                let e = if r < E { r } else { 3 * (r - E) % E };
                if !chosen.contains(&e) {
                    chosen.push(e);
                }
            }
            chosen.iter().for_each(|&e| rows[e].push(t as i32));
        }
        let mut offsets = vec![0i32];
        let sorted: Vec<i32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        rows.iter()
            .for_each(|r| offsets.push(offsets.last().copied().unwrap_or(0) + r.len() as i32));
        let te = tokens * TOP_K;
        let i32s = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let offsets_d = upload(&gpu, &i32s(&offsets))?;
        let sorted_d = upload(&gpu, &i32s(&sorted))?;
        let act = |rows: usize, k: usize, rng: &mut Rng| -> Result<(DevicePtr, DevicePtr)> {
            let a: Vec<u8> = (0..rows * k).map(|_| rng.e4m3()).collect();
            let s: Vec<f32> = (0..rows * k / 128).map(|_| rng.scale()).collect();
            Ok((upload(&gpu, &a)?, upload(&gpu, &f32s(&s))?))
        };
        let (a_gu, s_gu) = act(tokens, H, &mut rng)?;
        let (a_dn, s_dn) = act(te, INTER, &mut rng)?;
        let wl_items = (te.div_ceil(64) + E + 1) * (H / 64);
        let wl = gpu.alloc(wl_items * 8)?;
        let tt = gpu.alloc(16)?;
        println!(
            "tokens {tokens} ({} routing):",
            if skewed { "skewed" } else { "uniform" }
        );
        let cases = [
            Case {
                name: "gate/up",
                a: a_gu,
                a_scale: s_gu,
                table: gate,
                sorted: sorted_d,
                n: INTER as u32,
                k: H as u32,
                kernel: gu,
                tile: ops::MOE_E4M3_GU,
            },
            Case {
                name: "down",
                a: a_dn,
                a_scale: s_dn,
                table: down,
                sorted: DevicePtr::NULL,
                n: H as u32,
                k: INTER as u32,
                kernel: dn,
                tile: ops::MOE_E4M3_DN,
            },
        ];
        for c in &cases {
            run_case(&gpu, c, build, pm4, offsets_d, te, wl, tt, sms)?;
        }
        for p in [offsets_d, sorted_d, a_gu, s_gu, a_dn, s_dn, wl, tt] {
            gpu.free(p)?;
        }
    }
    println!("PASS: e4m3 grouped GEMMs are bit-identical to PM4 and grid-independent");
    Ok(())
}
