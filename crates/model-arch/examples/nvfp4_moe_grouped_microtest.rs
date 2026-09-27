// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: The grouped NVFP4 MoE decode (`moe_sort_by_expert`, `moe_fp8_grouped_compact`,
//! `moe_expert_{gate_up,down}_act_nvfp4_grouped`, `moe_weighted_sum_blend_fp8_grouped`) at
//! Qwen3.6-35B-A3B shapes.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exits 1 unless (a) for every M, each row's blended output bytes equal that row's bytes at
//!   M = MAX_M (the same routing and input), (b) the gate+up SiLU products and down outputs
//!   at M = 4 agree with an f64 host reference within a relative 2 % of the row's largest
//!   value, and (c) the invariance oracle is live: comparing each row against its neighbour's
//!   bytes finds a difference.
//!
//! Row t routes to the same experts at every M (the first M rows of one MAX_M-row draw), so
//! only the number of rows sharing the launch changes. Weights are random NVFP4 (packed E2M1,
//! E4M3 block scales of 16, per-tensor scale 2). Mean times of the two expert kernels over 20
//! launches are printed per M.
//!
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example nvfp4_moe_grouped_microtest -- [experts] [zipf_alpha]
//!
//! Arguments: routed experts (default 32, so rows share experts; 256 is the model's) and a
//! Zipf exponent for the routing (default 0.9).

use anyhow::{Context, Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::QuantizedWeight;

const H: usize = 2048;
const INTER: usize = 512;
const TOP_K: usize = 8;
const MAX_M: usize = 64;
const WIDTHS: [usize; 10] = [1, 2, 3, 4, 5, 8, 13, 16, 32, 63];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f64 {
        self.next() as f64 / u32::MAX as f64
    }
}

fn arg<T: std::str::FromStr>(i: usize, default: T) -> T {
    std::env::args()
        .nth(i)
        .and_then(|a| a.parse().ok())
        .unwrap_or(default)
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

const E2M1: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// 2026-09-27: E4M3 byte to value (no NaN bytes are drawn).
fn e4m3(b: u8) -> f64 {
    let (s, e, m) = ((b >> 7) & 1, (b >> 3) & 0xF, b & 7);
    let v = if e == 0 {
        m as f64 / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7)
    };
    if s == 1 { -v } else { v }
}

/// 2026-09-27: One NVFP4 matrix `[n, k]` on host and device.
struct Mat {
    packed: Vec<u8>,
    scale: Vec<u8>,
    s2: f32,
    w: QuantizedWeight,
    k: usize,
}

impl Mat {
    fn new(g: &dyn GpuBackend, rng: &mut Rng, n: usize, k: usize) -> Result<Self> {
        let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
        // 2026-09-27: Scale bytes 0x28..0x3f: E4M3 values 1/32 .. 15/16.
        let scale: Vec<u8> = (0..n * k / 16)
            .map(|_| 0x28 + (rng.next() % 24) as u8)
            .collect();
        let s2 = 0.02 + 0.01 * rng.unit() as f32;
        let mut w = QuantizedWeight::null();
        w.weight = upload(g, &packed)?;
        w.weight_scale = upload(g, &scale)?;
        w.weight_scale_2 = s2;
        Ok(Self {
            packed,
            scale,
            s2,
            w,
            k,
        })
    }
    fn at(&self, row: usize, col: usize) -> f64 {
        let byte = self.packed[(row * self.k + col) / 2];
        let nib = if col % 2 == 0 { byte & 0xF } else { byte >> 4 };
        E2M1[nib as usize] * e4m3(self.scale[(row * self.k + col) / 16]) * self.s2 as f64
    }
    fn dot(&self, row: usize, x: &[f64]) -> f64 {
        (0..self.k).map(|c| self.at(row, c) * x[c]).sum()
    }
}

struct Expert {
    gate: Mat,
    up: Mat,
    down: Mat,
}

fn table(g: &dyn GpuBackend, mats: &[&Mat]) -> Result<ops::Nvfp4ExpertTables> {
    let ptrs = |f: &dyn Fn(&Mat) -> u64| -> Vec<u8> {
        mats.iter().flat_map(|m| f(m).to_le_bytes()).collect()
    };
    Ok(ops::Nvfp4ExpertTables {
        packed_ptrs: upload(g, &ptrs(&|m| m.w.weight.0))?,
        scale_ptrs: upload(g, &ptrs(&|m| m.w.weight_scale.0))?,
        scale2_vals: upload(
            g,
            &mats
                .iter()
                .flat_map(|m| m.s2.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?,
    })
}

fn bf16_to_f64(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
        .collect()
}

fn f32s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect()
}

fn round_bf16(x: f64) -> f64 {
    bf16::from_f64(x).to_f64()
}

/// 2026-09-27: `got` within 2 % of the largest |want| of its row, element by element.
fn close(got: &[f64], want: &[f64], what: &str) -> Result<()> {
    let scale = want.iter().fold(1e-30f64, |a, v| a.max(v.abs()));
    for (i, (x, y)) in got.iter().zip(want).enumerate() {
        ensure!(
            x.is_finite() && (x - y).abs() <= 0.02 * scale,
            "{what}: element {i} is {x}, the reference {y} (row max {scale})"
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let num_experts: usize = arg(1, 32);
    let alpha: f64 = arg(2, 0.9);
    let set = metrale_kernels::ptx_for_exact_target("qwen3.6-35b-a3b", "nvfp4")
        .context("no compiled qwen3.6-35b-a3b/nvfp4 kernel set")?;
    let backend = MetraleCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let stream = g.default_stream();
    let k_gate_up = g.kernel("moe_nvfp4_grouped", "moe_expert_gate_up_act_nvfp4_grouped")?;
    let k_down = g.kernel("moe_nvfp4_grouped", "moe_expert_down_act_nvfp4_grouped")?;
    let k_sort = g.kernel("moe", "moe_sort_by_expert")?;
    let k_compact = g.kernel(
        "moe_shared_expert_fused_fp8_grouped",
        "moe_fp8_grouped_compact",
    )?;
    let k_blend = g.kernel(
        "moe_fp8_grouped_blend",
        "moe_weighted_sum_blend_fp8_grouped",
    )?;

    let mut rng = Rng(7);
    let experts: Vec<Expert> = (0..num_experts)
        .map(|_| {
            Ok(Expert {
                gate: Mat::new(g, &mut rng, INTER, H)?,
                up: Mat::new(g, &mut rng, INTER, H)?,
                down: Mat::new(g, &mut rng, H, INTER)?,
            })
        })
        .collect::<Result<_>>()?;
    let shared = Expert {
        gate: Mat::new(g, &mut rng, INTER, H)?,
        up: Mat::new(g, &mut rng, INTER, H)?,
        down: Mat::new(g, &mut rng, H, INTER)?,
    };
    let gate_t = table(g, &experts.iter().map(|e| &e.gate).collect::<Vec<_>>())?;
    let up_t = table(g, &experts.iter().map(|e| &e.up).collect::<Vec<_>>())?;
    let down_t = table(g, &experts.iter().map(|e| &e.down).collect::<Vec<_>>())?;

    // 2026-09-27: MAX_M rows of input, routing and slot weights; width M uses the first M.
    let input_bytes: Vec<u8> = (0..MAX_M * H)
        .flat_map(|_| {
            bf16::from_f64(rng.unit() * 2.0 - 1.0)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let input = upload(g, &input_bytes)?;
    let input_f = bf16_to_f64(&input_bytes);
    let pop: Vec<f64> = (0..num_experts)
        .map(|e| ((e + 1) as f64).powf(-alpha))
        .collect();
    let total: f64 = pop.iter().sum();
    let mut routing: Vec<u32> = Vec::with_capacity(MAX_M * TOP_K);
    for _ in 0..MAX_M {
        let mut row: Vec<u32> = Vec::new();
        while row.len() < TOP_K {
            let (u, mut acc, mut e) = (rng.unit() * total, 0.0, 0usize);
            while e + 1 < num_experts && acc + pop[e] < u {
                acc += pop[e];
                e += 1;
            }
            if !row.contains(&(e as u32)) {
                row.push(e as u32);
            }
        }
        routing.extend(row);
    }
    let idx = upload(
        g,
        &routing
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let slot_w: Vec<u8> = (0..MAX_M * TOP_K)
        .flat_map(|_| (rng.unit() as f32 / TOP_K as f32).to_le_bytes())
        .collect();
    let slot_w = upload(g, &slot_w)?;
    let sh_gate_vec = upload(
        g,
        &(0..H)
            .flat_map(|_| {
                bf16::from_f64((rng.unit() - 0.5) * 0.05)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect::<Vec<u8>>(),
    )?;

    let te_max = MAX_M * TOP_K;
    let sorted = g.alloc(te_max * 4)?;
    let sorted_e = g.alloc(te_max * 4)?;
    let offsets = g.alloc((num_experts + 1) * 4)?;
    let to_perm = g.alloc(te_max * 4)?;
    let active = g.alloc(te_max.min(num_experts) * 4 + 16)?;
    let active_count = g.alloc(16)?;
    let act = g.alloc(te_max * INTER * 4)?;
    let sh_act = g.alloc(MAX_M * INTER * 4)?;
    let down_out = g.alloc(te_max * H * 2)?;
    let sh_out = g.alloc(MAX_M * H * 2)?;
    let output = g.alloc(MAX_M * H * 2)?;

    // 2026-09-27: One grouped dispatch of the first m rows; returns (blended output, act,
    // down_out, sorted token ids, expert offsets) and the mean time of the expert kernels.
    type Run = (Vec<u8>, Vec<f64>, Vec<u8>, Vec<u32>, Vec<u32>, f64);
    let run = |m: usize, iters: usize| -> Result<Run> {
        let (te, n) = (m * TOP_K, m as u32);
        let cap = ops::fp8_grouped_active_cap(n, TOP_K as u32, num_experts as u32);
        ops::moe_sort_by_expert(
            g,
            k_sort,
            idx,
            sorted,
            sorted_e,
            offsets,
            to_perm,
            te as u32,
            num_experts as u32,
            TOP_K as u32,
            stream,
        )?;
        ops::moe_fp8_grouped_compact(
            g,
            k_compact,
            offsets,
            active,
            active_count,
            num_experts as u32,
            stream,
        )?;
        let experts_once = || -> Result<()> {
            ops::moe_expert_gate_up_act_nvfp4_grouped(
                g,
                k_gate_up,
                input,
                gate_t,
                up_t,
                act,
                offsets,
                sorted,
                active,
                active_count,
                &shared.gate.w,
                &shared.up.w,
                sh_act,
                INTER as u32,
                H as u32,
                cap,
                n,
                stream,
            )?;
            ops::moe_expert_down_act_nvfp4_grouped(
                g,
                k_down,
                act,
                down_t,
                down_out,
                offsets,
                active,
                active_count,
                sh_act,
                &shared.down.w,
                sh_out,
                H as u32,
                INTER as u32,
                cap,
                n,
                stream,
            )
        };
        experts_once()?;
        g.synchronize(stream)?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            experts_once()?;
        }
        g.synchronize(stream)?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters.max(1) as f64;
        ops::moe_weighted_sum_blend_fp8_grouped(
            g,
            k_blend,
            output,
            down_out,
            slot_w,
            to_perm,
            sh_out,
            input,
            sh_gate_vec,
            H as u32,
            TOP_K as u32,
            H as u32,
            n,
            stream,
        )?;
        g.synchronize(stream)?;
        let u32s = |b: Vec<u8>| -> Vec<u32> {
            b.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        Ok((
            read(g, output, m * H * 2)?,
            f32s(&read(g, act, te * INTER * 4)?),
            read(g, down_out, te * H * 2)?,
            u32s(read(g, sorted, te * 4)?),
            u32s(read(g, offsets, (num_experts + 1) * 4)?),
            us,
        ))
    };

    let mut failures = 0usize;
    let full = run(MAX_M, 0)?.0;
    let row = H * 2;
    for &m in &WIDTHS {
        let (out, .., us) = run(m, 20)?;
        let bad = (0..m)
            .filter(|t| out[t * row..(t + 1) * row] != full[t * row..(t + 1) * row])
            .count();
        println!(
            "M={m:<3} rows equal to M={MAX_M}: {}/{m}  experts {us:.1} us",
            m - bad
        );
        failures += usize::from(bad > 0);
    }
    let live = (0..MAX_M - 1)
        .filter(|t| full[t * row..(t + 1) * row] != full[(t + 1) * row..(t + 2) * row])
        .count();
    println!("oracle live: {live}/{} neighbour rows differ", MAX_M - 1);
    failures += usize::from(live == 0);

    // 2026-09-27: Host reference at M = 4 for the routed products and down outputs.
    let m = 4;
    let (_, act_h, down_h, sorted_h, offsets_h, _) = run(m, 0)?;
    let down_f = bf16_to_f64(&down_h);
    let mut ref_fail = None;
    for e in 0..num_experts {
        for pos in offsets_h[e] as usize..offsets_h[e + 1] as usize {
            let x = &input_f[sorted_h[pos] as usize * H..][..H];
            let a: Vec<f64> = (0..INTER)
                .map(|c| {
                    let gv = round_bf16(experts[e].gate.dot(c, x));
                    let uv = round_bf16(experts[e].up.dot(c, x));
                    gv / (1.0 + (-gv).exp()) * uv
                })
                .collect();
            let d: Vec<f64> = (0..H).map(|c| experts[e].down.dot(c, &a)).collect();
            let r = close(
                &act_h[pos * INTER..][..INTER],
                &a,
                &format!("act e{e} pos{pos}"),
            )
            .and_then(|()| close(&down_f[pos * H..][..H], &d, &format!("down e{e} pos{pos}")));
            if let Err(err) = r {
                ref_fail.get_or_insert(err);
            }
        }
    }
    match ref_fail {
        None => println!("host reference at M={m}: PASS"),
        Some(e) => {
            println!("host reference at M={m}: FAIL {e:#}");
            failures += 1;
        }
    }
    println!(
        "nvfp4_moe_grouped_microtest experts={num_experts} alpha={alpha}: {}",
        if failures == 0 {
            "ALL PASS"
        } else {
            "FAILURES"
        }
    );
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
