// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The per-token FP8 MoE loop (single-token gate_up, silu_down and blend
//! kernels, once per row) against one cross-row grouped dispatch (`moe_sort_by_expert`,
//! `moe_fp8_grouped_compact` and the `_fp8_grouped` kernels), at Qwen3.6-35B-A3B shapes.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every M, the grouped output bytes equal the loop's for
//!   every live row, rows past M and the guard bands still hold the sentinel, and the
//!   compacted active-expert list is the ascending set of routed experts.
//!
//! Both legs get the same routing. Shapes follow the 35B MODEL.toml (hidden 2048, expert
//! and shared-expert intermediate 512, top-8), but by default with 32 experts so rows share experts;
//! at M=32 some experts get more rows than the grouped kernel's GROUP_ROWS (8). Each leg's
//! mean time over 20 iterations is printed.
//!
//! Run (GB10), optionally with the routed-expert count (default 32; 256 is the model's,
//! where rows rarely share an expert and the printed bandwidth is representative):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example fp8_moe_grouped_decode_microtest -- [experts] [zipf_alpha]

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::Fp8Weight;

#[path = "common/fp8_moe_grouped_fixture.rs"]
mod fixture;
use fixture::{Rng, bf16_bytes, check, fp8_bytes, fp8w, scale_bytes, upload};

const H: usize = 2048;
const INTER: usize = 512;
/// 2026-09-26: Routed experts: the first argument, default 32 (rows share experts).
/// `256` is the model's count, where the distinct-expert bandwidth is representative.
fn experts() -> usize {
    std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(32)
}

/// 2026-09-26: The second argument, a Zipf exponent for the routing (default 0,
/// uniform): expert e is drawn with weight (e + 1)^-alpha. At 256 experts,
/// alpha 0.9 gives about the distinct-expert counts the 35B verify step
/// routes to (121 at 32 rows, 153 at 64, measured with
/// METRALE_DUMP_EXPERT_IDS=1 on the concurrency ladder).
fn zipf_alpha() -> f64 {
    std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0.0)
}
const TOP_K: usize = 8;
const MAX_M: usize = 64;
const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;

struct Handles {
    gate_up: KernelHandle,
    silu_down: KernelHandle,
    blend: KernelHandle,
    sort: KernelHandle,
    g_gate_up: KernelHandle,
    g_silu_down: KernelHandle,
    g_blend: KernelHandle,
    g_compact: KernelHandle,
}

struct Experts {
    gate_w: DevicePtr,
    gate_s: DevicePtr,
    up_w: DevicePtr,
    up_s: DevicePtr,
    down_w: DevicePtr,
    down_s: DevicePtr,
    sh_gate: Fp8Weight,
    sh_up: Fp8Weight,
    sh_down: Fp8Weight,
    sh_gate_vec: DevicePtr,
}

/// 2026-09-25: Scratch shared by both legs, sized for MAX_M rows and sentinel-filled once
/// at allocation.
struct Scratch {
    e: usize,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    down_out: DevicePtr,
    sh_gate_out: DevicePtr,
    sh_up_out: DevicePtr,
    sh_down_out: DevicePtr,
    sort: DevicePtr,
    // 2026-09-26: The grouped leg's FP32 SiLU products, routed `[te, INTER]` and
    // shared `[MAX_M, INTER]`.
    act: DevicePtr,
    sh_act: DevicePtr,
}

#[allow(clippy::too_many_arguments)]
fn run_loop(
    gpu: &dyn GpuBackend,
    h: &Handles,
    x: &Experts,
    s: &Scratch,
    input: DevicePtr,
    indices: DevicePtr,
    weights: DevicePtr,
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    for t in 0..m {
        let in_t = input.offset(t * H * 2);
        let idx_t = indices.offset(t * TOP_K * 4);
        ops::moe_expert_gate_up_shared_fp8(
            gpu,
            h.gate_up,
            in_t,
            x.gate_w,
            x.gate_s,
            s.gate_out,
            x.up_w,
            x.up_s,
            s.up_out,
            idx_t,
            &x.sh_gate,
            s.sh_gate_out,
            &x.sh_up,
            s.sh_up_out,
            INTER as u32,
            H as u32,
            TOP_K as u32,
            0,
        )?;
        ops::moe_expert_silu_down_shared_fp8(
            gpu,
            h.silu_down,
            s.gate_out,
            s.up_out,
            x.down_w,
            x.down_s,
            s.down_out,
            idx_t,
            s.sh_gate_out,
            s.sh_up_out,
            &x.sh_down,
            s.sh_down_out,
            H as u32,
            INTER as u32,
            TOP_K as u32,
            0,
        )?;
        ops::moe_weighted_sum_blend(
            gpu,
            h.blend,
            out.offset(t * H * 2),
            s.down_out,
            weights.offset(t * TOP_K * 4),
            s.sh_down_out,
            in_t,
            x.sh_gate_vec,
            H as u32,
            TOP_K as u32,
            H as u32,
            0,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_grouped(
    gpu: &dyn GpuBackend,
    h: &Handles,
    x: &Experts,
    s: &Scratch,
    input: DevicePtr,
    indices: DevicePtr,
    weights: DevicePtr,
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    let te = m * TOP_K;
    let sorted_token_ids = s.sort;
    let sorted_expert_ids = s.sort.offset(te * 4);
    let expert_offsets = s.sort.offset(te * 8);
    let token_to_perm = s.sort.offset(te * 8 + (s.e + 1) * 4);
    let cap = ops::fp8_grouped_active_cap(m as u32, TOP_K as u32, s.e as u32);
    let active_experts = token_to_perm.offset(te * 4);
    let active_count = active_experts.offset(cap as usize * 4);
    ops::moe_sort_by_expert(
        gpu,
        h.sort,
        indices,
        sorted_token_ids,
        sorted_expert_ids,
        expert_offsets,
        token_to_perm,
        te as u32,
        s.e as u32,
        TOP_K as u32,
        0,
    )?;
    ops::moe_fp8_grouped_compact(
        gpu,
        h.g_compact,
        expert_offsets,
        active_experts,
        active_count,
        s.e as u32,
        0,
    )?;
    ops::moe_expert_gate_up_act_fp8_grouped(
        gpu,
        h.g_gate_up,
        input,
        x.gate_w,
        x.gate_s,
        x.up_w,
        x.up_s,
        s.act,
        expert_offsets,
        sorted_token_ids,
        active_experts,
        active_count,
        &x.sh_gate,
        &x.sh_up,
        s.sh_act,
        INTER as u32,
        H as u32,
        cap,
        m as u32,
        0,
    )?;
    ops::moe_expert_down_act_fp8_grouped(
        gpu,
        h.g_silu_down,
        s.act,
        x.down_w,
        x.down_s,
        s.down_out,
        expert_offsets,
        active_experts,
        active_count,
        s.sh_act,
        &x.sh_down,
        s.sh_down_out,
        H as u32,
        INTER as u32,
        cap,
        m as u32,
        0,
    )?;
    ops::moe_weighted_sum_blend_fp8_grouped(
        gpu,
        h.g_blend,
        out,
        s.down_out,
        weights,
        token_to_perm,
        s.sh_down_out,
        input,
        x.sh_gate_vec,
        H as u32,
        TOP_K as u32,
        H as u32,
        m as u32,
        0,
    )
}

/// 2026-09-25: Reads back the compacted active-expert list and errors unless it is the
/// ascending set of experts routed to by the first M rows.
fn check_compaction(gpu: &dyn GpuBackend, s: &Scratch, idx: &[u32], m: usize) -> Result<()> {
    let te = m * TOP_K;
    let cap = ops::fp8_grouped_active_cap(m as u32, TOP_K as u32, s.e as u32) as usize;
    let active_experts = s.sort.offset(te * 8 + (s.e + 1) * 4 + te * 4);
    let mut buf = vec![0u8; (cap + 1) * 4];
    gpu.copy_d2h(active_experts, &mut buf)?;
    let words: Vec<i32> = buf
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let count = words[cap] as usize;
    let mut expect: Vec<i32> = idx[..te].iter().map(|&e| e as i32).collect();
    expect.sort_unstable();
    expect.dedup();
    ensure!(
        count == expect.len() && words[..count] == expect[..],
        "compaction mismatch at M={m}: got {} {:?} want {} {:?}",
        count,
        &words[..count.min(cap)],
        expect.len(),
        expect
    );
    Ok(())
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let h = Handles {
        gate_up: gpu.kernel(
            "moe_shared_expert_fused_fp8",
            "moe_expert_gate_up_shared_fp8",
        )?,
        silu_down: gpu.kernel(
            "moe_shared_expert_fused_fp8",
            "moe_expert_silu_down_shared_fp8",
        )?,
        blend: gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?,
        sort: gpu.kernel("moe", "moe_sort_by_expert")?,
        g_gate_up: gpu.kernel(
            "moe_shared_expert_fused_fp8_grouped",
            "moe_expert_gate_up_act_fp8_grouped",
        )?,
        g_silu_down: gpu.kernel(
            "moe_shared_expert_fused_fp8_grouped",
            "moe_expert_down_act_fp8_grouped",
        )?,
        g_blend: gpu.kernel(
            "moe_fp8_grouped_blend",
            "moe_weighted_sum_blend_fp8_grouped",
        )?,
        g_compact: gpu.kernel(
            "moe_shared_expert_fused_fp8_grouped",
            "moe_fp8_grouped_compact",
        )?,
    };
    let mut rng = Rng(0x6d6f_6520_6739_2026);
    let e_count = experts();

    // 2026-09-25: Routed experts: per-expert [INTER, H] gate/up and [H, INTER] down FP8
    // weights with block scales, passed as device tables of per-expert pointers.
    let table = |n: usize, k: usize, rng: &mut Rng| -> Result<(DevicePtr, DevicePtr)> {
        let mut wp = Vec::with_capacity(e_count * 8);
        let mut sp = Vec::with_capacity(e_count * 8);
        for _ in 0..e_count {
            wp.extend_from_slice(&(upload(&gpu, &fp8_bytes(rng, n * k))?.0).to_le_bytes());
            sp.extend_from_slice(&(upload(&gpu, &scale_bytes(rng, n, k))?.0).to_le_bytes());
        }
        Ok((upload(&gpu, &wp)?, upload(&gpu, &sp)?))
    };
    let (gate_w, gate_s) = table(INTER, H, &mut rng)?;
    let (up_w, up_s) = table(INTER, H, &mut rng)?;
    let (down_w, down_s) = table(H, INTER, &mut rng)?;
    let shared = |n: usize, k: usize, rng: &mut Rng| -> Result<Fp8Weight> {
        Ok(fp8w(
            upload(&gpu, &fp8_bytes(rng, n * k))?,
            upload(&gpu, &scale_bytes(rng, n, k))?,
            n,
            k,
        ))
    };
    let x = Experts {
        gate_w,
        gate_s,
        up_w,
        up_s,
        down_w,
        down_s,
        sh_gate: shared(INTER, H, &mut rng)?,
        sh_up: shared(INTER, H, &mut rng)?,
        sh_down: shared(H, INTER, &mut rng)?,
        sh_gate_vec: upload(&gpu, &bf16_bytes(&mut rng, H, 0.05))?,
    };
    let input = upload(&gpu, &bf16_bytes(&mut rng, MAX_M * H, 1.0))?;

    // 2026-09-25: Routing: distinct experts within a row; rows draw from the same
    // `e_count` experts, with the weights of `zipf_alpha`.
    let alpha = zipf_alpha();
    let weights: Vec<f64> = (0..e_count)
        .map(|e| ((e + 1) as f64).powf(-alpha))
        .collect();
    let total: f64 = weights.iter().sum();
    let cdf: Vec<f64> = weights
        .iter()
        .scan(0.0, |acc, w| {
            *acc += w / total;
            Some(*acc)
        })
        .collect();
    let mut idx = Vec::with_capacity(MAX_M * TOP_K);
    for _ in 0..MAX_M {
        let mut row: Vec<u32> = Vec::new();
        while row.len() < TOP_K {
            let u = rng.next() as f64 / u32::MAX as f64;
            let e = cdf.partition_point(|&c| c < u).min(e_count - 1) as u32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        idx.extend(row);
    }
    let idx_bytes: Vec<u8> = idx.iter().flat_map(|e| e.to_le_bytes()).collect();
    let w_bytes: Vec<u8> = (0..MAX_M * TOP_K)
        .flat_map(|_| ((rng.next() % 1000) as f32 / 1000.0).to_le_bytes())
        .collect();
    let indices = upload(&gpu, &idx_bytes)?;
    let weights = upload(&gpu, &w_bytes)?;

    let te = MAX_M * TOP_K;
    let scratch = Scratch {
        e: e_count,
        gate_out: upload(&gpu, &vec![SENTINEL; te * INTER * 2])?,
        up_out: upload(&gpu, &vec![SENTINEL; te * INTER * 2])?,
        down_out: upload(&gpu, &vec![SENTINEL; te * H * 2])?,
        sh_gate_out: upload(&gpu, &vec![SENTINEL; MAX_M * INTER * 2])?,
        sh_up_out: upload(&gpu, &vec![SENTINEL; MAX_M * INTER * 2])?,
        sh_down_out: upload(&gpu, &vec![SENTINEL; MAX_M * H * 2])?,
        sort: upload(
            &gpu,
            &vec![
                0u8;
                te * 12
                    + (e_count + 1) * 4
                    + (ops::fp8_grouped_active_cap(MAX_M as u32, TOP_K as u32, e_count as u32)
                        as usize
                        + 1)
                        * 4
            ],
        )?,
        act: upload(&gpu, &vec![SENTINEL; te * INTER * 4])?,
        sh_act: upload(&gpu, &vec![SENTINEL; MAX_M * INTER * 4])?,
    };
    let sentinel = vec![SENTINEL; MAX_M * H * 2 + 2 * GUARD];
    let loop_base = upload(&gpu, &sentinel)?;
    let grouped_base = upload(&gpu, &sentinel)?;
    let (loop_out, grouped_out) = (loop_base.offset(GUARD), grouped_base.offset(GUARD));

    let mut failures = 0usize;
    let mut first = true;
    for m in [2usize, 3, 4, 8, 16, 32, 48, 64] {
        gpu.copy_h2d(&sentinel, loop_base)?;
        gpu.copy_h2d(&sentinel, grouped_base)?;
        run_loop(&gpu, &h, &x, &scratch, input, indices, weights, loop_out, m)?;
        run_grouped(
            &gpu,
            &h,
            &x,
            &scratch,
            input,
            indices,
            weights,
            grouped_out,
            m,
        )?;
        gpu.synchronize(0)?;
        let mut baseline = vec![0u8; sentinel.len()];
        let mut observed = vec![0u8; sentinel.len()];
        gpu.copy_d2h(loop_base, &mut baseline)?;
        gpu.copy_d2h(grouped_base, &mut observed)?;

        if first {
            for mutation in ["output-bit", "past-m", "guard", "nonfinite"] {
                let mut bad = baseline.clone();
                match mutation {
                    "output-bit" => bad[GUARD + 7] ^= 1,
                    "past-m" => bad[GUARD + m * H * 2] ^= 1,
                    "guard" => bad[0] ^= 1,
                    _ => bad[GUARD..GUARD + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes()),
                }
                let err = check(&bad, &baseline, &sentinel, m)
                    .expect_err("known-bad output was admitted by the oracle");
                println!("KNOWN_BAD {mutation}: refused: {err}");
            }
            first = false;
        }

        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            gpu.synchronize(0)?;
            let t = std::time::Instant::now();
            for _ in 0..20 {
                f()?;
            }
            gpu.synchronize(0)?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / 20.0)
        };
        let us_loop =
            time(&|| run_loop(&gpu, &h, &x, &scratch, input, indices, weights, loop_out, m))?;
        let us_grouped = time(&|| {
            run_grouped(
                &gpu,
                &h,
                &x,
                &scratch,
                input,
                indices,
                weights,
                grouped_out,
                m,
            )
        })?;
        let distinct = {
            let mut seen = vec![false; e_count];
            idx[..m * TOP_K]
                .iter()
                .for_each(|&e| seen[e as usize] = true);
            seen.iter().filter(|&&s| s).count()
        };
        match check(&observed, &baseline, &sentinel, m)
            .and_then(|()| check_compaction(&gpu, &scratch, &idx, m))
        {
            Ok(()) => println!(
                "M={m:2} distinct_experts={distinct:3}/{} loop={us_loop:8.1}us grouped={us_grouped:8.1}us \
                 speedup={:.2}x grouped_bw={:.0}GB/s  BIT-IDENTICAL",
                m * TOP_K,
                us_loop / us_grouped,
                ((distinct + 1) * 3 * INTER * H) as f64 / us_grouped / 1e3
            ),
            Err(e) => {
                println!("FAIL M={m}: {e}");
                failures += 1;
            }
        }
    }
    ensure!(failures == 0, "{failures} case(s) failed");
    println!("ALL PASS: grouped FP8 MoE decode == per-token loop, bit for bit, M=2..64");
    Ok(())
}
