// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Byte gate and microbench of the grouped W4A4 routed-expert GEMM
//! (`kernels/gb10/common/w4a4_grouped_mx.cu`) at the per-rank shapes of GLM-5.3's TP=3 `tp`
//! expert layout (hidden 4096, all 288 experts, rank 0's 704-wide slice, top 8).
//!
//! 1. Byte gate, at `GLM_BENCH_ROWS` rows (default 17, 64, 512, 1036, 2048, 8208; 2026-10-14:
//!    8208 x top 8 slots pass CUDA's grid-Y limit, 1036 ends on a ragged tile) of random
//!    routing: every sorted slot of the grouped gate (gathered through `sorted_token_ids`)
//!    against the slot GEMV `w4a4_gemv_mx8_moe_slots` on the same slot, and every row of the
//!    grouped `_k64` down against the `_k64` slot GEMV on the same activation row, on both tile
//!    entries.
//! 2. Timing (CUDA-graph replay, median of `REPS` replays with p10 and max): at each row count
//!    the grouped gate + up + down, the whole sequence (sort, tiles, quantizations, SwiGLU), and
//!    the grouped W4A16 GEMM the prefill ran before (`moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128`,
//!    gate + up + down on the same weights), with TFLOPS over the routed MACs.
//! 3. 2026-10-13: The dense MLP's one-group launch (`dense.rs`): byte gate against the mx32 chunk
//!    path and the slot GEMV, then its time against the chunk path.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on the first byte mismatch.
//!
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! GLM_BENCH_GPU_ORDINAL=0 cargo run -p metrale-gpu-examples --release \
//!     --example glm5next_moe_prefill_w4a4_bench --features cuda,gpu-examples
//! ```

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

#[allow(dead_code)]
#[path = "../glm5next_moe_wide_bench/device.rs"]
mod device;
use device::{Rng, env_or, fmt, read, read_i32, time_set, up_bf16};

mod dense;
mod launch;
use launch::*;

/// 2026-10-10: Part 1 at one routing, on tile entry `ti`.
fn byte_gate(
    gpu: &dyn GpuBackend,
    kn: &Kern,
    b: &Bufs,
    (gate, down): (&Table, &Table),
    rng: &mut Rng,
    ti: usize,
) -> Result<()> {
    let (rows, slots, mt) = (b.rows, b.rows * TOP_K, MT[ti]);
    sort(gpu, kn, b, 0)?;
    tiles(gpu, kn, b, mt, 0)?;
    quant(gpu, kn.quant, b, b.x, rows, HIDDEN, GS_GATE_UP, 0)?;
    grouped(
        gpu,
        kn.mx[ti],
        ti,
        b,
        gate,
        b.gate,
        b.sorted,
        (MI, HIDDEN),
        0,
    )?;
    slots_gemv(gpu, kn.slots, b, gate, b.upo, (MI, HIDDEN, TOP_K))?;
    let perm = read_i32(gpu, b.perm, slots)?;
    let (got, want) = (
        read(gpu, b.gate, slots * MI * 2)?,
        read(gpu, b.upo, slots * MI * 2)?,
    );
    let row = MI * 2;
    for (s, &p) in perm.iter().enumerate() {
        let p = p as usize;
        ensure!(
            got[p * row..(p + 1) * row] == want[s * row..(s + 1) * row],
            "rows={rows} mt={mt}: gate slot {s} (sorted {p}) differs from the slot GEMV"
        );
    }
    ensure!(
        got.iter().any(|&x| x != 0),
        "rows={rows}: gate wrote nothing"
    );
    // 2026-10-10: Down: per-slot activations in slot order for the GEMV, the same rows in
    // sorted order for the grouped GEMM; each row quantizes on its own, so the bytes match.
    let act: Vec<f32> = (0..slots * MI).map(|_| rng.unit() * 10.0).collect();
    let mut sorted = vec![0f32; slots * MI];
    for (s, &p) in perm.iter().enumerate() {
        sorted[p as usize * MI..(p as usize + 1) * MI].copy_from_slice(&act[s * MI..(s + 1) * MI]);
    }
    let (a_slot, a_sorted) = (up_bf16(gpu, &act)?, up_bf16(gpu, &sorted)?);
    quant(gpu, kn.quant_k64, b, a_sorted, slots, MI, GS_DOWN, 0)?;
    grouped(
        gpu,
        kn.mx_k64[ti],
        ti,
        b,
        down,
        b.out,
        DevicePtr::NULL,
        (HIDDEN, MI),
        0,
    )?;
    gpu.synchronize(0)?;
    let got = read(gpu, b.out, slots * HIDDEN * 2)?;
    quant(gpu, kn.quant_k64, b, a_slot, slots, MI, GS_DOWN, 0)?;
    slots_gemv(gpu, kn.slots_k64, b, down, b.out_ref, (HIDDEN, MI, 1))?;
    let want = read(gpu, b.out_ref, slots * HIDDEN * 2)?;
    let row = HIDDEN * 2;
    for (s, &p) in perm.iter().enumerate() {
        let p = p as usize;
        ensure!(
            got[p * row..(p + 1) * row] == want[s * row..(s + 1) * row],
            "rows={rows} mt={mt}: down slot {s} (sorted {p}) differs from the slot GEMV"
        );
    }
    for p in [a_slot, a_sorted] {
        gpu.free(p)?;
    }
    println!("byte gate rows={rows:4} mt={mt}: gate and k64 down identical to the slot GEMV");
    Ok(())
}

/// 2026-10-10: Part 2 at one routing.
fn timing(gpu: &dyn GpuBackend, kn: &Kern, b: &Bufs, t: [&Table; 3], s: u64) -> Result<()> {
    let (rows, slots) = (b.rows, b.rows * TOP_K);
    let macs = (slots * 3 * MI * HIDDEN) as f64;
    let report = |name: &str, x: [f64; 3]| {
        println!(
            "rows={rows:4} {name:30} {}  {:6.1} TFLOPS  per layer {:7.3} ms per 1k tokens",
            fmt(x),
            2.0 * macs / x[0] / 1e6,
            x[0] / 1e3 * 1000.0 / rows as f64
        );
    };
    for ti in 0..4 {
        let mt = MT[ti];
        sort(gpu, kn, b, 0)?;
        tiles(gpu, kn, b, mt, 0)?;
        quant(gpu, kn.quant, b, b.x, rows, HIDDEN, GS_GATE_UP, 0)?;
        let gemms = || -> Result<()> {
            grouped(
                gpu,
                kn.mx[ti],
                ti,
                b,
                t[0],
                b.gate,
                b.sorted,
                (MI, HIDDEN),
                s,
            )?;
            grouped(
                gpu,
                kn.mx[ti],
                ti,
                b,
                t[1],
                b.upo,
                b.sorted,
                (MI, HIDDEN),
                s,
            )?;
            grouped(
                gpu,
                kn.mx_k64[ti],
                ti,
                b,
                t[2],
                b.out,
                DevicePtr::NULL,
                (HIDDEN, MI),
                s,
            )
        };
        let whole = || -> Result<()> {
            sort(gpu, kn, b, s)?;
            tiles(gpu, kn, b, mt, s)?;
            quant(gpu, kn.quant, b, b.x, rows, HIDDEN, GS_GATE_UP, s)?;
            gemms()?;
            KernelLaunch::new(gpu, kn.swiglu)
                .grid([div_ceil((slots * MI) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(b.gate)
                .arg_ptr(b.upo)
                .arg_ptr(b.act)
                .arg_u32((slots * MI) as u32)
                .arg_f32(10.0)
                .launch(s)?;
            quant(gpu, kn.quant_k64, b, b.act, slots, MI, GS_DOWN, s)
        };
        let gate_only = || {
            grouped(
                gpu,
                kn.mx[ti],
                ti,
                b,
                t[0],
                b.gate,
                b.sorted,
                (MI, HIDDEN),
                s,
            )
        };
        let down_only = || {
            grouped(
                gpu,
                kn.mx_k64[ti],
                ti,
                b,
                t[2],
                b.out,
                DevicePtr::NULL,
                (HIDDEN, MI),
                s,
            )
        };
        let r = time_set(gpu, s, &[&gemms, &whole, &gate_only, &down_only])?;
        println!(
            "rows={rows:4} m{mt}: gate {:.1} us, down {:.1} us",
            r[2][0], r[3][0]
        );
        report(&format!("w4a4 m{mt} s{} gate+up+down", STAGES[ti]), r[0]);
        report(&format!("w4a4 m{mt} s{} whole", STAGES[ti]), r[1]);
    }
    let off = read_i32(gpu, b.offsets, EXPERTS + 1)?;
    let max_m = off
        .windows(2)
        .map(|w| w[1] - w[0])
        .max()
        .unwrap_or(0)
        .max(1) as u32;
    let m16 = max_m.div_ceil(16);
    let old = || -> Result<()> {
        w4a16(
            gpu,
            kn,
            b,
            t[0],
            b.x,
            b.gate,
            b.sorted,
            (MI, HIDDEN),
            m16,
            s,
        )?;
        w4a16(gpu, kn, b, t[1], b.x, b.upo, b.sorted, (MI, HIDDEN), m16, s)?;
        w4a16(
            gpu,
            kn,
            b,
            t[2],
            b.act,
            b.out,
            DevicePtr::NULL,
            (HIDDEN, MI),
            m16,
            s,
        )
    };
    report("w4a16 grouped gate+up+down", time_set(gpu, s, &[&old])?[0]);
    Ok(())
}

fn main() -> Result<()> {
    let ordinal = std::env::var("GLM_BENCH_GPU_ORDINAL")
        .context("GLM_BENCH_GPU_ORDINAL names the GPU to run on")?
        .parse()?;
    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(ordinal, &target.modules)?;
    let stream = gpu.create_stream()?;
    let g: &dyn GpuBackend = &gpu;
    let m = "w4a4_grouped_mx";
    let kn = Kern {
        sort: g.kernel("moe", "moe_sort_by_expert")?,
        tiles: g.kernel(m, "w4a4_grouped_tiles")?,
        quant: g.kernel("w4a4_gemv_mx_moe", "w4a4_quant_rows_static")?,
        quant_k64: g.kernel("w4a4_gemv_mx_moe", "w4a4_quant_rows_static_k64")?,
        swiglu: g.kernel("glm5next_ffn", "glm5next_swiglu_clamp")?,
        slots: g.kernel("w4a4_gemv_mx_moe", "w4a4_gemv_mx8_moe_slots")?,
        slots_k64: g.kernel("w4a4_gemv_mx_moe", "w4a4_gemv_mx8_moe_slots_k64")?,
        mx: [
            g.kernel(m, "w4a4_grouped_mx_m32")?,
            g.kernel(m, "w4a4_grouped_mx_m64")?,
            g.kernel(m, "w4a4_grouped_mx_m128")?,
            g.kernel(m, "w4a4_grouped_mx_m128_s6")?,
        ],
        mx_k64: [
            g.kernel(m, "w4a4_grouped_mx_m32_k64")?,
            g.kernel(m, "w4a4_grouped_mx_m64_k64")?,
            g.kernel(m, "w4a4_grouped_mx_m128_k64")?,
            g.kernel(m, "w4a4_grouped_mx_m128_k64")?,
        ],
        mx32: g.kernel("w4a4_gemv_mx", "w4a4_gemv_mx32")?,
        w4a16: g.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128")?,
    };
    let mut rng = Rng(0x243f_6a88_85a3_08d3);
    let gate = table(g, &mut rng, MI, HIDDEN)?;
    let upt = table(g, &mut rng, MI, HIDDEN)?;
    let down = table(g, &mut rng, HIDDEN, MI)?;
    let rows_list: Vec<usize> = std::env::var("GLM_BENCH_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![17, 64, 512, 1036, 2048, 8208]);
    let gate_only = env_or("GLM_BENCH_GATE_ONLY", 0) != 0;
    for &rows in &rows_list {
        let b = bufs(g, &mut rng, rows)?;
        for ti in 0..4 {
            byte_gate(g, &kn, &b, (&gate, &down), &mut rng, ti)?;
        }
        if !gate_only {
            timing(g, &kn, &b, [&gate, &upt, &down], stream)?;
            dense::dense_one_group(g, &kn, &b, &mut rng, stream)?;
        }
        for p in [
            b.ids, b.sorted, b.sorted_e, b.offsets, b.perm, b.tile_off, b.x, b.aq, b.as_, b.ag,
            b.gate, b.upo, b.act, b.out, b.out_ref,
        ] {
            g.free(p)?;
        }
    }
    Ok(())
}
