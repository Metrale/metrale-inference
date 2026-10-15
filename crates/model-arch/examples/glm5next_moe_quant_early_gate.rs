// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Byte gate and microbench of `METRALE_GLM_MOE_QUANT_EARLY`'s premise on one GPU:
//! the grouped W4A4 launch's gate/up quantization (`w4a4_quant_rows_static`) of a rank's shard
//! of the rows, launched through `GateUpCodes::launch_rows` at the shard's offset, then
//! all-gathered in place, is byte for byte the launch of every row the grouped launch makes.
//!
//! 1. Reference: one launch of every row (hidden 4096, the static gate/up scale) into its own
//!    buffers, argument for argument the grouped launch's quantization.
//! 2. Shards: for each of `WORLD` emulated ranks, a scratch filled with 0xA5, the rank's shard
//!    quantized through `launch_rows` on a side stream after an event on the main stream, and
//!    every peer's shard copied in at its rows (the in-place all-gather's layout). Every
//!    scratch's codes, group scales and global scales must equal the reference.
//! 3. Negative controls (each must differ from the reference): rank 1's shard read one row
//!    off its input, and every row under a gate/up scale one ulp off.
//! 4. Timing (CUDA-graph replay, median of `GLM_BENCH_REPS` replays with p10 and max): the
//!    launch of every row (what the lever takes off the grouped launch's critical path) and
//!    the largest shard's launch (what it adds on the side stream).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on the first byte mismatch or a negative control that does
//! not fire; prints `RESULT: PASS` last.
//!
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! GLM_BENCH_GPU_ORDINAL=0 cargo run -p metrale-model-arch --release \
//!     --example glm5next_moe_quant_early_gate --features cuda,gpu-examples
//! ```

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mlp::forward::GateUpCodes;
use metrale_model_arch::glm5next_sp_mhc::shards;

#[allow(dead_code)]
#[path = "glm5next_moe_wide_bench/device.rs"]
mod device;
use device::{Rng, fmt, read, time_set, up_bf16};

/// 2026-10-15: GLM-5.3 Flash's hidden width, the TP=3 world, and a static gate/up scale of the
/// checkpoint's order (amax about 8 over E2M1 6 x E4M3 448).
const HIDDEN: usize = 4096;
const WORLD: usize = 3;
const GS: f32 = 0.002_975;

/// 2026-10-15: The three outputs of `rows` rows: codes, group scales, global scales.
fn alloc_out(gpu: &dyn GpuBackend, c: &GateUpCodes, rows: usize) -> Result<[DevicePtr; 3]> {
    let b = c.row_bytes();
    let mut out = [DevicePtr::NULL; 3];
    for (o, rb) in out.iter_mut().zip(b) {
        *o = gpu.alloc(rows * rb)?;
        gpu.memset(*o, 0xA5, rows * rb)?;
    }
    Ok(out)
}

fn read_out(gpu: &dyn GpuBackend, c: &GateUpCodes, rows: usize) -> Result<[Vec<u8>; 3]> {
    let b = c.row_bytes();
    Ok([
        read(gpu, c.out[0], rows * b[0])?,
        read(gpu, c.out[1], rows * b[1])?,
        read(gpu, c.out[2], rows * b[2])?,
    ])
}

/// 2026-10-15: The grouped launch's quantization of every row, written out independently of
/// `launch_rows` (grid one block per row, 256 threads, x, codes, scales, globals, k, scale).
fn reference(
    gpu: &dyn GpuBackend,
    c: &GateUpCodes,
    x: DevicePtr,
    rows: usize,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, c.kern)
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(c.out[0])
        .arg_ptr(c.out[1])
        .arg_ptr(c.out[2])
        .arg_u32(c.k as u32)
        .arg_f32(c.gs)
        .launch(s)
}

/// 2026-10-15: Rank `rank`'s scratch after its shard's quantization on `side` (behind an event
/// on `main`) and the in-place gather of every peer's rows from `peers`.
#[allow(clippy::too_many_arguments)]
fn rank_scratch(
    gpu: &dyn GpuBackend,
    c: &GateUpCodes,
    x: DevicePtr,
    rows: usize,
    rank: usize,
    peers: &[GateUpCodes],
    (main, side, ev): (u64, u64, u64),
) -> Result<()> {
    let s = shards(rows, WORLD);
    let (lo, hi) = s[rank];
    gpu.record_event(ev, main)?;
    gpu.stream_wait_event(side, ev)?;
    c.launch_rows(gpu, x, lo, hi, side)?;
    gpu.record_event(ev, side)?;
    gpu.stream_wait_event(main, ev)?;
    for (p, &(plo, phi)) in s.iter().enumerate() {
        if p == rank {
            continue;
        }
        let rb = c.row_bytes();
        let (src, dst) = (peers[p].at(plo), c.at(plo));
        for i in 0..3 {
            gpu.copy_d2d_async(src[i], dst[i], (phi - plo) * rb[i], main)?;
        }
    }
    gpu.synchronize(main)
}

fn compare(what: &str, got: &[Vec<u8>; 3], want: &[Vec<u8>; 3]) -> Result<bool> {
    let mut same = true;
    for (i, name) in ["codes", "group scales", "global scales"]
        .iter()
        .enumerate()
    {
        ensure!(got[i].len() == want[i].len(), "{what}: {name} length");
        let diff = got[i].iter().zip(&want[i]).filter(|(a, b)| a != b).count();
        println!(
            "  {what:<34} {name:<14} {diff:>9} of {:>9} bytes differ",
            want[i].len()
        );
        same &= diff == 0;
    }
    Ok(same)
}

fn free_out(gpu: &dyn GpuBackend, c: &GateUpCodes) -> Result<()> {
    for p in c.out {
        gpu.free(p)?;
    }
    Ok(())
}

fn gate_at(
    gpu: &dyn GpuBackend,
    kern: KernelHandle,
    rng: &mut Rng,
    rows: usize,
    streams: (u64, u64, u64),
) -> Result<()> {
    let main = streams.0;
    // 2026-10-15: One row more than the gate reads, so the negative control's shifted shard
    // stays inside the input.
    let xs: Vec<f32> = (0..(rows + 1) * HIDDEN)
        .map(|_| (rng.unit() - 0.5) * 16.0)
        .collect();
    let x = up_bf16(gpu, &xs)?;
    let mk = |gs: f32| GateUpCodes {
        kern,
        gs,
        k: HIDDEN,
        out: [DevicePtr::NULL; 3],
        rows_cap: rows,
    };
    let mut want = mk(GS);
    want.out = alloc_out(gpu, &want, rows)?;
    reference(gpu, &want, x, rows, main)?;
    gpu.synchronize(main)?;
    let want_b = read_out(gpu, &want, rows)?;
    ensure!(
        want_b[2].iter().any(|&b| b != 0xA5),
        "rows={rows}: the reference wrote nothing"
    );

    let mut ranks = Vec::with_capacity(WORLD);
    for _ in 0..WORLD {
        let mut c = mk(GS);
        c.out = alloc_out(gpu, &c, rows)?;
        ranks.push(c);
    }
    // 2026-10-15: Every rank quantizes its shard first; the gathers then read finished peers.
    let s = shards(rows, WORLD);
    for (r, c) in ranks.iter().enumerate() {
        c.launch_rows(gpu, x, s[r].0, s[r].1, streams.1)?;
    }
    gpu.synchronize(streams.1)?;
    let mut pass = true;
    for r in 0..WORLD {
        rank_scratch(gpu, &ranks[r], x, rows, r, &ranks, streams)?;
        let got = read_out(gpu, &ranks[r], rows)?;
        pass &= compare(
            &format!("rows={rows} rank {r} shard {:?}", s[r]),
            &got,
            &want_b,
        )?;
    }
    ensure!(
        pass,
        "rows={rows}: a gathered scratch differs from the launch of every row"
    );

    // 2026-10-15: Negative controls on rank 1's scratch: its shard one input row off, then
    // every row one scale ulp off.
    let c = &ranks[1];
    let (lo, hi) = s[1];
    c.launch_rows(gpu, x.offset(HIDDEN * 2), lo, hi, main)?;
    gpu.synchronize(main)?;
    let off = !compare(
        &format!("NEGCTL rows={rows} shard one row off"),
        &read_out(gpu, c, rows)?,
        &want_b,
    )?;
    let ulp = GateUpCodes {
        gs: f32::from_bits(GS.to_bits() + 1),
        ..*c
    };
    ulp.launch_rows(gpu, x, 0, rows, main)?;
    gpu.synchronize(main)?;
    let scale = !compare(
        &format!("NEGCTL rows={rows} scale one ulp off"),
        &read_out(gpu, c, rows)?,
        &want_b,
    )?;
    ensure!(
        off && scale,
        "rows={rows}: a negative control did not fire (row {off}, scale {scale})"
    );
    println!("rows={rows}: byte gate PASS, negative controls fired (row, scale)");

    let big = s.iter().map(|&(a, b)| b - a).max().context("no shards")?;
    let (blo, bhi) = *s.iter().find(|&&(a, b)| b - a == big).context("no shard")?;
    let whole = || reference(gpu, &want, x, rows, main);
    let shard = || c.launch_rows(gpu, x, blo, bhi, main);
    let t = time_set(gpu, main, &[&whole, &shard])?;
    println!("rows={rows}: quant every row        {}", fmt(t[0]));
    println!("rows={rows}: quant one shard ({big:>5}) {}", fmt(t[1]));
    for c in ranks.iter().chain([&want]) {
        free_out(gpu, c)?;
    }
    gpu.free(x)
}

fn main() -> Result<()> {
    let ordinal = std::env::var("GLM_BENCH_GPU_ORDINAL")
        .context("GLM_BENCH_GPU_ORDINAL names the GPU to run on")?
        .parse()?;
    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(ordinal, &target.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let kern = g.kernel("w4a4_gemv_mx_moe", "w4a4_quant_rows_static")?;
    let streams = (g.create_stream()?, g.create_stream()?, g.create_event()?);
    let rows_list: Vec<usize> = std::env::var("GLM_BENCH_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![17, 1036, 2048, 8208]);
    let mut rng = Rng(0x1319_8a2e_0370_7344);
    for &rows in &rows_list {
        gate_at(g, kern, &mut rng, rows, streams)?;
    }
    g.destroy_event(streams.2)?;
    println!("RESULT: PASS");
    Ok(())
}
