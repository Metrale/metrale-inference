// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Part 3 of the bench: the dense MLP's one-group launch
//! (`METRALE_GLM_DENSE_W4A4_GROUPED_MIN_ROWS`). One rank-0 dense projection of TP=3 (hidden 4096,
//! intermediate 12288 / 3 = 4096: gate, up and down are all [4096, 4096]) over `rows` rows as
//! expert 0 of the 128-slot grouped GEMM with offsets [0, rows] and no gather, byte-gated
//! against BOTH the path it replaces (`w4a4_gemv_mx32` per 32-row chunk, each chunk quantized
//! on its own, as `w4a4_dense::forward_dense_w4a4` runs it) and the slot GEMV on all-zero ids
//! (the grouped kernel's own contract), then timed against the chunk path with the
//! quantizations included on both sides.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on the first byte mismatch; frees what it allocates.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::device::{Rng, fmt, read, time_set, up, up_bf16};
use super::launch::{Bufs, GS_GATE_UP, HIDDEN, Kern, Table, grouped_one, quant, slots_gemv_ids};

/// 2026-10-13: The dense chunk rows (`DENSE_W4A4_CHUNK_ROWS`) and the weight's global scale.
const CHUNK: usize = 32;
const SCALE2: f32 = 0.01;

/// 2026-10-13: The chunk path: per 32 rows, quantize the chunk then `w4a4_gemv_mx32` into
/// `c` at the chunk's rows.
#[allow(clippy::too_many_arguments)]
fn chunk_path(
    gpu: &dyn GpuBackend,
    kn: &Kern,
    b: &Bufs,
    x: DevicePtr,
    (packed, scale): (DevicePtr, DevicePtr),
    c: DevicePtr,
    s: u64,
) -> Result<()> {
    for r0 in (0..b.rows).step_by(CHUNK) {
        let n = (b.rows - r0).min(CHUNK);
        quant(
            gpu,
            kn.quant,
            b,
            x.offset(r0 * HIDDEN * 2),
            n,
            HIDDEN,
            GS_GATE_UP,
            s,
        )?;
        KernelLaunch::new(gpu, kn.mx32)
            .grid([div_ceil(HIDDEN as u32, 16), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(b.aq)
            .arg_ptr(b.as_)
            .arg_ptr(b.ag)
            .arg_ptr(packed)
            .arg_ptr(scale)
            .arg_f32(SCALE2)
            .arg_ptr(c.offset(r0 * HIDDEN * 2))
            .arg_u32(n as u32)
            .arg_u32(HIDDEN as u32)
            .arg_u32(HIDDEN as u32)
            .launch(s)?;
    }
    Ok(())
}

/// 2026-10-13: Part 3 at `b.rows` rows.
pub fn dense_one_group(
    gpu: &dyn GpuBackend,
    kn: &Kern,
    b: &Bufs,
    rng: &mut Rng,
    s: u64,
) -> Result<()> {
    let rows = b.rows;
    let (pb, sb) = (HIDDEN * HIDDEN / 2, HIDDEN * HIDDEN / 16);
    let packed: Vec<u8> = (0..pb).map(|_| rng.next() as u8).collect();
    let scales: Vec<u8> = (0..sb).map(|_| 0x30 + (rng.next() % 9) as u8).collect();
    let (wp, wsc) = (up(gpu, &packed)?, up(gpu, &scales)?);
    let t = Table {
        packed: up(gpu, &wp.0.to_le_bytes())?,
        scale: up(gpu, &wsc.0.to_le_bytes())?,
        scale2: up(gpu, &SCALE2.to_le_bytes())?,
    };
    let x: Vec<f32> = (0..rows * HIDDEN).map(|_| rng.unit() * 3.0).collect();
    let xd = up_bf16(gpu, &x)?;
    let zero_ids = up(gpu, &vec![0u8; rows * 4])?;
    let mut off = [0u8; 8];
    off[4..].copy_from_slice(&(rows as i32).to_le_bytes());
    let (offsets, tile_off) = (up(gpu, &off)?, up(gpu, &[0u8; 8])?);
    let chunked = gpu.alloc(rows * HIDDEN * 2)?;
    let tiles = |st: u64| {
        KernelLaunch::new(gpu, kn.tiles)
            .grid([1, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(offsets)
            .arg_ptr(tile_off)
            .arg_u32(1)
            .arg_u32(128)
            .launch(st)
    };
    let one = |st: u64| -> Result<()> {
        tiles(st)?;
        quant(gpu, kn.quant, b, xd, rows, HIDDEN, GS_GATE_UP, st)?;
        grouped_one(
            gpu,
            kn.mx[2],
            b,
            &t,
            b.out,
            offsets,
            tile_off,
            (HIDDEN, HIDDEN),
            st,
        )
    };
    chunk_path(gpu, kn, b, xd, (wp, wsc), chunked, 0)?;
    let want_chunks = read(gpu, chunked, rows * HIDDEN * 2)?;
    one(0)?;
    let got = read(gpu, b.out, rows * HIDDEN * 2)?;
    slots_gemv_ids(
        gpu,
        kn.slots,
        b,
        zero_ids,
        &t,
        b.out_ref,
        (HIDDEN, HIDDEN, 1),
        rows,
    )?;
    let want_slots = read(gpu, b.out_ref, rows * HIDDEN * 2)?;
    let diff = |w: &[u8]| got.iter().zip(w).filter(|(a, b)| a != b).count();
    ensure!(
        got.iter().any(|&v| v != 0),
        "rows={rows}: dense one-group output is all zero"
    );
    ensure!(
        got == want_chunks,
        "rows={rows}: dense one-group differs from the mx32 chunk path in {} bytes",
        diff(&want_chunks)
    );
    ensure!(
        got == want_slots,
        "rows={rows}: dense one-group differs from the slot GEMV in {} bytes",
        diff(&want_slots)
    );
    let one_s = || one(s);
    let chunks_s = || chunk_path(gpu, kn, b, xd, (wp, wsc), chunked, s);
    let r = time_set(gpu, s, &[&one_s, &chunks_s])?;
    println!(
        "rows={rows:5} dense [4096 x 4096] incl. quant: one-group {} vs mx32 chunks {}  x{:.2}  \
         (bytes == chunk path == slot GEMV)",
        fmt(r[0]),
        fmt(r[1]),
        r[1][0] / r[0][0]
    );
    for p in [
        wp, wsc, t.packed, t.scale, t.scale2, xd, zero_ids, offsets, tile_off, chunked,
    ] {
        gpu.free(p)?;
    }
    Ok(())
}
