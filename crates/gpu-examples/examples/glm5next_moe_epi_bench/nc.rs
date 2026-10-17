// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Part 4 of the `glm5next_moe_epi_bench` example: the m128 down's `_n` twins
//! (`METRALE_GLM_MOE_W4A4_NC`, `w4a4_grouped_mx_m128[_k64]_n`, nc column tiles per CTA).
//!
//! - Byte gate: the `_n` down of every `nc` in [`NC_GATE`] at every raster group in
//!   [`RASTER_NC`] (0 = row-major) on the separate launches' down input, against the plain
//!   down's output (outputs poisoned first, so an unwritten row fails too).
//! - Negative control: the `_n` down (nc 4, raster 4) on that input with every per-slot global
//!   doubled must differ from the reference output, or the bench fails.
//! - Timing (CUDA-graph replay, interleaved): the production down (the raster-4 `_r` twin, the
//!   `METRALE_GLM_MOE_W4A4_RASTER` default) against the `_n` down at raster 0 and 4 for every
//!   `GLM_BENCH_NC` value (default 2,4,8), with the saving per call and per 32k prefill.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on the first byte mismatch.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mlp::forward::GroupedEntry;

use crate::device::{fmt, read, time_set};
use crate::launch::{Bufs, HIDDEN};
use crate::route::{Width, down, down_entry};
use crate::{MOE_LAYERS, PREFILL, same_rows};

/// 2026-10-15: Column tiles per CTA of the byte gate: one, divisors and non-divisors of the 64
/// column tiles (hidden 4096), and one past them (a single CTA column).
const NC_GATE: [u32; 5] = [1, 3, 4, 8, 65];
/// 2026-10-15: Raster groups of the byte gate: row-major, a non-divisor, the
/// production down group, and a single band.
const RASTER_NC: [u32; 4] = [0, 3, 4, 4096];
/// 2026-10-15: The production down raster group (`METRALE_GLM_MOE_W4A4_RASTER` default `0,4`).
const PROD_RASTER: u32 = 4;

fn entry(w: &Width, raster: u32, nc: u32) -> GroupedEntry {
    GroupedEntry {
        kern: w.down_n,
        raster: Some(raster),
        nc: Some(nc),
    }
}

/// 2026-10-15: The byte gate and its negative control at one routing, width and limit: `dq`
/// holds the separate launches' down input (`want`: its codes, block scales and globals),
/// `want_out` the plain m128 down's output from it.
pub fn byte_gate(
    gpu: &dyn GpuBackend,
    w: &Width,
    b: &Bufs,
    dq: [DevicePtr; 3],
    want: &[Vec<u8>],
    want_out: &[u8],
    at: &str,
) -> Result<()> {
    for (p, v) in dq.into_iter().zip(want) {
        gpu.copy_h2d(v, p)?;
    }
    for nc in NC_GATE {
        for raster in RASTER_NC {
            let at = format!("{at} nc={nc} raster={raster}");
            gpu.copy_h2d(&vec![0xA5u8; want_out.len()], b.out_ref)?;
            down_entry(gpu, w, b, dq, b.out_ref, 2, entry(w, raster, nc), 0)?;
            let got = read(gpu, b.out_ref, want_out.len())?;
            same_rows("nc down output", &got, want_out, HIDDEN * 2, &at)?;
        }
    }
    let globals: Vec<u8> = want[2]
        .chunks_exact(4)
        .flat_map(|c| (2.0 * f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes())
        .collect();
    gpu.copy_h2d(&globals, dq[2])?;
    gpu.copy_h2d(&vec![0xA5u8; want_out.len()], b.out_ref)?;
    down_entry(gpu, w, b, dq, b.out_ref, 2, entry(w, PROD_RASTER, 4), 0)?;
    let got = read(gpu, b.out_ref, want_out.len())?;
    ensure!(
        same_rows("nc down output", &got, want_out, HIDDEN * 2, at).is_err(),
        "{at}: negative control (globals x2 into the nc down) matched the reference: the nc byte \
         gate cannot see a difference"
    );
    Ok(())
}

/// 2026-10-15: The `GLM_BENCH_NC` list.
fn nc_list() -> Result<Vec<u32>> {
    let v = std::env::var("GLM_BENCH_NC").unwrap_or_else(|_| "2,4,8".to_string());
    v.split(',')
        .map(|x| {
            x.trim()
                .parse::<u32>()
                .ok()
                .filter(|n| *n >= 1)
                .with_context(|| format!("GLM_BENCH_NC={v:?}: expected n,n,.. with n >= 1"))
        })
        .collect()
}

/// 2026-10-15: The timing at one routing and width on the m128 entry: the production down
/// against the `_n` down at raster 0 and [`PROD_RASTER`] per nc, interleaved in one set.
pub fn timing(gpu: &dyn GpuBackend, w: &Width, b: &Bufs, at: &str, s: u64) -> Result<()> {
    let act = [b.aq, b.as_, b.ag];
    let ncs = nc_list()?;
    let prod = || down(gpu, w, b, act, b.out, 2, Some(PROD_RASTER), s);
    let variants: Vec<(u32, u32)> = ncs
        .iter()
        .flat_map(|&nc| [(nc, 0), (nc, PROD_RASTER)])
        .collect();
    let runs: Vec<Box<dyn Fn() -> Result<()> + '_>> = variants
        .iter()
        .map(|&(nc, r)| -> Box<dyn Fn() -> Result<()> + '_> {
            Box::new(move || down_entry(gpu, w, b, act, b.out, 2, entry(w, r, nc), s))
        })
        .collect();
    let mut fs: Vec<&dyn Fn() -> Result<()>> = vec![&prod];
    fs.extend(runs.iter().map(|f| f.as_ref()));
    let t = time_set(gpu, s, &fs)?;
    let chunks = PREFILL / b.rows as f64;
    println!(
        "{at} down GEMM raster {PROD_RASTER} (production) {}",
        fmt(t[0])
    );
    for (&(nc, r), x) in variants.iter().zip(&t[1..]) {
        let d = t[0][0] - x[0];
        println!(
            "{at} down GEMM nc {nc} raster {r}: {} saving {d:.1} us per call ({:.1}%), {:.3} s \
             per {PREFILL} prefill",
            fmt(*x),
            100.0 * d / t[0][0],
            d * 1e-6 * MOE_LAYERS * chunks
        );
    }
    Ok(())
}
