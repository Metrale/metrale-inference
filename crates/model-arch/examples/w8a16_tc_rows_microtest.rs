// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `w8a16_tc_rows` (`kernels/gb10/common/w8a16_tc_rows.cu`) at Qwen3.6-35B-A3B
//! decode projection shapes: row invariance and closeness to `w8a16_gemm_pipelined_m32`.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every shape and M, each row of the M-row launch equals,
//!   byte for byte, the same row launched alone, rows past M and the pitch padding still
//!   hold the sentinel, and the output is within `MAX_REL_L2` (relative L2) of the 32-row
//!   tile twin on the same inputs.
//! - Before the sweep, a flipped output bit and a zeroed row must be refused by those checks.
//!
//! Shapes: GDN `in_proj_qkvz` (12288 x 2048), `out_proj` (2048 x 4096), attention Q with its
//! gate (8192 x 2048) and K/V (512 x 2048). The output pitch is N + 64 so a write past a
//! row's N columns shows. Each launch's mean time over 20 iterations is printed.
//!
//! Run (GB10):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example w8a16_tc_rows_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;

const MAX_M: u32 = 256;
const PAD: u32 = 64;
const SENTINEL: u8 = 0x5a;
/// 2026-09-28: Both kernels round the FP32 sum once to BF16; they differ in summation order
/// only, measured ~2e-3 relative L2, so 1e-2.
const MAX_REL_L2: f64 = 1e-2;
const SHAPES: [(u32, u32); 4] = [(12288, 2048), (2048, 4096), (8192, 2048), (512, 2048)];
const ROWS: [u32; 18] = [
    1, 2, 3, 4, 7, 8, 9, 16, 17, 32, 33, 64, 65, 96, 127, 128, 129, 256,
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn rows_f32(b: &[u8], m: u32, n: u32, ldc: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity((m * n) as usize);
    for r in 0..m {
        for c in 0..n {
            let i = ((r * ldc + c) * 2) as usize;
            v.push(bf16::from_bits(u16::from_le_bytes([b[i], b[i + 1]])).to_f32());
        }
    }
    v
}

/// 2026-09-28: Row r of `got` equals `alone[r]`; bytes past each row's N columns and past
/// row M hold the sentinel.
fn check_rows(got: &[u8], alone: &[Vec<u8>], m: u32, n: u32, ldc: u32) -> Result<()> {
    let row_bytes = (ldc * 2) as usize;
    for (r, want) in alone.iter().enumerate().take(m as usize) {
        let row = &got[r * row_bytes..(r + 1) * row_bytes];
        ensure!(
            row[..(n * 2) as usize] == want[..],
            "M={m}: row {r} differs from the row alone"
        );
        ensure!(
            row[(n * 2) as usize..].iter().all(|&b| b == SENTINEL),
            "M={m}: row {r} written past N"
        );
    }
    ensure!(
        got[m as usize * row_bytes..].iter().all(|&b| b == SENTINEL),
        "M={m}: rows past M were written"
    );
    Ok(())
}

fn check_close(got: &[f32], want: &[f32]) -> Result<f64> {
    ensure!(got.iter().all(|v| v.is_finite()), "non-finite output");
    let num: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
    ensure!(
        rel <= MAX_REL_L2,
        "relative L2 {rel:.3e} from the tile twin exceeds {MAX_REL_L2:.0e}"
    );
    Ok(rel)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let m32 = gpu.kernel("w8a16_gemm_pipelined_m32", "w8a16_gemm_pipelined_m32")?;
    let full = gpu.kernel("w8a16_gemm_pipelined", "w8a16_gemm_pipelined")?;
    let mut rng = Rng(0x7472_2d72_6f77_2028);
    let mut first = true;
    for (n, k) in SHAPES {
        let ldc = n + PAD;
        let w: Vec<u8> = (0..n * k)
            .map(|_| {
                let x = rng.next();
                ((x % 120) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect();
        let s: Vec<u8> = (0..(n / 128) * (k / 128))
            .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 512.0).to_le_bytes())
            .collect();
        let a: Vec<u8> = (0..MAX_M * k)
            .flat_map(|_| {
                bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let (wd, sd, ad) = (upload(&gpu, &w)?, upload(&gpu, &s)?, upload(&gpu, &a)?);
        let sentinel = vec![SENTINEL; (MAX_M * ldc * 2) as usize];
        let out = upload(&gpu, &sentinel)?;
        let run = |m: u32, input: DevicePtr, tc: bool| -> Result<Vec<u8>> {
            gpu.copy_h2d(&sentinel, out)?;
            if tc {
                ops::w8a16_tc_rows(&gpu, input, wd, sd, out, m, n, k, k, ldc, 0)?;
            } else {
                ops::w8a16_gemm_pipelined_m32_strided(
                    &gpu, m32, input, wd, sd, out, m, n, k, k, ldc, 0,
                )?;
            }
            gpu.synchronize(0)?;
            let mut b = vec![0u8; sentinel.len()];
            gpu.copy_d2h(out, &mut b)?;
            Ok(b)
        };
        let alone: Vec<Vec<u8>> = (0..MAX_M)
            .map(|r| {
                run(1, ad.offset((r * k * 2) as usize), true)
                    .map(|b| b[..(n * 2) as usize].to_vec())
            })
            .collect::<Result<_>>()?;
        for m in ROWS {
            let tc = run(m, ad, true)?;
            let twin = run(m, ad, false)?;
            if first {
                let mut bad = tc.clone();
                bad[5] ^= 1;
                ensure!(
                    check_rows(&bad, &alone, m, n, ldc).is_err(),
                    "a flipped bit was admitted"
                );
                println!("KNOWN_BAD flipped-bit: refused");
                let zero = vec![0.0f32; (m * n) as usize];
                ensure!(
                    check_close(&zero, &rows_f32(&twin, m, n, ldc)).is_err(),
                    "a zeroed row was admitted"
                );
                println!("KNOWN_BAD zeroed-row: refused");
                first = false;
            }
            check_rows(&tc, &alone, m, n, ldc)?;
            let rel = check_close(&rows_f32(&tc, m, n, ldc), &rows_f32(&twin, m, n, ldc))?;
            let time = |tc: bool| -> Result<f64> {
                gpu.synchronize(0)?;
                let t = std::time::Instant::now();
                for _ in 0..20 {
                    if tc {
                        ops::w8a16_tc_rows(&gpu, ad, wd, sd, out, m, n, k, k, ldc, 0)?;
                    } else {
                        ops::w8a16_gemm_pipelined_m32_strided(
                            &gpu, m32, ad, wd, sd, out, m, n, k, k, ldc, 0,
                        )?;
                    }
                }
                gpu.synchronize(0)?;
                Ok(t.elapsed().as_secs_f64() * 1e6 / 20.0)
            };
            let (us_tc, us_twin) = (time(true)?, time(false)?);
            // 2026-09-30: Above 64 rows the canonical tiers ran the 128-row tile before the
            // chunked row-tile entry replaced it: its time, for the cost of the swap.
            let us_full = if m > 64 {
                gpu.synchronize(0)?;
                let t = std::time::Instant::now();
                for _ in 0..20 {
                    ops::w8a16_gemm_pipelined(&gpu, full, ad, wd, sd, out, m, n, k, 0)?;
                }
                gpu.synchronize(0)?;
                t.elapsed().as_secs_f64() * 1e6 / 20.0
            } else {
                f64::NAN
            };
            println!(
                "N={n:5} K={k} M={m:3} rows == alone, rel_l2 vs m32 {rel:.2e}; m32 {us_twin:6.1}us tc {us_tc:6.1}us full {us_full:6.1}us  PASS"
            );
        }
    }
    println!(
        "ALL PASS: w8a16_tc_rows is row-invariant and within {MAX_REL_L2:.0e} of w8a16_gemm_pipelined_m32, M=1..256"
    );
    Ok(())
}
