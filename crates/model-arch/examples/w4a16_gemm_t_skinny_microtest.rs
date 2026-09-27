// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: Byte oracle for the skinny arm of the 35B `w4a16_gemm_t` (the
//! NVFP4 lm_head tile GEMM that canonical row tiers run at every row count).
//!
//! At M <= 16 the kernel has every warp take rows 0..15 and a quarter of the
//! N tiles instead of a 16-row slab. The same launch at M = 17 takes the slab
//! layout, and a row's arithmetic does not depend on M in either, so rows
//! 0..M-1 of the M = 17 launch are the reference: the run exits 0 only if, at
//! every M in 1..=16 and every shape, those rows are byte-identical and the
//! skinny launch wrote nothing past row M. It also times both launches over
//! weight copies that do not fit in L2.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//!   cargo run -p metrale-model-arch --release --example w4a16_gemm_t_skinny_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

const REF_M: usize = 17;
const SENTINEL: u8 = 0x5a;
const REPS: usize = 10;
/// 2026-09-27: Weight bytes the timing rotates through, several times L2.
const COLD_BYTES: usize = 320 << 20;
/// 2026-09-27: `(label, N, K)`: the 35B lm_head (vocab padded to 128) and a
/// narrow shape with a partial last N tile.
const SHAPES: &[(&str, usize, usize)] = &[
    ("35B lm_head N=248320 K=2048", 248_320, 2048),
    ("narrow     N=1000   K=512 ", 1000, 512),
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(16))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    bs: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldb: usize,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n as u32, 128), div_ceil(m as u32, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(bs)
        .arg_f32(1.0 / 64.0)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(ldb as u32)
        .launch(0)
}

fn main() -> Result<()> {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.6-35b-a3b", "nvfp4")
        .context("no compiled qwen3.6-35b-a3b/nvfp4 kernel set")?;
    let backend = MetraleCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let kh = g.kernel("w4a16", "w4a16_gemm_t")?;
    let mut rng = Rng(0x2026_0927_1d4a_5eed);
    let mut failures = 0usize;

    for &(label, n, k) in SHAPES {
        let ldb = n.div_ceil(128) * 128;
        let packed: Vec<u8> = (0..k / 2 * ldb).map(|_| rng.next() as u8).collect();
        // 2026-09-27: E4M3 scale bytes with exponent field 4..11 (no NaN, no
        // saturation), either sign.
        let scales: Vec<u8> = (0..k / 16 * ldb)
            .map(|_| {
                let x = rng.next();
                (((x % 8 + 4) as u8) << 3) | ((x >> 8) as u8 & 7) | (((x >> 16) & 1) as u8) << 7
            })
            .collect();
        let acts: Vec<u8> = (0..REF_M * k)
            .flat_map(|_| {
                half::bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let (b, bs, a) = (up(g, &packed)?, up(g, &scales)?, up(g, &acts)?);
        let out_bytes = REF_M * n * 2;
        let reference = up(g, &vec![SENTINEL; out_bytes])?;
        let skinny = up(g, &vec![SENTINEL; out_bytes])?;
        launch(g, kh, a, b, bs, reference, REF_M, n, k, ldb)?;
        g.synchronize(0)?;
        let mut want = vec![0u8; out_bytes];
        g.copy_d2h(reference, &mut want)?;
        let before = failures;
        for m in 1..=16usize {
            g.copy_h2d(&vec![SENTINEL; out_bytes], skinny)?;
            launch(g, kh, a, b, bs, skinny, m, n, k, ldb)?;
            g.synchronize(0)?;
            let mut got = vec![0u8; out_bytes];
            g.copy_d2h(skinny, &mut got)?;
            let live = m * n * 2;
            let diff = (0..live).find(|&i| got[i] != want[i]);
            let stray = (live..out_bytes).find(|&i| got[i] != SENTINEL);
            match (diff, stray) {
                (None, None) => {}
                (d, s) => {
                    println!(
                        "FAIL {label} M={m}: first differing byte {d:?}, write past row M at {s:?}"
                    );
                    failures += 1;
                }
            }
        }
        if failures == before {
            println!("{label}: M=1..16 rows byte-identical to the M={REF_M} slab layout");
        }

        let copies = COLD_BYTES.div_ceil(packed.len()).clamp(2, 64);
        let cold: Vec<DevicePtr> = (0..copies).map(|_| up(g, &packed)).collect::<Result<_>>()?;
        for m in [1usize, 2, 4, 8, 16, REF_M] {
            g.synchronize(0)?;
            let t = std::time::Instant::now();
            for i in 0..REPS {
                launch(g, kh, a, cold[i % copies], bs, skinny, m, n, k, ldb)?;
            }
            g.synchronize(0)?;
            let us = t.elapsed().as_secs_f64() * 1e6 / REPS as f64;
            println!(
                "TIME {label} M={m:2}: {us:8.1} us ({:.0} GB/s of weight)",
                (packed.len() + scales.len()) as f64 / us / 1e3
            );
        }
        for p in cold {
            g.free(p)?;
        }
    }
    ensure!(failures == 0, "{failures} case(s) failed");
    println!("ALL PASS: w4a16_gemm_t skinny arm byte-identical at M=1..16");
    Ok(())
}
