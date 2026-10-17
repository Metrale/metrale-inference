// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Byte gate and timing of `METRALE_GLM_DEQUANT_PERSIST` at every rank-0 site shape
//! of GLM-5.3 at TP=3 that the dequantize + cuBLASLt prefill levers serve: the production
//! `glm5next_w4a16_wide::proj` / `glm5next_fp8_wide::proj` without a kept copy (dequantize into
//! the shared scratch, then the cuBLASLt GEMM) against the same call once
//! `glm5next_dequant_persist::keep_*` has kept the weight's BF16 copy (the GEMM alone).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every shape and every row count in `ROWS`, the two output
//!   buffers are equal byte for byte and not all zero, and unless the negative control (one
//!   element of the kept copy overwritten) makes the copy path's output differ, which proves the
//!   second call read the copy.
//!
//! Times: median, min and max of 7 host-timed batches of 10 calls at `TIME_ROWS` rows; the
//! per-chunk total weights each shape by its count on one rank.
//!
//! Run (GB10, < 1.5 GB of device memory):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example glm5next_dequant_persist_gate

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::{
    glm5next_dequant_persist as persist, glm5next_dequant_scratch, glm5next_fp8_wide,
    glm5next_w4a16_wide,
};
use metrale_model_layers::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};

/// 2026-10-15: The row counts gated: 1, odd small, a sub-chunk, the KDA chunk, and the two
/// prefill chunk widths.
const ROWS: [usize; 6] = [1, 17, 268, 1036, 8192, 8208];
const TIME_ROWS: usize = 8208;
const MMAX: usize = 8208;
const KMAX: usize = 11264;
const NMAX: usize = 11264;

/// 2026-10-15: `(n, k, sites on one rank, fp8, name)`.
const SHAPES: [(usize, usize, usize, bool, &str); 12] = [
    (2816, 4096, 34 * 3, false, "kda q/k/v"),
    (4096, 2816, 34, false, "kda o"),
    (768, 4096, 42 * 2, false, "shared gate/up"),
    (4096, 768, 42, false, "shared down"),
    (128, 4096, 34 * 2, false, "kda f_a/g_a"),
    (22, 4096, 34, false, "kda b"),
    (4096, 11264, 11, true, "dsa o_absorb"),
    (11264, 1536, 11, true, "dsa q_absorb"),
    (1536, 4096, 11, true, "dsa q_a"),
    (512, 4096, 11, true, "dsa kv_a"),
    (2816, 128, 34 * 2, true, "kda f_b/g_b"),
    (128, 4096, 11 * 2, true, "dsa wk/compress_gate"),
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

fn download(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-15: One site's weight, either tier.
enum Weight {
    W4(QuantizedWeight),
    Fp8(Fp8Weight),
}

impl Weight {
    fn random(gpu: &dyn GpuBackend, rng: &mut Rng, n: usize, k: usize, fp8: bool) -> Result<Self> {
        if fp8 {
            // E4M3 with the exponent field below 9 (no NaN code), per-row scales near 1/64.
            let q: Vec<u8> = (0..n * k)
                .map(|_| {
                    let u = rng.next();
                    ((u & 0x80) | ((u >> 8) % 9) << 3 | ((u >> 16) & 7)) as u8
                })
                .collect();
            let s: Vec<u8> = (0..n)
                .flat_map(|_| (0.01f32 + (rng.next() % 1000) as f32 * 1e-5).to_le_bytes())
                .collect();
            return Ok(Weight::Fp8(Fp8Weight {
                weight: upload(gpu, &q)?,
                row_scale: upload(gpu, &s)?,
                n: n as u32,
                k: k as u32,
                scale_format: WeightQuantFormat::Fp8PerRow,
            }));
        }
        let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
        let scale: Vec<u8> = (0..n * k / 16)
            .map(|_| (0x08 + rng.next() % 0x70) as u8)
            .collect();
        Ok(Weight::W4(QuantizedWeight {
            weight: upload(gpu, &packed)?,
            weight_scale: upload(gpu, &scale)?,
            weight_scale_2: 0.0123,
            ..QuantizedWeight::null()
        }))
    }

    fn key(&self) -> DevicePtr {
        match self {
            Weight::W4(w) => w.weight,
            Weight::Fp8(w) => w.weight,
        }
    }

    /// The production projection: the kept copy when there is one, else the scratch path.
    #[allow(clippy::too_many_arguments)]
    fn proj(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        c: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        match self {
            Weight::W4(w) => glm5next_w4a16_wide::proj(gpu, w, a, c, m, n, k, 0),
            Weight::Fp8(w) => glm5next_fp8_wide::proj(gpu, w, a, c, m, n, k, 0),
        }
    }

    fn keep(&self, gpu: &dyn GpuBackend, n: usize, k: usize, what: &str) -> Result<()> {
        match self {
            Weight::W4(w) => persist::keep_w4a16(gpu, w, n, k, what),
            Weight::Fp8(w) => persist::keep_fp8(gpu, w, what),
        }
    }
}

/// 2026-10-15: `(median, min, max)` microseconds per call over 7 batches of 10 calls.
fn time_us(gpu: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<(f64, f64, f64)> {
    f()?;
    let mut t = Vec::new();
    for _ in 0..7 {
        gpu.synchronize(0)?;
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            f()?;
        }
        gpu.synchronize(0)?;
        t.push(t0.elapsed().as_secs_f64() * 1e6 / 10.0);
    }
    t.sort_by(f64::total_cmp);
    Ok((t[3], t[0], t[6]))
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let mut rng = Rng(0x0dee_9e25_15a7_e001);
    let a: Vec<u8> = (0..MMAX * KMAX)
        .flat_map(|_| {
            let u = rng.next();
            let bits = ((u & 1) << 15) | ((115 + (u >> 1) % 17) << 7) | ((u >> 8) & 0x7f);
            (bits as u16).to_le_bytes()
        })
        .collect();
    let ad = upload(&gpu, &a)?;
    drop(a);
    let (c_old, c_new) = (gpu.alloc(MMAX * NMAX * 2)?, gpu.alloc(MMAX * NMAX * 2)?);
    for &(n, k, _, _, _) in &SHAPES {
        glm5next_dequant_scratch::reserve(&gpu, n, k, "persist gate")?;
    }
    let (mut saved, mut old_total) = (0.0f64, 0.0f64);
    for (n, k, count, fp8, name) in SHAPES {
        let w = Weight::random(&gpu, &mut rng, n, k, fp8)?;
        let olds = ROWS
            .iter()
            .map(|&m| {
                w.proj(&gpu, ad, c_old, m, n, k)?;
                download(&gpu, c_old, m * n * 2)
            })
            .collect::<Result<Vec<_>>>()?;
        let t_old = time_us(&gpu, || w.proj(&gpu, ad, c_old, TIME_ROWS, n, k))?;
        w.keep(&gpu, n, k, name)?;
        let copy = persist::copy_of(w.key(), n, k)?.expect("kept above");
        for (&m, old) in ROWS.iter().zip(&olds) {
            ensure!(old.iter().any(|&b| b != 0), "{name} M={m}: zero output");
            w.proj(&gpu, ad, c_new, m, n, k)?;
            let new = download(&gpu, c_new, m * n * 2)?;
            let diff = new.iter().zip(old).filter(|(x, y)| x != y).count();
            ensure!(diff == 0, "{name} [{n} x {k}] M={m}: {diff} bytes differ");
        }
        // Negative control: element (0, 0) of the kept copy set to 1e30 must move output column 0.
        let saved_elem = download(&gpu, copy, 2)?;
        gpu.copy_h2d(&half::bf16::from_f32(1e30).to_le_bytes(), copy)?;
        for (&m, old) in ROWS.iter().zip(&olds) {
            w.proj(&gpu, ad, c_new, m, n, k)?;
            let new = download(&gpu, c_new, m * n * 2)?;
            ensure!(
                new != *old,
                "{name} M={m}: negative control (corrupted copy) matched the scratch path"
            );
        }
        gpu.copy_h2d(&saved_elem, copy)?;
        let t_new = time_us(&gpu, || w.proj(&gpu, ad, c_new, TIME_ROWS, n, k))?;
        println!(
            "{name:>20} [{n:>5} x {k:>5}] x{count:<3} bytes == at M {ROWS:?}, control differs; \
             M={TIME_ROWS}: per-call dequant {:8.1} us [{:.1}..{:.1}]  kept copy {:8.1} us \
             [{:.1}..{:.1}]  saved {:7.1} us/call",
            t_old.0,
            t_old.1,
            t_old.2,
            t_new.0,
            t_new.1,
            t_new.2,
            t_old.0 - t_new.0
        );
        saved += (t_old.0 - t_new.0) * count as f64;
        old_total += t_old.0 * count as f64;
    }
    let (copies, bytes) = persist::kept();
    println!(
        "GATE PASS: {} shapes x {} row counts byte-identical, every negative control fired; \
         per {TIME_ROWS}-row chunk on one rank: {:.2} ms -> {:.2} ms (saved {:.2} ms); {copies} \
         copies, {bytes} bytes kept",
        SHAPES.len(),
        ROWS.len(),
        old_total / 1e3,
        (old_total - saved) / 1e3,
        saved / 1e3
    );
    Ok(())
}
