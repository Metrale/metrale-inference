// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: GPU tests of the pinned cuBLASLt BF16 GEMM ([`super::cublas_bf16_proj_dense_pinned`],
//! the `METRALE_DENSE_BF16_CUBLAS_PINNED` arm of `dense_gemm_bf16_pipelined`) on the dense
//! prefill projections of the 2,560-hidden model: GDN in/out, attention o, the hyper-connection
//! down/up and the shared expert's down.
//!
//! - `..._rows_are_m_invariant` is the gate: the bytes of a row are the same whether the row is
//!   computed in a call of 16, 20, 2,048 or 4,096 rows, at row 0 or at an offset, and on a rerun.
//!   Sampled rows are first checked against an FP32 host product, so an invariant but wrong (or
//!   unwritten) result fails. For contrast it prints how many elements the per-call heuristic
//!   path and the pipelined kernel change at each M (not asserted).
//! - `..._timing` prints microseconds per call of the pipelined kernel, the per-call heuristic
//!   path and the pinned path at 2,048 and 4,096 rows.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release dense_cublas_pinned_tests -- --ignored --nocapture --test-threads 1
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// 2026-10-14: (N, K) of the projections: GDN in (q, k, v, z), GDN out and attention o, the
/// hyper-connection down and up, the shared expert's down.
const SHAPES: [(u32, u32); 5] = [
    (16_384, 2_560),
    (2_560, 6_144),
    (320, 10_240),
    (10_240, 320),
    (2_560, 640),
];
/// 2026-10-14: Rows of the reference call, and of the activation every case slices.
const ROWS: u32 = 4_096;
/// 2026-10-14: (rows, first row) of each compared call: decode width, the prefill tail chunk at
/// row 0 and near the end, one slab at row 0 and at its offset, and the full call again.
const CASES: [(u32, u32); 6] = [
    (16, 0),
    (20, 0),
    (20, 4_070),
    (2_048, 0),
    (2_048, 2_048),
    (4_096, 0),
];
/// 2026-10-14: Rows of the reference call checked against the host product.
const CHECKED_ROWS: [u32; 4] = [0, 1, 2_047, 4_095];

fn backend() -> metrale_gpu_runtime::cuda_backend::MetraleCudaBackend {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend")
}

/// 2026-10-14: `n` BF16 values uniform in [-scale, scale] from an xorshift stream.
fn bf16s(n: usize, seed: u32, scale: f32) -> Vec<u16> {
    let mut x = seed.wrapping_mul(2_654_435_761).max(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let v = (x >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            ((v * scale).to_bits() >> 16) as u16
        })
        .collect()
}

fn f32_of(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn upload(g: &dyn GpuBackend, v: &[u16]) -> DevicePtr {
    let bytes: Vec<u8> = v.iter().flat_map(|b| b.to_le_bytes()).collect();
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(&bytes, p).unwrap();
    p
}

fn download(g: &dyn GpuBackend, p: DevicePtr, elements: usize) -> Vec<u16> {
    let mut bytes = vec![0u8; elements * 2];
    g.copy_d2h(p, &mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// 2026-10-14: The three ways one projection can run.
#[derive(Clone, Copy)]
enum Arm {
    Pinned,
    Heuristic,
    Pipelined(KernelHandle),
}

/// 2026-10-14: One projection's device operands.
struct Problem {
    act: DevicePtr,
    weight: DevicePtr,
    out: DevicePtr,
    n: u32,
    k: u32,
}

impl Problem {
    fn new(g: &dyn GpuBackend, n: u32, k: u32) -> (Self, Vec<u16>, Vec<u16>) {
        let act = bf16s((ROWS * k) as usize, n ^ k, 1.0);
        let weight = bf16s((n * k) as usize, n.wrapping_add(k), 1.0 / (k as f32).sqrt());
        let p = Self {
            act: upload(g, &act),
            weight: upload(g, &weight),
            out: g.alloc((ROWS * n) as usize * 2).unwrap(),
            n,
            k,
        };
        (p, act, weight)
    }

    /// 2026-10-14: Launch `rows` rows from `first` on `arm` into the same rows of `out`.
    fn launch(&self, g: &dyn GpuBackend, arm: Arm, rows: u32, first: u32, s: u64) {
        let (n, k) = (self.n, self.k);
        let a = self.act.offset((first * k) as usize * 2);
        let c = self.out.offset((first * n) as usize * 2);
        let w = self.weight;
        match arm {
            Arm::Pinned => ops::cublas_bf16_proj_dense_pinned(a, w, c, rows, n, k, s),
            Arm::Heuristic => ops::cublas_bf16_proj_dense(a, w, c, rows, n, k, s),
            Arm::Pipelined(kernel) => {
                let dw = DenseWeight { weight: w };
                ops::dense_gemm_bf16_pipelined(g, kernel, a, &dw, c, rows, n, k, s)
            }
        }
        .unwrap_or_else(|e| panic!("{} N={n} K={k} M={rows}: {e:#}", name(arm)));
    }

    /// 2026-10-14: The output rows `first..first + rows` after one call on `arm`.
    fn run(&self, g: &dyn GpuBackend, arm: Arm, rows: u32, first: u32, s: u64) -> Vec<u16> {
        g.memset_async(self.out, 0xFF, (ROWS * self.n) as usize * 2, s)
            .unwrap();
        self.launch(g, arm, rows, first, s);
        g.synchronize(s).unwrap();
        let all = download(g, self.out, (ROWS * self.n) as usize);
        let n = self.n as usize;
        all[first as usize * n..(first + rows) as usize * n].to_vec()
    }

    fn free(self, g: &dyn GpuBackend) {
        for p in [self.act, self.weight, self.out] {
            g.free(p).unwrap();
        }
    }
}

fn name(arm: Arm) -> &'static str {
    match arm {
        Arm::Pinned => "pinned",
        Arm::Heuristic => "heuristic",
        Arm::Pipelined(_) => "pipelined",
    }
}

/// 2026-10-14: Elements of `got` whose bytes differ from `want`.
fn differing(want: &[u16], got: &[u16]) -> usize {
    want.iter().zip(got).filter(|(w, g)| w != g).count()
}

/// 2026-10-14: Panics unless every element of the checked rows of `out` (the reference call) is
/// within 1% of the FP32 product's absolute-value sum (plus 1e-3) of the host product.
fn check_against_host(out: &[u16], act: &[u16], weight: &[u16], n: u32, k: u32) {
    let (n, k) = (n as usize, k as usize);
    for row in CHECKED_ROWS.map(|r| r as usize) {
        let a = &act[row * k..(row + 1) * k];
        for col in 0..n {
            let w = &weight[col * k..(col + 1) * k];
            let (dot, mag) = a.iter().zip(w).fold((0f32, 0f32), |(d, m), (&x, &y)| {
                let p = f32_of(x) * f32_of(y);
                (d + p, m + p.abs())
            });
            let got = f32_of(out[row * n + col]);
            assert!(
                (got - dot).abs() <= 1e-2 * mag + 1e-3,
                "N={n} K={k} row {row} col {col}: got {got}, host {dot}"
            );
        }
    }
}

#[test]
#[ignore]
fn pinned_cublas_bf16_rows_are_m_invariant() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let pipe = g.kernel("gemm", "dense_gemm_bf16_pipelined").unwrap();
    let s = g.create_stream().unwrap();
    let mut failures = Vec::new();
    for (n, k) in SHAPES {
        let (p, act, weight) = Problem::new(g, n, k);
        let full: Vec<Vec<u16>> = [Arm::Pinned, Arm::Heuristic, Arm::Pipelined(pipe)]
            .map(|arm| p.run(g, arm, ROWS, 0, s))
            .into();
        check_against_host(&full[0], &act, &weight, n, k);
        for (rows, first) in CASES {
            let range = (first * n) as usize..((first + rows) * n) as usize;
            let mut line = format!("N={n:5} K={k:5} M={rows:4} first={first:4}:");
            for (arm, want) in [Arm::Pinned, Arm::Heuristic, Arm::Pipelined(pipe)]
                .into_iter()
                .zip(&full)
            {
                let d = differing(&want[range.clone()], &p.run(g, arm, rows, first, s));
                line += &format!(" {} {d}", name(arm));
                if matches!(arm, Arm::Pinned) && d != 0 {
                    failures.push(format!("N={n} K={k} M={rows} first={first}: {d} differ"));
                }
            }
            println!("{line} (of {} differ)", range.len());
        }
        p.free(g);
    }
    assert!(failures.is_empty(), "pinned rows depend on M: {failures:?}");
}

#[test]
#[ignore]
fn pinned_cublas_bf16_timing() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let pipe = g.kernel("gemm", "dense_gemm_bf16_pipelined").unwrap();
    let s = g.create_stream().unwrap();
    for (n, k) in SHAPES {
        let (p, _, _) = Problem::new(g, n, k);
        for rows in [2_048, 4_096] {
            let us = [Arm::Pipelined(pipe), Arm::Heuristic, Arm::Pinned].map(|arm| {
                p.launch(g, arm, rows, 0, s);
                g.synchronize(s).unwrap();
                let iters = 20;
                let t = std::time::Instant::now();
                for _ in 0..iters {
                    p.launch(g, arm, rows, 0, s);
                }
                g.synchronize(s).unwrap();
                t.elapsed().as_secs_f64() * 1e6 / f64::from(iters)
            });
            let tflops = 2.0 * f64::from(rows) * f64::from(n) * f64::from(k) / us[2] / 1e6;
            println!(
                "N={n:5} K={k:5} M={rows:4}: pipelined {:8.1} us, heuristic {:8.1} us, pinned \
                 {:8.1} us ({tflops:.0} TFLOP/s), pinned {:.2}x the pipelined",
                us[0],
                us[1],
                us[2],
                us[0] / us[2]
            );
        }
        p.free(g);
    }
}
