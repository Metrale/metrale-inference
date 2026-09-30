// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: GPU check that the Nemotron-H routed-expert grouped GEMMs write every output
//! column, through the launcher production uses, against a host reference: Nemotron-3-Nano's
//! intermediate size (N = 1856) and Nemotron-3-Super's intermediate and latent sizes (2688,
//! 1024), each on its own target's kernels.
//!
//! On `gb10/common` both `moe_w4a16_grouped_gemm_ptrtable` and `_ptrtable_t` are 64 columns
//! wide. Launched on a grid sized for 128 they wrote 960 of 1856 columns; the launcher now
//! sizes the grid from the published tile. Each output starts at a sentinel, so an
//! unwritten column fails the comparison.
//!
//! `#[ignore]`: it needs a GPU and the built GB10 kernels. Run with:
//! ```text
//! cargo test -p metrale-model-layers --release grouped_gemm_writes_every_column \
//!   -- --ignored --nocapture
//! ```
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::moe_w4a16_grouped_gemm_ptrtable_n128;

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
const K: usize = 256;
const ROWS: usize = 3;
const SENTINEL: u16 = 0x7FC1;

fn e4m3(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    s * if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

/// 2026-09-29: One expert's NVFP4 weight, `[n, K/2]` packed and `[n, K/16]` scales, from a
/// fixed LCG so the run is reproducible; scales stay in E4M3's normal range.
fn weight(seed: u32, n: usize) -> (Vec<u8>, Vec<u8>) {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 8) as u8
    };
    let packed = (0..n * K / 2).map(|_| next()).collect();
    let scales = (0..n * K / 16).map(|_| 0x30 + (next() & 0x0F)).collect();
    (packed, scales)
}

fn dequant(packed: &[u8], scales: &[u8], n: usize, k: usize) -> f32 {
    let b = packed[n * K / 2 + k / 2];
    let code = if k.is_multiple_of(2) { b & 0xF } else { b >> 4 };
    E2M1[code as usize] * e4m3(scales[n * K / 16 + k / 16])
}

fn transpose(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let mut t = vec![0u8; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = src[r * cols + c];
        }
    }
    t
}

#[test]
#[ignore]
fn grouped_gemm_writes_every_column_at_nemotron_h_sizes() {
    for (model, n) in [
        ("nemotron-3-nano-30b-a3b", 1856),
        ("nemotron-super-120b-a12b", 2688),
        ("nemotron-super-120b-a12b", 1024),
    ] {
        check(model, n);
    }
}

fn check(model: &str, n_cols: usize) {
    let set = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|t| t.target.model == model && t.ptx_arch.starts_with("sm_121"))
        .expect("build the gb10 Nemotron-H kernels first");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let a: Vec<f32> = (0..ROWS * K)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
        .collect();
    let a_bytes: Vec<u8> = a.iter().flat_map(|&x| bf16(x).to_le_bytes()).collect();
    let a_dev = upload(g, &a_bytes);
    let (packed, scales) = weight(7, n_cols);
    for (entry, transposed) in [
        ("moe_w4a16_grouped_gemm_ptrtable", false),
        ("moe_w4a16_grouped_gemm_ptrtable_t", true),
    ] {
        let kernel = g.kernel("moe_w4a16", entry).unwrap();
        assert_eq!(
            g.kernel_n_tile(kernel).unwrap(),
            64,
            "{entry}: gb10/common is 64 wide"
        );
        let (p, s) = if transposed {
            (
                transpose(&packed, n_cols, K / 2),
                transpose(&scales, n_cols, K / 16),
            )
        } else {
            (packed.clone(), scales.clone())
        };
        let ptrs = |d: DevicePtr| upload(g, &d.0.to_le_bytes());
        let packed_ptrs = ptrs(upload(g, &p));
        let scale_ptrs = ptrs(upload(g, &s));
        let scale2 = upload(g, &1.0f32.to_le_bytes());
        let offsets = upload(g, &[0i32, ROWS as i32].map(|v| v.to_le_bytes()).concat());
        let tokens = upload(g, &[0i32, 1, 2].map(|v| v.to_le_bytes()).concat());
        let c = upload(g, &SENTINEL.to_le_bytes().repeat(ROWS * n_cols));
        moe_w4a16_grouped_gemm_ptrtable_n128(
            g,
            kernel,
            a_dev,
            packed_ptrs,
            scale_ptrs,
            scale2,
            c,
            offsets,
            tokens,
            1,
            n_cols as u32,
            K as u32,
            1,
            g.default_stream(),
        )
        .unwrap();
        g.synchronize(g.default_stream()).unwrap();
        let mut out = vec![0u8; ROWS * n_cols * 2];
        g.copy_d2h(c, &mut out).unwrap();
        let mut unwritten = 0;
        let (mut err, mut norm) = (0f64, 0f64);
        for r in 0..ROWS {
            for n in 0..n_cols {
                let bits =
                    u16::from_le_bytes([out[(r * n_cols + n) * 2], out[(r * n_cols + n) * 2 + 1]]);
                if bits == SENTINEL {
                    unwritten += 1;
                    continue;
                }
                let got = f32::from_bits((bits as u32) << 16) as f64;
                let want: f64 = (0..K)
                    .map(|k| {
                        let x = f32::from_bits((bf16(a[r * K + k]) as u32) << 16);
                        (x * dequant(&packed, &scales, n, k)) as f64
                    })
                    .sum();
                err += (got - want).powi(2);
                norm += want.powi(2);
            }
        }
        assert_eq!(
            unwritten,
            0,
            "{model} {entry} N={n_cols}: {unwritten} of {} outputs unwritten",
            ROWS * n_cols
        );
        let rel = (err / norm).sqrt();
        assert!(rel < 1e-2, "{entry}: relative error {rel}");
        println!("{model} {entry}: all {n_cols} columns written, rel err {rel:.2e}");
    }
}
