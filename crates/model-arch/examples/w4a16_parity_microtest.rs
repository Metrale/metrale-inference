// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Parity gate for the transposed-B W4A16 (NVFP4) GEMMs at M = 17.
//!
//! On the same random NVFP4 data, each of `w4a16_gemm_t`, `w4a16_gemm_t_k64`,
//! `w4a16_gemm_t_m128` and `w4a16_gemm_t_k64_p3` that the target has (absent
//! ones are skipped) is compared with the base `w4a16_gemm`:
//!
//!   C_base = A · dequant(B)        (`w4a16_gemm`, B as [N, K/2])
//!   C_x    = A · dequant(B_t)      (the kernel under test, B_t as [K/2, N])
//!
//! B_t is the byte transpose `QuantizedWeight::transpose_for_gemm` performs. A
//! kernel passes at cosine >= `PASS_COS`; max |delta| is printed, not gated.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//!   cargo run -p metrale-model-arch --release --example w4a16_parity_microtest \
//!       --features cuda,gpu-examples
//!
//! 2026-10-05: The transposed tiles are also compared with EACH OTHER byte for byte (every tile
//! against `w4a16_gemm_t_p3` at the same rows): a tile choice that keeps each output's K-sum order
//! (the N tile, the CTA count, the pipeline depth) must not change a bit, which is what lets a
//! class pick its tile from its SM count. Each output also prints an `XCLASS-DIGEST` line, so the
//! same run on two hardware classes is a cross-class byte check. Rows 17, 64 and 256.
//!
//! Exit 0 = every tested kernel passes; 1 = at least one fails (named).

#[path = "common/xclass_digest.rs"]
mod xclass_digest;

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

/// 2026-10-05: The row counts: the original M = 17, one M64 tile, four M64 / two M128 tiles.
const ROWS: [usize; 3] = [17, 64, 256];
const M_MAX: usize = 256;
const GROUP: usize = 16;
const PASS_COS: f64 = 0.999;

/// 2026-09-25: `(label, N, K)`.
const SHAPES: &[(&str, usize, usize)] = &[
    ("ffn_gate/up N=17408 K=5120 ", 17408, 5120),
    ("ffn_down    N=5120  K=17408", 5120, 17408),
    ("attn_q      N=12288 K=5120 ", 12288, 5120),
    ("attn_o      N=5120  K=6144 ", 5120, 6144),
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
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn dn_raw(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    grid: [u32; 3],
    block: u32,
    m: usize,
    a: DevicePtr,
    b: DevicePtr,
    bs: DevicePtr,
    c: DevicePtr,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid(grid)
        .block([block, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(bs)
        .arg_f32(0.01)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        // 2026-09-25: `ldb`, the transposed-B row stride (`n` when packed), for the
        // kernels that declare a ninth argument; the others do not read it.
        .arg_u32(n as u32)
        .launch(0)
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;

    let base_k = g.kernel("w4a16", "w4a16_gemm")?;
    // 2026-10-05: `(name, module, N tile, M tile, block)`. `w4a16_gemm_t_p3` first: the reference
    // of the tile-to-tile byte comparison.
    let t_kernels: Vec<(&str, KernelHandle, u32, u32, u32)> = [
        ("w4a16_gemm_t_p3", "w4a16", 128, 64, 128),
        ("w4a16_gemm_t", "w4a16", 128, 64, 128),
        ("w4a16_gemm_t_k64", "w4a16", 128, 64, 128),
        ("w4a16_gemm_t_k64_p3", "w4a16", 128, 64, 128),
        ("w4a16_gemm_t_k64_n64_p3", "w4a16", 64, 64, 128),
        ("w4a16_gemm_t_m128", "w4a16", 128, 128, 128),
        ("w4a16_gemm_t_m128_v2", "w4a16_v2", 128, 128, 256),
    ]
    .into_iter()
    .filter_map(|(name, module, nt, mt, block)| {
        g.kernel(module, name)
            .ok()
            .map(|h| (name, h, nt, mt, block))
    })
    .collect();
    let mut tiles_identical = true;

    let mut all_ok = true;
    for &(label, n, k) in SHAPES {
        let mut r = Rng(0x517A_C0DE ^ (n as u64) << 20 ^ k as u64);
        let half_k = k / 2;
        let groups = k / GROUP;

        // 2026-09-25: A `[M, K]` BF16: uniform draws in [-0.25, 0.25), rounded to BF16.
        let a_host: Vec<u8> = (0..M_MAX * k)
            .flat_map(|_| {
                bf16::from_f32((r.unit() - 0.5) * 0.5)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        // 2026-09-25: B packed E2M1 `[N, K/2]`; every nibble is a valid E2M1 value.
        let b_host: Vec<u8> = (0..n * half_k).map(|_| r.byte()).collect();
        // 2026-09-25: B scales E4M3 `[N, K/16]`, bytes 0x28..=0x47: finite
        // positive values 0.25 to 3.75.
        let bs_host: Vec<u8> = (0..n * groups).map(|_| 0x28 + (r.byte() & 0x1F)).collect();

        let mut bt_host = vec![0u8; n * half_k];
        for i in 0..n {
            for j in 0..half_k {
                bt_host[j * n + i] = b_host[i * half_k + j];
            }
        }
        let mut bst_host = vec![0u8; n * groups];
        for i in 0..n {
            for j in 0..groups {
                bst_host[j * n + i] = bs_host[i * groups + j];
            }
        }

        let a = up(g, &a_host)?;
        let b = up(g, &b_host)?;
        let bs = up(g, &bs_host)?;
        let bt = up(g, &bt_host)?;
        let bst = up(g, &bst_host)?;
        let c_base = g.alloc(M_MAX * n * 2)?;
        let c_test = g.alloc(M_MAX * n * 2)?;

        for m in ROWS {
            launch(
                g,
                base_k,
                [div_ceil(n as u32, 64), div_ceil(m as u32, 64), 1],
                128,
                m,
                a,
                b,
                bs,
                c_base,
                n,
                k,
            )?;
            g.synchronize(0)?;
            let base_out = to_f32(&dn_raw(g, c_base, m * n)?);
            let mut reference: Option<Vec<u8>> = None;

            for &(name, kh, nt, mt, block) in &t_kernels {
                let grid = [div_ceil(n as u32, nt), div_ceil(m as u32, mt), 1];
                g.memset(c_test, 0, m * n * 2)?;
                launch(g, kh, grid, block, m, a, bt, bst, c_test, n, k)?;
                g.synchronize(0)?;
                let raw = dn_raw(g, c_test, m * n)?;
                xclass_digest::print(&format!("w4a16-tile n={n} k={k} m={m} {name}"), &raw);
                let same = match &reference {
                    None => {
                        reference = Some(raw.clone());
                        true
                    }
                    Some(r) => *r == raw,
                };
                tiles_identical &= same;
                let out = to_f32(&raw);

                let c = cos(&out, &base_out);
                let max_abs_base = base_out.iter().fold(0f32, |m, v| m.max(v.abs()));
                let max_d = out
                    .iter()
                    .zip(&base_out)
                    .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                let ok = c >= PASS_COS;
                all_ok &= ok;
                eprintln!(
                    "{label}  m={m:<3} {name:<24} cos={c:.7}  max|Δ|={max_d:.5} (base max|C|={max_abs_base:.3})  \
                 bytes-vs-t_p3={same}  {}",
                    if ok {
                        "PASS"
                    } else {
                        "FAIL ← kernel disagrees with base"
                    }
                );
            }
            eprintln!();
        }
        // 2026-10-05: The row tiles' N tile: each `_w2` entry (32 columns, 2 warps) against its
        // 4-warp twin on the row-major weight, byte for byte, at every row count it serves.
        // 2026-10-09: The prefetch-distance points (`_pf2`, `_pf3`, 4 warps) against their PF 1
        // twin the same way; `cols` below is 64 for them.
        for (four, two, rows) in [
            ("w4a16_tc_rows_16", "w4a16_tc_rows_16_w2", 16usize),
            ("w4a16_tc_rows_32", "w4a16_tc_rows_32_w2", 32),
            ("w4a16_tc_rows_64", "w4a16_tc_rows_64_w2", 64),
            ("w4a16_tc_rows_16", "w4a16_tc_rows_16_pf2", 16),
            ("w4a16_tc_rows_32", "w4a16_tc_rows_32_pf2", 32),
            ("w4a16_tc_rows_32", "w4a16_tc_rows_32_pf3", 32),
            ("w4a16_tc_rows_64", "w4a16_tc_rows_64_pf2", 64),
            ("w4a16_tc_rows_64", "w4a16_tc_rows_64_pf3", 64),
        ] {
            let (Ok(k4), Ok(k2)) = (
                g.kernel("w4a16_tc_rows", four),
                g.kernel("w4a16_tc_rows", two),
            ) else {
                eprintln!("SKIP {four}/{two}: not in this target's module set");
                continue;
            };
            for m in [1, rows / 2 + 1, rows] {
                let mut out = Vec::new();
                let cols2 = if two.ends_with("_w2") { 32u32 } else { 64 };
                for (kh, cols) in [(k4, 64u32), (k2, cols2)] {
                    g.memset(c_test, 0, m * n * 2)?;
                    KernelLaunch::new(g, kh)
                        .grid([div_ceil(n as u32, cols), 1, 1])
                        .block([cols * 2, 1, 1])
                        .arg_ptr(a)
                        .arg_ptr(b)
                        .arg_ptr(bs)
                        .arg_f32(0.01)
                        .arg_ptr(c_test)
                        .arg_u32(m as u32)
                        .arg_u32(n as u32)
                        .arg_u32(k as u32)
                        .arg_u32(k as u32)
                        .arg_u32(n as u32)
                        .launch(0)?;
                    g.synchronize(0)?;
                    out.push(dn_raw(g, c_test, m * n)?);
                }
                xclass_digest::print(&format!("w4a16-tc-rows n={n} k={k} m={m} {four}"), &out[0]);
                let same = out[0] == out[1];
                tiles_identical &= same;
                eprintln!("{label}  m={m:<3} {two} vs {four}: bytes identical={same}");
            }
        }
        for p in [a, b, bs, bt, bst, c_base, c_test] {
            let _ = g.free(p);
        }
    }

    eprintln!(
        "W4A16 tile byte identity (every transposed tile vs w4a16_gemm_t_p3, and every tc_rows \
         N tile vs its 4-warp twin, same rows): {}",
        if tiles_identical {
            "IDENTICAL"
        } else {
            "DIFFERS (see bytes-vs-t_p3=false)"
        }
    );
    eprintln!(
        "W4A16 parity GATE (all transposed kernels vs base, cos≥{PASS_COS}): {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
