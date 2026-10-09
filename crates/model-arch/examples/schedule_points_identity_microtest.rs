// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Byte-identity gate for schedule points: every point of a row band or tier must
//! write exactly the bytes of its band's baseline entry, on the same random operands, at every
//! row count the band serves. A point may change only K unroll, blocks per SM and the column
//! tiles per CTA, none of which may move a bit; a class picks its points from speed alone on
//! the strength of this gate.
//!
//! - W8A8 GEMV (`w8a8_gemv.cu`): each point of `metrale_kernels::w8a8_gemv_entries::
//!   W8A8_GEMV_POINTS[b]` against the band's first point, both scale layouts.
//! - W4A16 tensor-core GEMV (`w4a16_gemv_tc.cu`): each point of `metrale_kernels::
//!   w4a16_gemv_tc_entries::W4A16_GEMV_TC_POINTS[t]` against `tc8` / `tc16`.
//!
//! Both sides are checked non-trivial (a nonzero, non-NaN output byte count) before a verdict.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//!   cargo run -p metrale-model-arch --release --example schedule_points_identity_microtest \
//!       --features cuda,gpu-examples
//!
//! Exit 0 = every compiled point is identical; 1 = at least one differs (named).

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const M_MAX: usize = 128;

/// 2026-10-09: `(label, N, K)`; N a multiple of 128 so both W8A8 scale layouts apply.
const SHAPES: &[(&str, usize, usize)] = &[
    ("N=17408 K=5120", 17408, 5120),
    ("N=5120 K=17408", 5120, 17408),
    ("N=6144 K=6144 ", 6144, 6144),
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
    /// 2026-10-09: An E4M3 byte that is not NaN (0x7F, 0xFF).
    fn e4m3(&mut self) -> u8 {
        let b = self.next() as u8;
        if b & 0x7F == 0x7F { b & 0xF0 } else { b }
    }
    /// 2026-10-09: A small positive E4M3 scale byte (2^-3 .. 2^1 range).
    fn e4m3_scale(&mut self) -> u8 {
        0x30 + (self.next() % 0x10) as u8
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len())?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    g.synchronize(0)?;
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

/// 2026-10-09: BF16 bytes of `n` values in [-1, 1).
fn bf16_bytes(r: &mut Rng, n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|_| ((r.unit().to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// 2026-10-09: FP32 bytes of `n` positive scales near `base`.
fn f32_bytes(r: &mut Rng, n: usize, base: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| (base * (1.0 + 0.5 * r.unit())).to_le_bytes())
        .collect()
}

/// 2026-10-09: The first `rows * n` BF16 outputs are non-trivial: some nonzero, none NaN.
fn non_trivial(out: &[u8]) -> bool {
    let vals: Vec<u16> = out
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    vals.iter().any(|&v| v & 0x7FFF != 0)
        && !vals
            .iter()
            .any(|&v| (v & 0x7F80) == 0x7F80 && v & 0x7F != 0)
}

#[allow(clippy::too_many_arguments)]
fn w8a8_launch(
    g: &dyn GpuBackend,
    h: KernelHandle,
    aq: DevicePtr,
    a_s: DevicePtr,
    w: DevicePtr,
    s: DevicePtr,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(g, h)
        .grid([(n as u32).div_ceil(16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(aq)
        .arg_ptr(a_s)
        .arg_ptr(w)
        .arg_ptr(s)
        .arg_ptr(w)
        .arg_ptr(s)
        .arg_ptr(w)
        .arg_ptr(s)
        .arg_ptr(out)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(k as u32)
        .arg_u32(n as u32)
        .arg_u32(n as u32)
        .arg_u32(n as u32)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn w4tc_launch(
    g: &dyn GpuBackend,
    h: KernelHandle,
    nt: u32,
    a: DevicePtr,
    packed: DevicePtr,
    scales: DevicePtr,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(g, h)
        .grid([(n as u32).div_ceil(8 * nt), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(packed)
        .arg_ptr(scales)
        .arg_f32(0.75)
        .arg_ptr(out)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(0)
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let mut r = Rng(0x5EED_2026_1009);
    let mut failures: Vec<String> = Vec::new();
    let mut compared = 0usize;
    for &(label, n, k) in SHAPES {
        // 2026-10-09: W8A8 operands: E4M3 weights and activations, positive FP32 scales sized
        // for the larger (block-128) layout.
        let w = up(g, &(0..n * k).map(|_| r.e4m3()).collect::<Vec<_>>())?;
        let s = up(g, &f32_bytes(&mut r, n * k.div_ceil(128), 0.01))?;
        let aq = up(g, &(0..M_MAX * k).map(|_| r.e4m3()).collect::<Vec<_>>())?;
        let a_s = up(g, &f32_bytes(&mut r, M_MAX * k.div_ceil(128), 0.02))?;
        // 2026-10-09: W4A16 operands: E2M1 pairs (any byte), small E4M3 group scales, BF16
        // activations.
        let packed = up(
            g,
            &(0..n * k / 2).map(|_| r.next() as u8).collect::<Vec<_>>(),
        )?;
        let scales = up(
            g,
            &(0..n * k / 16).map(|_| r.e4m3_scale()).collect::<Vec<_>>(),
        )?;
        let a = up(g, &bf16_bytes(&mut r, 16 * k))?;
        let out_ref = g.alloc(M_MAX * n * 2)?;
        let out_pt = g.alloc(M_MAX * n * 2)?;

        for layout in ["rowscale", "blk128"] {
            for (band, points) in metrale_kernels::w8a8_gemv_entries::W8A8_GEMV_POINTS
                .iter()
                .enumerate()
            {
                let base = format!("w8a8_gemv_{layout}_{}", points[0]);
                let Ok(hb) = g.kernel("w8a8_gemv", &base) else {
                    continue;
                };
                let hi = 8usize << band;
                let rows: Vec<usize> = [hi / 2 + 1, hi].into_iter().filter(|&m| m >= 1).collect();
                for p in &points[1..] {
                    let name = format!("w8a8_gemv_{layout}_{p}");
                    let Ok(hp) = g.kernel("w8a8_gemv", &name) else {
                        continue;
                    };
                    for &m in &rows {
                        w8a8_launch(g, hb, aq, a_s, w, s, out_ref, m, n, k)?;
                        w8a8_launch(g, hp, aq, a_s, w, s, out_pt, m, n, k)?;
                        let (x, y) = (down(g, out_ref, m * n * 2)?, down(g, out_pt, m * n * 2)?);
                        if !non_trivial(&x) || !non_trivial(&y) {
                            bail!("{label} {name} m={m}: trivial output, the gate cannot judge");
                        }
                        compared += 1;
                        let same = x == y;
                        println!("{label} {name:<34} vs {base:<28} m={m:<3} identical={same}");
                        if !same {
                            failures.push(format!("{label} {name} m={m}"));
                        }
                    }
                }
            }
        }

        let nt = metrale_kernels::w4a16_gemv_tc_entries::w4a16_gemv_tc_nt;
        for (tier, points) in metrale_kernels::w4a16_gemv_tc_entries::W4A16_GEMV_TC_POINTS
            .iter()
            .enumerate()
        {
            let base = format!("w4a16_gemv_{}", points[0]);
            let Ok(hb) = g.kernel("w4a16_gemv_tc", &base) else {
                continue;
            };
            let rows: &[usize] = if tier == 0 { &[1, 4, 8] } else { &[9, 12, 16] };
            for p in &points[1..] {
                let name = format!("w4a16_gemv_{p}");
                let Ok(hp) = g.kernel("w4a16_gemv_tc", &name) else {
                    continue;
                };
                for &m in rows {
                    w4tc_launch(g, hb, nt(points[0]), a, packed, scales, out_ref, m, n, k)?;
                    w4tc_launch(g, hp, nt(p), a, packed, scales, out_pt, m, n, k)?;
                    let (x, y) = (down(g, out_ref, m * n * 2)?, down(g, out_pt, m * n * 2)?);
                    if !non_trivial(&x) || !non_trivial(&y) {
                        bail!("{label} {name} m={m}: trivial output, the gate cannot judge");
                    }
                    compared += 1;
                    let same = x == y;
                    println!("{label} {name:<34} vs {base:<28} m={m:<3} identical={same}");
                    if !same {
                        failures.push(format!("{label} {name} m={m}"));
                    }
                }
            }
        }
        for p in [w, s, aq, a_s, packed, scales, a, out_ref, out_pt] {
            let _ = g.free(p);
        }
    }
    if compared == 0 {
        bail!("no schedule point resolved: nothing was compared");
    }
    if failures.is_empty() {
        println!("SCHEDULE-POINTS IDENTICAL: {compared} comparisons");
        Ok(())
    } else {
        println!("SCHEDULE-POINTS DIFFER ({} of {compared}):", failures.len());
        for f in &failures {
            println!("  {f}");
        }
        std::process::exit(1);
    }
}
