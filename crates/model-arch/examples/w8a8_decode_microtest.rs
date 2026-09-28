// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: GPU microtest of the W8A8 decode family (`ops::w8a8_proj`:
//! `w8a8_act_quant.cu` + `w8a8_gemv.cu`) at the Qwen3.8-27B dense and
//! Qwen3.6-35B-A3B MoE decode shapes, both weight-scale layouts (per-row and
//! 128x128 block), and 1..=256 rows.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants: the run exits 0 only if, for every shape, layout and row count:
//! - accuracy: against a host FP64 reference that also quantizes the
//!   activation (per token, or per token and 128-wide group, to E4M3 with
//!   round-to-nearest-even), the relative RMS error is <= `REL_RMS_GATE` and the
//!   cosine >= `COSINE_GATE` over the checked output rows;
//! - determinism: two launches of the same inputs are byte-identical;
//! - row invariance: row r of an M-row launch is byte-identical to row r of the
//!   256-row call (prefix; 128-row launches), and to the same activation launched alone (M=1);
//! - bounds: the `LDC_PAD` BF16 columns right of N in every output row still
//!   hold the sentinel.
//!
//! `W8A8_MICROTEST_TIME=1` also times the quantizer and the GEMV at 1, 4, 8,
//! 16, 32, 64, 128 and 256 rows (median of `REPS` launches over `ROTATE` weight copies,
//! so the weights come from DRAM) and prints weight-stream GB/s.
//!
//! Run: `METRALE_TARGET_MODEL=qwen3.8-27b cargo run --release -p
//! metrale-model-arch --features cuda,gpu-examples --example
//! w8a8_decode_microtest`.

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{self, W8a8Kernels, W8a8Scale, W8a8Scratch, W8a8Weight};
use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};
use std::time::Instant;

const MAX_M: usize = 256;
const ROWS: [usize; 17] = [
    1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 63, 64, 65, 128, 129, 200,
];
const ALONE: [usize; 7] = [0, 7, 8, 31, 63, 127, 255];
const LDC_PAD: usize = 16;
const SENTINEL: u16 = 0x7fc1;
const REL_RMS_GATE: f64 = 1e-2;
const COSINE_GATE: f64 = 0.9999;
const REPS: usize = 30;
const ROTATE: usize = 4;
/// 2026-09-28: Host reference work cap (multiply-adds) per check.
const REF_BUDGET: usize = 400_000_000;

/// 2026-09-28: (name, segment rows, K). Attention Q|K|V and GDN QKV|Z and FFN
/// gate|up are stacked segments, read in place.
const SHAPES: [(&str, &[u32], u32); 11] = [
    ("dense gdn qkv|z", &[10240, 6144], 5120),
    ("dense gdn out", &[5120], 6144),
    ("dense attn q|k|v", &[12288, 1024, 1024], 5120),
    ("dense attn o", &[5120], 6144),
    ("dense ffn gate|up", &[17408, 17408], 5120),
    ("dense ffn down", &[5120], 17408),
    ("dense lm_head", &[248077], 5120),
    ("moe gdn qkv|z", &[8192, 4096], 2048),
    ("moe attn q|k|v", &[8192, 512, 512], 2048),
    ("moe attn o", &[2048], 4096),
    ("moe expert gate|up", &[512, 512], 2048),
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn e4m3(b: u8) -> f64 {
    let (s, e, m) = (b >> 7, ((b >> 3) & 15) as i32, (b & 7) as f64);
    if e == 15 && m == 7.0 {
        return f64::NAN;
    }
    let v = if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    if s == 1 { -v } else { v }
}

/// 2026-09-28: Round `x` to the nearest E4M3 value (ties to the even code),
/// saturating at +-448, in FP64.
fn e4m3_rne(x: f64, table: &[f64; 256]) -> f64 {
    let x = x.clamp(-448.0, 448.0);
    let mut best = (f64::INFINITY, 0u8);
    for b in 0..=255u8 {
        let v = table[b as usize];
        if v.is_nan() {
            continue;
        }
        let d = (v - x).abs();
        if d < best.0 || (d == best.0 && b & 1 == 0) {
            best = (d, b);
        }
    }
    table[best.1 as usize]
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Case {
    w: W8a8Weight,
    host_w: Vec<Vec<u8>>,
    host_s: Vec<Vec<f32>>,
    seg_rows: Vec<u32>,
    k: u32,
    scale: W8a8Scale,
    dev: Vec<DevicePtr>,
}

impl Case {
    fn free(&self, g: &dyn GpuBackend) -> Result<()> {
        self.dev.iter().try_for_each(|p| g.free(*p))
    }
}

fn build(
    g: &dyn GpuBackend,
    rng: &mut Rng,
    seg_rows: &[u32],
    k: u32,
    scale: W8a8Scale,
) -> Result<Case> {
    let format = match scale {
        W8a8Scale::PerRow => WeightQuantFormat::Fp8PerRow,
        W8a8Scale::Block128 => WeightQuantFormat::Fp8BlockScaled,
    };
    let (mut segs, mut host_w, mut host_s, mut dev) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &n in seg_rows {
        let bytes: Vec<u8> = (0..n as usize * k as usize)
            .map(|_| {
                let b = rng.next() as u8;
                if b & 0x7f == 0x7f { b ^ 1 } else { b }
            })
            .collect();
        let count = match scale {
            W8a8Scale::PerRow => n as usize,
            W8a8Scale::Block128 => n.div_ceil(128) as usize * (k / 128) as usize,
        };
        let s: Vec<f32> = (0..count)
            .map(|_| ((0.5 + rng.unit()) * 20.0 / 448.0 / (k as f64).sqrt()) as f32)
            .collect();
        let (weight, row_scale) = (up(g, &bytes)?, up(g, &f32_bytes(&s))?);
        dev.extend([weight, row_scale]);
        segs.push(Fp8Weight {
            weight,
            row_scale,
            n,
            k,
            scale_format: format,
        });
        host_w.push(bytes);
        host_s.push(s);
    }
    Ok(Case {
        w: W8a8Weight::new(&segs)?,
        host_w,
        host_s,
        seg_rows: seg_rows.to_vec(),
        k,
        scale,
        dev,
    })
}

/// 2026-09-28: The FP64 activation quantization of the reference: per token
/// (group = K) or per token and 128-wide group, E4M3 round-to-nearest-even of
/// x / s with s = max(amax / 448, 1e-12). Returns the quantized values and the
/// scales, `[MAX_M, K]` and `[MAX_M, K / group]`.
fn quantize_ref(x: &[f32], k: usize, group: usize, table: &[f64; 256]) -> (Vec<f64>, Vec<f64>) {
    let (mut q, mut s) = (Vec::with_capacity(x.len()), Vec::new());
    for chunk in x.chunks_exact(group) {
        let amax = chunk.iter().fold(0f64, |a, v| a.max((*v as f64).abs()));
        let sa = (amax / 448.0).max(1e-12);
        s.push(sa);
        q.extend(chunk.iter().map(|v| e4m3_rne(*v as f64 / sa, table)));
    }
    debug_assert_eq!(q.len(), MAX_M * k);
    (q, s)
}

/// 2026-09-28: FP64 reference of output (row m, column n) from the quantized
/// activation: sum over groups of (q . w) * w_scale * a_scale.
fn reference(c: &Case, q: &[f64], sa: &[f64], table: &[f64; 256], m: usize, n: usize) -> f64 {
    let k = c.k as usize;
    let (mut seg, mut local) = (0, n);
    while local >= c.seg_rows[seg] as usize {
        local -= c.seg_rows[seg] as usize;
        seg += 1;
    }
    let g = if c.scale == W8a8Scale::PerRow { k } else { 128 };
    let w = &c.host_w[seg][local * k..(local + 1) * k];
    let mut out = 0.0;
    for u in 0..k / g {
        let dot: f64 = (u * g..(u + 1) * g)
            .map(|i| q[m * k + i] * table[w[i] as usize])
            .sum();
        let sw = match c.scale {
            W8a8Scale::PerRow => c.host_s[seg][local] as f64,
            W8a8Scale::Block128 => c.host_s[seg][(local / 128) * (k / 128) + u] as f64,
        };
        out += dot * sw * sa[m * (k / g) + u];
    }
    out
}

fn download(g: &dyn GpuBackend, p: DevicePtr, elems: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; elems * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kernels = W8a8Kernels::load(g);
    ensure!(
        kernels.resolved(W8a8Scale::PerRow) && kernels.resolved(W8a8Scale::Block128),
        "w8a8 kernels not compiled into this target"
    );
    let time = std::env::var_os("W8A8_MICROTEST_TIME").is_some();
    let table: [f64; 256] = std::array::from_fn(|b| e4m3(b as u8));
    let scratch = W8a8Scratch::alloc(g, 17408)?;
    let stream = g.default_stream();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut failures = 0usize;

    for (name, seg_rows, k) in SHAPES {
        for scale in [W8a8Scale::PerRow, W8a8Scale::Block128] {
            let c = build(g, &mut rng, seg_rows, k, scale)?;
            let (n, ku) = (c.w.n() as usize, k as usize);
            let ldc = n + LDC_PAD;
            let x: Vec<f32> = (0..MAX_M * ku)
                .map(|_| {
                    ((rng.unit() * 2.0 - 1.0) * if rng.unit() < 0.01 { 4.0 } else { 1.0 }) as f32
                })
                .collect();
            let xb: Vec<u8> = x
                .iter()
                .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
                .collect();
            let x: Vec<f32> = xb
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect();
            let x_d = up(g, &xb)?;
            let out = g.alloc(MAX_M * ldc * 2)?;
            let fill = vec![SENTINEL; MAX_M * ldc];
            let fill_b: Vec<u8> = fill.iter().flat_map(|v| v.to_le_bytes()).collect();
            let run = |rows: usize, x_at: DevicePtr| -> Result<Vec<u16>> {
                g.copy_h2d(&fill_b, out)?;
                ops::w8a8_proj(
                    g, &kernels, &c.w, x_at, k, rows, out, ldc as u32, &scratch, stream,
                )?;
                g.synchronize(stream)?;
                download(g, out, MAX_M * ldc)
            };
            let full = run(MAX_M, x_d)?;
            let mut ok = run(MAX_M, x_d)? == full;
            let det = ok;
            let mut prefix_ok = true;
            let mut bounds_ok = full
                .iter()
                .enumerate()
                .all(|(i, v)| i % ldc < n || *v == SENTINEL);
            for rows in ROWS {
                let o = run(rows, x_d)?;
                prefix_ok &=
                    (0..rows).all(|r| o[r * ldc..r * ldc + n] == full[r * ldc..r * ldc + n]);
                bounds_ok &= o
                    .iter()
                    .enumerate()
                    .all(|(i, v)| (i / ldc < rows && i % ldc < n) || *v == SENTINEL);
            }
            let mut alone_ok = true;
            for r in ALONE {
                let o = run(1, x_d.offset(r * ku * 2))?;
                alone_ok &= o[..n] == full[r * ldc..r * ldc + n];
            }
            let step = (n * MAX_M * ku / REF_BUDGET).max(1) | 1;
            let group = if scale == W8a8Scale::PerRow { ku } else { 128 };
            let (q, sa) = quantize_ref(&x, ku, group, &table);
            let (mut se, mut sr, mut dot, mut ng) = (0f64, 0f64, 0f64, 0f64);
            for col in (0..n).step_by(step) {
                for m in 0..MAX_M {
                    let r = reference(&c, &q, &sa, &table, m, col);
                    let gv = bf16::from_bits(full[m * ldc + col]).to_f64();
                    se += (gv - r).powi(2);
                    sr += r * r;
                    dot += gv * r;
                    ng += gv * gv;
                }
            }
            let (rel, cos) = ((se / sr).sqrt(), dot / (sr.sqrt() * ng.sqrt()));
            ok &= prefix_ok && alone_ok && bounds_ok && rel <= REL_RMS_GATE && cos >= COSINE_GATE;
            failures += usize::from(!ok);
            println!(
                "{} {name:<20} {scale:<8?} N={n:<6} K={k:<5} rel_rms={rel:.2e} cos={cos:.6} deterministic={det} \
                 prefix_invariant={prefix_ok} alone_invariant={alone_ok} bounds={bounds_ok}",
                if ok { "PASS" } else { "FAIL" }
            );
            if time {
                time_case(g, &kernels, &c, x_d, &scratch, stream)?;
            }
            c.free(g)?;
            g.free(x_d)?;
            g.free(out)?;
        }
    }
    failures += usize::from(!silu_fused_matches(
        g, &kernels, &scratch, stream, &mut rng,
    )?);
    ensure!(failures == 0, "{failures} case(s) failed");
    println!("all cases passed");
    Ok(())
}

/// 2026-09-28: The fused SiLU quantizer (`w8a8_act_quant_silu`) against `moe_silu_mul`
/// followed by the plain quantizer, at the dense FFN width (17408) and 1, 5, 64 and 256 rows, both
/// scale layouts: the E4M3 bytes and the scales must be identical.
fn silu_fused_matches(
    g: &dyn GpuBackend,
    kernels: &W8a8Kernels,
    scratch: &W8a8Scratch,
    stream: u64,
    rng: &mut Rng,
) -> Result<bool> {
    const INTER: usize = 17408;
    let silu = g.kernel("moe_silu_mul", "moe_silu_mul")?;
    let bf = |rng: &mut Rng| -> Vec<u8> {
        (0..MAX_M * INTER)
            .flat_map(|_| {
                bf16::from_f64((rng.unit() * 2.0 - 1.0) * 6.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    };
    let (gate, up) = (up(g, &bf(rng))?, up(g, &bf(rng))?);
    let h = g.alloc(MAX_M * INTER * 2)?;
    let mut all = true;
    for scale in [W8a8Scale::PerRow, W8a8Scale::Block128] {
        for rows in [1usize, 5, 64, 256] {
            let read = |g: &dyn GpuBackend| -> Result<(Vec<u8>, Vec<u8>)> {
                let (mut q, mut s) = (
                    vec![0u8; rows * INTER],
                    vec![0u8; rows * scale.act_scales_per_row(INTER as u32) * 4],
                );
                g.synchronize(stream)?;
                g.copy_d2h(scratch.q, &mut q)?;
                g.copy_d2h(scratch.scale, &mut s)?;
                Ok((q, s))
            };
            ops::w8a8_act_quant_silu(
                g,
                kernels,
                scale,
                gate,
                up,
                INTER as u32,
                rows,
                INTER as u32,
                scratch,
                stream,
            )?;
            let fused = read(g)?;
            ops::silu_mul(g, silu, gate, up, h, (rows * INTER) as u32, stream)?;
            ops::w8a8_act_quant(
                g,
                kernels,
                scale,
                h,
                INTER as u32,
                rows,
                INTER as u32,
                scratch,
                stream,
            )?;
            let ok = read(g)? == fused;
            all &= ok;
            println!(
                "{} fused silu quant {scale:<8?} rows={rows:<2} bytes and scales identical={ok}",
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }
    [gate, up, h].into_iter().try_for_each(|p| g.free(p))?;
    Ok(all)
}

/// 2026-09-28: Median time of the quantizer and of the GEMV over `ROTATE`
/// copies of the weight (so each launch streams from DRAM), with weight-stream
/// GB/s: (weight + scale bytes) / GEMV time.
fn time_case(
    g: &dyn GpuBackend,
    kernels: &W8a8Kernels,
    c: &Case,
    x_d: DevicePtr,
    scratch: &W8a8Scratch,
    stream: u64,
) -> Result<()> {
    let mut copies = vec![c.w];
    let mut extra = Vec::new();
    let mut rng = Rng(7);
    let wbytes_one: usize = c.host_w.iter().map(Vec::len).sum();
    let rotate = if wbytes_one > 64 << 20 { 1 } else { ROTATE };
    for _ in 1..rotate {
        let e = build(g, &mut rng, &c.seg_rows, c.k, c.scale)?;
        copies.push(e.w);
        extra.push(e);
    }
    let n = c.w.n() as usize;
    let out = g.alloc(MAX_M * n * 2)?;
    let wbytes: usize = c.host_w.iter().map(Vec::len).sum::<usize>()
        + c.host_s.iter().map(|s| s.len() * 4).sum::<usize>();
    let median = |f: &mut dyn FnMut(usize) -> Result<()>| -> Result<f64> {
        let mut t = Vec::with_capacity(REPS);
        for i in 0..REPS {
            g.synchronize(stream)?;
            let t0 = Instant::now();
            f(i)?;
            g.synchronize(stream)?;
            t.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        t.sort_by(f64::total_cmp);
        Ok(t[REPS / 2])
    };
    for rows in [1usize, 4, 8, 16, 32, 64, 128, 256] {
        let tq = median(&mut |_| {
            ops::w8a8_act_quant(g, kernels, c.scale, x_d, c.k, rows, c.k, scratch, stream)
        })?;
        let tg = median(&mut |i| {
            ops::w8a8_gemv(
                g,
                kernels,
                &copies[i % copies.len()],
                scratch,
                rows,
                out,
                n as u32,
                stream,
            )
        })?;
        println!(
            "    M={rows:<2} quant {tq:7.1} us  gemv {tg:8.1} us  {:6.1} GB/s (host-timed, includes launch)",
            wbytes as f64 / tg / 1e3
        );
    }
    extra.iter().try_for_each(|e| e.free(g))?;
    g.free(out)
}
