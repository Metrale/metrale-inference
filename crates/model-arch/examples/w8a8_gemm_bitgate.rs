// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Bit gate and timing of the W8A8 per-row GEMM (`w8a8_gemm.cu`) on every GLM-5.3
//! dense site shape of one TP=3 rank. For each shape, width M and tile:
//! - bits: the GEMM's `[M, N]` output equals, bit for bit, the GEMV family's (`w8a8_gemv`, 256-row
//!   launches over the same quantized activation);
//! - M invariance: probe rows computed alone (a one-row GEMM launch) equal the same rows inside
//!   the widest launch;
//! - timing (`GATE_TIME_ROWS`): quantize + GEMM, against the GEMV family over the same rows and
//!   against dequantize + BF16 GEMM (`glm5next_fp8_wide::dequant_gemm`, today's prefill path).
//!
//! `--control {none,scale,weight,class}` feeds the GEMM (never the GEMV oracle) altered data:
//! one row scale with its sign flipped, one weight byte flipped, or weights and activations
//! rotated by one 128-byte K chunk (same products, so the sum differs only through the class
//! order). Under a control the gate must report mismatches and exit nonzero.
//!
//! Owner: model-arch examples.
//! Invariants: exits 0 only when every compared value matched ("GATE PASS"); otherwise prints
//! the mismatch counts and "GATE FAIL" and returns an error.
//!
//! Run (GB10, < 3 GB of device memory):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example w8a8_gemm_bitgate -- --control none

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_fp8_wide::dequant_gemm;
use metrale_model_layers::layers::ops::{
    W8A8_MAX_ROWS, W8a8GemmKernels, W8a8GemmTile, W8a8Kernels, W8a8Scale, W8a8Scratch, W8a8Weight,
    w8a8_act_quant, w8a8_gemm, w8a8_gemv, w8a8_proj, w8a8_proj_wide,
};
use metrale_model_layers::weight_map::{
    DenseWeight, Fp8Weight, WeightQuantFormat, quantize_to_fp8,
};

/// 2026-10-14: `(segment rows, K, name)` per site shape on one TP=3 rank.
const SHAPES: [(&[usize], usize, &str); 14] = [
    (&[2816, 2816, 2816], 4096, "kda.q|k|v (stacked)"),
    (&[2816], 4096, "kda.q"),
    (&[128, 128], 4096, "kda.f_a|g_a"),
    (&[22], 4096, "kda.b_proj"),
    (&[4096], 2816, "kda.o_proj"),
    (&[2048, 2048], 4096, "shared gate|up"),
    (&[4096], 2048, "shared down"),
    (&[2816], 128, "kda.f_b"),
    (&[1536], 4096, "dsa.q_a"),
    (&[512], 4096, "dsa.kv_a"),
    (&[11264], 1536, "dsa.q_absorb"),
    (&[4096], 11264, "dsa.o_absorb"),
    (&[128, 16], 4096, "indexer wk|gate"),
    (&[16], 4096, "minimal"),
];
const ROWS: [usize; 10] = [17, 31, 64, 127, 128, 129, 255, 1000, 4096, 8192];
const PROBES: [usize; 4] = [0, 63, 1000, 8191];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Control {
    None,
    Scale,
    Weight,
    Class,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn bf16_bits(&mut self) -> u16 {
        let u = self.next();
        (((u & 1) << 15) | ((115 + (u >> 1) % 17) << 7) | ((u >> 8) & 0x7f)) as u16
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

fn seg(
    gpu: &dyn GpuBackend,
    rng: &mut Rng,
    quant: KernelHandle,
    n: usize,
    k: usize,
) -> Result<Fp8Weight> {
    let bf: Vec<u8> = (0..n * k)
        .flat_map(|_| rng.bf16_bits().to_le_bytes())
        .collect();
    let dev = upload(gpu, &bf)?;
    let q = quantize_to_fp8(&DenseWeight { weight: dev }, n, k, gpu, quant, 0)?;
    gpu.free(dev)?;
    Ok(Fp8Weight {
        weight: q.weight,
        row_scale: q.row_scale,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    })
}

/// 2026-10-14: A device copy of `s` with the control applied (the oracle keeps `s`).
fn altered(gpu: &dyn GpuBackend, s: &Fp8Weight, control: Control) -> Result<Fp8Weight> {
    let (n, k) = (s.n as usize, s.k as usize);
    let mut w = download(gpu, s.weight, n * k)?;
    let mut sc = download(gpu, s.row_scale, n * 4)?;
    match control {
        Control::None => {}
        Control::Scale => sc[(5 % n) * 4 + 3] ^= 0x80,
        Control::Weight => w[(7 % n) * k + 3] ^= 0x01,
        Control::Class => w.chunks_exact_mut(k).for_each(|r| r.rotate_left(128)),
    }
    Ok(Fp8Weight {
        weight: upload(gpu, &w)?,
        row_scale: upload(gpu, &sc)?,
        ..*s
    })
}

fn view(base: &W8a8Scratch, row0: usize, k: usize) -> W8a8Scratch {
    W8a8Scratch {
        q: base.q.offset(row0 * k),
        q_bytes: base.q_bytes - row0 * k,
        scale: base.scale.offset(row0 * 4),
        scale_bytes: base.scale_bytes - row0 * 4,
    }
}

/// 2026-10-14: The oracle: the GEMV family over `rows` rows of `q`, 256 rows per call.
fn gemv_all(
    gpu: &dyn GpuBackend,
    k8: &W8a8Kernels,
    w: &W8a8Weight,
    q: &W8a8Scratch,
    rows: usize,
    out: DevicePtr,
) -> Result<()> {
    let (n, k) = (w.n() as usize, w.k() as usize);
    let mut done = 0;
    while done < rows {
        let m = (rows - done).min(W8A8_MAX_ROWS);
        w8a8_gemv(
            gpu,
            k8,
            w,
            &view(q, done, k),
            m,
            out.offset(done * n * 2),
            n as u32,
            0,
        )?;
        done += m;
    }
    Ok(())
}

fn median_us(
    gpu: &dyn GpuBackend,
    launches: usize,
    mut f: impl FnMut(usize) -> Result<()>,
) -> Result<f64> {
    let mut t = Vec::new();
    for rep in 0..5 {
        gpu.synchronize(0)?;
        let t0 = std::time::Instant::now();
        for i in 0..launches {
            f(rep * launches + i)?;
        }
        gpu.synchronize(0)?;
        t.push(t0.elapsed().as_secs_f64() * 1e6 / launches as f64);
    }
    t.sort_by(f64::total_cmp);
    Ok(t[2])
}

fn parse_args() -> Result<Control> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["--control", "none"] => Ok(Control::None),
        ["--control", "scale"] => Ok(Control::Scale),
        ["--control", "weight"] => Ok(Control::Weight),
        ["--control", "class"] => Ok(Control::Class),
        other => bail!("usage: --control {{none,scale,weight,class}} (got {other:?})"),
    }
}

fn main() -> Result<()> {
    let control = parse_args()?;
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k8 = W8a8Kernels::load(&gpu);
    let gk = W8a8GemmKernels::load(&gpu);
    ensure!(
        k8.resolved(W8a8Scale::PerRow),
        "the per-row W8A8 kernels are not in this target"
    );
    for t in W8a8GemmTile::ALL {
        ensure!(gk.resolved(t), "{} is not in this target", t.entry());
    }
    let time_rows: Vec<usize> = std::env::var("GATE_TIME_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    let quant = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
    let mmax = ROWS[ROWS.len() - 1];
    let kmax = SHAPES.iter().map(|s| s.1).max().unwrap_or(4096);
    let nmax = SHAPES
        .iter()
        .map(|s| s.0.iter().sum::<usize>())
        .max()
        .unwrap_or(4096);
    let mut rng = Rng(0x00f8_d0f0_9a7e_0001);
    let x_bits: Vec<u8> = (0..mmax * kmax)
        .flat_map(|_| rng.bf16_bits().to_le_bytes())
        .collect();
    let x = upload(&gpu, &x_bits)?;
    let q = W8a8Scratch::alloc_rows(&gpu, mmax, kmax as u32)?;
    let q_alt = W8a8Scratch::alloc_rows(&gpu, mmax, kmax as u32)?;
    let (o_ref, o_gemm, o_one) = (
        gpu.alloc(mmax * nmax * 2)?,
        gpu.alloc(mmax * nmax * 2)?,
        gpu.alloc(nmax * 2)?,
    );
    let (mut compared, mut mismatched, mut shapes_failed) = (0usize, 0usize, Vec::new());
    for (segs_n, k, name) in SHAPES {
        let segs: Vec<Fp8Weight> = segs_n
            .iter()
            .map(|&n| seg(&gpu, &mut rng, quant, n, k))
            .collect::<Result<_>>()?;
        let alt: Vec<Fp8Weight> = segs
            .iter()
            .map(|s| altered(&gpu, s, control))
            .collect::<Result<_>>()?;
        let (w_ref, w_gemm) = (W8a8Weight::new(&segs)?, W8a8Weight::new(&alt)?);
        let n = w_ref.n() as usize;
        w8a8_act_quant(
            &gpu,
            &k8,
            W8a8Scale::PerRow,
            x,
            kmax as u32,
            mmax,
            k as u32,
            &q,
            0,
        )?;
        let q_gemm = if control == Control::Class {
            let mut b = download(&gpu, q.q, mmax * k)?;
            b.chunks_exact_mut(k).for_each(|r| r.rotate_left(128));
            gpu.copy_h2d(&b, q_alt.q)?;
            gpu.copy_h2d(&download(&gpu, q.scale, mmax * 4)?, q_alt.scale)?;
            q_alt
        } else {
            q
        };
        gemv_all(&gpu, &k8, &w_ref, &q, mmax, o_ref)?;
        let reference = download(&gpu, o_ref, mmax * n * 2)?;
        let mut bad_shape = 0usize;
        for tile in W8a8GemmTile::ALL {
            for m in ROWS {
                w8a8_gemm(&gpu, &gk, tile, &w_gemm, &q_gemm, m, o_gemm, n as u32, 0)?;
                let got = download(&gpu, o_gemm, m * n * 2)?;
                let bad = got
                    .chunks_exact(2)
                    .zip(reference.chunks_exact(2))
                    .filter(|(a, b)| a != b)
                    .count();
                compared += m * n;
                bad_shape += bad;
                if bad > 0 {
                    let first = got
                        .chunks_exact(2)
                        .zip(reference.chunks_exact(2))
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    println!(
                        "MISMATCH {name:<22} {:?} M={m:5}: {bad} of {} values differ (first at row {}, col {})",
                        tile,
                        m * n,
                        first / n,
                        first % n
                    );
                }
            }
            for p in PROBES {
                w8a8_gemm(
                    &gpu,
                    &gk,
                    tile,
                    &w_gemm,
                    &view(&q_gemm, p, k),
                    1,
                    o_one,
                    n as u32,
                    0,
                )?;
                let one = download(&gpu, o_one, n * 2)?;
                let wide = &reference[p * n * 2..(p + 1) * n * 2];
                let bad = one
                    .chunks_exact(2)
                    .zip(wide.chunks_exact(2))
                    .filter(|(a, b)| a != b)
                    .count();
                compared += n;
                bad_shape += bad;
                if bad > 0 {
                    println!(
                        "MISMATCH {name:<22} {:?} row {p} alone: {bad} of {n} values differ from the M={mmax} oracle",
                        tile
                    );
                }
            }
        }
        mismatched += bad_shape;
        if bad_shape > 0 {
            shapes_failed.push(name);
        }
        println!(
            "bits {name:<22} [{n:>5} x {k:>5}] {} segs: {} mismatches over 4 tiles x {} widths + {} probes",
            segs_n.len(),
            bad_shape,
            ROWS.len(),
            PROBES.len()
        );
        if control == Control::None && !time_rows.is_empty() {
            time_shape(&gpu, &k8, &gk, &segs, x, kmax, &q, o_gemm, &time_rows, name)?;
        }
        for s in segs.iter().chain(&alt) {
            gpu.free(s.weight)?;
            gpu.free(s.row_scale)?;
        }
    }
    println!(
        "control={control:?}: {mismatched} of {compared} compared values differ; shapes failing: {shapes_failed:?}"
    );
    if mismatched > 0 {
        println!("GATE FAIL");
        bail!("{mismatched} mismatches");
    }
    println!("GATE PASS");
    Ok(())
}

/// 2026-10-14: Per-launch times over rotated weight copies (>= 64 MB per batch, so they stream
/// from DRAM): quantize + GEMM per tile, the GEMV family, and dequantize + BF16 GEMM.
#[allow(clippy::too_many_arguments)]
fn time_shape(
    gpu: &dyn GpuBackend,
    k8: &W8a8Kernels,
    gk: &W8a8GemmKernels,
    segs: &[Fp8Weight],
    x: DevicePtr,
    kmax: usize,
    q: &W8a8Scratch,
    out: DevicePtr,
    time_rows: &[usize],
    name: &str,
) -> Result<()> {
    let (n, k) = (
        segs.iter().map(|s| s.n as usize).sum::<usize>(),
        segs[0].k as usize,
    );
    let copies = (64usize << 20).div_ceil(n * k).clamp(2, 32);
    let mut rng = Rng(0x7e57 + n as u64);
    let quant = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
    let ws: Vec<Vec<Fp8Weight>> = (0..copies)
        .map(|_| {
            segs.iter()
                .map(|s| seg(gpu, &mut rng, quant, s.n as usize, k))
                .collect()
        })
        .collect::<Result<_>>()?;
    let stacked: Vec<W8a8Weight> = ws
        .iter()
        .map(|c| W8a8Weight::new(c))
        .collect::<Result<_>>()?;
    let mmax = time_rows.iter().copied().max().unwrap_or(1);
    // 2026-10-14: dequant_gemm reads A with lda = K and one segment; the stacked shapes time it
    // as one [N, K] weight (its cost depends on N x K only).
    let whole: Vec<Fp8Weight> = (0..copies)
        .map(|_| seg(gpu, &mut rng, quant, n, k))
        .collect::<Result<_>>()?;
    let xk_host = download(gpu, x, mmax * kmax * 2)?;
    let xk: Vec<u8> = xk_host
        .chunks_exact(kmax * 2)
        .flat_map(|r| r[..k * 2].iter().copied())
        .collect();
    let xk_dev = upload(gpu, &xk)?;
    let dq = gpu.alloc(n * k * 2)?;
    for &m in time_rows {
        let launches = copies.max(8);
        let mut line = format!("time {name:<22} M={m:5}");
        for tile in W8a8GemmTile::ALL {
            let t = median_us(gpu, launches, |i| {
                w8a8_proj_wide(
                    gpu,
                    k8,
                    gk,
                    tile,
                    &stacked[i % copies],
                    x,
                    kmax as u32,
                    m,
                    out,
                    n as u32,
                    q,
                    0,
                )
            })?;
            line += &format!(" {:?} {t:8.1}", tile);
        }
        let tv = median_us(gpu, launches, |i| {
            let mut done = 0;
            while done < m {
                let r = (m - done).min(W8A8_MAX_ROWS);
                w8a8_proj(
                    gpu,
                    k8,
                    &stacked[i % copies],
                    x.offset(done * kmax * 2),
                    kmax as u32,
                    r,
                    out.offset(done * n * 2),
                    n as u32,
                    q,
                    0,
                )?;
                done += r;
            }
            Ok(())
        })?;
        let td = median_us(gpu, launches, |i| {
            dequant_gemm(gpu, &whole[i % copies], dq, n * k, xk_dev, out, m, n, k, 0)
        })?;
        println!("{line}  gemv {tv:8.1}  dequant+gemm {td:8.1} us");
    }
    gpu.free(xk_dev)?;
    gpu.free(dq)?;
    for s in ws.iter().flatten().chain(&whole) {
        gpu.free(s.weight)?;
        gpu.free(s.row_scale)?;
    }
    Ok(())
}
