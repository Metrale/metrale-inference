// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Byte gate and microbench of `METRALE_GLM_MHC_PRE_SLICE` (the mHC pre-mix of a wide
//! prefill sub-chunk launched per slice of rows): for the fused `glm_hc_pre_norm` and the
//! unfused `glm_hc_pre` (mix + finish), one launch over T rows against T / slice launches over
//! consecutive slices, at GLM-5.3's shape (hidden 4096, hc 4, 20 Sinkhorn iterations).
//! 2026-10-12: Also `METRALE_GLM_MHC_PRE_TOK` (`glm5next_hc_pre_norm_tok4`, four tokens per
//! block) against the fused 8-row grid, whose bytes `glm5next_hc_fused_gate` ties to the unfused
//! kernels: byte gate with a negative control, and the time of each.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - Returns an error unless every output byte (y, post, comb, and the normed row on the fused
//!   path) of the sliced launches equals the single launch's, at every width and slice, and
//!   unless the negative control (one stream value of the last slice changed) changes them.
//! - Likewise the token-tiled grid against the 8-row grid, at token counts that do and do not
//!   fill the last block of four.
//!
//! Run (GB10, < 1 GB of device memory; the token bound needs `METRALE_GLM_PREFILL_ROWS`):
//!   METRALE_GLM_PREFILL_ROWS=4096 cargo run --release -p metrale-gpu-examples \
//!     --features cuda,gpu-examples --example glm5next_mhc_slice_bench

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mhc::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights, glm_hc_pre};
use metrale_model_arch::glm5next_mhc_fused::{
    PreNormChoice, glm_hc_pre_norm, glm_hc_pre_norm_with,
};

const H: usize = 4096;
const HC: usize = 4;
const MIX: usize = (2 + HC) * HC;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(16))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16b(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect()
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-10: The outputs of one arm: y, normed, post, comb.
struct Out([DevicePtr; 4]);

impl Out {
    fn alloc(gpu: &dyn GpuBackend, t: usize) -> Result<Self> {
        Ok(Self([
            gpu.alloc(t * H * 2)?,
            gpu.alloc(t * H * 2)?,
            gpu.alloc(t * HC * 4)?,
            gpu.alloc(t * HC * HC * 4)?,
        ]))
    }
    fn poison(&self, gpu: &dyn GpuBackend, t: usize) -> Result<()> {
        for (p, n) in self.0.iter().zip(sizes(t)) {
            gpu.copy_h2d(&vec![0xABu8; n], *p)?;
        }
        Ok(())
    }
    fn bytes(&self, gpu: &dyn GpuBackend, t: usize) -> Result<Vec<u8>> {
        let mut v = Vec::new();
        for (p, n) in self.0.iter().zip(sizes(t)) {
            v.extend(read(gpu, *p, n)?);
        }
        Ok(v)
    }
}

fn sizes(t: usize) -> [usize; 4] {
    [t * H * 2, t * H * 2, t * HC * 4, t * HC * HC * 4]
}

/// 2026-10-10: Rows `r0..r0 + n` of the pre-mix, on the fused or unfused path.
#[allow(clippy::too_many_arguments)]
fn pre(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    w: &Glm5NextMhcSiteWeights,
    streams: DevicePtr,
    norm_w: DevicePtr,
    o: &Out,
    r0: usize,
    n: usize,
    fused: bool,
) -> Result<()> {
    let [y, normed, post, comb] = o.0;
    let (s, y, normed, post, comb) = (
        streams.offset(r0 * HC * H * 4),
        y.offset(r0 * H * 2),
        normed.offset(r0 * H * 2),
        post.offset(r0 * HC * 4),
        comb.offset(r0 * HC * HC * 4),
    );
    if fused {
        glm_hc_pre_norm(
            gpu, k, s, w, norm_w, y, normed, post, comb, n as u32, H as u32, HC as u32, 20, 1e-5,
            1e-6, 1e-5, 0,
        )
    } else {
        glm_hc_pre(
            gpu, k, s, w, y, post, comb, n as u32, H as u32, HC as u32, 20, 1e-5, 1e-6, 0,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn sliced(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    w: &Glm5NextMhcSiteWeights,
    streams: DevicePtr,
    norm_w: DevicePtr,
    o: &Out,
    t: usize,
    slice: usize,
    fused: bool,
) -> Result<()> {
    let mut r0 = 0;
    while r0 < t {
        let n = slice.min(t - r0);
        pre(gpu, k, w, streams, norm_w, o, r0, n, fused)?;
        r0 += n;
    }
    Ok(())
}

/// 2026-10-12: Rows `0..t` on the fused pre-norm, the token-tiled grid when `tok` (else the
/// 8-row grid; both need at least 64 rows).
#[allow(clippy::too_many_arguments)]
fn pre_tok(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    w: &Glm5NextMhcSiteWeights,
    streams: DevicePtr,
    norm_w: DevicePtr,
    o: &Out,
    t: usize,
    tok: bool,
) -> Result<()> {
    let [y, normed, post, comb] = o.0;
    glm_hc_pre_norm_with(
        gpu,
        k,
        streams,
        w,
        norm_w,
        y,
        normed,
        post,
        comb,
        t as u32,
        H as u32,
        HC as u32,
        20,
        1e-5,
        1e-6,
        1e-5,
        PreNormChoice {
            prefetch: false,
            splitk: false,
            tok,
            c1: false,
        },
        0,
    )
}

/// 2026-10-12: The token-tiled gate and timing (module doc).
#[allow(clippy::too_many_arguments)]
fn tok_gate(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    w: &Glm5NextMhcSiteWeights,
    streams: DevicePtr,
    streams_h: &[f32],
    norm_w: DevicePtr,
    a: &Out,
    b: &Out,
) -> Result<()> {
    ensure!(
        k.hc_pre_norm_tok.0 != 0,
        "glm5next_hc_pre_norm_tok4 is not in this build"
    );
    for t in [64usize, 66, 127, 512, 2048, 2051, 4096] {
        a.poison(gpu, t)?;
        b.poison(gpu, t)?;
        pre_tok(gpu, k, w, streams, norm_w, a, t, false)?;
        pre_tok(gpu, k, w, streams, norm_w, b, t, true)?;
        let want = a.bytes(gpu, t)?;
        ensure!(
            b.bytes(gpu, t)? == want,
            "token-tiled T={t}: bytes differ from the 8-row grid"
        );
        let row = (t - 1) * HC * H;
        let mut bad = streams_h[row..row + HC * H].to_vec();
        bad[7] += 1.0;
        gpu.copy_h2d(&f32b(&bad), streams.offset(row * 4))?;
        pre_tok(gpu, k, w, streams, norm_w, b, t, true)?;
        let changed = b.bytes(gpu, t)? != want;
        gpu.copy_h2d(
            &f32b(&streams_h[row..row + HC * H]),
            streams.offset(row * 4),
        )?;
        ensure!(changed, "token-tiled T={t}: negative control unchanged");
        let rows = median_us(gpu, || pre_tok(gpu, k, w, streams, norm_w, a, t, false))?;
        let tok = median_us(gpu, || pre_tok(gpu, k, w, streams, norm_w, b, t, true))?;
        println!(
            "token-tiled T={t:5}: identical; 8-row grid {rows:8.1} us, token-tiled {tok:8.1} us ({:.2}x)",
            rows / tok
        );
    }
    Ok(())
}

fn median_us(gpu: &dyn GpuBackend, f: impl Fn() -> Result<()>) -> Result<f64> {
    let mut v = Vec::new();
    for _ in 0..9 {
        gpu.synchronize(0)?;
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            f()?;
        }
        gpu.synchronize(0)?;
        v.push(t0.elapsed().as_secs_f64() * 2e5);
    }
    v.sort_by(f64::total_cmp);
    Ok(v[4])
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k = Glm5NextMhcKernels::resolve(&gpu, metrale_gpu_examples::glm_levers()?.mhc)?;
    let mut rng = Rng(0x3c6e_f372_fe94_f82b);
    let tmax = 4096usize;
    let hc_fn: Vec<f32> = (0..MIX * HC * H).map(|_| rng.unit() * 0.05).collect();
    let base: Vec<f32> = (0..MIX).map(|_| rng.unit() * 0.5).collect();
    let w = Glm5NextMhcSiteWeights {
        hc_fn: up(&gpu, &bf16b(&hc_fn))?,
        hc_fn_bf16: true,
        hc_scale: up(&gpu, &f32b(&[0.7, 1.3, 0.9]))?,
        hc_base: up(&gpu, &f32b(&base))?,
        mix: gpu.alloc(tmax * MIX * 4)?,
        done: up(&gpu, &vec![0u8; tmax * 4])?,
        mix_tokens: tmax,
    };
    let norm: Vec<f32> = (0..H).map(|_| 1.0 + 0.2 * rng.unit()).collect();
    let norm_w = up(&gpu, &bf16b(&norm))?;
    let streams_h: Vec<f32> = (0..tmax * HC * H).map(|_| rng.unit() * 2.0).collect();
    let streams = up(&gpu, &f32b(&streams_h))?;
    let (a, b) = (Out::alloc(&gpu, tmax)?, Out::alloc(&gpu, tmax)?);
    tok_gate(&gpu, &k, &w, streams, &streams_h, norm_w, &a, &b)?;
    for fused in [true, false] {
        let path = if fused {
            "fused pre-norm"
        } else {
            "mix + finish"
        };
        for (t, slice) in [(2048usize, 256usize), (2048, 512), (4096, 256), (1000, 256)] {
            a.poison(&gpu, t)?;
            b.poison(&gpu, t)?;
            pre(&gpu, &k, &w, streams, norm_w, &a, 0, t, fused)?;
            sliced(&gpu, &k, &w, streams, norm_w, &b, t, slice, fused)?;
            let want = a.bytes(&gpu, t)?;
            let got = b.bytes(&gpu, t)?;
            // 2026-10-10: The unfused path writes no normed row; its two poisoned copies match.
            ensure!(
                got == want,
                "{path} T={t} slice {slice}: sliced bytes differ"
            );
            let row = (t - 1) * HC * H;
            let mut bad = streams_h[row..row + HC * H].to_vec();
            bad[5] += 1.0;
            gpu.copy_h2d(&f32b(&bad), streams.offset(row * 4))?;
            sliced(&gpu, &k, &w, streams, norm_w, &b, t, slice, fused)?;
            let changed = b.bytes(&gpu, t)? != want;
            gpu.copy_h2d(
                &f32b(&streams_h[row..row + HC * H]),
                streams.offset(row * 4),
            )?;
            ensure!(
                changed,
                "{path} T={t} slice {slice}: negative control unchanged"
            );
            let one = median_us(&gpu, || pre(&gpu, &k, &w, streams, norm_w, &a, 0, t, fused))?;
            let sl = median_us(&gpu, || {
                sliced(&gpu, &k, &w, streams, norm_w, &b, t, slice, fused)
            })?;
            println!(
                "{path:15} T={t:5} slice {slice:4}: identical; one launch {one:8.1} us, sliced {sl:8.1} us ({:.2}x)",
                one / sl
            );
        }
    }
    Ok(())
}
