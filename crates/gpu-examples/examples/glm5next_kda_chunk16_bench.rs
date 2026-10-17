// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The chunked KDA scan (`kda_chunk16_prepare` + `kda_chunk16_scan`, the
//! `METRALE_GLM_KDA_CHUNK_SCAN` path) against the per-token walk the prefill runs today
//! (`kda_recurrent_decode_bf16_seq_reg`) at GLM-5.3's per-rank shape (22 heads of 128 at TP=3),
//! on synthetic inputs shaped like the layer's: q and k L2-normalised per head, BF16; gates in
//! (-5, 0) per channel, mostly slow; beta in (0, 1); a non-zero carried state.
//!
//! Reports, per prompt length: the relative RMS and max abs difference of the outputs and of
//! the final state against the per-token walk, and the median time of each path.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error when the output's relative RMS difference exceeds 1e-4 or the state's
//!   1e-4 (FP32 reorder drift is ~1e-6), and when the negative control (one chunk's beta
//!   dropped to zero in the scan input) stays under that bound.
//!
//! Run (GB10, < 1 GB of device memory):
//!   GLM_BENCH_ROWS=64,512,2048,8192 cargo run --release -p metrale-gpu-examples \
//!     --features cuda,gpu-examples --example glm5next_kda_chunk16_bench

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const H: usize = 22;
const D: usize = 128;
const QKV: usize = H * D;
const CD: usize = 3 * QKV;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let (a, b) = (self.unit().max(1e-7), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn up_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(b.len())?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

fn read_f32(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// 2026-10-10: (relative RMS, max abs) of `a - b`.
fn diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut num, mut den, mut mx) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = f64::from(x - y);
        num += d * d;
        den += f64::from(y) * f64::from(y);
        mx = mx.max(d.abs());
    }
    ((num / den.max(1e-30)).sqrt(), mx)
}

struct Bufs {
    qkv: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    state0: Vec<f32>,
    state: DevicePtr,
    core: DevicePtr,
    scratch: [DevicePtr; 6],
}

fn setup(gpu: &dyn GpuBackend, rng: &mut Rng, t: usize) -> Result<Bufs> {
    let mut qkv = vec![0f32; t * CD];
    for row in 0..t {
        for part in 0..3 {
            for h in 0..H {
                let base = row * CD + part * QKV + h * D;
                let v: Vec<f32> = (0..D).map(|_| rng.normal()).collect();
                let n = if part < 2 {
                    v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6)
                } else {
                    1.0
                };
                for d in 0..D {
                    qkv[base + d] = v[d] / n;
                }
            }
        }
    }
    let qb: Vec<u8> = qkv
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
        .collect();
    let qp = gpu.alloc(qb.len())?;
    gpu.copy_h2d(&qb, qp)?;
    // 2026-10-10: Mostly slow channels (sigmoid(N - 3) of the bound), a few fast ones.
    let gate: Vec<f32> = (0..t * QKV)
        .map(|_| -5.0 * sigmoid(2.0 * rng.normal() - 3.0))
        .collect();
    let beta: Vec<f32> = (0..t * H).map(|_| sigmoid(rng.normal())).collect();
    let state0: Vec<f32> = (0..H * D * D).map(|_| 0.05 * rng.normal()).collect();
    let tp = t.div_ceil(32) * 32;
    let mut scratch = [DevicePtr::NULL; 6];
    for s in scratch.iter_mut() {
        *s = gpu.alloc(tp * QKV * 4)?;
    }
    Ok(Bufs {
        qkv: qp,
        gate: up_f32(gpu, &gate)?,
        beta: up_f32(gpu, &beta)?,
        state: up_f32(gpu, &state0)?,
        state0,
        core: gpu.alloc(t * QKV * 4)?,
        scratch,
    })
}

fn reset(gpu: &dyn GpuBackend, b: &Bufs) -> Result<()> {
    let s: Vec<u8> = b.state0.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.copy_h2d(&s, b.state)
}

fn token_walk(gpu: &dyn GpuBackend, k: KernelHandle, b: &Bufs, t: usize) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([H as u32, (D / 32) as u32, 1])
        .block([32, 1, 1])
        .shared_mem((3 * D * 4) as u32)
        .arg_ptr(b.qkv)
        .arg_ptr(b.qkv.offset(QKV * 2))
        .arg_ptr(b.qkv.offset(QKV * 4))
        .arg_ptr(b.gate)
        .arg_ptr(b.beta)
        .arg_ptr(b.state)
        .arg_ptr(b.core)
        .arg_u32(H as u32)
        .arg_f32(1.0 / (D as f32).sqrt())
        .arg_u32(t as u32)
        .arg_u32(CD as u32)
        .arg_u32(QKV as u32)
        .arg_u32(H as u32)
        .arg_u32(QKV as u32)
        .launch(0)
}

fn chunked(
    gpu: &dyn GpuBackend,
    k: [KernelHandle; 2],
    b: &Bufs,
    t: usize,
    beta: DevicePtr,
) -> Result<()> {
    prepare(gpu, k[0], b, t, beta)?;
    scan(gpu, k[1], b, t)
}

fn prepare(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    b: &Bufs,
    t: usize,
    beta: DevicePtr,
) -> Result<()> {
    let nc = t.div_ceil(16);
    let s = &b.scratch;
    KernelLaunch::new(gpu, k)
        .grid([nc as u32, H as u32, 1])
        .block([D as u32, 1, 1])
        .arg_ptr(b.qkv)
        .arg_ptr(b.qkv.offset(QKV * 2))
        .arg_ptr(b.qkv.offset(QKV * 4))
        .arg_ptr(b.gate)
        .arg_ptr(beta)
        .arg_ptr(s[0])
        .arg_ptr(s[1])
        .arg_ptr(s[2])
        .arg_ptr(s[3])
        .arg_ptr(s[4])
        .arg_ptr(s[5])
        .arg_u32(H as u32)
        .arg_u32(t as u32)
        .arg_u32(CD as u32)
        .arg_u32(QKV as u32)
        .arg_f32(1.0 / (D as f32).sqrt())
        .launch(0)
}

fn scan(gpu: &dyn GpuBackend, k: KernelHandle, b: &Bufs, t: usize) -> Result<()> {
    let nc = t.div_ceil(16);
    let s = &b.scratch;
    KernelLaunch::new(gpu, k)
        .grid([H as u32, (D / 32) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(s[0])
        .arg_ptr(s[1])
        .arg_ptr(s[2])
        .arg_ptr(s[3])
        .arg_ptr(s[4])
        .arg_ptr(s[5])
        .arg_ptr(b.state)
        .arg_ptr(b.core)
        .arg_u32(H as u32)
        .arg_u32(t as u32)
        .arg_u32(nc as u32)
        .arg_u32(QKV as u32)
        .launch(0)
}

fn median_us(gpu: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    let mut v = Vec::new();
    for _ in 0..7 {
        gpu.synchronize(0)?;
        let t0 = std::time::Instant::now();
        f()?;
        gpu.synchronize(0)?;
        v.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    v.sort_by(f64::total_cmp);
    Ok(v[3])
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let walk = gpu.kernel("kda_recurrent", "kda_recurrent_decode_bf16_seq_reg")?;
    let ck = [
        gpu.kernel("kda_chunk16", "kda_chunk16_prepare")?,
        gpu.kernel("kda_chunk16", "kda_chunk16_scan")?,
    ];
    let rows: Vec<usize> = std::env::var("GLM_BENCH_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![64, 512, 2048]);
    let mut rng = Rng(0x0dd_ba11_cafe_f00d);
    for t in rows {
        let b = setup(&gpu, &mut rng, t)?;
        reset(&gpu, &b)?;
        token_walk(&gpu, walk, &b, t)?;
        let (o_ref, s_ref) = (
            read_f32(&gpu, b.core, t * QKV)?,
            read_f32(&gpu, b.state, H * D * D)?,
        );
        reset(&gpu, &b)?;
        chunked(&gpu, ck, &b, t, b.beta)?;
        let (o, s) = (
            read_f32(&gpu, b.core, t * QKV)?,
            read_f32(&gpu, b.state, H * D * D)?,
        );
        let (od, sd) = (diff(&o, &o_ref), diff(&s, &s_ref));
        ensure!(
            od.0 <= 1e-4 && sd.0 <= 1e-4,
            "T={t}: chunked scan off the token walk: out rel RMS {:.2e}, state {:.2e}",
            od.0,
            sd.0
        );
        // 2026-10-10: Negative control: beta zeroed for the first 16 tokens.
        let mut beta = read_f32(&gpu, b.beta, t * H)?;
        for x in beta.iter_mut().take(16 * H) {
            *x = 0.0;
        }
        let bad = up_f32(&gpu, &beta)?;
        reset(&gpu, &b)?;
        chunked(&gpu, ck, &b, t, bad)?;
        let cd = diff(&read_f32(&gpu, b.core, t * QKV)?, &o_ref);
        ensure!(
            cd.0 > 1e-4,
            "T={t}: negative control stayed within the bound ({:.2e})",
            cd.0
        );
        let tw = median_us(&gpu, || {
            reset(&gpu, &b)?;
            token_walk(&gpu, walk, &b, t)
        })?;
        let tc = median_us(&gpu, || {
            reset(&gpu, &b)?;
            chunked(&gpu, ck, &b, t, b.beta)
        })?;
        let tp = median_us(&gpu, || prepare(&gpu, ck[0], &b, t, b.beta))?;
        let tsc = median_us(&gpu, || scan(&gpu, ck[1], &b, t))?;
        let trs = median_us(&gpu, || reset(&gpu, &b))?;
        println!("T={t:6}: prepare {tp:8.1} us, scan {tsc:8.1} us, state reset {trs:6.1} us");
        println!(
            "T={t:6}: out rel RMS {:.2e} max {:.2e}; state rel RMS {:.2e} max {:.2e}; control {:.2e} | token walk {tw:9.1} us ({:.2} us/token), chunked {tc:9.1} us ({:.3} us/token) incl. state reset",
            od.0,
            od.1,
            sd.0,
            sd.1,
            cd.0,
            tw / t as f64,
            tc / t as f64
        );
        for p in [b.qkv, b.gate, b.beta, b.state, b.core, bad]
            .into_iter()
            .chain(b.scratch)
        {
            gpu.free(p)?;
        }
    }
    Ok(())
}
