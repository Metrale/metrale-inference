// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: The 64-token tensor-core chunked KDA scan (`kda_chunk64_prepare` +
//! `kda_chunk64_scan` in L2-sized slabs, the `METRALE_GLM_KDA_CHUNK64` path) against the per-token walk the prefill
//! runs today (`kda_recurrent_decode_bf16_seq_reg`) at GLM-5.3's per-rank shape (22 heads of 128
//! at TP=3), on the inputs `glm5next_kda_chunk16_bench` uses: q and k L2-normalised per head,
//! BF16; gates in (-5, 0) per channel, mostly slow; beta in (0, 1); a non-zero carried state.
//!
//! Reports, per prompt length: the relative RMS of each WY factor of one (chunk, head) against a
//! host f64 reference (`glm5next_kda_chunk64_ref.rs`), the relative RMS and max abs difference of
//! the outputs and of the final state against the per-token walk, and the median time of each
//! kernel and path.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error when a factor is off its reference (the decay, FP32, past 1e-4; A and U,
//!   whose terms carry the BF16 rounding of the prepare's Y table, and the BF16 factors past
//!   5e-3), when the output's or the state's relative RMS difference exceeds 3e-2 (a broken scan,
//!   not BF16 rounding, which is ~2e-3), and when the negative control (the first half's beta
//!   dropped to zero) stays under that bound. The numerics are reported; the gate is the serve's
//!   accuracy microtest.
//!
//! Run (GB10, < 1 GB of device memory); `GLM_BENCH_SLAB` overrides the slab (chunks per launch
//! pair; `0` = every chunk in one slab) to sweep the L2 fit:
//!   GLM_BENCH_ROWS=512,2048,8192 cargo run --release -p metrale-model-arch \
//!     --features cuda,gpu-examples --example glm5next_kda_chunk64_bench

#[path = "glm5next_kda_chunk64_ref.rs"]
mod host_ref;

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_kda::chunk64::{
    CHUNK64 as C, CHUNK64_AD_STRIDE, CHUNK64_D as D, CHUNK64_PREP_SMEM, CHUNK64_PREP_THREADS,
    CHUNK64_SCAN_SMEM, CHUNK64_SCAN_THREADS, CHUNK64_SLAB, CHUNK64_VB,
};

const H: usize = 22;
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

fn read_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
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
    /// 2026-10-12: Host copies for the factor reference: q/k/v as the BF16 values uploaded.
    qkv_h: Vec<f32>,
    gate_h: Vec<f32>,
    beta_h: Vec<f32>,
    state0: Vec<f32>,
    state: DevicePtr,
    core: DevicePtr,
    /// 2026-10-12: U, W, Qg, Kd, P, A + decay (`kda_chunk64.cuh`), each `t_pad * QKV` floats.
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
    let qkv: Vec<f32> = qkv.iter().map(|x| bf16::from_f32(*x).to_f32()).collect();
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
    let tp = t.div_ceil(C) * C;
    let mut scratch = [DevicePtr::NULL; 6];
    for s in scratch.iter_mut() {
        *s = gpu.alloc(tp * QKV * 4)?;
    }
    Ok(Bufs {
        qkv: qp,
        gate: up_f32(gpu, &gate)?,
        beta: up_f32(gpu, &beta)?,
        qkv_h: qkv,
        gate_h: gate,
        beta_h: beta,
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

/// 2026-10-12: Chunks per prepare + scan launch pair: `GLM_BENCH_SLAB` (0 = all), else the
/// crate's [`CHUNK64_SLAB`].
fn slab(t: usize) -> usize {
    let nc = t.div_ceil(C);
    match std::env::var("GLM_BENCH_SLAB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(0) => nc,
        Some(s) => s.min(nc),
        None => CHUNK64_SLAB.min(nc),
    }
}

fn chunked(
    gpu: &dyn GpuBackend,
    k: [KernelHandle; 2],
    b: &Bufs,
    t: usize,
    beta: DevicePtr,
) -> Result<()> {
    let (nc, sl) = (t.div_ceil(C), slab(t));
    for c0 in (0..nc).step_by(sl) {
        let n = sl.min(nc - c0);
        prepare(gpu, k[0], b, t, beta, c0, n)?;
        scan(gpu, k[1], b, t, c0, n)?;
    }
    Ok(())
}

fn scale() -> f32 {
    1.0 / (D as f32).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    b: &Bufs,
    t: usize,
    beta: DevicePtr,
    c0: usize,
    n: usize,
) -> Result<()> {
    let s = &b.scratch;
    KernelLaunch::new(gpu, k)
        .grid([n as u32, H as u32, 1])
        .block([CHUNK64_PREP_THREADS, 1, 1])
        .shared_mem(CHUNK64_PREP_SMEM)
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
        .arg_u32(c0 as u32)
        .arg_u32(CD as u32)
        .arg_u32(QKV as u32)
        .arg_f32(scale())
        .launch(0)
}

fn scan(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    b: &Bufs,
    t: usize,
    c0: usize,
    n: usize,
) -> Result<()> {
    let s = &b.scratch;
    KernelLaunch::new(gpu, k)
        .grid([H as u32, (D / CHUNK64_VB) as u32, 1])
        .block([CHUNK64_SCAN_THREADS, 1, 1])
        .shared_mem(CHUNK64_SCAN_SMEM)
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
        .arg_u32(c0 as u32)
        .arg_u32(n as u32)
        .arg_u32(QKV as u32)
        .launch(0)
}

/// 2026-10-12: Every device-visible factor of chunk `c` (in the scratch slot it occupies after
/// the last slab, `c - c0` of its slab), head `h`, against the host reference: (name, rel RMS).
fn check_factors(
    gpu: &dyn GpuBackend,
    b: &Bufs,
    t: usize,
    c: usize,
    h: usize,
) -> Result<Vec<(&'static str, f64)>> {
    let r = host_ref::factors(
        &b.qkv_h,
        &b.gate_h,
        &b.beta_h,
        t,
        H,
        c,
        h,
        f64::from(scale()),
    );
    let s = &b.scratch;
    let cb = (c % slab(t)) * H + h;
    let ad = read_f32(
        gpu,
        s[5].offset(cb * CHUNK64_AD_STRIDE * 4),
        CHUNK64_AD_STRIDE,
    )?;
    let rows: [(&str, Vec<f32>, &[f64], f64); 7] = [
        (
            "U",
            read_f32(gpu, s[0].offset(cb * C * D * 4), C * D)?,
            &r.u,
            5e-3,
        ),
        (
            "W",
            read_bf16(gpu, s[1].offset(cb * C * D * 2), C * D)?,
            &r.w,
            5e-3,
        ),
        (
            "Qg",
            read_bf16(gpu, s[2].offset(cb * C * D * 2), C * D)?,
            &r.qg,
            5e-3,
        ),
        (
            "Kd",
            read_bf16(gpu, s[3].offset(cb * C * D * 2), C * D)?,
            &r.kd,
            5e-3,
        ),
        (
            "P",
            read_bf16(gpu, s[4].offset(cb * C * C * 2), C * C)?,
            &r.p,
            5e-3,
        ),
        ("A", ad[..C * C].to_vec(), &r.a, 5e-3),
        ("decay", ad[C * C..].to_vec(), &r.decay, 1e-4),
    ];
    let mut out = Vec::new();
    for (name, dev, reference, bound) in rows {
        let e = host_ref::rel_rms(&dev, reference);
        ensure!(
            std::env::var("GLM_BENCH_NOGATE").as_deref() == Ok("1") || e <= bound,
            "T={t}: factor {name} of chunk {c} head {h} off the host reference: rel RMS {e:.2e} > {bound:.0e}"
        );
        out.push((name, e));
    }
    Ok(out)
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
        gpu.kernel("kda_chunk64_prepare", "kda_chunk64_prepare")?,
        gpu.kernel("kda_chunk64", "kda_chunk64_scan")?,
    ];
    let rows: Vec<usize> = std::env::var("GLM_BENCH_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![64, 512, 2048, 8192]);
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
        // 2026-10-12: The factors of the last full-or-partial chunk of head 3 (a partial one
        // exercises the t >= T zeroing); it sits in the last slab's scratch.
        let fx = check_factors(&gpu, &b, t, t.div_ceil(C) - 1, 3)?;
        let fx: Vec<String> = fx.iter().map(|(n, e)| format!("{n} {e:.1e}")).collect();
        println!("T={t:6}: factors vs host f64: {}", fx.join(", "));
        let (od, sd) = (diff(&o, &o_ref), diff(&s, &s_ref));
        // 2026-10-12: `GLM_BENCH_NOGATE=1` keeps timing a kernel variant known to be wrong.
        let gate = std::env::var("GLM_BENCH_NOGATE").as_deref() != Ok("1");
        ensure!(
            !gate || (od.0 <= 3e-2 && sd.0 <= 3e-2),
            "T={t}: chunked scan off the token walk: out rel RMS {:.2e}, state {:.2e}",
            od.0,
            sd.0
        );
        // 2026-10-12: Negative control: beta zeroed for the first half of the tokens (at the
        // looser 3e-2 bound, 16 tokens of a long prompt would not stand out).
        let mut beta = read_f32(&gpu, b.beta, t * H)?;
        for x in beta.iter_mut().take((t / 2).max(16) * H) {
            *x = 0.0;
        }
        let bad = up_f32(&gpu, &beta)?;
        reset(&gpu, &b)?;
        chunked(&gpu, ck, &b, t, bad)?;
        let cd = diff(&read_f32(&gpu, b.core, t * QKV)?, &o_ref);
        ensure!(
            !gate || cd.0 > 3e-2,
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
        let (nc, sl) = (t.div_ceil(C), slab(t));
        let tpr = median_us(&gpu, || {
            (0..nc)
                .step_by(sl)
                .try_for_each(|c0| prepare(&gpu, ck[0], &b, t, b.beta, c0, sl.min(nc - c0)))
        })?;
        let tsc = median_us(&gpu, || {
            (0..nc)
                .step_by(sl)
                .try_for_each(|c0| scan(&gpu, ck[1], &b, t, c0, sl.min(nc - c0)))
        })?;
        let trs = median_us(&gpu, || reset(&gpu, &b))?;
        println!(
            "T={t:6}: slab {sl:2} chunks: prepare {tpr:8.1} us, scan {tsc:8.1} us (each summed over the slabs, back to back), state reset {trs:6.1} us"
        );
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
