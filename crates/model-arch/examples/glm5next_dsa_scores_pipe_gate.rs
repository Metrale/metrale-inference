// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tier-1 gate of `METRALE_GLM_DSA_SCORES_PIPE`: `dsa_pool_prep_bf16` +
//! `dsa_index_scores_mma_pipe` against `dsa_index_scores_mma`, on pools `dsa_kpool_compress`
//! builds from a synthetic flat indexer cache (random BF16 keys and gates, about 1% of the keys
//! invalid, a partial trailing pool where S % 4 != 0, and one shape with a nonzero first key).
//!
//! 1. The prepared BF16 keys equal a host round-to-nearest-even conversion of the FP32 pool
//!    keys, byte for byte, and each visibility position equals the host rule.
//! 2. Every score and candidacy byte of the pipe equals the MMA kernel's (outputs pre-filled
//!    with different patterns before each run).
//! 3. Negative controls: a flipped sign bit of one prepared key changes the scores; one
//!    perturbed visibility entry changes the candidacy.
//! 4. Timing (median of 7 synchronous launches): mma, prep, pipe.
//!
//! Owner: model-arch examples.
//! Invariants: exits with an error unless every check holds; the last line reads
//! `RESULT: PASS` only then.
//!
//! Run (GB10, < 3 GB of device memory):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example glm5next_dsa_scores_pipe_gate

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::DSA_SCORES_PIPE_MODULE;
use metrale_model_arch::glm5next_dsa::select::SCORES_PIPE_SMEM;

const H: usize = 32;
const D: usize = 128;
const KP: usize = 4;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// 2026-10-14: Round-to-nearest-even FP32 -> BF16 bits (finite inputs; NaN is refused).
fn bf16_rne(f: f32) -> Result<u16> {
    ensure!(!f.is_nan(), "a NaN pool key");
    let b = f.to_bits();
    Ok((b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16)
}

fn up<T: Copy>(gpu: &dyn GpuBackend, v: &[T]) -> Result<DevicePtr> {
    let n = std::mem::size_of_val(v);
    // 2026-10-14: SAFETY: `v` is a live slice of `Copy` plain data; the byte view covers exactly
    // its `n` bytes and is only read.
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) };
    let p = gpu.alloc(n.max(16))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct Kernels {
    compress: KernelHandle,
    mma: KernelHandle,
    prep: KernelHandle,
    pipe: KernelHandle,
}

/// 2026-10-14: One shape: Q query rows over S tokens; `first_key` tokens of left padding;
/// `shuffled` draws each row's position at random in [0, S) instead of the causal S - Q + r.
struct Shape {
    q: usize,
    s: usize,
    first_key: usize,
    shuffled: bool,
}

struct Io {
    q: DevicePtr,
    w: DevicePtr,
    valid: DevicePtr,
    qpos: DevicePtr,
    keys: DevicePtr,
    idx: DevicePtr,
    pvalid: DevicePtr,
    keys_bf16: DevicePtr,
    vis: DevicePtr,
    out: DevicePtr,
    cand: DevicePtr,
    q_rows: usize,
    pools: usize,
    s: usize,
}

fn scale() -> f32 {
    (D as f32).powf(-0.5)
}

fn run_mma(gpu: &dyn GpuBackend, k: &Kernels, io: &Io) -> Result<()> {
    KernelLaunch::new(gpu, k.mma)
        .grid([io.q_rows.div_ceil(8) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(io.q)
        .arg_ptr(io.keys)
        .arg_ptr(io.w)
        .arg_ptr(io.idx)
        .arg_ptr(io.pvalid)
        .arg_ptr(io.valid)
        .arg_ptr(io.qpos)
        .arg_ptr(io.out)
        .arg_ptr(io.cand)
        .arg_u32(io.q_rows as u32)
        .arg_u32(io.pools as u32)
        .arg_u32(H as u32)
        .arg_u32(D as u32)
        .arg_u32(KP as u32)
        .arg_u32(io.s as u32)
        .arg_f32(scale())
        .launch(0)?;
    Ok(())
}

fn run_prep(gpu: &dyn GpuBackend, k: &Kernels, io: &Io) -> Result<()> {
    KernelLaunch::new(gpu, k.prep)
        .grid([io.pools.div_ceil(8) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(io.keys)
        .arg_ptr(io.idx)
        .arg_ptr(io.pvalid)
        .arg_ptr(io.valid)
        .arg_ptr(io.keys_bf16)
        .arg_ptr(io.vis)
        .arg_u32(io.pools as u32)
        .arg_u32(D as u32)
        .arg_u32(KP as u32)
        .arg_u32(io.s as u32)
        .launch(0)?;
    Ok(())
}

fn run_pipe(gpu: &dyn GpuBackend, k: &Kernels, io: &Io) -> Result<()> {
    KernelLaunch::new(gpu, k.pipe)
        .grid([io.q_rows.div_ceil(8) as u32, 1, 1])
        .block([256, 1, 1])
        .shared_mem(SCORES_PIPE_SMEM)
        .arg_ptr(io.q)
        .arg_ptr(io.keys_bf16)
        .arg_ptr(io.w)
        .arg_ptr(io.vis)
        .arg_ptr(io.qpos)
        .arg_ptr(io.out)
        .arg_ptr(io.cand)
        .arg_u32(io.q_rows as u32)
        .arg_u32(io.pools as u32)
        .arg_u32(H as u32)
        .arg_u32(D as u32)
        .arg_f32(scale())
        .launch(0)?;
    Ok(())
}

/// 2026-10-14: (scores, candidacy) bytes of one run, the outputs pre-filled with `fill`.
fn outputs(
    gpu: &dyn GpuBackend,
    io: &Io,
    fill: u8,
    f: impl Fn() -> Result<()>,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let n = io.q_rows * io.pools;
    gpu.memset(io.out, fill, n * 4)?;
    gpu.memset(io.cand, fill, n)?;
    f()?;
    Ok((read(gpu, io.out, n * 4)?, read(gpu, io.cand, n)?))
}

fn median_us(gpu: &dyn GpuBackend, f: impl Fn() -> Result<()>) -> Result<f64> {
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

fn rows_differ(a: &[u8], b: &[u8], row_bytes: usize) -> usize {
    a.chunks(row_bytes)
        .zip(b.chunks(row_bytes))
        .filter(|(x, y)| x != y)
        .count()
}

fn shape(gpu: &dyn GpuBackend, k: &Kernels, sh: &Shape, rng: &mut Rng) -> Result<()> {
    let (q_rows, s) = (sh.q, sh.s);
    let pools = s / KP;
    let pools_full = s.div_ceil(KP);
    let bf = |rng: &mut Rng, n: usize| -> Result<Vec<u16>> {
        (0..n).map(|_| bf16_rne(rng.unit())).collect()
    };
    let kc = bf(rng, s * D)?;
    let gate = bf(rng, s * D)?;
    let ape: Vec<f32> = (0..KP * D).map(|_| rng.unit() * 0.1).collect();
    let valid: Vec<u8> = (0..s)
        .map(|t| u8::from(t >= sh.first_key && !rng.next().is_multiple_of(100)))
        .collect();
    let qpos: Vec<i32> = (0..q_rows)
        .map(|r| {
            if sh.shuffled {
                (rng.next() % s as u64) as i32
            } else {
                (s - q_rows + r) as i32
            }
        })
        .collect();
    let q: Vec<f32> = (0..q_rows * H * D).map(|_| rng.unit()).collect();
    let w: Vec<f32> = (0..q_rows * H).map(|_| rng.unit() * 0.2).collect();
    let (kd, gd, ad) = (up(gpu, &kc)?, up(gpu, &gate)?, up(gpu, &ape)?);
    let io = Io {
        q: up(gpu, &q)?,
        w: up(gpu, &w)?,
        valid: up(gpu, &valid)?,
        qpos: up(gpu, &qpos)?,
        keys: gpu.alloc(pools_full * D * 4)?,
        idx: gpu.alloc(pools_full * KP * 4)?,
        pvalid: gpu.alloc(pools_full)?,
        keys_bf16: gpu.alloc(pools_full * D * 2)?,
        vis: gpu.alloc(pools_full * 4)?,
        out: gpu.alloc(q_rows * pools * 4)?,
        cand: gpu.alloc(q_rows * pools)?,
        q_rows,
        pools,
        s,
    };
    KernelLaunch::new(gpu, k.compress)
        .grid([pools_full as u32, 1, 1])
        .block([D as u32, 1, 1])
        .arg_ptr(kd)
        .arg_ptr(gd)
        .arg_ptr(io.valid)
        .arg_ptr(ad)
        .arg_ptr(io.keys)
        .arg_ptr(io.idx)
        .arg_ptr(io.pvalid)
        .arg_u32(s as u32)
        .arg_u32(D as u32)
        .arg_u32(KP as u32)
        .arg_i32(sh.first_key as i32)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .arg_u32(0)
        .launch(0)?;

    // 2026-10-14: 1. The prepared regions against the host conversion and the host rule.
    gpu.memset(io.keys_bf16, 0xA5, pools_full * D * 2)?;
    gpu.memset(io.vis, 0x5A, pools_full * 4)?;
    run_prep(gpu, k, &io)?;
    let keys_f = words(&read(gpu, io.keys, pools * D * 4)?);
    let want_bf: Vec<u8> = keys_f
        .iter()
        .map(|&b| bf16_rne(f32::from_bits(b)))
        .collect::<Result<Vec<u16>>>()?
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect();
    let got_bf = read(gpu, io.keys_bf16, pools * D * 2)?;
    let bad_keys = rows_differ(&got_bf, &want_bf, D * 2);
    let idx = words(&read(gpu, io.idx, pools * KP * 4)?);
    let pv = read(gpu, io.pvalid, pools)?;
    let want_vis: Vec<i32> = (0..pools)
        .map(|p| {
            let end = idx[p * KP + KP - 1] as i32;
            let end_c = end.clamp(0, s as i32 - 1);
            if pv[p] != 0 && valid[end_c as usize] != 0 {
                end_c
            } else {
                -1
            }
        })
        .collect();
    let got_vis: Vec<i32> = words(&read(gpu, io.vis, pools * 4)?)
        .iter()
        .map(|&b| b as i32)
        .collect();
    let bad_vis = got_vis
        .iter()
        .zip(&want_vis)
        .filter(|(a, b)| a != b)
        .count();
    let invisible = want_vis.iter().filter(|&&v| v < 0).count();

    // 2026-10-14: 2. Every score and candidacy byte.
    let (o_mma, c_mma) = outputs(gpu, &io, 0x00, || run_mma(gpu, k, &io))?;
    let (o_pipe, c_pipe) = outputs(gpu, &io, 0x7F, || run_pipe(gpu, k, &io))?;
    let bad_scores = rows_differ(&o_pipe, &o_mma, pools * 4);
    let bad_cand = rows_differ(&c_pipe, &c_mma, pools);
    let n_cand = c_mma.iter().filter(|&&c| c != 0).count();

    // 2026-10-14: 3. Negative controls on the first candidate (row, pool) pair's pool.
    let p_hit = c_mma.iter().position(|&c| c != 0).map(|i| i % pools);
    let (mut key_ctl, mut vis_ctl) = (false, false);
    if let Some(p) = p_hit {
        let mut flipped = got_bf.clone();
        flipped[p * D * 2 + 1] ^= 0x80;
        gpu.copy_h2d(&flipped, io.keys_bf16)?;
        let (o_ctl, _) = outputs(gpu, &io, 0x7F, || run_pipe(gpu, k, &io))?;
        key_ctl = o_ctl != o_pipe;
        let mut vis_bad = got_vis.clone();
        vis_bad[p] = -1;
        up_into(gpu, &vis_bad, io.vis)?;
        let (_, c_ctl) = outputs(gpu, &io, 0x7F, || run_pipe(gpu, k, &io))?;
        vis_ctl = c_ctl != c_pipe;
        run_prep(gpu, k, &io)?;
    }

    // 2026-10-14: 4. Timing.
    let tm = median_us(gpu, || run_mma(gpu, k, &io))?;
    let tp = median_us(gpu, || run_prep(gpu, k, &io))?;
    let ts = median_us(gpu, || run_pipe(gpu, k, &io))?;
    println!(
        "Q={q_rows:5} S={s:6} pools={pools:5} first_key={} {}: keys rows differ {bad_keys}, vis \
         differ {bad_vis} ({invisible} invisible), score rows differ {bad_scores}, candidacy \
         rows differ {bad_cand} ({n_cand} candidates); controls key={key_ctl} vis={vis_ctl}",
        sh.first_key,
        if sh.shuffled { "shuffled" } else { "causal" },
    );
    println!(
        "  mma {tm:9.1} us | prep {tp:7.1} us + pipe {ts:9.1} us = {:9.1} us ({:.3}x)",
        tp + ts,
        tm / (tp + ts)
    );
    for p in [
        kd,
        gd,
        ad,
        io.q,
        io.w,
        io.valid,
        io.qpos,
        io.keys,
        io.idx,
        io.pvalid,
        io.keys_bf16,
        io.vis,
        io.out,
        io.cand,
    ] {
        gpu.free(p)?;
    }
    ensure!(
        bad_keys == 0 && bad_vis == 0,
        "Q={q_rows} S={s}: prepared keys/visibility differ"
    );
    ensure!(
        bad_scores == 0 && bad_cand == 0,
        "Q={q_rows} S={s}: pipe differs from dsa_index_scores_mma"
    );
    ensure!(n_cand > 0, "Q={q_rows} S={s}: no candidate pool at all");
    ensure!(
        key_ctl && vis_ctl,
        "Q={q_rows} S={s}: a negative control left the outputs unchanged"
    );
    Ok(())
}

fn up_into(gpu: &dyn GpuBackend, v: &[i32], p: DevicePtr) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.copy_h2d(&b, p)?;
    Ok(())
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k = Kernels {
        compress: gpu.kernel("dsa_indexer", "dsa_kpool_compress")?,
        mma: gpu.kernel("dsa_indexer_rows", "dsa_index_scores_mma")?,
        prep: gpu.kernel(DSA_SCORES_PIPE_MODULE, "dsa_pool_prep_bf16")?,
        pipe: gpu.kernel(DSA_SCORES_PIPE_MODULE, "dsa_index_scores_mma_pipe")?,
    };
    let sh = |q, s, first_key, shuffled| Shape {
        q,
        s,
        first_key,
        shuffled,
    };
    let shapes = [
        sh(1024, 1024, 0, false),
        sh(8208, 8208, 0, false),
        sh(8192, 32792, 0, false),
        sh(288, 33072, 0, false),
        sh(37, 1030, 5, false),
        sh(300, 9001, 0, true),
    ];
    let mut rng = Rng(0x6a09_e667_f3bc_c908);
    for s in &shapes {
        shape(&gpu, &k, s, &mut rng)?;
    }
    println!("RESULT: PASS");
    Ok(())
}
