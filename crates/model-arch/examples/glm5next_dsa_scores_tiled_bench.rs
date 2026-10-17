// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `dsa_index_scores_tiled` (the `METRALE_GLM_DSA_SCORES_TILED` path) against
//! `dsa_index_scores` on a prefill sub-chunk's selection: GLM-5.3's indexer (32 heads of 128,
//! pools of 4 tokens), Q query rows at the end of a context of S tokens, every pool complete.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless the scores and the candidacy bytes of the two kernels are
//!   identical, and unless the negative control (one query value changed) changes the scores.
//!
//! Run (GB10, < 1 GB of device memory):
//!   GLM_BENCH_SHAPES=2048x4096,2048x32768 cargo run --release -p metrale-model-arch \
//!     --features cuda,gpu-examples --example glm5next_dsa_scores_tiled_bench

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const H: usize = 32;
const D: usize = 128;
const KP: usize = 4;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn up<T: Copy>(gpu: &dyn GpuBackend, v: &[T]) -> Result<DevicePtr> {
    let n = std::mem::size_of_val(v);
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

struct Io {
    q: DevicePtr,
    keys: DevicePtr,
    w: DevicePtr,
    idx: DevicePtr,
    pvalid: DevicePtr,
    qpos: DevicePtr,
    out: DevicePtr,
    cand: DevicePtr,
    q_rows: usize,
    pools: usize,
    s: usize,
}

fn launch(gpu: &dyn GpuBackend, k: KernelHandle, tiled: bool, io: &Io, q: DevicePtr) -> Result<()> {
    let (grid, block) = if tiled {
        (
            [
                io.pools.div_ceil(64) as u32,
                io.q_rows.div_ceil(32) as u32,
                1,
            ],
            256,
        )
    } else {
        ([io.pools as u32, io.q_rows as u32, 1], 128)
    };
    let l = KernelLaunch::new(gpu, k)
        .grid(grid)
        .block([block, 1, 1])
        .arg_ptr(q)
        .arg_ptr(io.keys)
        .arg_ptr(io.w)
        .arg_ptr(io.idx)
        .arg_ptr(io.pvalid)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(io.qpos)
        .arg_ptr(io.out)
        .arg_ptr(io.cand)
        .arg_u32(io.q_rows as u32)
        .arg_u32(io.pools as u32)
        .arg_u32(H as u32)
        .arg_u32(D as u32)
        .arg_u32(KP as u32)
        .arg_u32(io.s as u32)
        .arg_f32((D as f32).powf(-0.5));
    if tiled {
        l.shared_mem(128 * 96 * 4).launch(0)
    } else {
        l.shared_mem(128.max((H * 4) as u32))
            .arg_ptr(DevicePtr::NULL)
            .launch(0)
    }
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

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let base = gpu.kernel("dsa_indexer", "dsa_index_scores")?;
    let tiled = gpu.kernel("dsa_indexer_rows", "dsa_index_scores_tiled")?;
    let shapes: Vec<(usize, usize)> = std::env::var("GLM_BENCH_SHAPES")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|x| {
                    let (a, b) = x.trim().split_once('x')?;
                    Some((a.parse().ok()?, b.parse().ok()?))
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(2048, 4096), (2048, 16384)]);
    let mut rng = Rng(0x1d5a_5c0e_0bad_cafe);
    for (q_rows, s) in shapes {
        let pools = s / KP;
        let q: Vec<f32> = (0..q_rows * H * D).map(|_| rng.unit()).collect();
        let keys: Vec<f32> = (0..pools * D).map(|_| rng.unit()).collect();
        let w: Vec<f32> = (0..q_rows * H).map(|_| rng.unit() * 0.2).collect();
        let idx: Vec<i32> = (0..(pools * KP) as i32).collect();
        let pvalid = vec![1u8; pools];
        let qpos: Vec<i32> = (0..q_rows).map(|r| (s - q_rows + r) as i32).collect();
        let io = Io {
            q: up(&gpu, &q)?,
            keys: up(&gpu, &keys)?,
            w: up(&gpu, &w)?,
            idx: up(&gpu, &idx)?,
            pvalid: up(&gpu, &pvalid)?,
            qpos: up(&gpu, &qpos)?,
            out: gpu.alloc(q_rows * pools * 4)?,
            cand: gpu.alloc(q_rows * pools)?,
            q_rows,
            pools,
            s,
        };
        launch(&gpu, base, false, &io, io.q)?;
        let (o_ref, c_ref) = (
            read(&gpu, io.out, q_rows * pools * 4)?,
            read(&gpu, io.cand, q_rows * pools)?,
        );
        gpu.memset(io.out, 0, q_rows * pools * 4)?;
        gpu.memset(io.cand, 0x55, q_rows * pools)?;
        launch(&gpu, tiled, true, &io, io.q)?;
        let (o, c) = (
            read(&gpu, io.out, q_rows * pools * 4)?,
            read(&gpu, io.cand, q_rows * pools)?,
        );
        ensure!(c == c_ref, "Q={q_rows} S={s}: candidacy differs");
        ensure!(
            o == o_ref,
            "Q={q_rows} S={s}: scores differ from dsa_index_scores"
        );
        let mut q2 = q.clone();
        q2[(q_rows - 1) * H * D] += 0.5;
        let q2d = up(&gpu, &q2)?;
        launch(&gpu, tiled, true, &io, q2d)?;
        ensure!(
            read(&gpu, io.out, q_rows * pools * 4)? != o_ref,
            "Q={q_rows} S={s}: negative control left the scores unchanged"
        );
        let tb = median_us(&gpu, || launch(&gpu, base, false, &io, io.q))?;
        let tt = median_us(&gpu, || launch(&gpu, tiled, true, &io, io.q))?;
        println!(
            "Q={q_rows:5} S={s:6} pools={pools:5}: identical; dsa_index_scores {tb:9.1} us, tiled {tt:9.1} us ({:.1}x)",
            tb / tt
        );
        for p in [
            io.q, io.keys, io.w, io.idx, io.pvalid, io.qpos, io.out, io.cand, q2d,
        ] {
            gpu.free(p)?;
        }
    }
    Ok(())
}
