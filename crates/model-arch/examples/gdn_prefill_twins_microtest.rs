// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The GDN chunked-prefill kernels changed on 2026-09-28, against the kernels
//! they replace, at Qwen3.6-35B-A3B GDN shapes (16 key heads, 32 value heads, head dims 128):
//! - the state spine `gated_delta_rule_chunk_delta_h_pipe` (now the default) against
//!   `gated_delta_rule_chunk_delta_h_vfused`: S_c, uc and the final state h;
//! - the output kernel `gated_delta_rule_chunk_fwd_o_mma8` against
//!   `gated_delta_rule_chunk_fwd_o`: the output.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless every compared buffer is byte-identical at every T (both legs
//!   write into sentinel-filled buffers, so an element one leg skips shows as a difference).
//!
//! Inputs: q and k L2-normalized per head (as the conv output is), v normal, gate in
//! (0.5, 1), beta in (0, 1), a small nonzero initial state; `recompute_wu` produces W, U
//! and gc for both legs. T covers a partial last chunk. Mean times over 5 launches are
//! printed.
//!
//! Run (GB10): cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!   --example gdn_prefill_twins_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

const NK: u32 = 16;
const NV: u32 = 32;
const D: u32 = 128;
const C: u32 = 64;
const SENTINEL: u8 = 0x5a;
const SMEM_WU: u32 = C * D * 2 + C * C * 4 + C * 4;
const SMEM_VFUSED: u32 = 2 * C * D * 2 + C * D * 2 + (C + 1) * 4;
const SMEM_PIPE: u32 = 2 * (C * 3 * D * 2) + 2 * C * 4 + 2 * (C + 1) * 4;
const SMEM_FO: u32 = 2 * C * D * 2 + C * C * 4 + C * D * 2 + D * D * 2 + 2 * C * 4;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}
fn filled(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes)?;
    gpu.memset(p, SENTINEL, bytes)?;
    Ok(p)
}
fn same(gpu: &dyn GpuBackend, a: DevicePtr, b: DevicePtr, bytes: usize, what: &str) -> Result<()> {
    let (mut x, mut y) = (vec![0u8; bytes], vec![0u8; bytes]);
    gpu.copy_d2h(a, &mut x)?;
    gpu.copy_d2h(b, &mut y)?;
    let diff = x
        .chunks_exact(2)
        .zip(y.chunks_exact(2))
        .filter(|(p, q)| p != q)
        .count();
    ensure!(
        diff == 0,
        "{what}: {diff} of {} 16-bit words differ",
        bytes / 2
    );
    Ok(())
}

/// 2026-09-28: The spine's argument list, with its own h, S and uc buffers.
#[allow(clippy::too_many_arguments)]
fn spine(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    smem: u32,
    h: DevicePtr,
    w: DevicePtr,
    u: DevicePtr,
    key: DevicePtr,
    gate: DevicePtr,
    gc: DevicePtr,
    s: DevicePtr,
    uc: DevicePtr,
    t: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([NV, 1, 1])
        .block([256, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h)
        .arg_ptr(w)
        .arg_ptr(u)
        .arg_ptr(key)
        .arg_ptr(gate)
        .arg_ptr(gc)
        .arg_ptr(s)
        .arg_ptr(uc)
        .arg_u32(1)
        .arg_u32(t)
        .arg_u32(t.div_ceil(C))
        .arg_u32(NK)
        .arg_u32(NV)
        .arg_u32(D)
        .arg_u32(D)
        .arg_u32(NK * D)
        .arg_u32(NV)
        .arg_u32(0)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn fwd_o(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    q: DevicePtr,
    key: DevicePtr,
    gate: DevicePtr,
    gc: DevicePtr,
    s: DevicePtr,
    uc: DevicePtr,
    out: DevicePtr,
    t: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([t.div_ceil(C), NV, 1])
        .block([512, 1, 1])
        .shared_mem(SMEM_FO)
        .arg_ptr(q)
        .arg_ptr(key)
        .arg_ptr(gate)
        .arg_ptr(gc)
        .arg_ptr(s)
        .arg_ptr(uc)
        .arg_ptr(out)
        .arg_u32(1)
        .arg_u32(t)
        .arg_u32(t.div_ceil(C))
        .arg_u32(NK)
        .arg_u32(NV)
        .arg_u32(D)
        .arg_u32(D)
        .arg_u32(NK * D)
        .arg_u32(NV)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .launch(0)
}

fn time(gpu: &dyn GpuBackend, f: &dyn Fn() -> Result<()>) -> Result<f64> {
    gpu.synchronize(0)?;
    let t = Instant::now();
    for _ in 0..5 {
        f()?;
    }
    gpu.synchronize(0)?;
    Ok(t.elapsed().as_secs_f64() * 200.0)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let fla = "gated_delta_rule_fla";
    let wu = gpu.kernel(fla, "gated_delta_rule_recompute_wu")?;
    let vfused = gpu.kernel(fla, "gated_delta_rule_chunk_delta_h_vfused")?;
    let pipe = gpu.kernel(fla, "gated_delta_rule_chunk_delta_h_pipe")?;
    let fo_old = gpu.kernel(fla, "gated_delta_rule_chunk_fwd_o")?;
    let fo_new = gpu.kernel("gdn_chunk_fwd_o_mma8", "gated_delta_rule_chunk_fwd_o_mma8")?;
    let mut rng = Rng(0x6764_6e32_2026_0928);
    for t in [8200u32, 1624, 300] {
        let (ts, nc) = (t as usize, t.div_ceil(C) as usize);
        let qk = |rng: &mut Rng| -> Vec<u8> {
            let mut out = Vec::with_capacity(ts * (NK * D) as usize * 2);
            for _ in 0..ts * NK as usize {
                let v: Vec<f32> = (0..D).map(|_| rng.normal()).collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter()
                    .for_each(|x| out.extend(bf16::from_f32(x / n).to_bits().to_le_bytes()));
            }
            out
        };
        let q = upload(&gpu, &qk(&mut rng))?;
        let key = upload(&gpu, &qk(&mut rng))?;
        let v: Vec<u8> = (0..ts * (NV * D) as usize)
            .flat_map(|_| bf16::from_f32(rng.normal()).to_bits().to_le_bytes())
            .collect();
        let value = upload(&gpu, &v)?;
        let f32s = |n: usize, f: &mut dyn FnMut() -> f32| -> Vec<u8> {
            (0..n).flat_map(|_| f().to_le_bytes()).collect()
        };
        let gate = upload(
            &gpu,
            &f32s(ts * NV as usize, &mut || 0.5 + 0.5 * rng.unit()),
        )?;
        let beta = upload(&gpu, &f32s(ts * NV as usize, &mut || rng.unit()))?;
        let h0 = f32s((NV * D * D) as usize, &mut || 0.01 * rng.normal());
        let blocks = nc * NV as usize;
        let (wb, ub, gcb) = (
            blocks * (C * D) as usize * 2,
            blocks * (C * D) as usize * 2,
            blocks * C as usize * 4,
        );
        let (w, u, gc) = (filled(&gpu, wb)?, filled(&gpu, ub)?, filled(&gpu, gcb)?);
        KernelLaunch::new(&gpu, wu)
            .grid([nc as u32, NV, 1])
            .block([256, 1, 1])
            .shared_mem(SMEM_WU)
            .arg_ptr(key)
            .arg_ptr(value)
            .arg_ptr(gate)
            .arg_ptr(beta)
            .arg_ptr(w)
            .arg_ptr(u)
            .arg_ptr(gc)
            .arg_u32(1)
            .arg_u32(t)
            .arg_u32(nc as u32)
            .arg_u32(NK)
            .arg_u32(NV)
            .arg_u32(D)
            .arg_u32(D)
            .arg_u32(NK * D)
            .arg_u32(NV * D)
            .arg_u32(NV)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(DevicePtr::NULL)
            .arg_u32(0)
            .launch(0)?;
        let (sb, ucb, hb) = (
            blocks * (D * D) as usize * 2,
            blocks * (C * D) as usize * 2,
            h0.len(),
        );
        let legs: Vec<[DevicePtr; 3]> = (0..2)
            .map(|_| Ok([filled(&gpu, sb)?, filled(&gpu, ucb)?, upload(&gpu, &h0)?]))
            .collect::<Result<_>>()?;
        for (k, smem, l) in [(vfused, SMEM_VFUSED, 0), (pipe, SMEM_PIPE, 1)] {
            let [s, uc, h] = legs[l];
            spine(&gpu, k, smem, h, w, u, key, gate, gc, s, uc, t)?;
        }
        gpu.synchronize(0)?;
        same(
            &gpu,
            legs[0][0],
            legs[1][0],
            sb,
            &format!("T={t} spine S_c"),
        )?;
        same(
            &gpu,
            legs[0][1],
            legs[1][1],
            ucb,
            &format!("T={t} spine uc"),
        )?;
        same(&gpu, legs[0][2], legs[1][2], hb, &format!("T={t} spine h"))?;
        let ob = ts * (NV * D) as usize * 2;
        let outs = [filled(&gpu, ob)?, filled(&gpu, ob)?];
        let [s, uc, _] = legs[0];
        for (k, o) in [(fo_old, outs[0]), (fo_new, outs[1])] {
            fwd_o(&gpu, k, q, key, gate, gc, s, uc, o, t)?;
        }
        gpu.synchronize(0)?;
        same(&gpu, outs[0], outs[1], ob, &format!("T={t} fwd_o output"))?;
        let [s1, uc1, h1] = legs[1];
        let spine_ms = [
            time(&gpu, &|| {
                spine(
                    &gpu,
                    vfused,
                    SMEM_VFUSED,
                    h1,
                    w,
                    u,
                    key,
                    gate,
                    gc,
                    s1,
                    uc1,
                    t,
                )
            })?,
            time(&gpu, &|| {
                spine(&gpu, pipe, SMEM_PIPE, h1, w, u, key, gate, gc, s1, uc1, t)
            })?,
        ];
        let fo_ms = [
            time(&gpu, &|| {
                fwd_o(&gpu, fo_old, q, key, gate, gc, s, uc, outs[0], t)
            })?,
            time(&gpu, &|| {
                fwd_o(&gpu, fo_new, q, key, gate, gc, s, uc, outs[1], t)
            })?,
        ];
        println!(
            "T={t}: bit-identical; spine vfused {:.3} -> pipe {:.3} ms, fwd_o {:.3} -> mma8 {:.3} ms",
            spine_ms[0], spine_ms[1], fo_ms[0], fo_ms[1]
        );
        for p in [q, key, value, gate, beta, w, u, gc]
            .into_iter()
            .chain(legs.iter().flatten().copied())
            .chain(outs)
        {
            gpu.free(p)?;
        }
    }
    println!("PASS: pipe spine == vfused spine, fwd_o_mma8 == fwd_o");
    Ok(())
}
