// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `moe_unpermute_blend` against `moe_unpermute_reduce_indexed` followed by
//! `moe_batched_blend`, at Qwen3.6-35B-A3B shapes (hidden 2048, top-8), with and without the
//! shared-expert gate weight.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless every case's output is byte-identical.
//!
//! Inputs: random BF16 expert rows, shared output and normed input, top-k weights in
//! [0, 1/8), a random permutation as token_to_perm. Mean times over 10 launches are printed.
//!
//! Run (GB10): cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!   --example moe_unpermute_blend_microtest

use anyhow::{Context, Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use std::time::Instant;

const H: u32 = 2048;
const TOP_K: u32 = 8;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    fn bf16s(&mut self, n: usize, scale: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                bf16::from_f32((2.0 * self.unit() - 1.0) * scale)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let set = sets
        .iter()
        .find(|s| s.target.model == "qwen3.6-35b-a3b")
        .context("qwen3.6-35b-a3b kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &set.modules)?;
    let unpermute = gpu.kernel("moe", "moe_unpermute_reduce_indexed")?;
    let blend = gpu.kernel("moe", "moe_batched_blend")?;
    let fused = gpu.kernel("moe_unpermute_blend", "moe_unpermute_blend")?;
    let mut rng = Rng(0x7562_6c64_2026_0928);
    for (tokens, gated) in [(8192u32, true), (474, true), (1000, false)] {
        let te = (tokens * TOP_K) as usize;
        let t = tokens as usize;
        let eo = upload(&gpu, &rng.bf16s(te * H as usize, 4.0))?;
        let so = upload(&gpu, &rng.bf16s(t * H as usize, 4.0))?;
        let nm = upload(&gpu, &rng.bf16s(t * H as usize, 2.0))?;
        let gw = if gated {
            upload(&gpu, &rng.bf16s(H as usize, 0.05))?
        } else {
            DevicePtr::NULL
        };
        let mut perm: Vec<i32> = (0..te as i32).collect();
        for i in (1..te).rev() {
            perm.swap(i, (rng.unit() * (i + 1) as f32) as usize % (i + 1));
        }
        let perm_d = upload(
            &gpu,
            &perm
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let w: Vec<u8> = (0..te)
            .flat_map(|_| (rng.unit() / 8.0).to_le_bytes())
            .collect();
        let w_d = upload(&gpu, &w)?;
        let bytes = t * H as usize * 2;
        let outs = [gpu.alloc(bytes)?, gpu.alloc(bytes)?];
        let pair = || -> Result<()> {
            ops::moe_unpermute_reduce_indexed(
                &gpu, unpermute, eo, outs[0], perm_d, w_d, H, tokens, TOP_K, 0,
            )?;
            ops::moe_batched_blend(&gpu, blend, outs[0], so, nm, gw, H, tokens, 0)
        };
        let one = || {
            ops::moe_unpermute_blend(
                &gpu, fused, eo, outs[1], perm_d, w_d, so, nm, gw, H, tokens, TOP_K, 0,
            )
        };
        pair()?;
        one()?;
        gpu.synchronize(0)?;
        let (mut a, mut b) = (vec![0u8; bytes], vec![0u8; bytes]);
        gpu.copy_d2h(outs[0], &mut a)?;
        gpu.copy_d2h(outs[1], &mut b)?;
        ensure!(a == b, "tokens {tokens} gated {gated}: outputs differ");
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            gpu.synchronize(0)?;
            let s = Instant::now();
            for _ in 0..10 {
                f()?;
            }
            gpu.synchronize(0)?;
            Ok(s.elapsed().as_secs_f64() * 1e2)
        };
        let (t0, t1) = (time(&pair)?, time(&one)?);
        println!(
            "tokens {tokens} gated {gated}: bit-identical; pair {t0:.3} ms, fused {t1:.3} ms ({:.2}x)",
            t0 / t1
        );
        for p in [eo, so, nm, perm_d, w_d, outs[0], outs[1]] {
            gpu.free(p)?;
        }
        if gated {
            gpu.free(gw)?;
        }
    }
    println!("PASS: moe_unpermute_blend equals unpermute + blend");
    Ok(())
}
