// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: The 128-row prefill attention twins (`ops::AttnFa128Kernels`) against the
//! kernels they replace: `attn_prefill_fa128_paged` vs `attn_prefill_paged_64` over a
//! BF16 paged cache with a shuffled block table, and `attn_prefill_fa128` vs
//! `attn_prefill_64` over contiguous BF16 K/V, at HDIM 256 with the 35B's (16/2) and the
//! dense 27B's (24/4) head counts.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless every case's twin output is byte-identical to the original's
//!   over all q_len x heads x 256 values (both are first filled with a sentinel, so a row
//!   either leg skips shows up as a difference).
//!
//! Cases cover q_len not a multiple of 64 or 128, a chunk after a cached prefix
//! (q_offset > 0, paged only), and short prompts. Mean times over 5 launches are printed.
//!
//! Run (GB10): cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!   --example attn_prefill_fa128_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{self, AttnFa128Kernels};
use std::time::Instant;

const HD: usize = 256;
const SENTINEL: u8 = 0x5a;
/// 2026-09-28: (q_len, q_offset, num_q_heads, num_kv_heads, cache block size). The served block
/// size is 16; 32 and 24 cover the twin's other power-of-two and its non-power-of-two paths.
const CASES: [(usize, usize, usize, usize, usize); 8] = [
    (8200, 0, 16, 2, 16),
    (8200, 16392, 16, 2, 16),
    (1000, 0, 24, 4, 16),
    (777, 5000, 24, 4, 16),
    (300, 0, 16, 2, 16),
    (65, 130, 16, 2, 16),
    (777, 5000, 16, 2, 32),
    (777, 5000, 16, 2, 24),
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// 2026-09-27: A BF16 value in [-scale, scale).
    fn bf16_bytes(&mut self, n: usize, scale: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                let u = self.next() as f32 / u32::MAX as f32;
                bf16::from_f32((2.0 * u - 1.0) * scale)
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

fn compare(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    b: DevicePtr,
    bytes: usize,
    what: &str,
) -> Result<()> {
    let (mut x, mut y) = (vec![0u8; bytes], vec![0u8; bytes]);
    gpu.copy_d2h(a, &mut x)?;
    gpu.copy_d2h(b, &mut y)?;
    let diff = x
        .chunks_exact(2)
        .zip(y.chunks_exact(2))
        .filter(|(p, q)| p != q)
        .count();
    ensure!(diff == 0, "{what}: {diff} of {} values differ", bytes / 2);
    Ok(())
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
    ensure!(
        std::env::var_os("METRALE_NO_ATTN_FA128").is_none(),
        "unset METRALE_NO_ATTN_FA128: the twins would not launch"
    );
    let twins = AttnFa128Kernels::resolve(&gpu);
    let paged64 = gpu.kernel("prefill_paged", "attn_prefill_paged_64")?;
    let contig64 = gpu.kernel("attn_prefill", "attn_prefill_64")?;
    let mut rng = Rng(0x6661_3132_2026_0927);
    let isd = 1.0 / (HD as f32).sqrt();
    for (q_len, q_off, nq, nkv, bs) in CASES {
        let kv_len = q_off + q_len;
        let pages = kv_len.div_ceil(bs);
        // 2026-09-27: Block table: a pseudo-random permutation of the pages.
        let mut table: Vec<i32> = (0..pages as i32).collect();
        for i in (1..pages).rev() {
            table.swap(i, rng.next() as usize % (i + 1));
        }
        let bt: Vec<u8> = table.iter().flat_map(|x| x.to_le_bytes()).collect();
        let cache_elems = pages * bs * nkv * HD;
        let q = upload(&gpu, &rng.bf16_bytes(q_len * nq * HD, 3.0))?;
        let kc = upload(&gpu, &rng.bf16_bytes(cache_elems, 1.0))?;
        let vc = upload(&gpu, &rng.bf16_bytes(cache_elems, 1.0))?;
        let kx = upload(&gpu, &rng.bf16_bytes(q_len * nkv * HD, 1.0))?;
        let vx = upload(&gpu, &rng.bf16_bytes(q_len * nkv * HD, 1.0))?;
        let bt_d = upload(&gpu, &bt)?;
        let bytes = q_len * nq * HD * 2;
        let outs: Vec<DevicePtr> = (0..4).map(|_| gpu.alloc(bytes)).collect::<Result<_>>()?;
        for &o in &outs {
            gpu.memset(o, SENTINEL, bytes)?;
        }
        let (ql, kl, qo, h, g, b) = (
            q_len as u32,
            kv_len as u32,
            q_off as u32,
            nq as u32,
            nkv as u32,
            bs as u32,
        );
        let run_paged_old = || {
            ops::prefill_attention_paged_64(
                &gpu, paged64, q, kc, vc, outs[0], bt_d, ql, kl, qo, h, g, 256, b, 0, isd, 0,
            )
        };
        let run_paged_new = || -> Result<()> {
            let ran = twins.paged(
                &gpu, q, kc, vc, outs[1], bt_d, ql, kl, qo, h, g, 256, b, 0, isd, 0,
            )?;
            ensure!(ran, "attn_prefill_fa128_paged did not apply");
            Ok(())
        };
        let run_contig_old = || {
            ops::prefill_attention_64(
                &gpu, contig64, q, kx, vx, outs[2], ql, 1, h, g, 256, isd, true, 0, 0,
            )
        };
        let run_contig_new = || -> Result<()> {
            let ran =
                twins.contiguous(&gpu, q, kx, vx, outs[3], ql, 1, h, g, 256, isd, true, 0, 0)?;
            ensure!(ran, "attn_prefill_fa128 did not apply");
            Ok(())
        };
        run_paged_old()?;
        run_paged_new()?;
        run_contig_old()?;
        run_contig_new()?;
        gpu.synchronize(0)?;
        let case = format!("q_len {q_len} q_offset {q_off} heads {nq}/{nkv} block {bs}");
        compare(&gpu, outs[0], outs[1], bytes, &format!("paged, {case}"))?;
        compare(
            &gpu,
            outs[2],
            outs[3],
            bytes,
            &format!("contiguous, {case}"),
        )?;
        let (po, pn) = (time(&gpu, &run_paged_old)?, time(&gpu, &run_paged_new)?);
        let (co, cn) = (time(&gpu, &run_contig_old)?, time(&gpu, &run_contig_new)?);
        println!(
            "{case}: bit-identical; paged {po:.3} -> {pn:.3} ms ({:.2}x), contiguous {co:.3} -> {cn:.3} ms ({:.2}x)",
            po / pn,
            co / cn
        );
        for p in [q, kc, vc, kx, vx, bt_d].into_iter().chain(outs) {
            gpu.free(p)?;
        }
    }
    println!(
        "PASS: the fa128 twins are bit-identical to attn_prefill_paged_64 and attn_prefill_64"
    );
    Ok(())
}
