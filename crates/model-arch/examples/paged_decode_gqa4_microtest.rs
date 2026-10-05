// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Bit-identity and speed of the four-wide packed BF16 paged decode
//! (`paged_decode_attn_bf16_gqa4`) against the unpacked `paged_decode_attn` at the
//! Qwen3.6-35B-A3B attention shape: 16 query heads over 2 KV heads, head dim 256, block 16.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exits with an error when any case's packed output differs in any byte from the unpacked
//!   output, or when either leaves a head slice at the sentinel; times are printed per case.
//!
//! Run:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=qwen3.6-35b-a3b METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example paged_decode_gqa4_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_kernels::attn_splitk;
use metrale_model_layers::layers::ops;

const NQ: u32 = 16;
const NKV: u32 = 2;
const HD: u32 = 256;
const BS: u32 = 16;
const SENTINEL: u8 = 0xA5;
const ITERS: usize = 50;

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn r(&mut self) -> f32 {
        -1.0 + 2.0 * ((self.u() as f64) / ((1u64 << 53) as f64)) as f32
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(256))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn bf16s(rng: &mut Lcg, n: usize, mag: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| bf16::from_f32(rng.r() * mag).to_bits().to_le_bytes())
        .collect()
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let plain = g.kernel("paged_decode", "paged_decode_attn")?;
    let pack4 = g.kernel("paged_decode_attn_bf16_gqa", "paged_decode_attn_bf16_gqa4")?;
    let w = attn_splitk::DECODE_GQA_PACK4_WIDTH;
    let inv_sqrt_d = 1.0f32 / (HD as f32).sqrt();
    let mut rng = Lcg(0x5eed_0a77);
    let mut fail = 0usize;
    println!(
        "{:<26} {:>5} {:>12} {:>12} {:>8} {:>9}  bytes-identical",
        "case", "seqs", "plain (us)", "pack4 (us)", "speedup", "KV GB/s"
    );
    // 2026-10-05: (label, rows, min context, max context, sliding window).
    let cases: &[(&str, u32, u32, u32, u32)] = &[
        ("c12 short", 12, 37, 300, 0),
        ("c32 mixed", 32, 200, 1300, 0),
        ("c128x2 verify-like", 256, 200, 1300, 0),
        ("c128x2 long", 256, 1500, 2000, 0),
        ("c64 sliding 128", 64, 300, 900, 128),
        ("c16 prime 4093", 16, 4093, 4093, 0),
    ];
    for &(label, seqs, lo, hi, sliding) in cases {
        let lens: Vec<u32> = (0..seqs)
            .map(|_| lo + (rng.u() % u64::from(hi - lo + 1)) as u32)
            .collect();
        let mbps = lens.iter().max().unwrap().div_ceil(BS);
        let mut total_blocks = seqs * mbps + 3;
        if total_blocks.is_multiple_of(7) {
            total_blocks += 1;
        }
        let q = bf16s(&mut rng, (seqs * NQ * HD) as usize, 0.25);
        let pool = (total_blocks * BS * NKV * HD) as usize;
        let k = bf16s(&mut rng, pool, 1.0);
        let v = bf16s(&mut rng, pool, 1.0);
        let bt: Vec<u8> = (0..seqs * mbps)
            .flat_map(|i| (((i * 7 + 3) % total_blocks) as i32).to_le_bytes())
            .collect();
        let sl: Vec<u8> = lens
            .iter()
            .flat_map(|l| (*l as i32).to_le_bytes())
            .collect();
        let (qd, kd, vd, btd, sld) = (up(g, &q)?, up(g, &k)?, up(g, &v)?, up(g, &bt)?, up(g, &sl)?);
        let out = (seqs * NQ * HD * 2) as usize;
        let (oa, ob) = (g.alloc(out)?, g.alloc(out)?);
        g.memset_async(oa, SENTINEL, out, 0)?;
        g.memset_async(ob, SENTINEL, out, 0)?;
        let run_plain = |o: DevicePtr| {
            ops::paged_decode_attn_bf16(
                g,
                plain,
                qd,
                kd,
                vd,
                o,
                btd,
                sld,
                mbps,
                seqs,
                NQ,
                NKV,
                HD,
                BS,
                inv_sqrt_d,
                NQ * HD,
                sliding,
                0,
            )
        };
        let run_pack = |o: DevicePtr| {
            ops::paged_decode_attn_bf16_gqa(
                g,
                pack4,
                qd,
                kd,
                vd,
                o,
                btd,
                sld,
                mbps,
                seqs,
                NQ,
                NKV,
                HD,
                BS,
                inv_sqrt_d,
                NQ * HD,
                sliding,
                w,
                0,
            )
        };
        run_plain(oa)?;
        run_pack(ob)?;
        g.synchronize(0)?;
        let (mut a, mut b) = (vec![0u8; out], vec![0u8; out]);
        g.copy_d2h(oa, &mut a)?;
        g.copy_d2h(ob, &mut b)?;
        let head = (HD * 2) as usize;
        let unwritten = |x: &[u8]| {
            x.chunks_exact(head)
                .any(|h| h.iter().all(|&c| c == SENTINEL))
        };
        let same = a == b && !unwritten(&a) && !unwritten(&b);
        let time = |f: &dyn Fn(DevicePtr) -> Result<()>, o| -> Result<f64> {
            for _ in 0..5 {
                f(o)?;
            }
            g.synchronize(0)?;
            let t = std::time::Instant::now();
            for _ in 0..ITERS {
                f(o)?;
            }
            g.synchronize(0)?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
        };
        let tp = time(&run_plain, oa)?;
        let tq = time(&run_pack, ob)?;
        let attended: u64 = lens
            .iter()
            .map(|&l| u64::from(if sliding > 0 { l.min(sliding) } else { l }))
            .sum();
        let kv_bytes = attended * u64::from(NKV * HD) * 2 * 2;
        println!(
            "{label:<26} {seqs:>5} {tp:>12.1} {tq:>12.1} {:>7.2}x {:>9.0}  {}",
            tp / tq,
            kv_bytes as f64 / tq / 1e3,
            if same { "yes" } else { "NO" }
        );
        fail += usize::from(!same);
    }
    if fail > 0 {
        bail!("{fail} case(s) differ");
    }
    println!("ALL PASS: paged_decode_attn_bf16_gqa4 is byte-identical to paged_decode_attn");
    Ok(())
}
