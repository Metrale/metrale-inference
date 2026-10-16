// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The narrow unsplit segmented row tiles (`ops::w4a16_tc_rows_seg_narrow`, the
//! entries `METRALE_GLM_W4A16_SEG_FILL` launches) against the 4-warp unsplit launch
//! (`ops::w4a16_tc_rows_seg`, split 1) at GLM-5.3 TP=3 groups, with their times.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every group, M in 1..=16 and every warps count, each segment's
//!   bytes equal the 4-warp launch's and nothing is written past a segment's M x N.
//! - Per group, a launch on altered weights must change the output (the comparison can fail).
//!
//! Inputs as `w4a16_tc_rows_seg_microtest` (17 binary orders, scales 2^-8 .. 2^7), so a changed
//! summation order would show. Times: median of 7 batches of 20 launches over rotated weight
//! copies (>= 96 MB, weights stream from DRAM), host-timed; the rule's choice at fill 1, 2, 3.
//!
//! Run (GB10, < 1 GB of device memory):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example glm5next_w4a16_seg_narrow_gate

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{self, W4a16Seg};
use metrale_model_layers::weight_map::QuantizedWeight;

const SENTINEL: u8 = 0x5a;
const PAD: usize = 64;
/// 2026-10-16: GLM-5.3 TP=3 rank-0 groups `(widths, K)`: KDA q/k/v, f_a/g_a/b, o, the shared
/// expert's gate/up and down.
const GROUPS: [(&[u32], u32); 5] = [
    (&[2816, 2816, 2816], 4096),
    (&[128, 128, 22], 4096),
    (&[4096], 2816),
    (&[768, 768], 4096),
    (&[4096], 768),
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
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn weight(gpu: &dyn GpuBackend, rng: &mut Rng, n: u32, k: u32) -> Result<QuantizedWeight> {
    let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
    let scale: Vec<u8> = (0..n * k / 16)
        .map(|_| (0x08 + rng.next() % 0x70) as u8)
        .collect();
    Ok(QuantizedWeight {
        weight: upload(gpu, &packed)?,
        weight_scale: upload(gpu, &scale)?,
        weight_scale_2: 0.0123,
        ..QuantizedWeight::null()
    })
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn segs(ws: &[QuantizedWeight], ns: &[u32], outs: &[DevicePtr]) -> Vec<W4a16Seg> {
    (0..ws.len())
        .map(|i| W4a16Seg {
            weight: ws[i],
            output: outs[i],
            n: ns[i],
        })
        .collect()
}

fn median_us(mut f: impl FnMut(usize) -> Result<()>, gpu: &dyn GpuBackend) -> Result<f64> {
    let mut t = Vec::new();
    for rep in 0..7 {
        gpu.synchronize(0)?;
        let t0 = std::time::Instant::now();
        for i in 0..20 {
            f(rep * 20 + i)?;
        }
        gpu.synchronize(0)?;
        t.push(t0.elapsed().as_secs_f64() * 1e6 / 20.0);
    }
    t.sort_by(f64::total_cmp);
    Ok(t[3])
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let sms = gpu.sm_count()?;
    let mut rng = Rng(0x6e61_7272_6f77_2d73);
    let kmax = 4096usize;
    let a: Vec<u8> = (0..16 * kmax)
        .flat_map(|_| {
            let u = rng.next();
            let bits = ((u & 1) << 15) | ((115 + (u >> 1) % 17) << 7) | ((u >> 8) & 0x7f);
            (bits as u16).to_le_bytes()
        })
        .collect();
    let ad = upload(&gpu, &a)?;
    let out_bytes = 16 * 4096 * 2 + PAD * 2;
    let outs: Vec<DevicePtr> = (0..6)
        .map(|_| gpu.alloc(out_bytes))
        .collect::<Result<_>>()?;
    let sentinel = vec![SENTINEL; out_bytes];
    let mut total_us = [[0f64; 3]; 4];
    for (ns, k) in GROUPS {
        let ws: Vec<QuantizedWeight> = ns
            .iter()
            .map(|&n| weight(&gpu, &mut rng, n, k))
            .collect::<Result<_>>()?;
        let (old, new) = (&outs[..3], &outs[3..]);
        for m in 1..=ops::W4A16_TC_ROWS_SEG_NARROW_MAX_M {
            ops::w4a16_tc_rows_seg(&gpu, ad, &segs(&ws, ns, old), m, k, k, 1, 0)?;
            let want: Vec<Vec<u8>> = (0..ns.len())
                .map(|i| read(&gpu, old[i], (m * ns[i] * 2) as usize))
                .collect::<Result<_>>()?;
            for nw in ops::W4A16_TC_ROWS_SEG_WARPS {
                for o in new {
                    gpu.copy_h2d(&sentinel, *o)?;
                }
                ops::w4a16_tc_rows_seg_narrow(&gpu, ad, &segs(&ws, ns, new), m, k, k, nw, 0)?;
                for (i, want) in want.iter().enumerate() {
                    let got = read(&gpu, new[i], want.len() + PAD * 2)?;
                    ensure!(
                        got[want.len()..].iter().all(|&b| b == SENTINEL),
                        "{ns:?} M={m} warps={nw} segment {i}: written past M x N"
                    );
                    ensure!(
                        got[..want.len()] == want[..],
                        "{ns:?} M={m} warps={nw} segment {i}: bytes differ from the 4-warp launch"
                    );
                }
            }
        }
        // 2026-10-16: Altered weights must change the output (the comparison can fail).
        let keep = read(&gpu, ws[0].weight, 64)?;
        let flip: Vec<u8> = keep.iter().map(|b| b ^ 0x77).collect();
        ops::w4a16_tc_rows_seg_narrow(&gpu, ad, &segs(&ws, ns, new), 4, k, k, 1, 0)?;
        let base = read(&gpu, new[0], (4 * ns[0] * 2) as usize)?;
        gpu.copy_h2d(&flip, ws[0].weight)?;
        ops::w4a16_tc_rows_seg_narrow(&gpu, ad, &segs(&ws, ns, new), 4, k, k, 1, 0)?;
        ensure!(
            read(&gpu, new[0], base.len())? != base,
            "{ns:?}: altered weights were admitted"
        );
        println!("KNOWN_BAD altered weights {ns:?}: refused");
        gpu.copy_h2d(&keep, ws[0].weight)?;

        let bytes: usize = ns.iter().map(|&n| (n * k / 2 + n * k / 16) as usize).sum();
        let copies = (96 << 20) / bytes + 1;
        let rot: Vec<Vec<QuantizedWeight>> = (0..copies.min(400))
            .map(|_| ns.iter().map(|&n| weight(&gpu, &mut rng, n, k)).collect())
            .collect::<Result<_>>()?;
        let picks: Vec<u32> = (1..=3)
            .map(|fill| ops::w4a16_tc_rows_seg_warps_for(ns, fill, sms))
            .collect();
        for (mi, m) in [2u32, 4, 8, 16].into_iter().enumerate() {
            let mut line = format!("{ns:?} K={k} M={m:2}:");
            for (wi, nw) in ops::W4A16_TC_ROWS_SEG_WARPS.into_iter().enumerate() {
                let us = median_us(
                    |i| {
                        let w = &rot[i % rot.len()];
                        ops::w4a16_tc_rows_seg_narrow(&gpu, ad, &segs(w, ns, new), m, k, k, nw, 0)
                    },
                    &gpu,
                )?;
                total_us[mi][wi] += us;
                line += &format!("  w{nw} {us:7.1}us");
            }
            println!(
                "{line}  rule fill1/2/3 -> w{}/w{}/w{}",
                picks[0], picks[1], picks[2]
            );
        }
        for w in rot.iter().flatten().chain(&ws) {
            gpu.free(w.weight)?;
            gpu.free(w.weight_scale)?;
        }
    }
    for (mi, m) in [2u32, 4, 8, 16].into_iter().enumerate() {
        println!(
            "sum of one launch per group, M={m:2}: w4 {:.1}us w2 {:.1}us w1 {:.1}us",
            total_us[mi][0], total_us[mi][1], total_us[mi][2]
        );
    }
    println!(
        "ALL PASS: every narrow entry (1/2/4 warps, M 1..=16) writes the 4-warp unsplit bytes; nothing past M x N"
    );
    Ok(())
}
