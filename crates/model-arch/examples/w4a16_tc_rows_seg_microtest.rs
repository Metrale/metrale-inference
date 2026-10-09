// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `w4a16_tc_rows_seg` (`kernels/gb10/common/w4a16_tc_rows_seg.cu`) against
//! `w4a16_tc_rows` at GLM-5.3 TP=3 projection shapes, single and grouped.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every shape and M: split 1 writes, segment by segment, the
//!   bytes `w4a16_tc_rows` writes; every split is within `MAX_REL_L2` of them; for each split a
//!   row of an M-row launch equals, byte for byte, that row launched alone on each of the 16-,
//!   32- and 64-row entries; nothing is written past a segment's M x N.
//! - Per group, a launch on altered weights must change the output (the comparisons can fail).
//!
//! The inputs span 17 binary orders (random sign, exponent and mantissa) and the E4M3 scales
//! 2^-8 .. 2^7, so FP32 sums round and a split's other summation order shows (a narrow input
//! range sums exactly and would hide it). Times: median of 7 batches of 20 launches over
//! rotated weight copies (>= 96 MB, so the weights stream from DRAM), host-timed.
//!
//! Run (GB10, < 1 GB of device memory):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example w4a16_tc_rows_seg_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{self, W4a16Seg};
use metrale_model_layers::weight_map::QuantizedWeight;

const SENTINEL: u8 = 0x5a;
const PAD: usize = 64;
/// 2026-10-09: A split differs from the unsplit sum by FP32 reassociation only: measured
/// ~1e-4 relative L2 on these inputs, so 1e-2.
const MAX_REL_L2: f64 = 1e-2;
/// 2026-10-09: GLM-5.3 TP=3 rank-0 groups `(widths, K)`: KDA q/k/v, f_a/g_a/b, o, the shared
/// expert's gate/up and down, and single projections of each width.
const GROUPS: [(&[u32], u32); 9] = [
    (&[2816], 4096),
    (&[2816, 2816, 2816], 4096),
    (&[128], 4096),
    (&[128, 128, 22], 4096),
    (&[4096], 2816),
    (&[768], 4096),
    (&[768, 768], 4096),
    (&[4096], 768),
    (&[512], 4096),
];
const ROWS: [u32; 9] = [1, 2, 4, 8, 9, 16, 17, 33, 64];

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

fn rel_l2(a: &[u8], b: &[u8]) -> f64 {
    let f = |c: &[u8]| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64();
    let (mut num, mut den) = (0.0, 0.0);
    for (x, y) in a.chunks(2).zip(b.chunks(2)) {
        num += (f(x) - f(y)).powi(2);
        den += f(y).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

/// 2026-10-09: The split-`s` launch of `ws` over `m` rows of `a` into `outs`.
#[allow(clippy::too_many_arguments)]
fn seg(
    gpu: &dyn GpuBackend,
    ws: &[QuantizedWeight],
    ns: &[u32],
    outs: &[DevicePtr],
    a: DevicePtr,
    m: u32,
    k: u32,
    s: u32,
) -> Result<()> {
    let segs: Vec<W4a16Seg> = (0..ws.len())
        .map(|i| W4a16Seg {
            weight: ws[i],
            output: outs[i],
            n: ns[i],
        })
        .collect();
    ops::w4a16_tc_rows_seg(gpu, a, &segs, m, k, k, s, 0)
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
    let mut rng = Rng(0x7365_672d_726f_7773);
    let kmax = 4096usize;
    let a: Vec<u8> = (0..64 * kmax)
        .flat_map(|_| {
            let u = rng.next();
            let bits = ((u & 1) << 15) | ((115 + (u >> 1) % 17) << 7) | ((u >> 8) & 0x7f);
            (bits as u16).to_le_bytes()
        })
        .collect();
    let ad = upload(&gpu, &a)?;
    let out_bytes = 64 * 8448 * 2 + PAD * 2;
    let outs: Vec<DevicePtr> = (0..6)
        .map(|_| gpu.alloc(out_bytes))
        .collect::<Result<_>>()?;
    let sentinel = vec![SENTINEL; out_bytes];
    for (ns, k) in GROUPS {
        let ws: Vec<QuantizedWeight> = ns
            .iter()
            .map(|&n| weight(&gpu, &mut rng, n, k))
            .collect::<Result<_>>()?;
        let (old, new) = (&outs[..3], &outs[3..]);
        let splits: Vec<u32> = ops::W4A16_TC_ROWS_SPLITS
            .into_iter()
            .filter(|&s| s <= k / ops::W4A16_TC_ROWS_SPLIT_K)
            .collect();
        let rule = ops::w4a16_tc_rows_split_for(ops::w4a16_tc_rows_seg_tiles(ns), k, sms);
        for m in ROWS {
            for (i, w) in ws.iter().enumerate() {
                ops::w4a16_tc_rows(&gpu, ad, w, old[i], m, ns[i], k, k, ns[i], 0)?;
            }
            let want: Vec<Vec<u8>> = (0..ns.len())
                .map(|i| read(&gpu, old[i], (m * ns[i] * 2) as usize))
                .collect::<Result<_>>()?;
            for &s in &splits {
                for o in new {
                    gpu.copy_h2d(&sentinel, *o)?;
                }
                seg(&gpu, &ws, ns, new, ad, m, k, s)?;
                for (i, want) in want.iter().enumerate() {
                    let got = read(&gpu, new[i], want.len() + PAD * 2)?;
                    ensure!(
                        got[want.len()..].iter().all(|&b| b == SENTINEL),
                        "{ns:?} M={m} S={s} segment {i}: written past M x N"
                    );
                    let got = &got[..want.len()];
                    let rel = rel_l2(got, want);
                    ensure!(rel <= MAX_REL_L2, "{ns:?} M={m} S={s}: rel L2 {rel:.2e}");
                    ensure!(
                        s > 1 || got == &want[..],
                        "{ns:?} M={m} S=1 segment {i}: bytes differ from w4a16_tc_rows"
                    );
                }
                for r in (0..m).step_by(((m / 4).max(1)) as usize) {
                    let ar = ad.offset((r * k * 2) as usize);
                    for one in [1, 17, 33] {
                        let rows = one.min(64 - r).max(1);
                        seg(&gpu, &ws, ns, old, ar, rows, k, s)?;
                        for i in 0..ns.len() {
                            let alone = read(&gpu, old[i], (ns[i] * 2) as usize)?;
                            let all = read(&gpu, new[i], (m * ns[i] * 2) as usize)?;
                            let at = (r * ns[i] * 2) as usize;
                            ensure!(
                                alone == all[at..at + alone.len()],
                                "{ns:?} M={m} S={s} row {r}: differs from the row as row 0 of {rows}"
                            );
                        }
                    }
                }
            }
        }
        // 2026-10-09: Altered weights must change the output (the comparisons can fail).
        let flip: Vec<u8> = read(&gpu, ws[0].weight, 64)?
            .iter()
            .map(|b| b ^ 0x77)
            .collect();
        let keep = read(&gpu, ws[0].weight, 64)?;
        seg(&gpu, &ws, ns, new, ad, 4, k, rule)?;
        let base = read(&gpu, new[0], (4 * ns[0] * 2) as usize)?;
        gpu.copy_h2d(&flip, ws[0].weight)?;
        seg(&gpu, &ws, ns, new, ad, 4, k, rule)?;
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
        for m in [1u32, 4, 8, 16] {
            let us_old = median_us(
                |i| {
                    let w = &rot[i % rot.len()];
                    (0..ns.len()).try_for_each(|j| {
                        ops::w4a16_tc_rows(&gpu, ad, &w[j], old[j], m, ns[j], k, k, ns[j], 0)
                    })
                },
                &gpu,
            )?;
            let us_new = median_us(
                |i| seg(&gpu, &rot[i % rot.len()], ns, new, ad, m, k, rule),
                &gpu,
            )?;
            println!(
                "{ns:?} K={k} M={m:2}: w4a16_tc_rows x{} {us_old:7.1}us  seg k{rule} {us_new:7.1}us  PASS",
                ns.len()
            );
        }
        for w in rot.iter().flatten().chain(&ws) {
            gpu.free(w.weight)?;
            gpu.free(w.weight_scale)?;
        }
    }
    println!(
        "ALL PASS: split 1 is w4a16_tc_rows byte for byte; every split row-invariant and within {MAX_REL_L2:.0e}"
    );
    Ok(())
}
