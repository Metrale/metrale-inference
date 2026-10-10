// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The one-row W4A16 dense projections of GLM-5.3 (rank 0 of TP=3): today's
//! `w4a16_gemv` (one launch per projection) against `w4a16_gemv_seg_r<R>` (one persistent launch
//! per group of projections that share the input row).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless every output byte of `w4a16_gemv_seg` equals `w4a16_gemv`'s, for
//!   every shape, group, tile height R and block cap measured (the cap 4 makes warps stride over
//!   several tiles).
//! - Negative controls, both must fire: one changed scale byte in the fifth segment of a
//!   six-segment launch changes that segment's output; the row-tile tensor-core kernel at one
//!   row (another FP32 summation order) differs from `w4a16_gemv` in some byte.
//!
//! Times: a CUDA graph of launches over a 128 MB weight pool (each launch reads the next slice,
//! so weights stream from DRAM as in a decode step), replayed (`common/graph_timing.rs`); the
//! arms alternate over 5 rounds and each prints the median of its round medians [min..max].
//! GB/s counts the packed codes and E4M3 scales. `--sweep` adds tile heights and block caps.
//!
//! Run (GB10, < 1 GB of device memory):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example glm5next_w4a16_gemv_seg_bench

#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::{self, W4a16Seg};
use metrale_model_layers::weight_map::QuantizedWeight;

/// 2026-10-10: `(name, [n of each member], k, launches per decode step)`: the rank-0 W4A16
/// projections (34 KDA layers, 42 shared experts, 11 DSA layers under METRALE_GLM_DSA_W4A16=1).
const GROUPS: [(&str, &[u32], u32, u32); 13] = [
    (
        "kda q|k|v|f_a|b|g_a",
        &[2816, 2816, 2816, 128, 22, 128],
        4096,
        34,
    ),
    ("kda q|k|v", &[2816, 2816, 2816], 4096, 0),
    ("kda f_a|b|g_a", &[128, 22, 128], 4096, 0),
    ("kda q", &[2816], 4096, 0),
    ("kda f_a", &[128], 4096, 0),
    ("kda b", &[22], 4096, 0),
    ("kda o", &[4096], 2816, 34),
    ("shared gate|up", &[768, 768], 4096, 42),
    ("shared gate", &[768], 4096, 0),
    ("shared down", &[4096], 768, 42),
    ("dsa q_a", &[1536], 4096, 11),
    ("dsa q_absorb", &[11264], 1536, 11),
    ("dsa o_absorb", &[4096], 11264, 11),
];

const POOL: usize = 128 << 20;
const STREAM_BYTES: usize = 96 << 20;
const ROUNDS: usize = 5;

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

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn read(g: &dyn GpuBackend, s: u64, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(s)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-10: Random packed codes and E4M3 scale bytes (normal range, both signs of every code).
struct Pool {
    packed: DevicePtr,
    scale: DevicePtr,
}

impl Pool {
    fn new(g: &dyn GpuBackend, rng: &mut Rng) -> Result<Self> {
        let packed: Vec<u8> = (0..POOL).map(|_| rng.next() as u8).collect();
        let scale: Vec<u8> = (0..POOL / 8)
            .map(|_| (0x08 + rng.next() % 0x70) as u8)
            .collect();
        Ok(Self {
            packed: upload(g, &packed)?,
            scale: upload(g, &scale)?,
        })
    }

    /// 2026-10-10: Weight `[n, k]` number `i` in the rotation: consecutive slices of the pool.
    fn weight(&self, i: usize, n: u32, k: u32) -> QuantizedWeight {
        let pb = ((n as usize * k as usize / 2).div_ceil(256)) * 256;
        let slots = POOL / pb;
        let slot = i % slots;
        QuantizedWeight {
            weight: self.packed.offset(slot * pb),
            weight_scale: self.scale.offset(slot * pb / 8),
            weight_scale_2: 0.0123,
            ..QuantizedWeight::null()
        }
    }
}

fn nvfp4_bytes(ns: &[u32], k: u32) -> f64 {
    ns.iter()
        .map(|&n| f64::from(n) * f64::from(k) * (0.5 + 1.0 / 16.0))
        .sum()
}

fn old_launch(
    g: &dyn GpuBackend,
    gemv: KernelHandle,
    a: DevicePtr,
    ws: &[QuantizedWeight],
    outs: &[DevicePtr],
    ns: &[u32],
    k: u32,
    s: u64,
) -> Result<()> {
    for ((w, &c), &n) in ws.iter().zip(outs).zip(ns) {
        ops::w4a16_gemv(g, gemv, a, w, c, n, k, s)?;
    }
    Ok(())
}

fn segs(ws: &[QuantizedWeight], outs: &[DevicePtr], ns: &[u32]) -> Vec<W4a16Seg> {
    ws.iter()
        .zip(outs)
        .zip(ns)
        .map(|((&weight, &output), &n)| W4a16Seg { weight, output, n })
        .collect()
}

fn main() -> Result<()> {
    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let s = g.create_stream()?;
    let sms = g.sm_count()?;
    let gemv = g.kernel("w4a16_gemv", "w4a16_gemv")?;
    let mut rng = Rng(0x7365_6731_6765_6d76);
    let pool = Pool::new(&g, &mut rng)?;
    let kmax = 11264usize;
    let a: Vec<u8> = (0..kmax)
        .flat_map(|_| {
            let u = rng.next();
            let bits = ((u & 1) << 15) | ((115 + (u >> 1) % 17) << 7) | ((u >> 8) & 0x7f);
            (bits as u16).to_le_bytes()
        })
        .collect();
    let ad = upload(&g, &a)?;
    let outs: Vec<DevicePtr> = (0..6).map(|_| g.alloc(11264 * 2)).collect::<Result<_>>()?;
    let refs: Vec<DevicePtr> = (0..6).map(|_| g.alloc(11264 * 2)).collect::<Result<_>>()?;
    let caps = [4, sms * 2, sms * 3, sms * 4, sms * 6, u32::MAX];

    // 2026-10-10: Byte gate over every group, R and cap.
    for (name, ns, k, _) in GROUPS {
        let ws: Vec<_> = (0..ns.len()).map(|i| pool.weight(i, ns[i], k)).collect();
        old_launch(&g, gemv, ad, &ws, &refs, ns, k, s)?;
        let want: Vec<Vec<u8>> = (0..ns.len())
            .map(|i| read(&g, s, refs[i], ns[i] as usize * 2))
            .collect::<Result<_>>()?;
        for r in ops::W4A16_GEMV_SEG_TILES {
            for cap in caps {
                for &o in &outs {
                    g.memset(o, 0xA5, 11264 * 2)?;
                }
                ops::w4a16_gemv_seg(&g, ad, &segs(&ws, &outs, ns), k, r, cap, s)?;
                for (i, w) in want.iter().enumerate() {
                    let got = read(&g, s, outs[i], ns[i] as usize * 2)?;
                    ensure!(
                        &got == w,
                        "{name} segment {i} R={r} cap={cap}: w4a16_gemv_seg differs from w4a16_gemv"
                    );
                }
            }
        }
    }
    println!("byte gate: every segment of every group equals w4a16_gemv (R 1/2/4, 6 caps)");

    // 2026-10-10: Negative control 1: a changed scale byte in segment 4 of the six.
    let (_, ns, k, _) = GROUPS[0];
    let ws: Vec<_> = (0..6).map(|i| pool.weight(i, ns[i], k)).collect();
    ops::w4a16_gemv_seg(&g, ad, &segs(&ws, &outs, ns), k, 2, u32::MAX, s)?;
    let before = read(&g, s, outs[4], ns[4] as usize * 2)?;
    // 2026-10-10: Output 7's scale of chunk 12 set to E4M3 448 (the pool's scales are below
    // 0x78), so the change cannot round away.
    let mut b0 = [0u8; 1];
    let at = ws[4].weight_scale.offset(4096 / 16 * 7 + 12);
    g.copy_d2h(at, &mut b0)?;
    g.copy_h2d(&[0x7E], at)?;
    ops::w4a16_gemv_seg(&g, ad, &segs(&ws, &outs, ns), k, 2, u32::MAX, s)?;
    let after = read(&g, s, outs[4], ns[4] as usize * 2)?;
    g.copy_h2d(&b0, at)?;
    ensure!(
        before != after,
        "negative control: a changed scale byte left segment 4 equal"
    );
    // 2026-10-10: Negative control 2: another FP32 order at one row must differ somewhere.
    let w = pool.weight(0, 2816, 4096);
    ops::w4a16_gemv(&g, gemv, ad, &w, refs[0], 2816, 4096, s)?;
    ops::w4a16_tc_rows(&g, ad, &w, outs[0], 1, 2816, 4096, 4096, 2816, s)?;
    let x = read(&g, s, refs[0], 2816 * 2)?;
    let y = read(&g, s, outs[0], 2816 * 2)?;
    let diff = x.chunks(2).zip(y.chunks(2)).filter(|(p, q)| p != q).count();
    ensure!(
        diff > 0,
        "negative control: w4a16_tc_rows at one row equals w4a16_gemv"
    );
    println!(
        "negative controls fire: scale byte; w4a16_tc_rows (1 row) differs in {diff}/2816 outputs"
    );

    // 2026-10-10: Timing: the arms of a group alternate over ROUNDS rounds (so drift and other
    // GPU work spread over every arm); per arm, the median of the round medians and its range.
    // Arm 0 is today's one launch per member; arm 1 the policy (`w4a16_gemv_seg_policy`); with
    // `--sweep`, every tile height at the policy cap and at caps of 2 and 4 blocks per SM.
    let sweep = std::env::args().any(|a| a == "--sweep");
    let mut arms: Vec<Option<(u32, u32)>> = vec![None, Some((0, 0))];
    if sweep {
        for r in ops::W4A16_GEMV_SEG_TILES {
            for cap in [sms * 2, sms * 4, u32::MAX] {
                arms.push(Some((r, cap)));
            }
        }
    }
    let mut tot = [0.0f64; 2];
    for (name, ns, k, count) in GROUPS {
        let bytes = nvfp4_bytes(ns, k);
        let per = ns
            .iter()
            .map(|&n| n as usize * k as usize / 2)
            .sum::<usize>();
        let copies = STREAM_BYTES.div_ceil(per).clamp(4, 2048);
        let (pr, pcap) = ops::w4a16_gemv_seg_policy(k, sms);
        let mut med: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for _ in 0..ROUNDS {
            for (ai, arm) in arms.iter().enumerate() {
                let mut i = 0usize;
                let t = timing::time_graph(&g, s, copies, &mut |st| {
                    let ws: Vec<_> = (0..ns.len())
                        .map(|m| pool.weight(i * 6 + m, ns[m], k))
                        .collect();
                    i += 1;
                    match *arm {
                        None => old_launch(&g, gemv, ad, &ws, &refs, ns, k, st),
                        Some((r, cap)) => {
                            let (r, cap) = if r == 0 { (pr, pcap) } else { (r, cap) };
                            ops::w4a16_gemv_seg(&g, ad, &segs(&ws, &outs, ns), k, r, cap, st)
                        }
                    }
                })?;
                med[ai].push(t.0);
            }
        }
        let stat = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2], v[0], v[v.len() - 1])
        };
        let old = stat(&mut med[0]);
        println!(
            "{name:<20} K={k:<5} old    {:7.2} us [{:.2}..{:.2}] {:4.0} GB/s",
            old.0,
            old.1,
            old.2,
            bytes / old.0 / 1e3
        );
        for (ai, arm) in arms.iter().enumerate().skip(1) {
            let t = stat(&mut med[ai]);
            let (r, cap) = match arm {
                Some((0, _)) => (pr, pcap),
                Some(x) => *x,
                None => unreachable!(),
            };
            let what = if ai == 1 { "policy" } else { "sweep " };
            let cap_s = if cap == u32::MAX {
                "all".to_string()
            } else {
                format!("{}xSM", cap / sms)
            };
            println!(
                "{:<26} {what} {:7.2} us [{:.2}..{:.2}] {:4.0} GB/s  R={r} cap={cap_s:<5} x{:.3}",
                "",
                t.0,
                t.1,
                t.2,
                bytes / t.0 / 1e3,
                old.0 / t.0
            );
            if ai == 1 {
                tot[0] += old.0 * f64::from(count);
                tot[1] += t.0 * f64::from(count);
            }
        }
    }
    // 2026-10-10: Old: every member its own `w4a16_gemv`; new: each counted group one launch.
    println!(
        "per decode step (rank 0, counted groups): old {:.3} ms -> policy {:.3} ms",
        tot[0] / 1e3,
        tot[1] / 1e3
    );
    Ok(())
}
