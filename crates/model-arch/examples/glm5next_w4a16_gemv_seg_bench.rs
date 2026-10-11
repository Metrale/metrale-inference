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

#[path = "common/w4a16_seg_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Result, ensure};
use fixture::{
    GROUPS, Pool, ROUNDS, Rng, STREAM_BYTES, bf16_rows, nvfp4_bytes, read, segs, stat, upload,
};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::{self, W4a16GemvSegPoint};
use metrale_model_layers::weight_map::QuantizedWeight;

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

fn main() -> Result<()> {
    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let s = g.create_stream()?;
    let sms = g.sm_count()?;
    let gemv = g.kernel("w4a16_gemv", "w4a16_gemv")?;
    let mut rng = Rng(0x7365_6731_6765_6d76);
    let pool = Pool::new(&g, &mut rng)?;
    let ad = upload(&g, &bf16_rows(&mut rng, 1, 11264))?;
    let outs: Vec<DevicePtr> = (0..6).map(|_| g.alloc(11264 * 2)).collect::<Result<_>>()?;
    let refs: Vec<DevicePtr> = (0..6).map(|_| g.alloc(11264 * 2)).collect::<Result<_>>()?;
    let caps = [4, sms * 2, sms * 3, sms * 4, sms * 6, u32::MAX];

    // 2026-10-10: Byte gate over every group, R and cap.
    let points: Vec<W4a16GemvSegPoint> = ops::W4A16_GEMV_SEG_TILES
        .into_iter()
        .map(W4a16GemvSegPoint::Tile)
        .collect();
    for (name, ns, k, _) in GROUPS {
        let ws: Vec<_> = (0..ns.len()).map(|i| pool.weight(i, ns[i], k)).collect();
        old_launch(&g, gemv, ad, &ws, &refs, ns, k, s)?;
        let want: Vec<Vec<u8>> = (0..ns.len())
            .map(|i| read(&g, s, refs[i], ns[i] as usize * 2))
            .collect::<Result<_>>()?;
        for &pt in &points {
            for cap in caps {
                for &o in &outs {
                    g.memset(o, 0xA5, 11264 * 2)?;
                }
                ops::w4a16_gemv_seg_point(&g, pt, ad, &segs(&ws, &outs, ns), k, cap, s)?;
                for (i, w) in want.iter().enumerate() {
                    let got = read(&g, s, outs[i], ns[i] as usize * 2)?;
                    ensure!(
                        &got == w,
                        "{name} segment {i} {pt:?} cap={cap}: w4a16_gemv_seg differs from w4a16_gemv"
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
    // Arms: today's one launch per member and the policy (`w4a16_gemv_seg_policy`); with
    // `--sweep`, every tile height at caps of 2 and 4 blocks per SM and uncapped.
    let sweep = std::env::args().any(|a| a == "--sweep");
    let mut arms: Vec<Arm> = vec![Arm::Old, Arm::Policy];
    if sweep {
        for r in ops::W4A16_GEMV_SEG_TILES {
            for cap in [sms * 2, sms * 4, u32::MAX] {
                arms.push(Arm::At(W4a16GemvSegPoint::Tile(r), cap));
            }
        }
    }
    let mut tot = vec![0.0f64; arms.len()];
    for (name, ns, k, count) in GROUPS {
        let bytes = nvfp4_bytes(ns, k);
        let per = ns
            .iter()
            .map(|&n| n as usize * k as usize / 2)
            .sum::<usize>();
        let copies = STREAM_BYTES.div_ceil(per).clamp(4, 2048);
        let (pr, pcap) = ops::w4a16_gemv_seg_policy(k, sms);
        let resolve = |arm: Arm| match arm {
            Arm::Policy => Some((W4a16GemvSegPoint::Tile(pr), pcap)),
            Arm::At(p, c) => Some((p, c)),
            Arm::Old => None,
        };
        let mut med: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for _ in 0..ROUNDS {
            for (ai, &arm) in arms.iter().enumerate() {
                let mut i = 0usize;
                let t = timing::time_graph(&g, s, copies, &mut |st| {
                    let ws: Vec<_> = (0..ns.len())
                        .map(|m| pool.weight(i * 6 + m, ns[m], k))
                        .collect();
                    i += 1;
                    match resolve(arm) {
                        None => old_launch(&g, gemv, ad, &ws, &refs, ns, k, st),
                        Some((pt, cap)) => {
                            ops::w4a16_gemv_seg_point(&g, pt, ad, &segs(&ws, &outs, ns), k, cap, st)
                        }
                    }
                })?;
                med[ai].push(t.0);
            }
        }
        let old = stat(&mut med[0]);
        println!(
            "{name:<20} K={k:<5} old    {:7.2} us [{:.2}..{:.2}] {:4.0} GB/s",
            old.0,
            old.1,
            old.2,
            bytes / old.0 / 1e3
        );
        tot[0] += old.0 * f64::from(count);
        for (ai, &arm) in arms.iter().enumerate().skip(1) {
            let t = stat(&mut med[ai]);
            let (pt, cap) = resolve(arm).expect("a kernel arm");
            let what = match arm {
                Arm::Policy => "policy",
                _ => "sweep ",
            };
            let cap_s = if cap == u32::MAX {
                "all".to_string()
            } else {
                format!("{}xSM", cap / sms)
            };
            println!(
                "{:<26} {what} {:7.2} us [{:.2}..{:.2}] {:4.0} GB/s  {pt:?} cap={cap_s:<5} x{:.3}",
                "",
                t.0,
                t.1,
                t.2,
                bytes / t.0 / 1e3,
                old.0 / t.0
            );
            tot[ai] += t.0 * f64::from(count);
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

/// 2026-10-11: A timing arm: today's launches, the policy, or a point at a block cap.
#[derive(Clone, Copy)]
enum Arm {
    Old,
    Policy,
    At(W4a16GemvSegPoint, u32),
}
