// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Exhaustive bit-identity gate for the one-shot all-reduce at two ranks: over every
//! ordered pair of BF16 partials `(p0, p1)`, `bf16_add_rank_sum` launched as
//! `NcclBackend::all_reduce_oneshot` launches it (two ranks, each rank's own view) against
//! `bf16_add_inplace` launched as `NcclBackend::all_reduce_2rank` does (`own += partner`). No
//! communicator: each rank's peer slot is a device buffer holding the other partial.
//!
//! Owner: model-arch examples (comm kernels).
//! Invariants:
//! - The run fails unless, on both ranks, every pair with finite inputs gives the same bytes
//!   except `(-0, -0)`, which must differ in exactly the sign bit (+0 one-shot, -0 2-rank: the
//!   FP32 accumulator starts at +0), and every pair with a NaN input gives the same NaN bits on
//!   both arms (2026-10-16: was any NaN; a payload or sign difference now fails).
//! - The `(-0, -0)` pair is the positive control that the comparison sees a one-bit difference;
//!   a second control, the one-shot arm reading its peer one element off, must differ on most
//!   pairs.
//!
//!   cargo run -p metrale-gpu-examples --release --example bf16_world2_sum_gate \
//!       --features cuda,gpu-examples
//!
//! Device memory: six buffers of `CHUNK` BF16 elements (6 x 64 MiB).

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-16: First partials per chunk; each chunk pairs them with all 65,536 second partials.
const P0_PER_CHUNK: usize = 512;
const CHUNK: usize = P0_PER_CHUNK << 16;
const NEG_ZERO: u16 = 0x8000;

fn is_nan(h: u16) -> bool {
    (h & 0x7f80) == 0x7f80 && (h & 0x7f) != 0
}

fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn halves(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// 2026-10-16: `NcclBackend::all_reduce_2rank`'s add: `own[i] = __hadd(own[i], partner[i])`.
fn hadd(g: &dyn GpuBackend, k: KernelHandle, own: DevicePtr, partner: DevicePtr) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([(CHUNK as u32).div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(own)
        .arg_ptr(partner)
        .arg_i32(CHUNK as i32)
        .launch(0)
}

/// 2026-10-16: `NcclBackend::all_reduce_oneshot`'s sum at two ranks: one peer slot.
fn rank_sum(
    g: &dyn GpuBackend,
    k: KernelHandle,
    own: DevicePtr,
    peer: DevicePtr,
    rank: i32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([(CHUNK as u32).div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(own)
        .arg_ptr(peer)
        .arg_u64(CHUNK as u64)
        .arg_i32(2)
        .arg_i32(rank)
        .arg_i32(CHUNK as i32)
        .launch(0)
}

/// 2026-10-16: Per-rank tallies over the whole sweep.
#[derive(Default)]
struct Tally {
    pairs: usize,
    neg_zero_seen: usize,
    nan_pairs: usize,
    bad: usize,
}

fn compare(t: &mut Tally, rank: usize, p0: &[u16], p1: &[u16], two: &[u16], one: &[u16]) {
    for i in 0..p0.len() {
        t.pairs += 1;
        let (a, b) = (p0[i], p1[i]);
        if is_nan(a) || is_nan(b) {
            t.nan_pairs += 1;
            if !(is_nan(two[i]) && two[i] == one[i]) {
                if t.bad < 8 {
                    println!(
                        "  rank {rank}: NaN pair {a:#06x} + {b:#06x}: 2-rank {:#06x} one-shot {:#06x}",
                        two[i], one[i]
                    );
                }
                t.bad += 1;
            }
            continue;
        }
        if a == NEG_ZERO && b == NEG_ZERO {
            t.neg_zero_seen += 1;
            if !(two[i] == NEG_ZERO && one[i] == 0) {
                t.bad += 1;
                println!(
                    "  rank {rank}: (-0)+(-0) 2-rank {:#06x} one-shot {:#06x}",
                    two[i], one[i]
                );
            }
            continue;
        }
        if two[i] != one[i] {
            if t.bad < 8 {
                println!(
                    "  rank {rank}: {a:#06x} + {b:#06x}: 2-rank {:#06x} one-shot {:#06x}",
                    two[i], one[i]
                );
            }
            t.bad += 1;
        }
    }
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let add = g.kernel("bf16_add", "bf16_add_inplace")?;
    let sum = g.kernel("bf16_add", "bf16_add_rank_sum")?;
    println!("bf16 world-2 sum gate: bf16_add_inplace (the 2-rank exchange) is the ORACLE");

    let p1: Vec<u16> = (0..P0_PER_CHUNK).flat_map(|_| 0..=u16::MAX).collect();
    let p1_bytes = bytes(&p1);
    let len = CHUNK * 2;
    let [d0, two0, one0, two1, one1] = [(); 5].map(|_| g.alloc(len));
    let (d0, two0, one0, two1, one1) = (d0?, two0?, one0?, two1?, one1?);
    // 2026-10-16: One spare element, so the control (peer one element off) stays in bounds.
    let d1 = g.alloc(len + 2)?;
    g.memset_async(d1, 0, len + 2, 0)?;
    g.copy_h2d(&p1_bytes, d1)?;

    let mut tally = [Tally::default(), Tally::default()];
    let mut control_diffs = 0usize;
    for chunk in 0..(1usize << 16) / P0_PER_CHUNK {
        let p0: Vec<u16> = (0..CHUNK)
            .map(|i| (chunk * P0_PER_CHUNK + (i >> 16)) as u16)
            .collect();
        let p0_bytes = bytes(&p0);
        g.copy_h2d(&p0_bytes, d0)?;
        // 2026-10-16: Rank 0 owns p0 and receives p1; rank 1 owns p1 and receives p0.
        for (dst, src) in [
            (two0, &p0_bytes),
            (one0, &p0_bytes),
            (two1, &p1_bytes),
            (one1, &p1_bytes),
        ] {
            g.copy_h2d(src, dst)?;
        }
        hadd(g, add, two0, d1)?;
        rank_sum(g, sum, one0, d1, 0)?;
        hadd(g, add, two1, d0)?;
        rank_sum(g, sum, one1, d0, 1)?;
        g.synchronize(0)?;
        let mut out = vec![0u8; len];
        let mut read = |p: DevicePtr| -> Result<Vec<u16>> {
            g.copy_d2h(p, &mut out)?;
            Ok(halves(&out))
        };
        let (t0, o0, t1, o1) = (read(two0)?, read(one0)?, read(two1)?, read(one1)?);
        compare(&mut tally[0], 0, &p0, &p1, &t0, &o0);
        compare(&mut tally[1], 1, &p0, &p1, &t1, &o1);
        if chunk == 64 {
            // 2026-10-16: Control: the peer read one element off must move most results.
            g.copy_h2d(&p0_bytes, one0)?;
            rank_sum(g, sum, one0, d1.offset(2), 0)?;
            g.synchronize(0)?;
            let off = read(one0)?;
            control_diffs = (0..CHUNK - 1).filter(|&i| off[i] != t0[i]).count();
        }
    }
    for p in [d0, d1, two0, one0, two1, one1] {
        g.free(p)?;
    }

    let mut failed = false;
    for (rank, t) in tally.iter().enumerate() {
        println!(
            "  rank {rank}: {} pairs, {} with a NaN input, (-0)+(-0) seen {}x, mismatches {}",
            t.pairs, t.nan_pairs, t.neg_zero_seen, t.bad
        );
        failed |= t.pairs != 1usize << 32 || t.neg_zero_seen != 1 || t.bad != 0;
    }
    let control_ok = control_diffs > CHUNK / 2;
    println!(
        "  control (peer one element off): {control_diffs} of {} differ {}",
        CHUNK - 1,
        if control_ok { "(ok)" } else { "(BAD)" }
    );
    if failed || !control_ok {
        bail!("bf16 world-2 sum gate FAILED");
    }
    println!(
        "\nPASS: every finite pair byte-identical on both ranks except (-0)+(-0) (+0 vs -0); a NaN \
         input gives the same NaN bits on both arms; the control differs"
    );
    Ok(())
}
