// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `w4a16_gemv_pipe` (D-deep pipelined W4A16 GEMV) against today's dispatch of the
//! GLM-5.3 rank-0 W4A16 projections at 1 and 2 rows: a group or a lone projection below K 2048 on
//! `w4a16_gemv_seg` (`_r1` / `_m2`), a lone projection from K 2048 up on `w4a16_gemv` (1 row) or
//! `w4a16_gemv_batch2` (2 rows).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless the policy and every gated cap write, for every row m of every
//!   segment, the bytes `w4a16_gemv` writes on input row m alone, on random and adversarial
//!   inputs (every input element +-0, a signed zero sweep, the largest BF16 magnitudes).
//! - `--sweep` also times the block caps of `gate::sweep_arms`; otherwise only today's dispatch
//!   and the policy (`ops::w4a16_gemv_pipe`).
//!
//! Times: as `glm5next_w4a16_gemv_seg_bench` (CUDA graph over a rotated 128 MB pool, arms
//! alternating over 5 rounds, median of round medians [min..max]).
//!
//! Run (GB10): cargo build --release -p metrale-gpu-examples --features cuda,gpu-examples
//!   --example glm5next_w4a16_gemv_pipe_bench, then run the binary.

#[path = "common/w4a16_seg_fixture.rs"]
mod fixture;
#[path = "common/w4a16_pipe_gate.rs"]
mod gate;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::Result;
use fixture::{
    GROUPS, Pool, ROUNDS, Rng, STREAM_BYTES, bf16_rows, nvfp4_bytes, segs, stat, upload,
};
use gate::{Arm, old_launch};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;

fn main() -> Result<()> {
    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let s = g.create_stream()?;
    let sms = g.sm_count()?;
    let sweep = std::env::args().any(|a| a == "--sweep");
    let kernels = gate::Kernels::load(&g)?;
    let mut rng = Rng(0x7069_7065_6265_6e63);
    let pool = Pool::new(&g, &mut rng)?;
    let outs: Vec<DevicePtr> = (0..6)
        .map(|_| g.alloc(4 * 11264 * 2))
        .collect::<Result<_>>()?;

    gate::byte_gate(&g, s, &kernels, &pool, &outs, &mut rng)?;
    gate::negative_controls(&g, s, &kernels, &pool, &outs, &mut rng)?;

    let a_host = bf16_rows(&mut rng, 2, 11264);
    let ad = upload(&g, &a_host)?;
    for m in [1u32, 2] {
        let mut tot = Vec::new();
        let mut tot_glm = [0.0f64; 2];
        for (name, ns, k, count) in GROUPS {
            if count == 0 {
                continue;
            }
            let mut arms = vec![Arm::Old, Arm::New];
            if sweep {
                arms.push(Arm::Policy);
                arms.extend(gate::sweep_arms(sms));
            }
            tot.resize(arms.len(), 0.0f64);
            let bytes = nvfp4_bytes(ns, k);
            let per = ns
                .iter()
                .map(|&n| n as usize * k as usize / 2)
                .sum::<usize>();
            let copies = STREAM_BYTES.div_ceil(per).clamp(4, 2048);
            let mut med: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
            for _ in 0..ROUNDS {
                for (ai, &arm) in arms.iter().enumerate() {
                    let mut i = 0usize;
                    let t = timing::time_graph(&g, s, copies, &mut |st| {
                        let ws: Vec<_> = (0..ns.len())
                            .map(|j| pool.weight(i * 7 + j, ns[j], k))
                            .collect();
                        i += 1;
                        let sg = segs(&ws, &outs, ns);
                        match arm {
                            Arm::Old => old_launch(&g, &kernels, ad, &sg, m, k, st),
                            Arm::New => gate::new_launch(&g, &kernels, ad, &sg, m, k, st),
                            Arm::Policy => ops::w4a16_gemv_pipe(&g, ad, &sg, m, k, sms, st),
                            Arm::At(cap) => ops::w4a16_gemv_pipe_at(&g, ad, &sg, m, k, cap, st),
                        }
                    })?;
                    med[ai].push(t.0);
                }
            }
            let old = stat(&mut med[0]);
            if !name.starts_with("dsa") {
                tot_glm[0] += old.0 * f64::from(count);
                tot_glm[1] += stat(&mut med[1].clone()).0 * f64::from(count);
            }
            println!(
                "M={m} {name:<20} K={k:<5} old    {:7.2} us [{:.2}..{:.2}] {:4.0} GB/s",
                old.0,
                old.1,
                old.2,
                bytes / old.0 / 1e3
            );
            for (ai, &arm) in arms.iter().enumerate() {
                let t = stat(&mut med[ai]);
                tot[ai] += t.0 * f64::from(count);
                if ai == 0 {
                    continue;
                }
                println!(
                    "{:<34} {:<26} {:7.2} us [{:.2}..{:.2}] {:4.0} GB/s  x{:.3}",
                    "",
                    gate::arm_name(arm, sms),
                    t.0,
                    t.1,
                    t.2,
                    bytes / t.0 / 1e3,
                    old.0 / t.0
                );
            }
        }
        println!(
            "M={m} per step (rank 0, KDA + shared expert): old {:.3} ms -> new {:.3} ms (saves {:.3}); \
             with the DSA W4A16 groups: {:.3} -> {:.3} ms",
            tot_glm[0] / 1e3,
            tot_glm[1] / 1e3,
            (tot_glm[0] - tot_glm[1]) / 1e3,
            tot[0] / 1e3,
            tot[1] / 1e3
        );
    }
    Ok(())
}
