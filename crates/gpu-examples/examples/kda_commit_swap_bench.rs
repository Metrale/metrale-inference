// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: What the SSM commit swap (`METRALE_SSM_COMMIT_SWAP`) removes from a rejected
//! step at the GLM-5.3-Flash serving shape: the reject commit's copies of intermediate 0 over
//! the live state for all 34 KDA layers, laid out as the SSM pool lays them out (one contiguous
//! block per family, layer stride = the family's per-layer size), against the swap's host work
//! (one role bit and 34 pointer pairs).
//!
//! Timed: (a) the host commit, `copy_d2d_2d_async` h then conv (`run_ssm_state_copies`);
//! (b) the device-step commit, `predicated_copy_2d` h then conv with a matching verdict word
//! (`commit_k2_on_device`); (c) the swap, host only.
//!
//!   cargo run -p metrale-gpu-examples --release --example kda_commit_swap_bench \
//!       --features cuda,gpu-examples
//!
//! Owner: gpu-examples (GLM-5.3 KDA layer).
//! Invariants: none; a measurement.

use std::time::Instant;

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_kda::Glm5NextKdaConfig;
use metrale_model_layers::layer::SsmLayerState;
use metrale_model_layers::layers::ops::predicated_copy_2d;
use metrale_model_layers::ssm_reserve::SlotSwaps;

/// 2026-10-21: GLM-5.3-Flash's KDA layers.
const LAYERS: usize = 34;
/// 2026-10-21: Pool slots per family block (the serving `--max-batch-size 16` + dummy).
const SLOTS: usize = 17;
const REPS: usize = 200;

fn cfg(heads: usize) -> Glm5NextKdaConfig {
    Glm5NextKdaConfig {
        hidden: 4096,
        heads,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: -5.0,
        rms_norm_eps: 1e-5,
        l2_eps: 1e-6,
        chunk: 32,
    }
}

/// 2026-10-21: Mean microseconds of `f` over `REPS` launches on `s`, after 10 warm-ups.
fn time_us(g: &dyn GpuBackend, s: u64, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..10 {
        f()?;
    }
    g.synchronize(s)?;
    let t = Instant::now();
    for _ in 0..REPS {
        f()?;
    }
    g.synchronize(s)?;
    Ok(t.elapsed().as_secs_f64() * 1e6 / REPS as f64)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let s = g.default_stream();
    let k_copy = g.kernel("mtp_device_step", "predicated_copy_2d")?;
    let flag = g.alloc(4)?;
    g.copy_h2d(&0u32.to_le_bytes(), flag)?;
    println!(
        "| heads | h MB/layer | conv KB/layer | MB moved | 2-D copy us | GB/s | predicated us | GB/s | swap host us |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for heads in [22usize, 21] {
        let c = cfg(heads);
        let (hb, cb) = (c.recurrent_state_elems() * 4, c.conv_state_elems() * 4);
        // 2026-10-21: Live pools (SLOTS blobs per layer) and intermediate pools (one per slot),
        // slot 0 of each.
        let (h_live, c_live) = (g.alloc(LAYERS * SLOTS * hb)?, g.alloc(LAYERS * SLOTS * cb)?);
        let (h_int, c_int) = (g.alloc(LAYERS * SLOTS * hb)?, g.alloc(LAYERS * SLOTS * cb)?);
        for (p, b) in [(h_live, hb), (c_live, cb), (h_int, hb), (c_int, cb)] {
            g.memset(p, 0x3c, LAYERS * SLOTS * b)?;
        }
        let (hp, cp) = (SLOTS * hb, SLOTS * cb);
        let mb = (LAYERS * (hb + cb)) as f64 / 1e6;
        let copy2d = time_us(g, s, || {
            g.copy_d2d_2d_async(h_int, hp, h_live, hp, hb, LAYERS, s)?;
            g.copy_d2d_2d_async(c_int, cp, c_live, cp, cb, LAYERS, s)
        })?;
        let pred = time_us(g, s, || {
            predicated_copy_2d(g, k_copy, flag, 0, h_int, hp, h_live, hp, hb, LAYERS, s)?;
            predicated_copy_2d(g, k_copy, flag, 0, c_int, cp, c_live, cp, cb, LAYERS, s)
        })?;
        // 2026-10-21: The swap: one role bit and every layer's pointer pair.
        let swaps = SlotSwaps::new(SLOTS);
        let mut states: Vec<SsmLayerState> = (0..LAYERS)
            .map(|l| SsmLayerState {
                h_state: h_live.offset(l * hp),
                conv_state: c_live.offset(l * cp),
                h_state_checkpoint: None,
                conv_state_checkpoint: None,
                h_state_intermediates: vec![h_int.offset(l * hp)],
                conv_state_intermediates: vec![c_int.offset(l * cp)],
                replay_ring: None,
                h_is_f16: false,
                h_prefill_stage: None,
                ple: None,
            })
            .collect();
        let n = 100_000usize;
        let t = Instant::now();
        for _ in 0..n {
            swaps.toggle(0);
            for st in states.iter_mut() {
                st.swap_live_with_inter0()?;
            }
        }
        let swap_us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        std::hint::black_box(&states);
        println!(
            "| {heads} | {:.3} | {:.1} | {mb:.1} | {copy2d:.1} | {:.0} | {pred:.1} | {:.0} | {swap_us:.3} |",
            hb as f64 / 1e6,
            cb as f64 / 1e3,
            2.0 * mb * 1e3 / copy2d,
            2.0 * mb * 1e3 / pred,
        );
        for p in [h_live, c_live, h_int, c_int] {
            g.free(p)?;
        }
    }
    println!("GB/s counts read + write bytes.");
    Ok(())
}
