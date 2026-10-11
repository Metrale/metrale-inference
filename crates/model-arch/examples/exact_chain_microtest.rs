// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Kernel-level check of the single-sequence exact verify against the per-row chain
//! it replaces: `gdn_exact_chain{2,3,4}` (`gdn_exact_carry.cu` in the model directory) against K
//! launches of `gated_delta_rule_decode_f32_strided` with a copy of the state after each, and
//! `gdn_conv_chain_f32` against K launches of `causal_conv1d_update_l2norm_f32_strided` with a
//! copy of the window after each.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 on any mismatch below; each check prints its mismatch count and max ULP delta.
//!
//! One sequence, `ROUNDS` rounds per K = 2, 3, 4, each from the state the previous round left
//! (the first one's head norm above 1000, which no path rescales). Each round checks the
//! outputs, every intermediate (the state, or window, after rows 0..K-2) and the final state bit
//! for bit.
//!
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!       --example exact_chain_microtest
//!
//! Env: TARGET (default qwen3.6-35b-a3b; qwen3.8-27b for the dense), SEED (default 1), ROUNDS
//! (default 6), NK/NV (default 16/32).

use anyhow::{Context, Result};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

// 2026-10-01: Shared with gdn_carry_microtest, which uses the WY parents this check does not.
#[allow(dead_code)]
#[path = "common/gdn_carry_fixture.rs"]
mod gdn_carry_fixture;
use gdn_carry_fixture::*;

const D_CONV: usize = 4;

fn copy(g: &dyn GpuBackend, from: DevicePtr, to: DevicePtr, bytes: usize) -> Result<()> {
    g.copy_d2d_async(from, to, bytes, g.default_stream())
}

/// 2026-10-01: (mismatches, max ULP) of two device f32 buffers of `n` elements.
fn diff_dev(g: &dyn GpuBackend, a: DevicePtr, b: DevicePtr, n: usize) -> Result<(usize, u32)> {
    Ok(diff_f32(&read_f32(g, a, n)?, &read_f32(g, b, n)?))
}

fn main() -> Result<()> {
    let target = std::env::var("TARGET").unwrap_or_else(|_| "qwen3.6-35b-a3b".to_string());
    let seed = env_usize("SEED", 1) as u64;
    let rounds = env_usize("ROUNDS", 6);
    let nk = env_usize("NK", 16);
    let nv = env_usize("NV", 32);
    anyhow::ensure!(nv.is_multiple_of(nk));
    let set = metrale_kernels::ptx_for_exact_target(&target, "nvfp4")
        .with_context(|| format!("no compiled {target}/nvfp4 kernel set"))?;
    let backend = MetraleCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let s = g.default_stream();
    let gdn_parent = g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32_strided")?;
    let conv_parent = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32_strided")?;
    let conv_chain = g.kernel("gated_delta_rule_carry", "gdn_conv_chain_f32")?;
    let (key_dim, value_dim) = (nk * KD, nv * VD);
    let dim = 2 * key_dim + value_dim;
    let row = dim + value_dim;
    let h_numel = nv * KD * VD;
    let state = dim * D_CONV;
    let mut rng = Rng(seed);
    let weight = DenseWeight {
        weight: upload_bytes(
            g,
            &(0..state)
                .flat_map(|_| bf16::from_f32(rng.next_f32() * 0.5).to_bits().to_le_bytes())
                .collect::<Vec<u8>>(),
        )?,
    };
    let mut failures = 0usize;
    let mut report = |label: String, n: usize, ulp: u32, total: usize| {
        if n != 0 {
            failures += 1;
        }
        println!(
            "{label:<56} {}  mismatches={n}/{total}  max_ulp={ulp}",
            if n == 0 { "PASS" } else { "FAIL" }
        );
    };
    for kk in 2..=4usize {
        let gdn_chain = g.kernel("gdn_exact_carry", &format!("gdn_exact_chain{kk}"))?;
        // 2026-10-01: Head norm near 1500 at the start, past 1000.
        let h0: Vec<f32> = (0..h_numel).map(|_| rng.next_f32() * 20.0).collect();
        let c0: Vec<f32> = (0..state).map(|_| rng.next_f32()).collect();
        let (h_ref, h_c) = (
            upload_bytes(g, &f32_bytes(&h0))?,
            upload_bytes(g, &f32_bytes(&h0))?,
        );
        let (c_ref, c_c) = (
            upload_bytes(g, &f32_bytes(&c0))?,
            upload_bytes(g, &f32_bytes(&c0))?,
        );
        let alloc =
            |n: usize| -> Result<Vec<DevicePtr>> { (0..3).map(|_| g.alloc(n * 4)).collect() };
        let (hi_ref, hi_c, ci_ref, ci_c) = (
            alloc(h_numel)?,
            alloc(h_numel)?,
            alloc(state)?,
            alloc(state)?,
        );
        let (o_ref, o_c) = (g.alloc(kk * value_dim * 4)?, g.alloc(kk * value_dim * 4)?);
        let (r_ref, r_c) = (g.alloc(kk * row * 4)?, g.alloc(kk * row * 4)?);
        let mut bad = [(0usize, 0u32); 6];
        for _ in 0..rounds {
            let x: Vec<u8> = (0..kk * dim)
                .flat_map(|_| bf16::from_f32(rng.next_f32()).to_bits().to_le_bytes())
                .collect();
            let xd = upload_bytes(g, &x)?;
            let mut gb = vec![0f32; kk * 2 * nv];
            for r in 0..kk {
                for h in 0..nv {
                    gb[r * 2 * nv + h] = 0.5 + 0.49 * rng.next_f32();
                    gb[r * 2 * nv + nv + h] = 0.5 + 0.5 * rng.next_f32().abs();
                }
            }
            let gbd = upload_bytes(g, &f32_bytes(&gb))?;
            // 2026-10-01: The per-row chain: conv then GDN for each row, a copy after each.
            for t in 0..kk {
                ops::conv1d_update_l2norm_strided(
                    g,
                    conv_parent,
                    c_ref,
                    xd.offset(t * dim * 2),
                    &weight,
                    r_ref.offset(t * row * 4),
                    dim as u32,
                    D_CONV as u32,
                    1,
                    (2 * key_dim) as u32,
                    KD as u32,
                    1e-6,
                    dim as u32,
                    row as u32,
                    s,
                )?;
                if t + 1 < kk {
                    copy(g, c_ref, ci_ref[t], state * 4)?;
                }
                let r = r_ref.offset(t * row * 4);
                ops::gdn_decode_f32_strided(
                    g,
                    gdn_parent,
                    h_ref,
                    r,
                    r.offset(key_dim * 4),
                    r.offset(2 * key_dim * 4),
                    gbd.offset(t * 2 * nv * 4),
                    gbd.offset((t * 2 * nv + nv) * 4),
                    o_ref.offset(t * value_dim * 4),
                    1,
                    nk as u32,
                    nv as u32,
                    KD as u32,
                    VD as u32,
                    row as u32,
                    row as u32,
                    (2 * nv) as u32,
                    value_dim as u32,
                    s,
                )?;
                if t + 1 < kk {
                    copy(g, h_ref, hi_ref[t], h_numel * 4)?;
                }
            }
            ops::gdn_conv_chain_f32(
                g,
                conv_chain,
                c_c,
                xd,
                &weight,
                r_c,
                [ci_c[0], ci_c[1], ci_c[2]],
                kk as u32,
                dim as u32,
                D_CONV as u32,
                (2 * key_dim) as u32,
                KD as u32,
                1e-6,
                dim as u32,
                row as u32,
                s,
            )?;
            ops::gdn_exact_chain(
                g,
                gdn_chain,
                h_c,
                r_c,
                r_c.offset(key_dim * 4),
                r_c.offset(2 * key_dim * 4),
                gbd,
                gbd.offset(nv * 4),
                o_c,
                [hi_c[0], hi_c[1], hi_c[2]],
                nk as u32,
                nv as u32,
                KD as u32,
                [row as u32, row as u32, (2 * nv) as u32, value_dim as u32],
                s,
            )?;
            g.synchronize(s)?;
            let mut add = |i: usize, (n, u): (usize, u32)| {
                bad[i].0 += n;
                bad[i].1 = bad[i].1.max(u);
            };
            add(0, diff_dev(g, o_c, o_ref, kk * value_dim)?);
            add(1, diff_dev(g, h_c, h_ref, h_numel)?);
            add(3, diff_dev(g, c_c, c_ref, state)?);
            for t in 0..kk - 1 {
                add(2, diff_dev(g, hi_c[t], hi_ref[t], h_numel)?);
                add(4, diff_dev(g, ci_c[t], ci_ref[t], state)?);
            }
            for t in 0..kk {
                add(
                    5,
                    diff_dev(g, r_c.offset(t * row * 4), r_ref.offset(t * row * 4), dim)?,
                );
            }
        }
        let names = [
            "GDN outputs",
            "final h",
            "h intermediates",
            "final conv window",
            "conv intermediates",
            "conv rows",
        ];
        for (i, name) in names.iter().enumerate() {
            report(
                format!("K={kk} {rounds} rounds: {name}"),
                bad[i].0,
                bad[i].1,
                rounds,
            );
        }
    }
    println!(
        "exact_chain_microtest: target={target} nk={nk} nv={nv} seed={seed} rounds={rounds}: {}",
        if failures == 0 {
            "ALL PASS (bit-equal)"
        } else {
            "FAILURES"
        }
    );
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
