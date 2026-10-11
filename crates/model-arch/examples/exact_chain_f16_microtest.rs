// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Kernel-level check of the FP16 h-state exact verify against the decode chain it
//! must reproduce, on the qwen3.8-27b kernel set:
//! - `gdn_exact_chain_f16_{2,3,4}` (table form, n sequences) against K launches of
//!   `gated_delta_rule_decode_f16_strided_norm_half` (the batched decode) with a copy of each
//!   sequence's state after every row but the last;
//! - the single-sequence form against K launches of `gated_delta_rule_decode_f16_norm` (the C1
//!   decode), which also checks that the C1 and batched decodes agree;
//! - `gdn_conv_chain_f32_batched` against K launches of `causal_conv1d_update_l2norm_f32_strided`
//!   with a copy of each window after every row but the last.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 on any mismatch; each check prints its mismatch count.
//!
//! `ROUNDS` rounds per K, each from the states the previous round left; the first state's head
//! norm is above 1000 (no path rescales it).
//!
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!       --example exact_chain_f16_microtest
//!
//! Env: SEED (default 1), ROUNDS (default 6), SEQS (default 5), NK/NV (default 16/32). CONTROL=1
//! gives the chain row 0's first gate one ULP up, a negative control: the GDN checks must FAIL
//! and the conv checks PASS.

use anyhow::{Context, Result};
use half::{bf16, f16};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

// 2026-10-01: Shared with the other GDN microtests; this one uses part of it.
#[allow(dead_code)]
#[path = "common/gdn_carry_fixture.rs"]
mod gdn_carry_fixture;
use gdn_carry_fixture::*;

const D_CONV: usize = 4;

fn read_bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-01: Mismatching bytes of two device buffers of `n` bytes.
fn diff_bytes(g: &dyn GpuBackend, a: DevicePtr, b: DevicePtr, n: usize) -> Result<usize> {
    let (x, y) = (read_bytes(g, a, n)?, read_bytes(g, b, n)?);
    Ok(x.iter().zip(&y).filter(|(p, q)| p != q).count())
}

fn bf16_bytes(n: usize, scale: f32, rng: &mut Rng) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            bf16::from_f32(rng.next_f32() * scale)
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

fn main() -> Result<()> {
    let seed = env_usize("SEED", 1) as u64;
    let rounds = env_usize("ROUNDS", 6);
    let seqs = env_usize("SEQS", 5);
    let control = env_usize("CONTROL", 0) == 1;
    let nk = env_usize("NK", 16);
    let nv = env_usize("NV", 32);
    anyhow::ensure!(nv.is_multiple_of(nk));
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-27b", "nvfp4")
        .context("no compiled qwen3.8-27b/nvfp4 kernel set")?;
    let backend = MetraleCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let s = g.default_stream();
    let dec_batched = g.kernel(
        "gated_delta_rule",
        "gated_delta_rule_decode_f16_strided_norm_half",
    )?;
    let dec_single = g.kernel("gated_delta_rule", "gated_delta_rule_decode_f16_norm")?;
    let conv_parent = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32_strided")?;
    let conv_chain = g.kernel("gated_delta_rule_carry", "gdn_conv_chain_f32_batched")?;
    let (key_dim, value_dim) = (nk * KD, nv * VD);
    let dim = 2 * key_dim + value_dim;
    let row = dim + value_dim;
    let h_numel = nv * KD * VD;
    // 2026-10-01: The FP32-sized pool's pitch (`--ssm-h-dtype f16`): each slot holds its FP16
    // state in its first half.
    let h_pitch = 2 * h_numel;
    let conv_numel = dim * D_CONV;
    let eps = 1e-6f32;
    let mut rng = Rng(seed);
    let conv_w = DenseWeight {
        weight: upload_bytes(g, &bf16_bytes(conv_numel, 0.5, &mut rng))?,
    };
    let norm_w = upload_bytes(g, &bf16_bytes(VD, 1.0, &mut rng))?;
    let mut failures = 0usize;
    let mut report = |label: String, n: usize| {
        if n != 0 {
            failures += 1;
        }
        println!(
            "{label:<60} {}  mismatched bytes={n}",
            if n == 0 { "PASS" } else { "FAIL" }
        );
    };
    for kk in 2..=4usize {
        let chain = g.kernel("gdn_exact_carry", &format!("gdn_exact_chain_f16_{kk}"))?;
        for n in [1usize, seqs] {
            // 2026-10-01: Head norm near 1500 at the start, past 1000.
            let h0: Vec<u8> = (0..n * h_pitch)
                .flat_map(|_| f16::from_f32(rng.next_f32() * 20.0).to_bits().to_le_bytes())
                .collect();
            let c0: Vec<f32> = (0..n * conv_numel).map(|_| rng.next_f32()).collect();
            let (h_ref, h_c) = (upload_bytes(g, &h0)?, upload_bytes(g, &h0)?);
            let (c_ref, c_c) = (
                upload_bytes(g, &f32_bytes(&c0))?,
                upload_bytes(g, &f32_bytes(&c0))?,
            );
            // 2026-10-01: Intermediates t of sequence b at (b * 3 + t) * pitch.
            let (hi_ref, hi_c) = (g.alloc(n * 3 * h_pitch * 2)?, g.alloc(n * 3 * h_pitch * 2)?);
            let (ci_ref, ci_c) = (
                g.alloc(n * 3 * conv_numel * 4)?,
                g.alloc(n * 3 * conv_numel * 4)?,
            );
            let h_at = |base: DevicePtr, b: usize| base.offset(b * h_pitch * 2);
            let hi_at =
                |base: DevicePtr, b: usize, t: usize| base.offset((b * 3 + t) * h_pitch * 2);
            let ci_at =
                |base: DevicePtr, b: usize, t: usize| base.offset((b * 3 + t) * conv_numel * 4);
            let tables: Vec<DevicePtr> = (0..4)
                .map(|i| {
                    let ptrs: Vec<DevicePtr> = (0..n)
                        .map(|b| {
                            if i == 0 {
                                h_at(h_c, b)
                            } else {
                                hi_at(hi_c, b, i - 1)
                            }
                        })
                        .collect();
                    ptr_table(g, &ptrs)
                })
                .collect::<Result<_>>()?;
            let rows_n = n * kk;
            let (o_ref, o_c) = (
                g.alloc(rows_n * value_dim * 2)?,
                g.alloc(rows_n * value_dim * 2)?,
            );
            let (r_ref, r_c) = (g.alloc(rows_n * row * 4)?, g.alloc(rows_n * row * 4)?);
            let mut bad = [0usize; 6];
            for _ in 0..rounds {
                let x = upload_bytes(g, &bf16_bytes(rows_n * dim, 1.0, &mut rng))?;
                let z = upload_bytes(g, &bf16_bytes(rows_n * value_dim, 2.0, &mut rng))?;
                let mut gb = vec![0f32; rows_n * 2 * nv];
                for r in 0..rows_n {
                    for h in 0..nv {
                        gb[r * 2 * nv + h] = 0.5 + 0.49 * rng.next_f32();
                        gb[r * 2 * nv + nv + h] = 0.5 + 0.5 * rng.next_f32().abs();
                    }
                }
                let gbd = upload_bytes(g, &f32_bytes(&gb))?;
                if control {
                    gb[0] = f32::from_bits(gb[0].to_bits() + 1);
                }
                let gbd_c = upload_bytes(g, &f32_bytes(&gb))?;
                // 2026-10-01: The reference: per row, the conv and the decode over n sequences,
                // then a copy of each sequence's window and state.
                for t in 0..kk {
                    ops::conv1d_update_l2norm_strided(
                        g,
                        conv_parent,
                        c_ref,
                        x.offset(t * dim * 2),
                        &conv_w,
                        r_ref.offset(t * row * 4),
                        dim as u32,
                        D_CONV as u32,
                        n as u32,
                        (2 * key_dim) as u32,
                        KD as u32,
                        eps,
                        (kk * dim) as u32,
                        (kk * row) as u32,
                        s,
                    )?;
                    let r = r_ref.offset(t * row * 4);
                    let gt = gbd.offset(t * 2 * nv * 4);
                    if n == 1 {
                        ops::gdn_decode_f32_norm(
                            g,
                            dec_single,
                            h_ref,
                            r,
                            r.offset(key_dim * 4),
                            r.offset(2 * key_dim * 4),
                            gt,
                            gt.offset(nv * 4),
                            z.offset(t * value_dim * 2),
                            norm_w,
                            o_ref.offset(t * value_dim * 2),
                            1,
                            nk as u32,
                            nv as u32,
                            KD as u32,
                            VD as u32,
                            eps,
                            s,
                        )?;
                    } else {
                        ops::gdn_decode_f16_strided_norm(
                            g,
                            dec_batched,
                            h_ref,
                            r,
                            r.offset(key_dim * 4),
                            r.offset(2 * key_dim * 4),
                            gt,
                            gt.offset(nv * 4),
                            z.offset(t * value_dim * 2),
                            norm_w,
                            o_ref.offset(t * value_dim * 2),
                            n as u32,
                            nk as u32,
                            nv as u32,
                            KD as u32,
                            VD as u32,
                            (kk * row) as u32,
                            (kk * row) as u32,
                            (kk * 2 * nv) as u32,
                            (kk * value_dim) as u32,
                            (kk * value_dim) as u32,
                            h_pitch as u64,
                            eps,
                            s,
                        )?;
                    }
                    if t + 1 < kk {
                        for b in 0..n {
                            g.copy_d2d_async(h_at(h_ref, b), hi_at(hi_ref, b, t), h_numel * 2, s)?;
                            g.copy_d2d_async(
                                c_ref.offset(b * conv_numel * 4),
                                ci_at(ci_ref, b, t),
                                conv_numel * 4,
                                s,
                            )?;
                        }
                    }
                }
                ops::gdn_conv_chain_f32_batched(
                    g,
                    conv_chain,
                    c_c,
                    x,
                    &conv_w,
                    r_c,
                    ci_c,
                    n as u32,
                    kk as u32,
                    [dim as u32, D_CONV as u32, (2 * key_dim) as u32, KD as u32],
                    eps,
                    [dim as u32, row as u32],
                    [conv_numel as u64, (3 * conv_numel) as u64],
                    s,
                )?;
                let states = if n == 1 {
                    ops::F16ChainStates::Single([
                        h_c,
                        hi_at(hi_c, 0, 0),
                        hi_at(hi_c, 0, 1),
                        hi_at(hi_c, 0, 2),
                    ])
                } else {
                    ops::F16ChainStates::Tables([tables[0], tables[1], tables[2], tables[3]])
                };
                ops::gdn_exact_chain_f16(
                    g,
                    chain,
                    states,
                    r_c,
                    r_c.offset(key_dim * 4),
                    r_c.offset(2 * key_dim * 4),
                    gbd_c,
                    gbd_c.offset(nv * 4),
                    z,
                    norm_w,
                    o_c,
                    n as u32,
                    [nk as u32, nv as u32, KD as u32],
                    [
                        row as u32,
                        row as u32,
                        (2 * nv) as u32,
                        value_dim as u32,
                        value_dim as u32,
                    ],
                    eps,
                    s,
                )?;
                g.synchronize(s)?;
                bad[0] += diff_bytes(g, o_c, o_ref, rows_n * value_dim * 2)?;
                for b in 0..n {
                    bad[1] += diff_bytes(g, h_at(h_c, b), h_at(h_ref, b), h_numel * 2)?;
                    bad[3] += diff_bytes(
                        g,
                        c_c.offset(b * conv_numel * 4),
                        c_ref.offset(b * conv_numel * 4),
                        conv_numel * 4,
                    )?;
                    for t in 0..kk - 1 {
                        bad[2] +=
                            diff_bytes(g, hi_at(hi_c, b, t), hi_at(hi_ref, b, t), h_numel * 2)?;
                        bad[4] +=
                            diff_bytes(g, ci_at(ci_c, b, t), ci_at(ci_ref, b, t), conv_numel * 4)?;
                    }
                }
                for r in 0..rows_n {
                    bad[5] += diff_bytes(
                        g,
                        r_c.offset(r * row * 4),
                        r_ref.offset(r * row * 4),
                        dim * 4,
                    )?;
                }
            }
            let names = [
                "normed outputs",
                "final h",
                "h intermediates",
                "final conv windows",
                "conv intermediates",
                "conv rows",
            ];
            for (i, name) in names.iter().enumerate() {
                report(format!("K={kk} n={n} {rounds} rounds: {name}"), bad[i]);
            }
        }
    }
    println!(
        "exact_chain_f16_microtest: nk={nk} nv={nv} seqs={seqs} seed={seed} rounds={rounds}: {}",
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
