// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Kernel-level check of the carried-state conv verify (`gdn_carry_conv` and
//! `gdn_carry_conv_flush`, kernels/gb10/common/gated_delta_rule_carry.cu) against the
//! batched parent `gdn_verify_fused_conv_kn_batched`.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 on any mismatch below; each check prints its mismatch count.
//!
//! `SEQS` sequences on scattered slots, two verify rounds of K rows, for K = 2, 3, 4:
//!
//! * round-1 outputs bit-equal to the parent's, and the carried kernel leaves every conv
//!   state alone;
//! * for na in 1..=K: `gdn_carry_conv_flush` with `pend = na` leaves the conv state
//!   bit-equal to the parent's snapshot na-1;
//! * for na in 1..=K: a round-2 carried launch from the round-1 state with `pend = na`
//!   gives the outputs of a parent launch from snapshot na-1, and leaves that snapshot as
//!   the conv state.
//!
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!       --example gdn_carry_conv_microtest
//!
//! Env: SEQS (default 5), SEED (default 1).

use anyhow::{Context, Result};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

// 2026-09-26: GDN dimensions and conv width of kernels/gb10/qwen3.6-35b-a3b/MODEL.toml.
const KD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const VD: usize = 128;
const D_CONV: usize = 4;
const CONV_DIM: usize = NK * KD * 2 + NV * VD;
const QK_CH: usize = NK * KD * 2;
const QKVZ_SIZE: usize = CONV_DIM + NV * VD;
const STATE_ELEMS: usize = CONV_DIM * D_CONV;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// 2026-09-26: Deterministic LCG in [-1, 1).
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}
fn bf16_bytes(n: usize, rng: &mut Rng, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            bf16::from_f32(rng.next_f32() * scale)
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}
fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn mismatches(a: &[u8], b: &[u8], width: usize) -> usize {
    a.chunks(width)
        .zip(b.chunks(width))
        .filter(|(x, y)| x != y)
        .count()
}

fn main() -> Result<()> {
    let seqs = env_usize("SEQS", 5);
    let seed = env_usize("SEED", 1) as u64;
    anyhow::ensure!((1..=16).contains(&seqs));
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-27b", "nvfp4")
        .context("no compiled qwen3.8-27b/nvfp4 kernel set (METRALE_TARGET_MODEL=qwen3.8-27b)")?;
    let backend = MetraleCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let stream = g.default_stream();
    let parent = g.kernel(
        "gdn_verify_fused_conv_kn",
        "gdn_verify_fused_conv_kn_batched",
    )?;
    let carry = g.kernel("gated_delta_rule_carry", "gdn_carry_conv")?;
    let flush = g.kernel("gated_delta_rule_carry", "gdn_carry_conv_flush")?;

    let mut rng = Rng(seed);
    let weight = DenseWeight {
        weight: upload(g, &bf16_bytes(STATE_ELEMS, &mut rng, 0.5))?,
    };
    // 2026-09-26: Sequence b on slot 2b + 1; pend is indexed by slot.
    let n_slots = 2 * seqs + 2;
    let slots: Vec<u32> = (0..seqs as u32).map(|b| 2 * b + 1).collect();
    let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
    let slot_tab = upload(g, &slot_bytes)?;
    let seq_elems = ops::gdn_carry_conv_seq_elems(CONV_DIM);
    let stash = g.alloc(n_slots * seq_elems * 2)?;
    let pend = g.alloc(n_slots * 4)?;
    let set_pend = |na: u32| -> Result<()> {
        let mut v = vec![0u8; n_slots * 4];
        for &s in &slots {
            v[s as usize * 4..s as usize * 4 + 4].copy_from_slice(&na.to_le_bytes());
        }
        g.copy_h2d(&v, pend)
    };
    let state_bytes = STATE_ELEMS * 4;
    let init: Vec<u8> = (0..seqs * STATE_ELEMS)
        .flat_map(|_| (rng.next_f32() * 0.5).to_le_bytes())
        .collect();

    let mut failures = 0usize;
    let mut report = |label: &str, n: usize, total: usize| {
        if n != 0 {
            failures += 1;
        }
        println!(
            "{label:<50} {}  mismatches={n}/{total}",
            if n == 0 { "PASS" } else { "FAIL" }
        );
    };

    for kk in 2..=4usize {
        let rounds: Vec<DevicePtr> = (0..2)
            .map(|_| upload(g, &bf16_bytes(seqs * kk * QKVZ_SIZE, &mut rng, 1.0)))
            .collect::<Result<_>>()?;
        let out_bytes = seqs * kk * CONV_DIM * 2;
        let (out_p, out_c) = (g.alloc(out_bytes)?, g.alloc(out_bytes)?);
        let state_p = upload(g, &init)?;
        let inter = g.alloc(seqs * kk * state_bytes)?;
        let run_parent = |state: DevicePtr, input: DevicePtr| -> Result<()> {
            ops::gdn_verify_fused_conv_kn_batched(
                g,
                parent,
                state,
                input,
                &weight,
                out_p,
                inter,
                kk as u32,
                CONV_DIM as u32,
                D_CONV as u32,
                QK_CH as u32,
                KD as u32,
                QKVZ_SIZE as u32,
                CONV_DIM as u32,
                STATE_ELEMS as u32,
                1e-6,
                seqs as u32,
                STATE_ELEMS as u32,
                (kk * QKVZ_SIZE) as u32,
                (kk * CONV_DIM) as u32,
                (kk * STATE_ELEMS) as u32,
                stream,
            )?;
            g.synchronize(stream)
        };
        let state_c = upload(g, &init)?;
        let run_carry = |input: DevicePtr, na: u32| -> Result<()> {
            set_pend(na)?;
            ops::gdn_carry_conv(
                g,
                carry,
                state_c,
                input,
                &weight,
                out_c,
                stash,
                slot_tab,
                pend,
                seq_elems as u32,
                kk as u32,
                CONV_DIM as u32,
                D_CONV as u32,
                QK_CH as u32,
                KD as u32,
                QKVZ_SIZE as u32,
                CONV_DIM as u32,
                1e-6,
                seqs as u32,
                STATE_ELEMS as u32,
                (kk * QKVZ_SIZE) as u32,
                (kk * CONV_DIM) as u32,
                stream,
            )?;
            g.synchronize(stream)
        };

        run_parent(state_p, rounds[0])?;
        let out1 = read(g, out_p, out_bytes)?;
        let snaps = read(g, inter, seqs * kk * state_bytes)?;
        // 2026-09-26: Snapshot t of sequence b, the parent's state after t + 1 rows.
        let snap = |t: usize| -> Vec<u8> {
            (0..seqs)
                .flat_map(|b| {
                    let o = (b * kk + t) * state_bytes;
                    snaps[o..o + state_bytes].to_vec()
                })
                .collect()
        };
        run_carry(rounds[0], 0)?;
        let n = mismatches(&read(g, out_c, out_bytes)?, &out1, 2);
        report(
            &format!("K={kk} round-1 output vs parent"),
            n,
            out_bytes / 2,
        );
        let n = mismatches(&read(g, state_c, seqs * state_bytes)?, &init, 4);
        report(
            &format!("K={kk} carried launch wrote no state"),
            n,
            seqs * STATE_ELEMS,
        );

        let table: Vec<u8> = (0..32)
            .flat_map(|b| {
                let p = if b < seqs {
                    state_c.offset(b * state_bytes).0
                } else {
                    0
                };
                p.to_le_bytes()
            })
            .collect();
        let state_table = upload(g, &table)?;
        for na in 1..=kk {
            let want = snap(na - 1);
            // 2026-09-26: Standalone fold of the round-1 stash, which the round-1 launch
            // writes again first (the previous na's round 2 overwrote it).
            g.copy_h2d(&init, state_c)?;
            run_carry(rounds[0], 0)?;
            g.copy_h2d(&init, state_c)?;
            set_pend(na as u32)?;
            ops::gdn_carry_conv_flush(
                g,
                flush,
                state_table,
                0,
                stash,
                0,
                slot_tab,
                pend,
                0,
                seq_elems as u32,
                seqs as u32,
                CONV_DIM as u32,
                D_CONV as u32,
                1,
                stream,
            )?;
            g.synchronize(stream)?;
            let n = mismatches(&read(g, state_c, seqs * state_bytes)?, &want, 4);
            report(
                &format!("K={kk} flush na={na} vs snapshot"),
                n,
                seqs * STATE_ELEMS,
            );

            // 2026-09-26: Round 2: parent from snapshot na-1 against the carried launch from
            // the initial state with na pending.
            let from = upload(g, &want)?;
            run_parent(from, rounds[1])?;
            let out2 = read(g, out_p, out_bytes)?;
            g.copy_h2d(&init, state_c)?;
            run_carry(rounds[0], 0)?;
            run_carry(rounds[1], na as u32)?;
            let n = mismatches(&read(g, out_c, out_bytes)?, &out2, 2);
            report(
                &format!("K={kk} na={na} round-2 output vs parent"),
                n,
                out_bytes / 2,
            );
            let n = mismatches(&read(g, state_c, seqs * state_bytes)?, &want, 4);
            report(
                &format!("K={kk} na={na} round-2 state = snapshot"),
                n,
                seqs * STATE_ELEMS,
            );
        }
    }

    println!(
        "gdn_carry_conv_microtest: seqs={seqs} seed={seed}: {}",
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
