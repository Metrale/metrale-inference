// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Kernel-level check of the carried exact verify against the strided decode chain
//! it replaces: `gdn_exact_carry{2,3,4}` (and `_lazy`, `gdn_exact_carry.cu` in the model
//! directory) against K launches of `gated_delta_rule_decode_f32_strided`, and
//! `gdn_carry_conv_f32` against K launches of `causal_conv1d_update_l2norm_f32_strided`.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 on any mismatch below; each check prints its mismatch count and max ULP delta.
//!
//! `SEQS` sequences on scattered slots, `ROUNDS` verify rounds of K rows for K = 2, 3, 4. Each
//! round runs the parent chain from the committed reference state (one launch per row, with a
//! snapshot after each) and the carried kernel (eager or lazy, drawn per round) from the carried
//! state with each slot's pending count, draws an accepted count per sequence, commits the
//! parent's snapshot after that many rows into the reference and advances the pending counts by
//! the kernels' rule (`pend' = na` after a write-back, else `pend + na`). It checks every round's
//! outputs bit for bit, and that `gdn_carry_flush` / `gdn_carry_conv_flush` after the last round
//! leave the reference state.
//!
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!       --example exact_carry_microtest
//!
//! The fold is `gdn_exact_carry_flush` where the target has one, else `gdn_carry_flush`.
//!
//! Env: TARGET (default qwen3.6-35b-a3b; qwen3.8-27b for the dense), SEQS (default 5), SEED (default 1), ROUNDS
//! (default 9), NK/NV (default 16/32).

use anyhow::{Context, Result};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

// 2026-10-01: Shared with gdn_carry_microtest, which uses the WY parents this check does not.
#[allow(dead_code)]
#[path = "common/gdn_carry_fixture.rs"]
mod gdn_carry_fixture;
use gdn_carry_fixture::*;

const D_CONV: usize = 4;

/// 2026-10-01: Slot table and pending counts shared by one check's carried launches.
struct Slots {
    slots: Vec<usize>,
    n_slots: usize,
    slot_tab: DevicePtr,
    pend: DevicePtr,
}

impl Slots {
    fn new(g: &dyn GpuBackend, seqs: usize) -> Result<Self> {
        // 2026-10-01: Sequence b on slot 2b + 1, so the slot indirection is exercised.
        let n_slots = 2 * seqs + 2;
        let slots: Vec<usize> = (0..seqs).map(|b| 2 * b + 1).collect();
        let tab: Vec<u32> = slots.iter().map(|&s| s as u32).collect();
        Ok(Self {
            slot_tab: upload_bytes(g, &u32_bytes(&tab))?,
            pend: g.alloc(n_slots * 4)?,
            slots,
            n_slots,
        })
    }
    fn upload(&self, g: &dyn GpuBackend, np: &[usize]) -> Result<()> {
        let mut v = vec![0u32; self.n_slots];
        for (b, &s) in self.slots.iter().enumerate() {
            v[s] = np[b] as u32;
        }
        g.copy_h2d(&u32_bytes(&v), self.pend)
    }
}

/// 2026-10-01: The pending count after a verify of `kk` rows with `na` accepted.
fn next_pend(np: usize, kk: usize, na: usize, lazy: bool) -> usize {
    let kept = if !lazy || np + kk > ops::GDN_CARRY_CAP {
        0
    } else {
        np
    };
    kept + na
}

fn draw_accept(rng: &mut Rng, kk: usize) -> usize {
    1 + (rng.next_f32().abs() * kk as f32) as usize % kk
}

fn copy(g: &dyn GpuBackend, from: DevicePtr, to: DevicePtr, bytes: usize) -> Result<()> {
    g.copy_d2d_async(from, to, bytes, g.default_stream())
}

fn main() -> Result<()> {
    let target = std::env::var("TARGET").unwrap_or_else(|_| "qwen3.6-35b-a3b".to_string());
    let seqs = env_usize("SEQS", 5);
    let seed = env_usize("SEED", 1) as u64;
    let rounds = env_usize("ROUNDS", 9);
    let nk = env_usize("NK", 16);
    let nv = env_usize("NV", 32);
    anyhow::ensure!((2..=SLAB_ENTRIES / 2).contains(&seqs) && nv.is_multiple_of(nk));
    let set = metrale_kernels::ptx_for_exact_target(&target, "nvfp4")
        .with_context(|| format!("no compiled {target}/nvfp4 kernel set"))?;
    let backend = MetraleCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let mut failures = 0usize;
    let mut report = |label: &str, n: usize, ulp: u32, total: usize| {
        if n != 0 {
            failures += 1;
        }
        println!(
            "{label:<56} {}  mismatches={n}/{total}  max_ulp={ulp}",
            if n == 0 { "PASS" } else { "FAIL" }
        );
    };
    let mut rng = Rng(seed);
    gdn_check(g, seqs, nk, nv, rounds, &mut rng, &mut report)?;
    conv_check(g, seqs, nk, nv, rounds, &mut rng, &mut report)?;
    println!(
        "exact_carry_microtest: target={target} seqs={seqs} nk={nk} nv={nv} seed={seed} \
         rounds={rounds}: {}",
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

type Report<'a> = dyn FnMut(&str, usize, u32, usize) + 'a;

/// 2026-10-01: `gdn_exact_carry{K}` against the strided decode chain.
fn gdn_check(
    g: &dyn GpuBackend,
    seqs: usize,
    nk: usize,
    nv: usize,
    rounds: usize,
    rng: &mut Rng,
    report: &mut Report<'_>,
) -> Result<()> {
    let s = g.default_stream();
    let parent = g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32_strided")?;
    // 2026-10-01: The fold the serve uses under the exact verify (`carry_flush_kernel`).
    let flush = g
        .kernel("gdn_exact_carry", "gdn_exact_carry_flush")
        .or_else(|_| g.kernel("gated_delta_rule_carry", "gdn_carry_flush"))?;
    let carried = |kk: usize, lazy: bool| -> Result<KernelHandle> {
        let name = format!("gdn_exact_carry{kk}{}", if lazy { "_lazy" } else { "" });
        g.kernel("gdn_exact_carry", &name)
    };
    let key_dim = nk * KD;
    let value_dim = nv * VD;
    // 2026-10-01: Row r = b * K + t: q | k | v FP32, the layout of the FP32 conv rows.
    let row = 2 * key_dim + value_dim;
    let h_numel = nv * KD * VD;
    let sl = Slots::new(g, seqs)?;
    let seq_floats = ops::gdn_carry_seq_floats(nv, KD, VD);
    let stash = g.alloc(sl.n_slots * seq_floats * 4)?;
    let flags = g.alloc(SLAB_ENTRIES * 4)?;
    for kk in 2..=4usize {
        let rows = seqs * kk;
        // 2026-10-01: Sequence 0 starts with a head norm near 1500, above the 27B decode's clamp
        // threshold (1000), so the clamp path is exercised where the decode has one.
        let h_init: Vec<f32> = (0..seqs * h_numel)
            .map(|i| rng.next_f32() * if i < h_numel { 20.0 } else { 0.1 })
            .collect();
        // 2026-10-01: The reference is contiguous (the strided parent's layout); the carried
        // states are separate allocations reached through the pointer table.
        let h_ref = upload_bytes(g, &f32_bytes(&h_init))?;
        let h_work = g.alloc(seqs * h_numel * 4)?;
        let snaps: Vec<DevicePtr> = (0..kk)
            .map(|_| g.alloc(seqs * h_numel * 4))
            .collect::<Result<_>>()?;
        let h_c: Vec<DevicePtr> = (0..seqs)
            .map(|b| upload_bytes(g, &f32_bytes(&h_init[b * h_numel..(b + 1) * h_numel])))
            .collect::<Result<_>>()?;
        let t_c = ptr_table(g, &h_c)?;
        let (out_p, out_c) = (
            g.alloc(rows * value_dim * 4)?,
            g.alloc(rows * value_dim * 4)?,
        );
        let mut np = vec![0usize; seqs];
        let (mut bad, mut ulp) = (0usize, 0u32);
        for _ in 0..rounds {
            let x: Vec<f32> = (0..rows * row).map(|_| rng.next_f32() * 0.5).collect();
            let mut gb = vec![0f32; rows * 2 * nv];
            for r in 0..rows {
                for h in 0..nv {
                    gb[r * 2 * nv + h] = 0.5 + 0.49 * rng.next_f32();
                    gb[r * 2 * nv + nv + h] = 0.5 + 0.5 * rng.next_f32().abs();
                }
            }
            let (xd, gbd) = (
                upload_bytes(g, &f32_bytes(&x))?,
                upload_bytes(g, &f32_bytes(&gb))?,
            );
            let (q, k, v) = (xd, xd.offset(key_dim * 4), xd.offset(2 * key_dim * 4));
            copy(g, h_ref, h_work, seqs * h_numel * 4)?;
            for (t, &snap) in snaps.iter().enumerate() {
                ops::gdn_decode_f32_strided(
                    g,
                    parent,
                    h_work,
                    q.offset(t * row * 4),
                    k.offset(t * row * 4),
                    v.offset(t * row * 4),
                    gbd.offset(t * 2 * nv * 4),
                    gbd.offset((t * 2 * nv + nv) * 4),
                    out_p.offset(t * value_dim * 4),
                    seqs as u32,
                    nk as u32,
                    nv as u32,
                    KD as u32,
                    VD as u32,
                    (kk * row) as u32,
                    (kk * row) as u32,
                    (kk * 2 * nv) as u32,
                    (kk * value_dim) as u32,
                    s,
                )?;
                copy(g, h_work, snap, seqs * h_numel * 4)?;
            }
            sl.upload(g, &np)?;
            let lazy = rng.next_f32() > 0.0;
            ops::gdn_exact_carry(
                g,
                carried(kk, lazy)?,
                t_c,
                q,
                k,
                v,
                gbd,
                gbd.offset(nv * 4),
                out_c,
                stash,
                sl.slot_tab,
                sl.pend,
                seq_floats as u32,
                seqs as u32,
                nk as u32,
                nv as u32,
                KD as u32,
                [row as u32, row as u32, (2 * nv) as u32, value_dim as u32],
                flags,
                s,
            )?;
            g.synchronize(s)?;
            let (n, u) = diff_f32(
                &read_f32(g, out_c, rows * value_dim)?,
                &read_f32(g, out_p, rows * value_dim)?,
            );
            bad += n;
            ulp = ulp.max(u);
            for b in 0..seqs {
                let na = draw_accept(rng, kk);
                let off = b * h_numel * 4;
                copy(g, snaps[na - 1].offset(off), h_ref.offset(off), h_numel * 4)?;
                np[b] = next_pend(np[b], kk, na, lazy);
            }
        }
        report(
            &format!("GDN K={kk} {rounds} rounds: outputs vs chain"),
            bad,
            ulp,
            rounds * rows * value_dim,
        );
        sl.upload(g, &np)?;
        ops::gdn_carry_flush(
            g,
            flush,
            t_c,
            0,
            stash,
            0,
            sl.slot_tab,
            sl.pend,
            0,
            seq_floats as u32,
            seqs as u32,
            nv as u32,
            1,
            s,
        )?;
        g.synchronize(s)?;
        let reference = read_f32(g, h_ref, seqs * h_numel)?;
        let (mut n, mut u) = (0, 0);
        for (b, &h) in h_c.iter().enumerate() {
            let (dn, du) = diff_f32(
                &read_f32(g, h, h_numel)?,
                &reference[b * h_numel..(b + 1) * h_numel],
            );
            n += dn;
            u = u.max(du);
        }
        report(
            &format!("GDN K={kk} flushed state vs chain (pend {np:?})"),
            n,
            u,
            seqs * h_numel,
        );
    }
    Ok(())
}

/// 2026-10-01: `gdn_carry_conv_f32` against the FP32 strided conv chain.
fn conv_check(
    g: &dyn GpuBackend,
    seqs: usize,
    nk: usize,
    nv: usize,
    rounds: usize,
    rng: &mut Rng,
    report: &mut Report<'_>,
) -> Result<()> {
    let s = g.default_stream();
    let parent = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32_strided")?;
    let carried = g.kernel("gated_delta_rule_carry", "gdn_carry_conv_f32")?;
    let flush = g.kernel("gated_delta_rule_carry", "gdn_carry_conv_flush")?;
    let dim = 2 * nk * KD + nv * VD;
    let qk_ch = 2 * nk * KD;
    let state = dim * D_CONV;
    let weight = DenseWeight {
        weight: upload_bytes(
            g,
            &(0..state)
                .flat_map(|_| bf16::from_f32(rng.next_f32() * 0.5).to_bits().to_le_bytes())
                .collect::<Vec<u8>>(),
        )?,
    };
    let sl = Slots::new(g, seqs)?;
    let seq_elems = ops::gdn_carry_conv_seq_elems(dim);
    let stash = g.alloc(sl.n_slots * seq_elems * 2)?;
    for kk in 2..=4usize {
        let rows = seqs * kk;
        let init: Vec<f32> = (0..seqs * state).map(|_| rng.next_f32()).collect();
        let (c_ref, c_c) = (
            upload_bytes(g, &f32_bytes(&init))?,
            upload_bytes(g, &f32_bytes(&init))?,
        );
        let c_work = g.alloc(seqs * state * 4)?;
        let snaps: Vec<DevicePtr> = (0..kk)
            .map(|_| g.alloc(seqs * state * 4))
            .collect::<Result<_>>()?;
        let (out_p, out_c) = (g.alloc(rows * dim * 4)?, g.alloc(rows * dim * 4)?);
        let mut np = vec![0usize; seqs];
        let (mut bad, mut ulp) = (0usize, 0u32);
        for _ in 0..rounds {
            let input: Vec<u8> = (0..rows * dim)
                .flat_map(|_| bf16::from_f32(rng.next_f32()).to_bits().to_le_bytes())
                .collect();
            let x = upload_bytes(g, &input)?;
            copy(g, c_ref, c_work, seqs * state * 4)?;
            for (t, &snap) in snaps.iter().enumerate() {
                ops::conv1d_update_l2norm_strided(
                    g,
                    parent,
                    c_work,
                    x.offset(t * dim * 2),
                    &weight,
                    out_p.offset(t * dim * 4),
                    dim as u32,
                    D_CONV as u32,
                    seqs as u32,
                    qk_ch as u32,
                    KD as u32,
                    1e-6,
                    (kk * dim) as u32,
                    (kk * dim) as u32,
                    s,
                )?;
                copy(g, c_work, snap, seqs * state * 4)?;
            }
            sl.upload(g, &np)?;
            let lazy = rng.next_f32() > 0.0;
            ops::gdn_carry_conv_f32(
                g,
                carried,
                c_c,
                x,
                &weight,
                out_c,
                stash,
                sl.slot_tab,
                sl.pend,
                seq_elems as u32,
                kk as u32,
                dim as u32,
                D_CONV as u32,
                qk_ch as u32,
                KD as u32,
                1e-6,
                dim as u32,
                dim as u32,
                seqs as u32,
                lazy,
                s,
            )?;
            g.synchronize(s)?;
            let (n, u) = diff_f32(
                &read_f32(g, out_c, rows * dim)?,
                &read_f32(g, out_p, rows * dim)?,
            );
            bad += n;
            ulp = ulp.max(u);
            for b in 0..seqs {
                let na = draw_accept(rng, kk);
                let off = b * state * 4;
                copy(g, snaps[na - 1].offset(off), c_ref.offset(off), state * 4)?;
                np[b] = next_pend(np[b], kk, na, lazy);
            }
        }
        report(
            &format!("conv K={kk} {rounds} rounds: outputs vs chain"),
            bad,
            ulp,
            rounds * rows * dim,
        );
        sl.upload(g, &np)?;
        let tab: Vec<DevicePtr> = (0..seqs).map(|b| c_c.offset(b * state * 4)).collect();
        ops::gdn_carry_conv_flush(
            g,
            flush,
            ptr_table(g, &tab)?,
            0,
            stash,
            0,
            sl.slot_tab,
            sl.pend,
            0,
            seq_elems as u32,
            seqs as u32,
            dim as u32,
            D_CONV as u32,
            1,
            s,
        )?;
        g.synchronize(s)?;
        let (n, u) = diff_f32(
            &read_f32(g, c_c, seqs * state)?,
            &read_f32(g, c_ref, seqs * state)?,
        );
        report(
            &format!("conv K={kk} flushed window vs chain (pend {np:?})"),
            n,
            u,
            seqs * state,
        );
    }
    Ok(())
}
