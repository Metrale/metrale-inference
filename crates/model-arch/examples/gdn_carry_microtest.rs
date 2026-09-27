// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Kernel-level check of the carried-state GDN verify
//! (`kernels/gb10/common/gated_delta_rule_carry.cu`) against the parent WY kernels.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 on any mismatch below; each check prints its mismatch count and max ULP delta.
//!
//! One GDN layer, random H / q / k / v / gate / beta over `SEQS` sequences on scattered
//! slots, `ROUNDS` verify rounds. For K = 2, 3, 4 it runs, each round, the parent
//! `gated_delta_rule_wy{K}` (table form) from the committed reference state and
//! `gdn_carry_wy{K}` or `gdn_carry_wy{K}_lazy` (drawn per round) from the carried state
//! with each slot's pending count, draws an accepted count per sequence, commits the
//! parent's Hi(na-1) or final H into the reference and advances the pending counts by the
//! kernels' rule (`pend' = na` after a write-back, else `pend + na`; the eager form always
//! writes back, the lazy one when `pend + K > GDN_CARRY_CAP`). It checks:
//!
//! * every round's `output`, carry against parent, bit for bit;
//! * after the last round, `gdn_carry_flush` leaves H bit-equal to the reference.
//!
//!   cargo run -p metrale-model-arch --release --features gpu-examples \
//!       --example gdn_carry_microtest
//!
//! Env: SEQS (default 5), SEED (default 1), NK/NV (default 16/32), ROUNDS (default 9).

use anyhow::{Context, Result};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;

const KD: usize = 128;
const VD: usize = 128;
/// 2026-09-26: Pointer entries per table, equal to
/// `metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS`.
const SLAB_ENTRIES: usize = 32;

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

fn upload_bytes(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn read_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn ptr_table(g: &dyn GpuBackend, ptrs: &[DevicePtr]) -> Result<DevicePtr> {
    let mut bytes = vec![0u8; SLAB_ENTRIES * 8];
    for (i, p) in ptrs.iter().enumerate() {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&p.0.to_le_bytes());
    }
    upload_bytes(g, &bytes)
}

/// 2026-09-26: (mismatching elements, max ULP distance) between two f32 buffers.
fn diff_f32(a: &[f32], b: &[f32]) -> (usize, u32) {
    let mut n = 0usize;
    let mut ulp = 0u32;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            n += 1;
            let d = (x.to_bits() as i64 - y.to_bits() as i64).unsigned_abs();
            ulp = ulp.max(d.min(u32::MAX as u64) as u32);
        }
    }
    (n, ulp)
}

struct Dims {
    seqs: usize,
    nk: usize,
    nv: usize,
    conv_dim: usize,
    value_dim: usize,
    gb_stride: usize,
    h_numel: usize,
}

/// 2026-09-26: One round of verify inputs: bf16 q|k|v rows and f32 gate|beta rows.
struct Round {
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
}

fn make_round(g: &dyn GpuBackend, d: &Dims, kk: usize, rng: &mut Rng) -> Result<Round> {
    let rows = d.seqs * kk;
    let qkv: Vec<u8> = (0..rows * d.conv_dim)
        .flat_map(|_| bf16::from_f32(rng.next_f32() * 0.5).to_bits().to_le_bytes())
        .collect();
    let mut gb = vec![0f32; rows * d.gb_stride];
    for r in 0..rows {
        for h in 0..d.nv {
            gb[r * d.gb_stride + h] = 0.5 + 0.49 * rng.next_f32();
            gb[r * d.gb_stride + d.nv + h] = 0.5 + 0.5 * rng.next_f32().abs();
        }
    }
    let qkv_dev = upload_bytes(g, &qkv)?;
    let gb_dev = upload_bytes(g, &f32_bytes(&gb))?;
    let key_dim = d.nk * KD;
    Ok(Round {
        q: qkv_dev,
        k: qkv_dev.offset(key_dim * 2),
        v: qkv_dev.offset(key_dim * 2 * 2),
        gate: gb_dev,
        beta: gb_dev.offset(d.nv * 4),
    })
}

/// 2026-09-26: Parent verify of width `kk` in table form: `h` states, `hi[t]` intermediates.
fn run_parent(
    g: &dyn GpuBackend,
    k: KernelHandle,
    kk: usize,
    d: &Dims,
    r: &Round,
    h: DevicePtr,
    hi: &[DevicePtr],
    out: DevicePtr,
) -> Result<()> {
    let s = g.default_stream();
    let (n, nk, nv) = (d.seqs as u32, d.nk as u32, d.nv as u32);
    let (cd, gbs) = (d.conv_dim as u32, d.gb_stride as u32);
    match kk {
        2 => ops::gdn_decode_wy2(
            g, k, h, r.q, r.k, r.v, r.gate, r.beta, out, hi[0], n, nk, nv, KD as u32, VD as u32,
            cd, cd, gbs, true, s,
        )?,
        3 => ops::gdn_decode_wy3(
            g, k, h, r.q, r.k, r.v, r.gate, r.beta, out, hi[0], hi[1], n, nk, nv, KD as u32,
            VD as u32, cd, cd, gbs, true, s,
        )?,
        _ => ops::gdn_decode_wy4(
            g, k, h, r.q, r.k, r.v, r.gate, r.beta, out, hi[0], hi[1], hi[2], n, nk, nv, KD as u32,
            VD as u32, cd, cd, gbs, true, s,
        )?,
    }
    g.synchronize(s)
}

fn main() -> Result<()> {
    let seqs = env_usize("SEQS", 5);
    let seed = env_usize("SEED", 1) as u64;
    let nk = env_usize("NK", 16);
    let nv = env_usize("NV", 32);
    let rounds = env_usize("ROUNDS", 9);
    anyhow::ensure!((1..=SLAB_ENTRIES / 2).contains(&seqs) && nv.is_multiple_of(nk));
    let d = Dims {
        seqs,
        nk,
        nv,
        conv_dim: 2 * nk * KD + nv * VD,
        value_dim: nv * VD,
        gb_stride: nv * 2,
        h_numel: nv * KD * VD,
    };
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-27b", "nvfp4")
        .context("no compiled qwen3.8-27b/nvfp4 kernel set (METRALE_TARGET_MODEL=qwen3.8-27b)")?;
    let backend = MetraleCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let stream = g.default_stream();
    let parents = [
        g.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?,
        g.kernel("gated_delta_rule_wy3", "gated_delta_rule_wy3")?,
        g.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?,
    ];
    let carries = [
        g.kernel("gated_delta_rule_carry", "gdn_carry_wy2")?,
        g.kernel("gated_delta_rule_carry", "gdn_carry_wy3")?,
        g.kernel("gated_delta_rule_carry", "gdn_carry_wy4")?,
    ];
    let lazies = [
        g.kernel("gated_delta_rule_carry", "gdn_carry_wy2_lazy")?,
        g.kernel("gated_delta_rule_carry", "gdn_carry_wy3_lazy")?,
        g.kernel("gated_delta_rule_carry", "gdn_carry_wy4_lazy")?,
    ];
    let flush = g.kernel("gated_delta_rule_carry", "gdn_carry_flush")?;

    // 2026-09-26: Sequence b lives on slot 2b + 1, so the slot indirection is exercised.
    let n_slots = 2 * seqs + 2;
    let slots: Vec<usize> = (0..seqs).map(|b| 2 * b + 1).collect();
    let slot_bytes: Vec<u32> = slots.iter().map(|&s| s as u32).collect();
    let slot_tab = upload_bytes(g, &u32_bytes(&slot_bytes))?;
    let seq_floats = ops::gdn_carry_seq_floats(nv, KD, VD);
    let stash = g.alloc(n_slots * seq_floats * 4)?;
    let pend = g.alloc(n_slots * 4)?;
    let flags = g.alloc(SLAB_ENTRIES * 4)?;
    let upload_pend = |np: &[usize]| -> Result<()> {
        let mut v = vec![0u32; n_slots];
        for (b, &s) in slots.iter().enumerate() {
            v[s] = np[b] as u32;
        }
        g.copy_h2d(&u32_bytes(&v), pend)
    };

    let mut failures = 0usize;
    let mut report = |label: &str, n: usize, ulp: u32, total: usize| {
        if n != 0 {
            failures += 1;
        }
        println!(
            "{label:<52} {}  mismatches={n}/{total}  max_ulp={ulp}",
            if n == 0 { "PASS" } else { "FAIL" }
        );
    };

    let mut rng = Rng(seed);
    for kk in 2..=4usize {
        let rows = seqs * kk;
        let out_bytes = rows * d.value_dim * 2;
        let (out_p, out_c) = (g.alloc(out_bytes)?, g.alloc(out_bytes)?);
        let alloc_h =
            || -> Result<Vec<DevicePtr>> { (0..seqs).map(|_| g.alloc(d.h_numel * 4)).collect() };
        let h_init: Vec<Vec<f32>> = (0..seqs)
            .map(|_| (0..d.h_numel).map(|_| rng.next_f32() * 0.1).collect())
            .collect();
        let (h_ref, h_c) = (alloc_h()?, alloc_h()?);
        for b in 0..seqs {
            g.copy_h2d(&f32_bytes(&h_init[b]), h_ref[b])?;
            g.copy_h2d(&f32_bytes(&h_init[b]), h_c[b])?;
        }
        let hi: Vec<Vec<DevicePtr>> = (0..kk - 1).map(|_| alloc_h()).collect::<Result<_>>()?;
        let t_hi: Vec<DevicePtr> = hi.iter().map(|v| ptr_table(g, v)).collect::<Result<_>>()?;
        let (t_ref, t_c) = (ptr_table(g, &h_ref)?, ptr_table(g, &h_c)?);
        let mut np = vec![0usize; seqs];
        let (mut out_bad, mut out_ulp) = (0usize, 0u32);
        for _ in 0..rounds {
            let r = make_round(g, &d, kk, &mut rng)?;
            run_parent(g, parents[kk - 2], kk, &d, &r, t_ref, &t_hi, out_p)?;
            upload_pend(&np)?;
            // 2026-09-26: Eager or lazy form, drawn per round, so the two share the stash. As
            // the host does, a slot holding more rows than an eager launch folds is folded
            // by `gdn_carry_flush` first.
            let lazy = rng.next_f32() > 0.0;
            if !lazy && np.iter().any(|&p| p > ops::GDN_CARRY_EAGER_MAX_PENDING) {
                let deep: Vec<usize> = np
                    .iter()
                    .map(|&p| {
                        if p > ops::GDN_CARRY_EAGER_MAX_PENDING {
                            p
                        } else {
                            0
                        }
                    })
                    .collect();
                upload_pend(&deep)?;
                ops::gdn_carry_flush(
                    g,
                    flush,
                    t_c,
                    0,
                    stash,
                    0,
                    slot_tab,
                    pend,
                    0,
                    seq_floats as u32,
                    seqs as u32,
                    nv as u32,
                    1,
                    stream,
                )?;
                for p in np.iter_mut() {
                    if *p > ops::GDN_CARRY_EAGER_MAX_PENDING {
                        *p = 0;
                    }
                }
                upload_pend(&np)?;
            }
            ops::gdn_carry_wy(
                g,
                if lazy {
                    lazies[kk - 2]
                } else {
                    carries[kk - 2]
                },
                t_c,
                r.q,
                r.k,
                r.v,
                r.gate,
                r.beta,
                out_c,
                stash,
                slot_tab,
                pend,
                seq_floats as u32,
                seqs as u32,
                nk as u32,
                nv as u32,
                d.conv_dim as u32,
                d.conv_dim as u32,
                d.gb_stride as u32,
                KD as u32,
                flags,
                stream,
            )?;
            g.synchronize(stream)?;
            let (n, u) = diff_f32(
                &read_f32(g, out_c, out_bytes / 4)?,
                &read_f32(g, out_p, out_bytes / 4)?,
            );
            out_bad += n;
            out_ulp = out_ulp.max(u);
            // 2026-09-26: Commit a random accepted count per sequence: the parent's state
            // after na rows into the reference, and the pending counts by the kernels' rule.
            for b in 0..seqs {
                let na = 1 + (rng.next_f32().abs() * kk as f32) as usize % kk;
                if na < kk {
                    let v = read_f32(g, hi[na - 1][b], d.h_numel)?;
                    g.copy_h2d(&f32_bytes(&v), h_ref[b])?;
                }
                let kept = if !lazy || np[b] + kk > ops::GDN_CARRY_CAP {
                    0
                } else {
                    np[b]
                };
                np[b] = kept + na;
            }
        }
        report(
            &format!("K={kk} {rounds} rounds: outputs vs parent"),
            out_bad,
            out_ulp,
            rounds * out_bytes / 4,
        );
        upload_pend(&np)?;
        ops::gdn_carry_flush(
            g,
            flush,
            t_c,
            0,
            stash,
            0,
            slot_tab,
            pend,
            0,
            seq_floats as u32,
            seqs as u32,
            nv as u32,
            1,
            stream,
        )?;
        g.synchronize(stream)?;
        let (mut n, mut ulp) = (0, 0);
        for b in 0..seqs {
            let (dn, du) = diff_f32(
                &read_f32(g, h_c[b], d.h_numel)?,
                &read_f32(g, h_ref[b], d.h_numel)?,
            );
            n += dn;
            ulp = ulp.max(du);
        }
        report(
            &format!("K={kk} flushed state vs committed reference (pend {np:?})"),
            n,
            ulp,
            seqs * d.h_numel,
        );
    }

    println!(
        "gdn_carry_microtest: seqs={seqs} nk={nk} nv={nv} seed={seed} rounds={rounds}: {}",
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
