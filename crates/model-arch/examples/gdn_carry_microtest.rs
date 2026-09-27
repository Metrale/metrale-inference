// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Kernel-level check of the carried-state GDN verify
//! (`kernels/gb10/common/gated_delta_rule_carry.cu`) against the parent WY kernels.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 on any mismatch below; each check prints its mismatch count and max ULP delta.
//!
//! One GDN layer, random H / q / k / v / gate / beta over `SEQS` sequences on scattered
//! slots, two verify rounds. For K = 2, 3, 4 it runs the parent `gated_delta_rule_wy{K}`
//! (table form) and `gdn_carry_wy{K}`, then checks:
//!
//! * round-1 `output` bit-equal, the carry kernel leaves every H alone and sets every
//!   engaged word;
//! * for na in 1..=K: `gdn_carry_flush` with `pend = na` leaves H bit-equal to the parent's
//!   Hi(na-1) (na < K) or its final H (na == K);
//! * for na in 1..=K: a round-2 carry verify that starts from the round-1 H with
//!   `pend = na` gives the round-2 output of a parent verify that starts from the
//!   committed state, bit for bit, and leaves H at that committed state.
//!
//!   cargo run -p metrale-model-arch --release --features gpu-examples \
//!       --example gdn_carry_microtest
//!
//! Env: SEQS (default 5), SEED (default 1), NK/NV (default 16/32).

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
fn read_u32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
    let flush = g.kernel("gated_delta_rule_carry", "gdn_carry_flush")?;

    // 2026-09-26: Sequence b lives on slot 2b + 1, so the slot indirection is exercised.
    let n_slots = 2 * seqs + 2;
    let slots: Vec<u32> = (0..seqs as u32).map(|b| 2 * b + 1).collect();
    let slot_tab = upload_bytes(g, &u32_bytes(&slots))?;
    let seq_floats = ops::gdn_carry_seq_floats(nv, KD, VD);
    let stash = g.alloc(n_slots * seq_floats * 4)?;
    let pend = g.alloc(n_slots * 4)?;
    let flags = g.alloc(SLAB_ENTRIES * 4)?;
    let set_pend = |na: u32| -> Result<()> {
        let mut v = vec![0u32; n_slots];
        for &s in &slots {
            v[s as usize] = na;
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
        let r1 = make_round(g, &d, kk, &mut rng)?;
        let r2 = make_round(g, &d, kk, &mut rng)?;
        let h_init: Vec<Vec<f32>> = (0..seqs)
            .map(|_| (0..d.h_numel).map(|_| rng.next_f32() * 0.1).collect())
            .collect();
        let alloc_h =
            || -> Result<Vec<DevicePtr>> { (0..seqs).map(|_| g.alloc(d.h_numel * 4)).collect() };
        let load = |ptrs: &[DevicePtr], vals: &[Vec<f32>]| -> Result<()> {
            for (p, v) in ptrs.iter().zip(vals) {
                g.copy_h2d(&f32_bytes(v), *p)?;
            }
            Ok(())
        };
        let read_all = |ptrs: &[DevicePtr]| -> Result<Vec<Vec<f32>>> {
            ptrs.iter().map(|&p| read_f32(g, p, d.h_numel)).collect()
        };
        let out_bytes = rows * d.value_dim * 2;
        let (out_p, out_c) = (g.alloc(out_bytes)?, g.alloc(out_bytes)?);

        // 2026-09-26: Parent round 1 from h_init.
        let h_p = alloc_h()?;
        load(&h_p, &h_init)?;
        let hi_p: Vec<Vec<DevicePtr>> = (0..kk - 1).map(|_| alloc_h()).collect::<Result<_>>()?;
        let t_hi: Vec<DevicePtr> = hi_p
            .iter()
            .map(|v| ptr_table(g, v))
            .collect::<Result<_>>()?;
        run_parent(
            g,
            parents[kk - 2],
            kk,
            &d,
            &r1,
            ptr_table(g, &h_p)?,
            &t_hi,
            out_p,
        )?;
        let out1_parent = read_f32(g, out_p, out_bytes / 4)?;
        let committed: Vec<Vec<Vec<f32>>> = (1..=kk)
            .map(|na| {
                if na == kk {
                    read_all(&h_p)
                } else {
                    read_all(&hi_p[na - 1])
                }
            })
            .collect::<Result<_>>()?;

        // 2026-09-26: Carry round 1 from h_init with nothing pending.
        let h_c = alloc_h()?;
        let t_h_c = ptr_table(g, &h_c)?;
        let carry_round = |r: &Round, na: u32, out: DevicePtr| -> Result<()> {
            set_pend(na)?;
            g.memset(flags, 0, SLAB_ENTRIES * 4)?;
            ops::gdn_carry_wy(
                g,
                carries[kk - 2],
                t_h_c,
                r.q,
                r.k,
                r.v,
                r.gate,
                r.beta,
                out,
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
            g.synchronize(stream)
        };
        load(&h_c, &h_init)?;
        carry_round(&r1, 0, out_c)?;
        let (n, ulp) = diff_f32(&read_f32(g, out_c, out_bytes / 4)?, &out1_parent);
        report(
            &format!("K={kk} round-1 output vs parent"),
            n,
            ulp,
            out_bytes / 4,
        );
        let now = read_all(&h_c)?;
        let bad: usize = now.iter().zip(&h_init).map(|(a, b)| diff_f32(a, b).0).sum();
        report(
            &format!("K={kk} carry verify wrote no state"),
            bad,
            0,
            seqs * d.h_numel,
        );
        let fl = read_u32(g, flags, SLAB_ENTRIES)?;
        let unset = (0..SLAB_ENTRIES)
            .filter(|&b| fl[b] != u32::from(b < seqs))
            .count();
        report(
            &format!("K={kk} engaged words = batch positions"),
            unset,
            0,
            SLAB_ENTRIES,
        );

        for na in 1..=kk {
            // 2026-09-26: Standalone fold of the round-1 stash, which the round-1 carry
            // writes again first (the previous na's round 2 overwrote it).
            load(&h_c, &h_init)?;
            carry_round(&r1, 0, out_c)?;
            load(&h_c, &h_init)?;
            set_pend(na as u32)?;
            ops::gdn_carry_flush(
                g,
                flush,
                t_h_c,
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
            let now = read_all(&h_c)?;
            let (mut n, mut ulp) = (0, 0);
            for (a, b) in now.iter().zip(&committed[na - 1]) {
                let (dn, du) = diff_f32(a, b);
                n += dn;
                ulp = ulp.max(du);
            }
            report(
                &format!("K={kk} flush na={na} vs parent state"),
                n,
                ulp,
                seqs * d.h_numel,
            );

            // 2026-09-26: Round 2: parent from the committed state against carry from h_init
            // with na pending. The round-1 carry runs again first to restore its stash.
            let h_q = alloc_h()?;
            load(&h_q, &committed[na - 1])?;
            let hi_q: Vec<Vec<DevicePtr>> =
                (0..kk - 1).map(|_| alloc_h()).collect::<Result<_>>()?;
            let t_hi_q: Vec<DevicePtr> = hi_q
                .iter()
                .map(|v| ptr_table(g, v))
                .collect::<Result<_>>()?;
            run_parent(
                g,
                parents[kk - 2],
                kk,
                &d,
                &r2,
                ptr_table(g, &h_q)?,
                &t_hi_q,
                out_p,
            )?;
            let out2_parent = read_f32(g, out_p, out_bytes / 4)?;
            load(&h_c, &h_init)?;
            carry_round(&r1, 0, out_c)?;
            carry_round(&r2, na as u32, out_c)?;
            let (n, ulp) = diff_f32(&read_f32(g, out_c, out_bytes / 4)?, &out2_parent);
            report(
                &format!("K={kk} na={na} round-2 output vs parent"),
                n,
                ulp,
                out_bytes / 4,
            );
            let now = read_all(&h_c)?;
            let (mut n, mut ulp) = (0, 0);
            for (a, b) in now.iter().zip(&committed[na - 1]) {
                let (dn, du) = diff_f32(a, b);
                n += dn;
                ulp = ulp.max(du);
            }
            report(
                &format!("K={kk} na={na} round-2 H = committed"),
                n,
                ulp,
                seqs * d.h_numel,
            );
        }
    }

    println!(
        "gdn_carry_microtest: seqs={seqs} nk={nk} nv={nv} seed={seed}: {}",
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
