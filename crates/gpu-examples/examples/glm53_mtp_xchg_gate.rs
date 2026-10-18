// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Bit-identity gate of the GLM-5.3 draft head's device exchange
//! (`glm5next_mtp_xchg_pack` / `_pick`, `METRALE_GLM_MTP_XCHG_DEVICE`) against the host exchange
//! (`glm5next_mtp_head::head_xchg_pack` / `head_xchg_pick`), on one GPU with the all-reduce
//! simulated on the host (each lane's contributions summed in FP32 and rounded to BF16, as
//! `bf16_add_rank_sum` does; exact here, since one rank writes each lane).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Fails unless, for every case at 2, 3 and 4 ranks, every rank's packed bytes equal the host
//!   pack's byte for byte and the device pick equals the host pick.
//! - Cases: random shards; a tie on the maximum between two ranks (the lower rank wins); every
//!   rank at the same logit; negative logits; +0.0 against -0.0; a NaN logit on one rank; the
//!   largest GLM-5.3 token id (154,855) and ids that use all three digits.
//! - Negative control: a pick on a buffer with rank 1's logit lane raised must change the
//!   answer, so the comparison can fail.
//! - 2026-10-11: The log-probability exchange (`glm5next_mtp_xchg_pack_lp` / `_pick_lp`): every
//!   rank's packed bytes equal `head_xchg_pack_lp`'s, the device pick's id equals the host's
//!   and its log-probability is within 1e-4 of the host's and within 1e-2 of a direct
//!   whole-vocab log-softmax.
//!
//! Run (GB10, a few MB of device memory):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p metrale-gpu-examples --features cuda,gpu-examples \
//!     --example glm53_mtp_xchg_gate

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mtp_head::{
    HEAD_XCHG_MAX_RANKS, head_xchg_bytes, head_xchg_lp_bytes, head_xchg_pack, head_xchg_pack_lp,
    head_xchg_pick, head_xchg_pick_lp,
};

/// 2026-10-11: One rank's shard: the vocab rows it starts at, its logits, its local argmax.
struct Shard {
    v0: usize,
    logits: Vec<bf16>,
    local: usize,
}

fn lane_f32(b: &[u8], i: usize) -> f32 {
    bf16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32()
}

/// 2026-10-11: The all-reduce: per lane, the FP32 sum of every rank's lane rounded to BF16.
fn all_reduce(packs: &[Vec<u8>]) -> Vec<u8> {
    let n = packs[0].len() / 2;
    let mut out = vec![0u8; n * 2];
    for i in 0..n {
        let s: f32 = packs.iter().map(|p| lane_f32(p, i)).sum();
        out[2 * i..2 * i + 2].copy_from_slice(&bf16::from_f32(s).to_bits().to_le_bytes());
    }
    out
}

fn shards(world: usize, case: usize, rng: &mut u64) -> Vec<Shard> {
    let mut next = || {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *rng
    };
    let len = 512usize;
    (0..world)
        .map(|r| {
            let mut logits: Vec<bf16> = (0..len)
                .map(|_| bf16::from_f32(((next() % 20_000) as f32 - 10_000.0) / 997.0))
                .collect();
            let mut local = (next() as usize) % len;
            let v0 = match case {
                6 => 154_855 - local - (world - 1 - r) * len,
                _ => r * 51_619 + r % 2,
            };
            let top = match case {
                1 if r == 1 || r == world - 1 => 30.0,
                2 => 12.5,
                3 => -0.75 - r as f32,
                4 => {
                    if r == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                5 if r == world - 1 => f32::NAN,
                _ => 20.0 + r as f32 * 0.125,
            };
            if case == 3 {
                for l in logits.iter_mut() {
                    *l = bf16::from_f32(-50.0);
                }
            }
            if case == 4 {
                for l in logits.iter_mut() {
                    *l = bf16::from_f32(-1.0);
                }
            }
            logits[local] = bf16::from_f32(top);
            if case == 0 {
                local = (0..len)
                    .max_by(|&a, &b| logits[a].to_f32().total_cmp(&logits[b].to_f32()))
                    .unwrap();
            }
            Shard { v0, logits, local }
        })
        .collect()
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let pack_k = g.kernel("glm5next_mtp_xchg", "glm5next_mtp_xchg_pack")?;
    let pick_k = g.kernel("glm5next_mtp_xchg", "glm5next_mtp_xchg_pick")?;
    let logits_d = g.alloc(512 * 2)?;
    let arg_d = g.alloc(4)?;
    let xchg_d = g.alloc(8 * HEAD_XCHG_MAX_RANKS)?;
    let out_d = g.alloc(4)?;
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut checked = 0usize;
    for world in 2..=HEAD_XCHG_MAX_RANKS {
        for case in 0..7 {
            let sh = shards(world, case, &mut rng);
            let mut host_packs = Vec::new();
            let mut dev_packs = Vec::new();
            for (r, s) in sh.iter().enumerate() {
                let lb = s.logits[s.local].to_bits().to_le_bytes();
                host_packs.push(head_xchg_pack(world, r, lb, s.v0 + s.local)?);
                let bytes: Vec<u8> = s
                    .logits
                    .iter()
                    .flat_map(|x| x.to_bits().to_le_bytes())
                    .collect();
                g.copy_h2d(&bytes, logits_d)?;
                g.copy_h2d(&(s.local as u32).to_le_bytes(), arg_d)?;
                KernelLaunch::new(g, pack_k)
                    .grid([1, 1, 1])
                    .block([32, 1, 1])
                    .arg_ptr(arg_d)
                    .arg_ptr(logits_d)
                    .arg_ptr(xchg_d)
                    .arg_u32(world as u32)
                    .arg_u32(r as u32)
                    .arg_u32(s.v0 as u32)
                    .launch(g.default_stream())?;
                let mut got = vec![0u8; head_xchg_bytes(world)];
                g.copy_d2h(xchg_d, &mut got)?;
                ensure!(
                    got == *host_packs.last().unwrap(),
                    "world {world} case {case} rank {r}: device pack {got:?} != host {:?}",
                    host_packs.last().unwrap()
                );
                dev_packs.push(got);
            }
            let summed = all_reduce(&dev_packs);
            let want = head_xchg_pick(world, &summed)?;
            g.copy_h2d(&summed, xchg_d)?;
            KernelLaunch::new(g, pick_k)
                .grid([1, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(xchg_d)
                .arg_u32(world as u32)
                .arg_ptr(out_d)
                .launch(g.default_stream())?;
            let mut out = [0u8; 4];
            g.copy_d2h(out_d, &mut out)?;
            let got = u32::from_le_bytes(out);
            ensure!(
                got == want,
                "world {world} case {case}: device pick {got} != host {want}"
            );
            checked += 1;
            // 2026-10-11: Negative control: raise rank 1's logit lane above every other, so
            // the pick must move to rank 1's id unless it already was it.
            if case == 0 {
                let mut raised = summed.clone();
                raised[2..4].copy_from_slice(&bf16::from_f32(1.0e4).to_bits().to_le_bytes());
                let r1 = head_xchg_pick(world, &raised)?;
                ensure!(
                    r1 as usize == sh[1].v0 + sh[1].local,
                    "control: the raised rank-1 lane did not win ({r1})"
                );
                g.copy_h2d(&raised, xchg_d)?;
                KernelLaunch::new(g, pick_k)
                    .grid([1, 1, 1])
                    .block([32, 1, 1])
                    .arg_ptr(xchg_d)
                    .arg_u32(world as u32)
                    .arg_ptr(out_d)
                    .launch(g.default_stream())?;
                g.copy_d2h(out_d, &mut out)?;
                ensure!(
                    u32::from_le_bytes(out) == r1,
                    "control: device pick did not follow"
                );
            }
            println!("  world {world} case {case}: pack x{world} and pick {got} identical");
        }
    }
    checked += lp_cases(g, &mut rng)?;
    println!("\nglm53 mtp xchg gate PASS ({checked} cases)");
    Ok(())
}

/// 2026-10-11: The log-probability exchange against the host form, at 2, 3 and 4 ranks.
fn lp_cases(g: &dyn GpuBackend, rng: &mut u64) -> Result<usize> {
    let pack_k = g.kernel("glm5next_mtp_xchg", "glm5next_mtp_xchg_pack_lp")?;
    let pick_k = g.kernel("glm5next_mtp_xchg", "glm5next_mtp_xchg_pick_lp")?;
    let logits_d = g.alloc(512 * 2)?;
    let arg_d = g.alloc(4)?;
    let lp_d = g.alloc(4)?;
    let xchg_d = g.alloc(12 * HEAD_XCHG_MAX_RANKS)?;
    let lse = |xs: &[f32]| {
        let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        m + xs.iter().map(|x| (x - m).exp()).sum::<f32>().ln()
    };
    let mut checked = 0;
    for world in 2..=HEAD_XCHG_MAX_RANKS {
        let sh = shards(world, 0, rng);
        let mut packs = Vec::new();
        for (r, s) in sh.iter().enumerate() {
            let xs: Vec<f32> = s.logits.iter().map(|x| x.to_f32()).collect();
            let top = xs[s.local];
            // 2026-10-11: What `argmax_bf16_batch_lp` hands the pack: the shard's log-softmax
            // at its argmax.
            let lp_local = top - lse(&xs);
            let host = head_xchg_pack_lp(
                world,
                r,
                s.logits[s.local].to_bits().to_le_bytes(),
                s.v0 + s.local,
                top - lp_local,
            )?;
            let bytes: Vec<u8> = s
                .logits
                .iter()
                .flat_map(|x| x.to_bits().to_le_bytes())
                .collect();
            g.copy_h2d(&bytes, logits_d)?;
            g.copy_h2d(&(s.local as u32).to_le_bytes(), arg_d)?;
            g.copy_h2d(&lp_local.to_le_bytes(), lp_d)?;
            KernelLaunch::new(g, pack_k)
                .grid([1, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(arg_d)
                .arg_ptr(logits_d)
                .arg_ptr(lp_d)
                .arg_ptr(xchg_d)
                .arg_u32(world as u32)
                .arg_u32(r as u32)
                .arg_u32(s.v0 as u32)
                .launch(g.default_stream())?;
            let mut got = vec![0u8; head_xchg_lp_bytes(world)];
            g.copy_d2h(xchg_d, &mut got)?;
            ensure!(
                got == host,
                "lp world {world} rank {r}: device pack {got:?} != host {host:?}"
            );
            packs.push(got);
        }
        let summed = all_reduce(&packs);
        let (want_id, want_lp) = head_xchg_pick_lp(world, &summed)?;
        g.copy_h2d(&summed, xchg_d)?;
        KernelLaunch::new(g, pick_k)
            .grid([1, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(xchg_d)
            .arg_u32(world as u32)
            .arg_ptr(arg_d)
            .arg_ptr(lp_d)
            .launch(g.default_stream())?;
        let (mut id, mut lp) = ([0u8; 4], [0u8; 4]);
        g.copy_d2h(arg_d, &mut id)?;
        g.copy_d2h(lp_d, &mut lp)?;
        let (id, lp) = (u32::from_le_bytes(id), f32::from_le_bytes(lp));
        let all: Vec<f32> = sh
            .iter()
            .flat_map(|s| s.logits.iter().map(|x| x.to_f32()))
            .collect();
        let win = sh
            .iter()
            .map(|s| s.logits[s.local].to_f32())
            .fold(f32::NEG_INFINITY, f32::max);
        let direct = win - lse(&all);
        ensure!(
            id == want_id,
            "lp world {world}: device id {id} != host {want_id}"
        );
        ensure!(
            (lp - want_lp).abs() < 1e-4 && (lp - direct).abs() < 1e-2,
            "lp world {world}: device {lp}, host {want_lp}, direct {direct}"
        );
        println!("  lp world {world}: id {id} lp {lp:.5} (direct {direct:.5})");
        checked += 1;
    }
    Ok(checked)
}
