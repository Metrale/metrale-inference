// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The byte gate and arms of `glm5next_w4a16_gemv_pipe_bench`: today's dispatch,
//! the sweep points of `w4a16_gemv_pipe`, and the proof that every point writes `w4a16_gemv`'s
//! bytes per input row.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
#![allow(dead_code)]

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops::{self, W4a16GemvSegPoint, W4a16Seg};
use metrale_model_layers::weight_map::QuantizedWeight;

use super::fixture::{GROUPS, Pool, Rng, bf16_rows, read, segs, upload};

pub struct Kernels {
    pub gemv: KernelHandle,
    pub batch2: KernelHandle,
}

impl Kernels {
    pub fn load(g: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gemv: g.kernel("w4a16_gemv", "w4a16_gemv")?,
            batch2: g.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Arm {
    Old,
    New,
    Policy,
    At(u32),
}

/// 2026-10-21: The swept block caps: one and two resident waves (4 blocks per SM) and uncapped.
pub fn sweep_arms(sms: u32) -> Vec<Arm> {
    let w = sms * ops::W4A16_GEMV_PIPE_BLOCKS_PER_SM;
    vec![Arm::At(w), Arm::At(2 * w), Arm::At(u32::MAX)]
}

pub fn arm_name(arm: Arm, sms: u32) -> String {
    match arm {
        Arm::Old => "old".into(),
        Arm::New => "new dispatch".into(),
        Arm::Policy => "pipe (policy cap)".into(),
        Arm::At(u32::MAX) => "pipe cap=all".into(),
        Arm::At(cap) => format!("pipe cap={}xSM", cap / sms),
    }
}

/// 2026-10-21: Today's launch of `segs` over `m` rows (`glm5next_w4a16_dense` with
/// METRALE_GLM_W4A16_GEMV_SEG=1 and METRALE_GLM_W4A16_GEMV_ROWS >= 2).
pub fn old_launch(
    g: &dyn GpuBackend,
    kn: &Kernels,
    a: DevicePtr,
    sg: &[W4a16Seg],
    m: u32,
    k: u32,
    s: u64,
) -> Result<()> {
    if sg.len() > 1 || ops::w4a16_gemv_seg_lone(k) {
        let pt = if m == 1 {
            W4a16GemvSegPoint::Tile(1)
        } else {
            W4a16GemvSegPoint::Rows(m)
        };
        return ops::w4a16_gemv_seg_point(g, pt, a, sg, k, u32::MAX, s);
    }
    let w = &sg[0];
    if m == 1 {
        return ops::w4a16_gemv(g, kn.gemv, a, &w.weight, w.output, w.n, k, s);
    }
    ensure!(m == 2, "old_launch: {m} rows");
    KernelLaunch::new(g, kn.batch2)
        .grid([div_ceil(w.n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.weight.weight)
        .arg_ptr(w.weight.weight_scale)
        .arg_f32(w.weight.weight_scale_2)
        .arg_ptr(w.output)
        .arg_u32(w.n)
        .arg_u32(k)
        .launch(s)
}

/// 2026-10-21: The dispatch with METRALE_GLM_W4A16_GEMV_PIPE on (`glm5next_w4a16_seg`): every
/// launch on `w4a16_gemv_pipe` except a lone one-row projection from K 2048 up, which keeps
/// `w4a16_gemv`.
pub fn new_launch(
    g: &dyn GpuBackend,
    kn: &Kernels,
    a: DevicePtr,
    sg: &[W4a16Seg],
    m: u32,
    k: u32,
    s: u64,
) -> Result<()> {
    if m == 1 && sg.len() == 1 && !ops::w4a16_gemv_seg_lone(k) {
        return old_launch(g, kn, a, sg, m, k, s);
    }
    ops::w4a16_gemv_pipe(g, a, sg, m, k, g.sm_count()?, s)
}

/// 2026-10-21: Input rows of a gate case: random, signed zeros, the largest BF16 magnitudes of
/// both signs (overflow to inf, inf - inf), sparse large values among zeros.
fn inputs(rng: &mut Rng, m: usize, k: usize) -> Vec<(&'static str, Vec<u8>)> {
    let fill = |rng: &mut Rng, f: &dyn Fn(u32) -> u16| -> Vec<u8> {
        (0..m * k).flat_map(|_| f(rng.next()).to_le_bytes()).collect()
    };
    vec![
        ("random", bf16_rows(rng, m, k)),
        ("signed zeros", fill(rng, &|u| ((u & 1) << 15) as u16)),
        ("max magnitude", fill(rng, &|u| (((u & 1) << 15) | 0x7F7F) as u16)),
        (
            "sparse large",
            fill(rng, &|u| {
                if u % 61 == 0 {
                    (((u >> 8) & 1) << 15 | 0x7E00 | ((u >> 9) & 0x7F)) as u16
                } else {
                    (((u >> 8) & 1) << 15) as u16
                }
            }),
        ),
    ]
}

/// 2026-10-21: `w4a16_gemv` of row r of `a` (each row alone), per segment.
fn reference(
    g: &dyn GpuBackend,
    s: u64,
    kn: &Kernels,
    a_host: &[u8],
    ws: &[QuantizedWeight],
    ns: &[u32],
    m: usize,
    k: u32,
) -> Result<Vec<Vec<Vec<u8>>>> {
    let ku = k as usize;
    let row = g.alloc(ku * 2)?;
    let one = g.alloc(11264 * 2)?;
    let mut want = vec![vec![Vec::new(); m]; ns.len()];
    for r in 0..m {
        g.copy_h2d(&a_host[r * ku * 2..(r + 1) * ku * 2], row)?;
        for (i, w) in ws.iter().enumerate() {
            ops::w4a16_gemv(g, kn.gemv, row, w, one, ns[i], k, s)?;
            want[i][r] = read(g, s, one, ns[i] as usize * 2)?;
        }
    }
    g.free(row)?;
    g.free(one)?;
    Ok(want)
}

/// 2026-10-21: Every swept point at every cap, and today's dispatch, against the reference, on
/// every group, at 1 and 2 rows, on every input case; plus a weight with adversarial scale bytes
/// (every E4M3 code, both signs, subnormals, NaN).
pub fn byte_gate(
    g: &dyn GpuBackend,
    s: u64,
    kn: &Kernels,
    pool: &Pool,
    outs: &[DevicePtr],
    rng: &mut Rng,
) -> Result<()> {
    let sms = g.sm_count()?;
    let adv_scale: Vec<u8> = (0..11264 * 11264 / 16).map(|_| rng.next() as u8).collect();
    let adv = upload(g, &adv_scale)?;
    let mut checked = 0usize;
    for (gi, (name, ns, k, _)) in GROUPS.iter().enumerate() {
        let ns = *ns;
        for m in [1u32, 2] {
            for (case, a_host) in inputs(rng, m as usize, *k as usize) {
                for adversarial in [false, true] {
                    let mut ws: Vec<_> = (0..ns.len()).map(|i| pool.weight(gi + i, ns[i], *k)).collect();
                    if adversarial {
                        ws[0].weight_scale = adv;
                    }
                    let ad = upload(g, &a_host)?;
                    let want = reference(g, s, kn, &a_host, &ws, ns, m as usize, *k)?;
                    let mut arms = vec![Arm::Old, Arm::New, Arm::Policy];
                    arms.extend([1, 7, sms, sms * 4, u32::MAX].map(Arm::At));
                    for arm in arms {
                        for &o in outs {
                            g.memset(o, 0xA5, 4 * 11264 * 2)?;
                        }
                        let sg = segs(&ws, outs, ns);
                        match arm {
                            Arm::Old => old_launch(g, kn, ad, &sg, m, *k, s)?,
                            Arm::New => new_launch(g, kn, ad, &sg, m, *k, s)?,
                            Arm::Policy => ops::w4a16_gemv_pipe(g, ad, &sg, m, *k, sms, s)?,
                            Arm::At(cap) => ops::w4a16_gemv_pipe_at(g, ad, &sg, m, *k, cap, s)?,
                        }
                        for (i, &n) in ns.iter().enumerate() {
                            let got = read(g, s, outs[i], m as usize * n as usize * 2)?;
                            for r in 0..m as usize {
                                let rowb = &got[r * n as usize * 2..(r + 1) * n as usize * 2];
                                if let Some(at) = rowb
                                    .chunks(2)
                                    .zip(want[i][r].chunks(2))
                                    .position(|(x, y)| x != y)
                                {
                                    anyhow::bail!(
                                        "{name} M={m} {case} adv={adversarial} arm={arm:?} seg {i} row {r}: \
                                         output {at} differs from w4a16_gemv ({:02x?} vs {:02x?})",
                                        &rowb[at * 2..at * 2 + 2],
                                        &want[i][r][at * 2..at * 2 + 2]
                                    );
                                }
                                checked += n as usize;
                            }
                        }
                    }
                    g.free(ad)?;
                }
            }
        }
    }
    g.free(adv)?;
    println!(
        "byte gate: {checked} outputs of the policy and every cap (and today's dispatch) equal w4a16_gemv \
         per row (13 groups, M 1/2, 4 input cases, adversarial scales)"
    );
    Ok(())
}

/// 2026-10-21: The gate's comparison can fail: a changed scale byte changes exactly the outputs
/// it feeds, in every row; a changed element of row 1 changes row 1 only.
pub fn negative_controls(
    g: &dyn GpuBackend,
    s: u64,
    _kn: &Kernels,
    pool: &Pool,
    outs: &[DevicePtr],
    rng: &mut Rng,
) -> Result<()> {
    let sms = g.sm_count()?;
    let (ns, k) = (&[768u32, 768][..], 4096u32);
    let ku = k as usize;
    let ws: Vec<_> = (0..2).map(|i| pool.weight(i, ns[i], k)).collect();
    let mut a2 = bf16_rows(rng, 2, ku);
    let ad = upload(g, &a2)?;
    let run = |ad: DevicePtr| -> Result<Vec<u8>> {
        ops::w4a16_gemv_pipe(g, ad, &segs(&ws, outs, ns), 2, k, sms, s)?;
        read(g, s, outs[1], 2 * 768 * 2)
    };
    let base = run(ad)?;
    a2[(ku + 100) * 2..(ku + 100) * 2 + 2].copy_from_slice(&0x4280u16.to_le_bytes());
    g.copy_h2d(&a2, ad)?;
    let moved = run(ad)?;
    ensure!(
        moved[..768 * 2] == base[..768 * 2] && moved[768 * 2..] != base[768 * 2..],
        "negative control: a changed row-1 element must change row 1 only"
    );
    let at = ws[1].weight_scale.offset(4096 / 16 * 7 + 12);
    let mut b0 = [0u8; 1];
    g.copy_d2h(at, &mut b0)?;
    g.copy_h2d(&[0x7E], at)?;
    let scaled = run(ad)?;
    g.copy_h2d(&b0, at)?;
    let diff: Vec<usize> = scaled
        .chunks(2)
        .zip(moved.chunks(2))
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i)
        .collect();
    ensure!(
        diff == [7, 768 + 7],
        "negative control: a changed scale byte must change output 7 of both rows only, got {diff:?}"
    );
    g.free(ad)?;
    println!("negative controls fire: row isolation; one scale byte -> exactly its outputs");
    Ok(())
}
