// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Part 1 of `glm5next_expert_tp_bench`: the `_k64` twins byte for byte against the
//! plain entries at the padded K. The down projection of a 704-wide expert slice (`[4096, 704]`
//! NVFP4, natural layout) on the `_k64` quantizer, slot GEMV and sweeps must equal the plain
//! quantizer, slot GEMV and sweeps at K 768 over the same weights zero-padded to 768 columns and
//! the same activations zero-padded to 768, at 1, 4, 8 and 16 rows. A control zeroes the slice's
//! last 64 columns and must change the output (the half chunk is read).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: returns an error on the first mismatch.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mlp::weights::Nvfp4Proj;

use crate::device::*;

/// 2026-10-10: The slice width under test and its 128-padded twin.
pub const K64: usize = 704;
pub const KPAD: usize = 768;

/// 2026-10-10: A pointer table over host experts (`[n, k]` NVFP4 each); global ids `0..LOCAL`.
pub fn table_from(
    gpu: &dyn GpuBackend,
    experts: &[(Vec<u8>, Vec<u8>)],
    n: usize,
    k: usize,
    gs: f32,
) -> Result<Table> {
    let mut projs = Vec::with_capacity(experts.len());
    for (p, s) in experts {
        ensure!(
            p.len() == n * k / 2 && s.len() == n * k / 16,
            "expert shape"
        );
        projs.push(Nvfp4Proj {
            packed: up(gpu, p)?,
            scale: up(gpu, s)?,
            scale_2: 0.01,
            input_scale: Some(gs),
        });
    }
    let ptrs = |f: &dyn Fn(&Nvfp4Proj) -> u64| -> Vec<u8> {
        (0..EXPERTS)
            .flat_map(|e| projs.get(e).map_or(0, f).to_le_bytes())
            .collect()
    };
    let packed = up(gpu, &ptrs(&|p| p.packed.0))?;
    let scale = up(gpu, &ptrs(&|p| p.scale.0))?;
    let s2: Vec<u8> = (0..EXPERTS)
        .flat_map(|e| projs.get(e).map_or(0.0f32, |p| p.scale_2).to_le_bytes())
        .collect();
    Ok(Table {
        packed,
        scale,
        scale2: up(gpu, &s2)?,
        projs,
        bytes_per_expert: n * k / 2 + n * k / 16,
    })
}

/// 2026-10-10: An `[n, k]` NVFP4 weight padded to `kp` columns with zero codes and zero scales.
pub fn pad_cols(p: &[u8], s: &[u8], n: usize, k: usize, kp: usize) -> (Vec<u8>, Vec<u8>) {
    let mut out_p = Vec::with_capacity(n * kp / 2);
    let mut out_s = Vec::with_capacity(n * kp / 16);
    for (rp, rs) in p.chunks_exact(k / 2).zip(s.chunks_exact(k / 16)) {
        out_p.extend_from_slice(rp);
        out_p.resize(out_p.len() + (kp - k) / 2, 0);
        out_s.extend_from_slice(rs);
        out_s.resize(out_s.len() + (kp - k) / 16, 0);
    }
    (out_p, out_s)
}

/// 2026-10-10: Quantize `rows` BF16 rows of width `k` with `kern` into a scratch of row width
/// `kp` (codes `kp / 2`, scales `kp / 16` per row).
fn quantize_into(
    gpu: &dyn GpuBackend,
    kern: KernelHandle,
    x: DevicePtr,
    rows: usize,
    k: usize,
    kp: usize,
) -> Result<Act> {
    let a = Act {
        aq: gpu.alloc(rows * kp / 2)?,
        as_: gpu.alloc(rows * kp / 16)?,
        ag: gpu.alloc(rows * 4)?,
    };
    gpu.memset(a.aq, 0xA5, rows * kp / 2)?;
    gpu.memset(a.as_, 0xA5, rows * kp / 16)?;
    KernelLaunch::new(gpu, kern)
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(a.aq)
        .arg_ptr(a.as_)
        .arg_ptr(a.ag)
        .arg_u32(k as u32)
        .arg_f32(GS_DOWN)
        .launch(0)?;
    Ok(a)
}

/// 2026-10-10: `rows` rows of `TOP_K` distinct experts each, all among the table's `LOCAL`, so
/// every slot is computed.
fn route_local(gpu: &dyn GpuBackend, rng: &mut Rng, rows: usize) -> Result<Route> {
    let mut host = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let mut row: Vec<i32> = Vec::with_capacity(TOP_K);
        while row.len() < TOP_K {
            let e = (rng.next() % LOCAL as u64) as i32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        host.extend(row);
    }
    let b: Vec<u8> = host.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok(Route {
        rows,
        ids: up(gpu, &b)?,
        u_eid: gpu.alloc(rows * TOP_K * 4)?,
        u_slot: gpu.alloc(rows * TOP_K * rows * 4)?,
        host,
    })
}

/// 2026-10-10: The plain and `_k64` kernels, as the bench's launcher sets.
pub struct Twins {
    pub plain: Kern,
    pub k64: Kern,
}

pub fn run(gpu: &dyn GpuBackend, tw: &Twins, rng: &mut Rng) -> Result<()> {
    let n = HIDDEN;
    let natural: Vec<(Vec<u8>, Vec<u8>)> = (0..LOCAL)
        .map(|_| {
            let p: Vec<u8> = (0..n * K64 / 2).map(|_| rng.next() as u8).collect();
            let s: Vec<u8> = (0..n * K64 / 16)
                .map(|_| 0x30 + (rng.next() % 9) as u8)
                .collect();
            (p, s)
        })
        .collect();
    let padded: Vec<_> = natural
        .iter()
        .map(|(p, s)| pad_cols(p, s, n, K64, KPAD))
        .collect();
    // 2026-10-10: Control: the slice with its half chunk (columns 640..704) zeroed.
    let no_tail: Vec<_> = natural
        .iter()
        .map(|(p, s)| {
            let (mut p, mut s) = (p.clone(), s.clone());
            for r in 0..n {
                p[r * K64 / 2 + 320..(r + 1) * K64 / 2].fill(0);
                s[r * K64 / 16 + 40..(r + 1) * K64 / 16].fill(0);
            }
            (p, s)
        })
        .collect();
    let t_nat = table_from(gpu, &natural, n, K64, GS_DOWN)?;
    let t_pad = table_from(gpu, &padded, n, KPAD, GS_DOWN)?;
    let t_ctl = table_from(gpu, &no_tail, n, K64, GS_DOWN)?;

    let slots_max = MAX_ROWS * TOP_K;
    let act: Vec<f32> = (0..slots_max * K64).map(|_| rng.unit() * 10.0).collect();
    let act_pad: Vec<f32> = act
        .chunks_exact(K64)
        .flat_map(|r| {
            r.iter()
                .copied()
                .chain(std::iter::repeat_n(0.0, KPAD - K64))
        })
        .collect();
    let (ad, ad_pad) = (up_bf16(gpu, &act)?, up_bf16(gpu, &act_pad)?);
    let a64 = quantize_into(gpu, tw.k64.quant, ad, slots_max, K64, KPAD)?;
    let aref = quantize_into(gpu, tw.plain.quant, ad_pad, slots_max, KPAD, KPAD)?;
    for (name, p, q, bytes) in [
        ("codes", a64.aq, aref.aq, slots_max * KPAD / 2),
        ("scales", a64.as_, aref.as_, slots_max * KPAD / 16),
        ("globals", a64.ag, aref.ag, slots_max * 4),
    ] {
        ensure!(
            read(gpu, p, bytes)? == read(gpu, q, bytes)?,
            "k64 quantizer {name} differ from the plain quantizer over zero-padded rows"
        );
    }
    println!("byte gate quant: k64 at K {K64} == plain at K {KPAD} over zero-padded rows");

    let out_bytes = slots_max * n * 2;
    let (o1, o2) = (gpu.alloc(out_bytes)?, gpu.alloc(out_bytes)?);
    let dn64 = (n, K64, 1);
    let dnp = (n, KPAD, 1);
    for rows in [1usize, 4, 8, 16] {
        let r = route_local(gpu, rng, rows)?;
        row_union(gpu, &tw.plain, &r, 0)?;
        let nb = rows * TOP_K * n * 2;
        let pair = |a: &dyn Fn(DevicePtr) -> Result<()>, b: &dyn Fn(DevicePtr) -> Result<()>| {
            gpu.memset(o1, 0, out_bytes)?;
            gpu.memset(o2, 0, out_bytes)?;
            a(o1)?;
            b(o2)?;
            Ok::<_, anyhow::Error>((read(gpu, o1, nb)?, read(gpu, o2, nb)?))
        };
        let (got, want) = pair(
            &|o| slots_gemv(gpu, &tw.k64, &a64, &r, &t_nat, o, dn64),
            &|o| slots_gemv(gpu, &tw.plain, &aref, &r, &t_pad, o, dnp),
        )?;
        ensure!(
            got.iter().any(|&b| b != 0),
            "rows={rows}: k64 slots wrote nothing"
        );
        ensure!(got == want, "rows={rows}: k64 slot GEMV differs");
        let (sw, _) = pair(
            &|o| sweep_gemv(gpu, &tw.k64, &a64, &r, &[&t_nat], &[o], dn64, 0),
            &|o| sweep_gemv(gpu, &tw.plain, &aref, &r, &[&t_pad], &[o], dnp, 0),
        )?;
        ensure!(sw == want, "rows={rows}: k64 sweep differs");
        let (ctl, _) = pair(
            &|o| slots_gemv(gpu, &tw.k64, &a64, &r, &t_ctl, o, dn64),
            &|o| slots_gemv(gpu, &tw.plain, &aref, &r, &t_pad, o, dnp),
        )?;
        ensure!(
            ctl != want,
            "rows={rows}: the zeroed half chunk did not change the output"
        );
        println!(
            "byte gate rows={rows:2}: k64 slot and sweep == plain at K {KPAD}; control (half \
             chunk zeroed) differs"
        );
    }
    Ok(())
}
