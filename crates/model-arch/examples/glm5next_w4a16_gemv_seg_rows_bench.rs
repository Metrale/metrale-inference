// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The W4A16 dense projections of GLM-5.3 (rank 0 of TP=3) at 2..=4 rows: today's
//! row-invariant CUDA-core tiers (`w4a16_gemv_batch{M}`, one launch per projection, as under
//! METRALE_GLM_W4A16_GEMV_ROWS) against the M-row points `w4a16_gemv_seg_m<M>` (one launch per
//! group of projections that share the input rows).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every group, M = 2..=4 and block caps 4 and uncapped, every row
//!   m of every segment equals, byte for byte, `w4a16_gemv` on input row m alone (so also the
//!   `w4a16_gemv_batch{M}` rows, which hold the same invariant).
//! - Negative controls, both must fire: one changed element of input row 1 changes output row 1
//!   and leaves row 0 equal (rows are not mixed); one changed scale byte changes every row.
//!
//! Times: as `glm5next_w4a16_gemv_seg_bench` (CUDA graph over a rotated 128 MB pool, arms
//! alternating over 5 rounds, median of round medians [min..max]).
//!
//! Run (GB10, < 1 GB of device memory):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example glm5next_w4a16_gemv_seg_rows_bench

#[path = "common/w4a16_seg_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Result, ensure};
use fixture::{
    GROUPS, Pool, ROUNDS, Rng, STREAM_BYTES, bf16_rows, nvfp4_bytes, read, segs, stat, upload,
};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops::{self, W4a16GemvSegPoint};
use metrale_model_layers::weight_map::QuantizedWeight;

const KMAX: usize = 11264;
const NMAX: usize = 11264;
const MMAX: usize = 4;

/// 2026-10-11: `w4a16_gemv_batch{m}` (m = 2..=4) on `ws`, one launch per member, as
/// `glm5next_w4a16_seg::launch_gemv_rows` makes it.
#[allow(clippy::too_many_arguments)]
fn old_rows(
    g: &dyn GpuBackend,
    batch: &[KernelHandle; 5],
    a: DevicePtr,
    ws: &[QuantizedWeight],
    outs: &[DevicePtr],
    ns: &[u32],
    m: u32,
    k: u32,
    s: u64,
) -> Result<()> {
    for ((w, &c), &n) in ws.iter().zip(outs).zip(ns) {
        let l = KernelLaunch::new(g, batch[m as usize])
            .grid([div_ceil(n, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(w.weight)
            .arg_ptr(w.weight_scale)
            .arg_f32(w.weight_scale_2)
            .arg_ptr(c);
        if m <= 3 {
            l.arg_u32(n).arg_u32(k).launch(s)?;
        } else {
            l.arg_u32(m).arg_u32(n).arg_u32(k).launch(s)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let s = g.create_stream()?;
    let gemv = g.kernel("w4a16_gemv", "w4a16_gemv")?;
    let mut batch = [KernelHandle(0); 5];
    for m in 2..=4 {
        batch[m] = g.kernel("w4a16_gemv", &format!("w4a16_gemv_batch{m}"))?;
    }
    let mut rng = Rng(0x7365_6731_726f_7773);
    let pool = Pool::new(&g, &mut rng)?;
    let a_host = bf16_rows(&mut rng, MMAX, KMAX);
    let outs: Vec<DevicePtr> = (0..6)
        .map(|_| g.alloc(MMAX * NMAX * 2))
        .collect::<Result<_>>()?;
    let one = g.alloc(NMAX * 2)?;
    let arow = g.alloc(KMAX * 2)?;

    // 2026-10-11: Byte gate: row m of each segment equals `w4a16_gemv` on input row m alone.
    for (name, ns, k, _) in GROUPS {
        let ku = k as usize;
        let ad = upload(&g, &a_host[..MMAX * ku * 2])?;
        let ws: Vec<_> = (0..ns.len()).map(|i| pool.weight(i, ns[i], k)).collect();
        let mut want = vec![vec![Vec::new(); MMAX]; ns.len()];
        for r in 0..MMAX {
            g.copy_h2d(&a_host[r * ku * 2..(r + 1) * ku * 2], arow)?;
            for (i, w) in ws.iter().enumerate() {
                ops::w4a16_gemv(&g, gemv, arow, w, one, ns[i], k, s)?;
                want[i][r] = read(&g, s, one, ns[i] as usize * 2)?;
            }
        }
        for m in 2..=MMAX as u32 {
            if !ops::w4a16_gemv_seg_rows_ok(k, ns, m) {
                println!("{name}: M={m} outside the contract at K {k}, skipped");
                continue;
            }
            for cap in [4, u32::MAX] {
                for &o in &outs {
                    g.memset(o, 0xA5, MMAX * NMAX * 2)?;
                }
                let pt = W4a16GemvSegPoint::Rows(m);
                ops::w4a16_gemv_seg_point(&g, pt, ad, &segs(&ws, &outs, ns), k, cap, s)?;
                for (i, &n) in ns.iter().enumerate() {
                    let got = read(&g, s, outs[i], m as usize * n as usize * 2)?;
                    for r in 0..m as usize {
                        let row = &got[r * n as usize * 2..(r + 1) * n as usize * 2];
                        ensure!(
                            row == want[i][r].as_slice(),
                            "{name} segment {i} M={m} cap={cap} row {r}: differs from the one-row GEMV"
                        );
                    }
                }
            }
        }
        g.free(ad)?;
    }
    println!("byte gate: every row of every segment equals the one-row w4a16_gemv (M 2..=4)");

    // 2026-10-11: Negative controls on the shared gate|up at M = 2.
    let (ns, k) = (&[768u32, 768][..], 4096u32);
    let ku = k as usize;
    let mut a2 = a_host[..2 * ku * 2].to_vec();
    let ad = upload(&g, &a2)?;
    let ws: Vec<_> = (0..2).map(|i| pool.weight(i, ns[i], k)).collect();
    let run = |ad: DevicePtr| -> Result<Vec<u8>> {
        ops::w4a16_gemv_seg_point(
            &g,
            W4a16GemvSegPoint::Rows(2),
            ad,
            &segs(&ws, &outs, ns),
            k,
            u32::MAX,
            s,
        )?;
        read(&g, s, outs[0], 2 * 768 * 2)
    };
    let base = run(ad)?;
    // 2026-10-11: Input row 1, element 100, set to BF16 64.0 (the rows hold |x| < 32).
    a2[(ku + 100) * 2..(ku + 100) * 2 + 2].copy_from_slice(&0x4280u16.to_le_bytes());
    g.copy_h2d(&a2, ad)?;
    let moved = run(ad)?;
    ensure!(
        moved[..768 * 2] == base[..768 * 2] && moved[768 * 2..] != base[768 * 2..],
        "negative control: a changed row-1 element must change row 1 only"
    );
    let mut b0 = [0u8; 1];
    let at = ws[0].weight_scale.offset(4096 / 16 * 7 + 12);
    g.copy_d2h(at, &mut b0)?;
    g.copy_h2d(&[0x7E], at)?;
    let scaled = run(ad)?;
    g.copy_h2d(&b0, at)?;
    ensure!(
        scaled[7 * 2..7 * 2 + 2] != moved[7 * 2..7 * 2 + 2]
            && scaled[(768 + 7) * 2..(768 + 7) * 2 + 2] != moved[(768 + 7) * 2..(768 + 7) * 2 + 2],
        "negative control: a changed scale byte must change output 7 of both rows"
    );
    g.free(ad)?;
    println!("negative controls fire: row isolation; scale byte");

    // 2026-10-11: Timing: old (one batch launch per member) vs one M-row seg launch per group.
    let ad = upload(&g, &a_host)?;
    for m in 2..=MMAX as u32 {
        let mut tot = [0.0f64; 2];
        for (name, ns, k, count) in GROUPS {
            if !ops::w4a16_gemv_seg_rows_ok(k, ns, m) {
                continue;
            }
            let bytes = nvfp4_bytes(ns, k);
            let per = ns
                .iter()
                .map(|&n| n as usize * k as usize / 2)
                .sum::<usize>();
            let copies = STREAM_BYTES.div_ceil(per).clamp(4, 2048);
            let mut med = [Vec::new(), Vec::new()];
            for _ in 0..ROUNDS {
                for (arm, v) in med.iter_mut().enumerate() {
                    let mut i = 0usize;
                    let t = timing::time_graph(&g, s, copies, &mut |st| {
                        let ws: Vec<_> = (0..ns.len())
                            .map(|j| pool.weight(i * 6 + j, ns[j], k))
                            .collect();
                        i += 1;
                        if arm == 0 {
                            old_rows(&g, &batch, ad, &ws, &outs, ns, m, k, st)
                        } else {
                            let pt = W4a16GemvSegPoint::Rows(m);
                            ops::w4a16_gemv_seg_point(
                                &g,
                                pt,
                                ad,
                                &segs(&ws, &outs, ns),
                                k,
                                u32::MAX,
                                st,
                            )
                        }
                    })?;
                    v.push(t.0);
                }
            }
            let old = stat(&mut med[0]);
            let new = stat(&mut med[1]);
            println!(
                "M={m} {name:<20} K={k:<5} batch {:7.2} us [{:.2}..{:.2}]  seg_m {:7.2} us \
                 [{:.2}..{:.2}] {:4.0} GB/s  x{:.3}",
                old.0,
                old.1,
                old.2,
                new.0,
                new.1,
                new.2,
                bytes / new.0 / 1e3,
                old.0 / new.0
            );
            tot[0] += old.0 * f64::from(count);
            tot[1] += new.0 * f64::from(count);
        }
        println!(
            "M={m} per step (rank 0, counted groups): batch tiers {:.3} ms -> seg_m {:.3} ms",
            tot[0] / 1e3,
            tot[1] / 1e3
        );
    }
    Ok(())
}
