// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Dense W4A16 leg of `glm5next_batch_inv_bench`: on every GLM-5.3 rank-0 W4A16 shape,
//! row r of an M-row `w4a16_tc_rows` launch (and of the one-segment `w4a16_tc_rows_seg`, split 1,
//! the recipe's 2+ row path) against row r launched alone, M = 1..=16, 24, 32, 48, 64; then the
//! one-row cost of the row tile against today's one-row GEMV.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Errors on the first row whose bytes differ from that row alone.
//! - Control: the one-row GEMV (another FP32 order) must differ from the row tile somewhere, or
//!   the comparison could not see an order change.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layers::ops::{self, W4a16Seg};

use crate::fixture::{Pool, Rng, bf16_rows, read, stat, upload};
use crate::timing;

/// 2026-10-11: `(name, n, k)`: the rank-0 W4A16 projections (DSA ones under the opt-in tier).
pub const SHAPES: [(&str, u32, u32); 9] = [
    ("kda q", 2816, 4096),
    ("kda f_a", 128, 4096),
    ("kda b", 22, 4096),
    ("kda o", 4096, 2816),
    ("shared gate", 768, 4096),
    ("shared down", 4096, 768),
    ("dsa q_a", 1536, 4096),
    ("dsa q_absorb", 11264, 1536),
    ("dsa o_absorb", 4096, 11264),
];

const ROWS: [u32; 20] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32, 48, 64,
];
const MAXR: usize = 64;

pub fn run(g: &dyn GpuBackend, s: u64, pool: &Pool, rng: &mut Rng) -> Result<()> {
    let gemv = g.kernel("w4a16_gemv", "w4a16_gemv")?;
    let kmax = 11264usize;
    let ad = upload(g, &bf16_rows(rng, MAXR, kmax))?;
    let out = g.alloc(MAXR * 11264 * 2)?;
    let one = g.alloc(11264 * 2)?;
    for (name, n, k) in SHAPES {
        let w = pool.weight(0, n, k);
        let nb = n as usize * 2;
        // 2026-10-11: Row r alone (input row r of a [MAXR, k] block at pitch k).
        let mut alone = Vec::with_capacity(MAXR);
        for r in 0..MAXR {
            let a = ad.offset(r * k as usize * 2);
            ops::w4a16_tc_rows(g, a, &w, one, 1, n, k, k, n, s)?;
            alone.push(read(g, s, one, nb)?);
        }
        for m in ROWS {
            for seg in [false, true] {
                g.memset(out, 0xA5, MAXR * 11264 * 2)?;
                if seg {
                    let segs = [W4a16Seg {
                        weight: w,
                        output: out,
                        n,
                    }];
                    ops::w4a16_tc_rows_seg(g, ad, &segs, m, k, k, 1, s)?;
                } else {
                    ops::w4a16_tc_rows(g, ad, &w, out, m, n, k, k, n, s)?;
                }
                let got = read(g, s, out, m as usize * nb)?;
                for r in 0..m as usize {
                    ensure!(
                        got[r * nb..(r + 1) * nb] == alone[r][..],
                        "{name} [{n} x {k}] M={m} seg={seg} row {r}: differs from the row alone"
                    );
                }
            }
        }
        // 2026-10-11: The narrow points (1 and 2 warps per block) at 1..=8 rows.
        for warps in [1u32, 2] {
            for m in 1..=ops::W4A16_TC_ROWS_NARROW_MAX_M {
                g.memset(out, 0xA5, MAXR * 11264 * 2)?;
                ops::w4a16_tc_rows_narrow(g, ad, &w, out, m, n, k, k, n, warps, s)?;
                let got = read(g, s, out, m as usize * nb)?;
                for r in 0..m as usize {
                    ensure!(
                        got[r * nb..(r + 1) * nb] == alone[r][..],
                        "{name} [{n} x {k}] narrow w{warps} M={m} row {r}: differs from the row alone"
                    );
                }
            }
        }
        // 2026-10-11: Control: the GEMV order differs from the row tile's somewhere.
        let mut diff = 0usize;
        for r in 0..8 {
            ops::w4a16_gemv(g, gemv, ad.offset(r * k as usize * 2), &w, one, n, k, s)?;
            let x = read(g, s, one, nb)?;
            diff += x
                .chunks(2)
                .zip(alone[r].chunks(2))
                .filter(|(p, q)| p != q)
                .count();
        }
        ensure!(
            diff > 0,
            "{name}: control: the GEMV equals the row tile on 8 rows"
        );
        println!(
            "dense {name:<13} [{n:>5} x {k:>5}]: rows 1..=64 equal alone (tc_rows, seg S=1, narrow w1/w2 at 1..=8); GEMV differs in {diff} of {} outputs",
            8 * n
        );
    }

    // 2026-10-11: One-row cost of the canonical tile (4, 2 and 1 warps per block) vs the GEMV
    // (graph, rotated weights).
    let mut tot = [0.0f64; 4];
    for (name, n, k) in SHAPES {
        let per = n as usize * k as usize / 2;
        let copies = (96usize << 20).div_ceil(per).clamp(4, 2048);
        let mut med = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..5 {
            for (arm, v) in med.iter_mut().enumerate() {
                let mut i = 0usize;
                let t = timing::time_graph(g, s, copies, &mut |st| {
                    let w = pool.weight(i, n, k);
                    i += 1;
                    match arm {
                        0 => ops::w4a16_gemv(g, gemv, ad, &w, out, n, k, st),
                        1 => ops::w4a16_tc_rows(g, ad, &w, out, 1, n, k, k, n, st),
                        a => ops::w4a16_tc_rows_narrow(
                            g,
                            ad,
                            &w,
                            out,
                            1,
                            n,
                            k,
                            k,
                            n,
                            a as u32 - 1,
                            st,
                        ),
                    }
                })?;
                v.push(t.0);
            }
        }
        let t: Vec<_> = med.iter_mut().map(|v| stat(v)).collect();
        println!(
            "dense 1 row {name:<13} gemv {:7.2} [{:.2}..{:.2}]  tc_rows w4 {:7.2} [{:.2}..{:.2}]  \
             w1 {:7.2} [{:.2}..{:.2}]  w2 {:7.2} [{:.2}..{:.2}] us",
            t[0].0,
            t[0].1,
            t[0].2,
            t[1].0,
            t[1].1,
            t[1].2,
            t[2].0,
            t[2].1,
            t[2].2,
            t[3].0,
            t[3].1,
            t[3].2
        );
        for (i, x) in t.iter().enumerate() {
            tot[i] += x.0;
        }
    }
    println!(
        "dense 1 row, one of each shape: gemv {:.1} us, tc_rows w4 {:.1}, w1 {:.1}, w2 {:.1}",
        tot[0], tot[1], tot[2], tot[3]
    );
    g.free(ad)?;
    Ok(())
}
