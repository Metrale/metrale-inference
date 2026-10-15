// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Byte gate and microbench of the GLM-5.3 MoE combine at eight elements per thread
//! (`glm5next_moe_combine[_indexed]_vec8`, `METRALE_GLM_MOE_COMBINE_VEC8=1`) against the scalar
//! combine at the serve's split grid (`METRALE_GLM_MOE_COMBINE_SPLIT=1`), in slot order and
//! through a shuffled `token_to_perm` (the expert-sorted order), at 1 and 16 decode rows, at 17,
//! 287, 341 and 1036 rows and at the 8192- and 8208-row prefill pieces.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Negative control: the indexed sum with slots 3 and 4 of every row swapped (weight and row
//!   together, so only the FP32 addition order changes) must NOT match: the gate sees an order
//!   change, the one way a re-vectorised sum could differ. Element 64 of row 0 is planted (`plant`)
//!   so the swap changes it by 1.0 at every row count, not by chance.
//! - Row 0 carries the edge values (NaN, infinities, signed zeros, BF16 subnormals, products that
//!   overflow FP32 and a sum that cancels), slot 0 of row 0 a zero weight and slot 1 a negative
//!   zero; the outputs must be byte-identical to the scalar kernel's at every grid tried, and at
//!   decode rows also to the host reference (`combine_ref`).

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::fixture::{Lcg, same, up, up_f32};
use crate::timing::report;
use crate::{HIDDEN, MOE_LAYERS, TOP_K, combine_ref, read, tg, zeros};

/// 2026-10-15: Decode rows, the odd and SP-piece prefill rows the cold plan gates (17, 287, 341,
/// 1036) and the 8192- and 8208-row prefill pieces.
const ROWS: [usize; 8] = [1, 16, 17, 287, 341, 1036, 8192, 8208];

/// 2026-10-14: Row 0's first 64 elements of every slot and of the shared row.
fn edges(eo: &mut [bf16], shared: &mut [bf16]) {
    let pick = [
        bf16::NAN,
        bf16::INFINITY,
        bf16::NEG_INFINITY,
        bf16::from_bits(0x8000),
        bf16::from_bits(0x0001),
        bf16::from_bits(0x807f),
        bf16::MAX,
        bf16::MIN,
        bf16::from_f32(1.0),
        bf16::from_f32(-1.0),
    ];
    for k in 0..TOP_K {
        for d in 0..64 {
            eo[k * HIDDEN + d] = pick[(d + 3 * k) % pick.len()];
        }
    }
    for (d, v) in shared.iter_mut().take(64).enumerate() {
        *v = pick[(d * 7) % pick.len()];
    }
}

/// 2026-10-14: `same` against the host reference, every BF16 NaN read as one value first: the
/// device rounds a NaN sum to `0x7fff` and the host's `bf16::from_f32` keeps its quieted payload
/// (`0x7fc0`), so only the NaN encoding may differ from the host. The kernel-to-kernel gates
/// compare raw bytes, NaN payloads included.
fn same_host(what: &str, host: &[u8], got: &[u8]) -> Result<()> {
    let canon = |b: &[u8]| -> Vec<u8> {
        b.chunks_exact(2)
            .flat_map(|c| {
                let v = u16::from_le_bytes([c[0], c[1]]);
                let nan = v & 0x7f80 == 0x7f80 && v & 0x007f != 0;
                if nan { 0x7fc0u16 } else { v }.to_le_bytes()
            })
            .collect()
    };
    same(what, &canon(host), &canon(got))
}

/// 2026-10-14: A shuffle of `0..n` (Fisher-Yates on the fixture's generator).
fn shuffled(rng: &mut Lcg, n: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (((rng.next() + 1.0) * 0.5 * (i + 1) as f32) as usize).min(i);
        p.swap(i, j);
    }
    p
}

/// 2026-10-14: Element 64 of row 0, routed through `perm`: slots 2, 3 and 4 at weight 1 carry
/// 1, 2^24 and -2^24. In slot order the FP32 sum is (1 + 2^24) - 2^24 = 0 (the first addition
/// rounds to even); with slots 3 and 4 swapped it is (1 - 2^24) + 2^24 = 1. Slots 0 and 1 keep
/// their zero weights, so the sum is still exactly 0 when slot 2 is added: a swap of slots 2 and
/// 3 alone could never change row 0.
fn plant(eo: &mut [bf16], w: &mut [f32], perm: &[usize]) {
    for (k, v) in [(2, 1.0f32), (3, 16_777_216.0), (4, -16_777_216.0)] {
        w[k] = 1.0;
        eo[perm[k] * HIDDEN + 64] = bf16::from_f32(v);
    }
}

/// 2026-10-14: `(weights, token_to_perm bytes)` with slots 3 and 4 of every row swapped, weight
/// and routed row together: the same terms, summed in another order.
fn swap_slots(w: &[f32], perm: &[usize]) -> (Vec<f32>, Vec<u8>) {
    let (mut w2, mut p2) = (w.to_vec(), perm.to_vec());
    for t in 0..w.len() / TOP_K {
        w2.swap(t * TOP_K + 3, t * TOP_K + 4);
        p2.swap(t * TOP_K + 3, t * TOP_K + 4);
    }
    let bytes = p2.iter().flat_map(|p| (*p as i32).to_le_bytes()).collect();
    (w2, bytes)
}

pub fn combine_vec8(g: &dyn GpuBackend, s: u64, rng: &mut Lcg) -> Result<()> {
    let scalar = [
        g.kernel("glm5next_ffn", "glm5next_moe_combine")?,
        g.kernel("glm5next_ffn", "glm5next_moe_combine_indexed")?,
    ];
    let vec8 = [
        g.kernel("glm5next_ffn", "glm5next_moe_combine_vec8")?,
        g.kernel("glm5next_ffn", "glm5next_moe_combine_indexed_vec8")?,
    ];
    for rows in ROWS {
        let n = rows * TOP_K;
        let mut eo: Vec<bf16> = (0..n * HIDDEN)
            .map(|_| bf16::from_f32(rng.next() * 2.0))
            .collect();
        let mut shared: Vec<bf16> = (0..rows * HIDDEN)
            .map(|_| bf16::from_f32(rng.next() * 2.0))
            .collect();
        edges(&mut eo, &mut shared);
        let mut w = rng.vec(n, 1.5);
        w[0] = 0.0;
        w[1] = -0.0;
        let perm = shuffled(rng, n);
        plant(&mut eo, &mut w, &perm);
        let to_bytes = |v: &[bf16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let perm_i: Vec<u8> = perm
            .iter()
            .flat_map(|p| (*p as i32).to_le_bytes())
            .collect();
        let bufs = [
            up(g, &to_bytes(&eo))?,
            up(g, &to_bytes(&shared))?,
            up_f32(g, &w)?,
            up(g, &perm_i)?,
        ];
        let [eo_d, sh_d, w_d, perm_d] = bufs;
        let ob = rows * HIDDEN * 2;
        let launch_on =
            |vec: bool, idx: bool, grid: (u32, u32), wp: [DevicePtr; 2], out: DevicePtr, s: u64| {
                let k = if vec { vec8 } else { scalar }[usize::from(idx)];
                let l = KernelLaunch::new(g, k)
                    .grid([rows as u32, grid.1, 1])
                    .block([grid.0, 1, 1])
                    .arg_ptr(eo_d);
                let l = if idx { l.arg_ptr(wp[1]) } else { l };
                l.arg_ptr(wp[0])
                    .arg_ptr(sh_d)
                    .arg_ptr(out)
                    .arg_u32(HIDDEN as u32)
                    .arg_u32(TOP_K as u32)
                    .launch(s)
            };
        let launch = |vec: bool, idx: bool, grid: (u32, u32), out: DevicePtr, s: u64| {
            launch_on(vec, idx, grid, [w_d, perm_d], out, s)
        };
        let identity: Vec<usize> = (0..n).collect();
        let (base, arm) = (zeros(g, ob)?, zeros(g, ob)?);
        for idx in [false, true] {
            let order = if idx { &perm } else { &identity };
            launch(false, idx, (256, 16), base, s)?;
            let want = read(g, s, base, ob)?;
            let host = (rows <= 16).then(|| combine_ref(&eo, &w, &shared, order));
            if let Some(h) = &host {
                same_host(
                    &format!("combine R={rows} idx={idx} scalar vs host"),
                    h,
                    &want,
                )?;
            }
            // 2026-10-14: The serve's grid (256 threads, hidden / 2048 blocks per row), then
            // narrower blocks and a grid wider than the row (the spare threads return).
            for grid in [(256u32, 2u32), (128, 4), (64, 8), (256, 3)] {
                g.memset(arm, 0xA5, ob)?;
                launch(true, idx, grid, arm, s)?;
                let got = read(g, s, arm, ob)?;
                same(
                    &format!("combine_vec8 R={rows} idx={idx} {grid:?}"),
                    &want,
                    &got,
                )?;
                if let Some(h) = &host {
                    same_host(&format!("combine_vec8 R={rows} idx={idx} vs host"), h, &got)?;
                }
            }
            if idx {
                let (w2, p2) = swap_slots(&w, &perm);
                let wp = [up_f32(g, &w2)?, up(g, &p2)?];
                g.memset(arm, 0xA5, ob)?;
                launch_on(true, true, (256, 2), wp, arm, s)?;
                let got = read(g, s, arm, ob)?;
                for p in wp {
                    g.free(p)?;
                }
                if got == want {
                    bail!(
                        "combine_vec8 R={rows}: negative control (slots 3 and 4 swapped) matched"
                    );
                }
                println!(
                    "combine_vec8 R={rows}: negative control (slots 3 and 4 swapped) caught: differs"
                );
            }
            let copies = if rows > 16 { 8 } else { MOE_LAYERS };
            report(
                &format!("combine vec8 idx={idx}"),
                rows,
                tg(g, s, copies, &|s| launch(false, idx, (256, 16), base, s))?,
                tg(g, s, copies, &|s| launch(true, idx, (256, 2), arm, s))?,
            );
        }
        for p in bufs.into_iter().chain([base, arm]) {
            g.free(p)?;
        }
    }
    Ok(())
}
