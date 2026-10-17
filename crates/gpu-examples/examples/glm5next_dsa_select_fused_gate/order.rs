// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The defined order (score descending by `dsa_tk`, then pool ascending): the host
//! oracle of one row (`oracle_row`, moved from `main.rs`), and the negative control at the cut
//! (`control`): in one causal, unmasked row whose order has
//! distinct keys around slot `k - 1`, the `k`-th and `(k + 1)`-th pools of the defined order
//! (`a`, selected last, and `b`, the best left out) swap their keys. Scores are a pure function
//! of the key, so `b` takes `a`'s exact score and `a` takes `b`'s: the fused row must change in
//! slot `k - 1` only, from `a` to `b`, and every row must still equal the reference over the
//! swapped keys. A cut comparator that is off by one slot, or that breaks the boundary
//! comparison, fails here where the whole-row checks of shapes far from the cut would not.
//!
//! Owner: model-arch examples.
//! Invariants: restores the shape's keys before returning; `Ok(Some(false))` when the control ran
//! and the output did not change as required; `Ok(None)` (2026-10-17) when no full row has more
//! than `k` visible candidates, so there is no cut to move (P = 513 at k = 512 with 1% of the
//! pools invisible); an error when a row has a cut but none has distinct keys around it.

use super::*;

/// 2026-10-17: `dsa_tk`, the kernels' order-preserving key (-0.0 keyed as +0.0).
fn tk(f: f32) -> u32 {
    let u = if f.to_bits() == 0x8000_0000 {
        0
    } else {
        f.to_bits()
    };
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}

/// 2026-10-17: The host selection over one row's reference scores, in the defined order, folded;
/// and whether the cut went through a tie (an equal score on both sides of it).
pub(super) fn oracle_row(io: &Io, r: usize, scores: &[f32]) -> (Vec<i32>, bool) {
    let cand = |p: usize| io.h_vis[p] >= 0 && io.h_vis[p] <= io.h_qpos[r];
    let key = |p: usize| {
        if cand(p) {
            tk(scores[p])
        } else {
            tk(-f32::MAX)
        }
    };
    let mut order: Vec<usize> = (0..io.pools).collect();
    let cmp = |a: &usize, b: &usize| key(*b).cmp(&key(*a)).then(a.cmp(b));
    let take = io.k.min(io.pools);
    if take < io.pools {
        order.select_nth_unstable_by(take - 1, cmp);
    }
    order[..take].sort_by(cmp);
    let tie = take < io.pools && {
        let cut = key(order[take - 1]);
        order[take..].iter().any(|&p| key(p) == cut && cand(p)) && cand(order[take - 1])
    };
    let row_k = ((io.h_qpos[r] + 1) as usize / KP).min(io.k);
    let on = io.h_qmask[r] != 0;
    let sel = (0..io.k)
        .map(|j| {
            let p = if j < take { order[j] } else { usize::MAX };
            if on && j < row_k && p != usize::MAX && cand(p) {
                p as i32
            } else {
                -1
            }
        })
        .collect();
    (sel, tie)
}

/// 2026-10-17: Row `r`'s candidates in the defined order (key descending, pool ascending).
fn order(io: &Io, r: usize, scores: &[f32]) -> Vec<(u32, usize)> {
    let mut o: Vec<(u32, usize)> = (0..io.pools)
        .filter(|&p| io.h_vis[p] >= 0 && io.h_vis[p] <= io.h_qpos[r])
        .map(|p| (tk(scores[r * io.pools + p]), p))
        .collect();
    o.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
    o
}

pub(super) fn control(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    io: &Io,
    scores: &[f32],
    want: &[i32],
) -> Result<Option<bool>> {
    let kk = io.k;
    let full = |r: usize| (io.h_qpos[r] + 1) as usize / KP >= kk && io.h_qmask[r] != 0;
    if !(0..io.rows).any(|r| full(r) && order(io, r, scores).len() > kk) {
        println!("  cut control: n/a, no full row has more than k = {kk} visible candidates");
        return Ok(None);
    }
    let pick = (0..io.rows).find_map(|r| {
        let full = full(r);
        let o = order(io, r, scores);
        let distinct =
            o.len() > kk && (kk < 2 || o[kk - 2].0 > o[kk - 1].0) && o[kk - 1].0 > o[kk].0;
        (full && distinct).then(|| (r, o[kk - 1].1, o[kk].1))
    });
    let (r, a, b) = pick.ok_or_else(|| anyhow::anyhow!("no row with a distinct cut"))?;
    ensure!(
        want[r * kk + kk - 1] == a as i32,
        "the reference's slot k - 1 is not the k-th pool of the defined order"
    );
    let mut keys = io.h_keys.clone();
    for d in 0..D {
        keys.swap(a * D + d, b * D + d);
    }
    let kb: Vec<u8> = keys.iter().flat_map(|h| h.to_le_bytes()).collect();
    gpu.copy_h2d(&kb, io.keys)?;
    let got = selected(gpu, io, 0x5A, || fused(gpu, k, io))?;
    let refd = selected(gpu, io, 0x00, || reference(gpu, k, io))?;
    let orig: Vec<u8> = io.h_keys.iter().flat_map(|h| h.to_le_bytes()).collect();
    gpu.copy_h2d(&orig, io.keys)?;
    let row = &got[r * kk..(r + 1) * kk];
    let changed: Vec<usize> = (0..kk).filter(|&j| row[j] != want[r * kk + j]).collect();
    let ok = changed == [kk - 1] && row[kk - 1] == b as i32 && rows_differ(&got, &refd, kk) == 0;
    println!(
        "  cut control: row {r} pools {a} <-> {b} at slot {}: slots changed {changed:?}, \
         fused vs swapped reference rows differ {}",
        kk - 1,
        rows_differ(&got, &refd, kk)
    );
    Ok(Some(ok))
}
