// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: A prefill-shaped DSA attend input for the prefill twin gates: one
//! 32k-token sequence, its query rows the last `rows` positions, all rows sharing one block
//! table (stride 0, as a prefill sub-chunk launches), and per row a selection laid out as
//! `dsa_expand_selection` lays it (kernels/gb10/common/dsa_indexer_body.cuh): the row's top
//! `min((pos + 1) / 4, 512)` pools by score, four consecutive tokens each, then the
//! in-progress pool's tail, then -1.
//!
//! The pool scores are correlated across rows the way an indexer's are: a score shared by
//! every row, attention-sink pools, a recency bump near the row's own position, and per-row
//! noise. No captured selections exist to replay, so this is a synthetic stand-in; the
//! overlap it produces is printed ([`overlap`]).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: every selected token is `<= pos`, so the reference and kernel agree on
//! validity whatever `seq_len` the launch passes.

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::fixture::{BLOCK, Case, KVL, Lcg, POOL_BLOCKS, WIDTH, upload_case};

/// 2026-10-14: The sequence length of the prefill case.
pub const SEQ32: usize = 32768;
/// 2026-10-14: Tokens per indexer pool and pools per row (index_topk 2048 / 4).
const KP: usize = 4;
const SELECT_K: usize = 512;

/// 2026-10-14: [`make_prefill_case_at`] a [`SEQ32`]-token sequence.
pub fn make_prefill_case(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    heads: usize,
    rows: usize,
    variants: usize,
    q_scale: f32,
) -> Result<Case> {
    make_prefill_case_at(g, rng, SEQ32, heads, rows, variants, q_scale)
}

/// 2026-10-14: A [`Case`] with `variants` block tables (each `seq / BLOCK` distinct pool
/// blocks, rounded up) and selections (each from its own per-row noise), `rows` query rows at
/// positions `seq - rows ..`, `seq_len = pos + 1`, `bps` 0 (one shared table). 2026-10-15: the
/// sequence length is a parameter (`glm5next_dsa_prefill_tc_nanor_gate` times a 1k one).
pub fn make_prefill_case_at(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    seq: usize,
    heads: usize,
    rows: usize,
    variants: usize,
    q_scale: f32,
) -> Result<Case> {
    let q: Vec<f32> = (0..rows * heads * KVL)
        .map(|_| bf16::from_f32(rng.f() * q_scale).to_f32())
        .collect();
    let pos = |r: usize| seq - rows + r;
    let sl: Vec<i32> = (0..rows).map(|r| (pos(r) + 1) as i32).collect();
    let n_pools = seq / KP;
    let (mut bt, mut sel) = (Vec::new(), Vec::new());
    for _ in 0..variants {
        let mut blocks: Vec<i32> = (0..POOL_BLOCKS as i32).collect();
        for i in 0..seq.div_ceil(BLOCK) {
            let j = i + rng.u() as usize % (POOL_BLOCKS - i);
            blocks.swap(i, j);
        }
        blocks.truncate(seq.div_ceil(BLOCK));
        bt.push(blocks);
        let base: Vec<f32> = (0..n_pools).map(|_| rng.f() + rng.f()).collect();
        let mut s = vec![-1i32; rows * WIDTH];
        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(n_pools);
        for r in 0..rows {
            let p_r = pos(r);
            let row_pools = (p_r + 1) / KP;
            scored.clear();
            scored.extend((0..row_pools).map(|p| {
                let sink = if p < 4 { 6.0 } else { 0.0 };
                let recent = 3.0 * (-((row_pools - 1 - p) as f32) / 64.0).exp();
                (base[p] + sink + recent + 0.5 * rng.f(), p)
            }));
            let k = row_pools.min(SELECT_K);
            let desc = |a: &(f32, usize), b: &(f32, usize)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
            if k < scored.len() {
                scored.select_nth_unstable_by(k - 1, desc);
            }
            scored[..k].sort_unstable_by(desc);
            let row = &mut s[r * WIDTH..][..WIDTH];
            for (j, (_, p)) in scored[..k].iter().enumerate() {
                for t in 0..KP {
                    row[j * KP + t] = (p * KP + t) as i32;
                }
            }
            let vis = p_r + 1;
            for t in 0..vis % KP {
                row[k * KP + t] = (vis - vis % KP + t) as i32;
            }
        }
        sel.push(s);
    }
    upload_case(g, heads, 0, q, sl, bt, sel)
}

/// 2026-10-14: Selection overlap of variant 0, as pools: the mean fraction of a row's pools
/// that the previous row also selected, and the mean union of `group`-row groups over
/// [`SELECT_K`] (1.0: every row of the group selects the same pools; `group`: disjoint).
pub fn overlap(c: &Case, group: usize) -> (f64, f64) {
    let rows = c.sl.len();
    let pools = |r: usize| -> Vec<usize> {
        c.sel[0][r * WIDTH..][..SELECT_K * KP]
            .iter()
            .step_by(KP)
            .filter(|t| **t >= 0)
            .map(|t| *t as usize / KP)
            .collect()
    };
    let mut seen = vec![u32::MAX; SEQ32 / KP];
    let mut adjacent = 0f64;
    for r in 1..rows {
        pools(r - 1).iter().for_each(|p| seen[*p] = r as u32);
        let cur = pools(r);
        adjacent += cur.iter().filter(|p| seen[**p] == r as u32).count() as f64 / cur.len() as f64;
    }
    let mut unions = 0f64;
    let groups = rows / group;
    for gi in 0..groups {
        let mut set = vec![false; SEQ32 / KP];
        (gi * group..(gi + 1) * group).for_each(|r| pools(r).iter().for_each(|p| set[*p] = true));
        unions += set.iter().filter(|x| **x).count() as f64 / SELECT_K as f64;
    }
    (adjacent / (rows - 1) as f64, unions / groups.max(1) as f64)
}
