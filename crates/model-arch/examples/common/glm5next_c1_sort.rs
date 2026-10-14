// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Byte gate and microbench of the multi-block stable expert sort
//! (`moe_sort_mb_hist` + `moe_sort_mb_scan` + `moe_sort_mb_scatter`,
//! `METRALE_GLM_MOE_SORT_MULTIBLOCK=1`) against the one-block `moe_sort_by_expert`, at 1, 16, 1000
//! and 8192 rows of top-8 ids over 288 experts, spread and skewed.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - The one-block sort's `expert_offsets` and the multi-block sort's are byte-equal, and both
//!   equal a host counting sort's; every multi-block table (`sorted_token_ids`,
//!   `sorted_expert_ids`, `token_to_perm`) equals the host STABLE sort's byte for byte. The
//!   one-block tables (atomic order) must be a consistent permutation of the same slots.
//! - Downstream: routed rows that depend only on (token, expert), as a grouped GEMM's do, laid
//!   out in each sort's order and summed by `glm5next_moe_combine_indexed` through each sort's
//!   `token_to_perm`, give byte-equal outputs.
//! - Negative controls: the scatter run with top_k 4 (each sorted token id halved) must NOT
//!   match, and the comparator must reject the host reference with two rows of one expert swapped.

use anyhow::{Result, bail, ensure};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::fixture::{Lcg, same, up, up_f32};
use crate::timing::report;
use crate::{EXPERTS, HIDDEN, MOE_LAYERS, TOP_K, read, tg, zeros};

/// 2026-10-14: `MOE_SORT_MB_CHUNK` in `moe_permute.cu`.
const CHUNK: usize = 1024;
const ROWS: [usize; 4] = [1, 16, 1000, 8192];

/// 2026-10-14: `rows` rows of `TOP_K` distinct ids: over every expert, or (`skew`) over the
/// first 16 only, so a few experts own most slots.
fn draw_ids(rng: &mut Lcg, rows: usize, skew: bool) -> Vec<u32> {
    let pool = if skew { 16 } else { EXPERTS };
    let mut ids = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let mut row: Vec<u32> = Vec::with_capacity(TOP_K);
        while row.len() < TOP_K {
            let e = (((rng.next() + 1.0) * 0.5 * pool as f32) as u32).min(pool as u32 - 1);
            if !row.contains(&e) {
                row.push(e);
            }
        }
        ids.extend(row);
    }
    ids
}

/// 2026-10-14: The host stable counting sort: `(offsets, sorted_token_ids, sorted_expert_ids,
/// token_to_perm)`.
fn stable_ref(ids: &[u32]) -> [Vec<i32>; 4] {
    let mut off = vec![0i32; EXPERTS + 1];
    for &e in ids {
        off[e as usize + 1] += 1;
    }
    for e in 0..EXPERTS {
        off[e + 1] += off[e];
    }
    let mut next = off.clone();
    let (mut tok, mut exp, mut ttp) = (vec![0; ids.len()], vec![0; ids.len()], vec![0; ids.len()]);
    for (i, &e) in ids.iter().enumerate() {
        let pos = next[e as usize] as usize;
        next[e as usize] += 1;
        tok[pos] = (i / TOP_K) as i32;
        exp[pos] = e as i32;
        ttp[i] = pos as i32;
    }
    [off, tok, exp, ttp]
}

fn bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ints(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 2026-10-14: Routed rows in a sort's order, row `pos` a function of its (token, expert) only.
fn routed_rows(tok: &[i32], exp: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tok.len() * HIDDEN * 2);
    for (t, e) in tok.iter().zip(exp) {
        for d in 0..HIDDEN {
            let h =
                (*t as u32).wrapping_mul(2654435761) ^ (*e as u32).wrapping_mul(40503) ^ d as u32;
            out.extend(bf16::from_f32((h % 977) as f32 / 61.0 - 8.0).to_le_bytes());
        }
    }
    out
}

pub fn sort_multiblock(g: &dyn GpuBackend, s: u64, rng: &mut Lcg) -> Result<()> {
    let old = g.kernel("moe", "moe_sort_by_expert")?;
    let [hist, scan, scatter] = [
        "moe_sort_mb_hist",
        "moe_sort_mb_scan",
        "moe_sort_mb_scatter",
    ]
    .map(|e| g.kernel("moe", e));
    let (hist, scan, scatter) = (hist?, scan?, scatter?);
    let combine = g.kernel("glm5next_ffn", "glm5next_moe_combine_indexed")?;
    for rows in ROWS {
        for skew in [false, true] {
            let n = rows * TOP_K;
            let nb = n.div_ceil(CHUNK) as u32;
            let ids = draw_ids(rng, rows, skew);
            let want = stable_ref(&ids);
            let ids_d = up(
                g,
                &ids.iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )?;
            let hist_d = zeros(g, nb as usize * EXPERTS * 4)?;
            let sizes = [(EXPERTS + 1) * 4, n * 4, n * 4, n * 4];
            // 2026-10-14: Every table starts as 0xA5 bytes, so an unwritten entry shows.
            let tabs = || -> Result<[DevicePtr; 4]> {
                let mut t = [DevicePtr::NULL; 4];
                for (p, b) in t.iter_mut().zip(sizes) {
                    *p = up(g, &vec![0xA5u8; b])?;
                }
                Ok(t)
            };
            let (o, m) = (tabs()?, tabs()?);
            let run_old = |s: u64| {
                KernelLaunch::new(g, old)
                    .grid([1, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(ids_d)
                    .arg_ptr(o[1])
                    .arg_ptr(o[2])
                    .arg_ptr(o[0])
                    .arg_ptr(o[3])
                    .arg_u32(n as u32)
                    .arg_u32(EXPERTS as u32)
                    .arg_u32(TOP_K as u32)
                    .launch(s)
            };
            let run_new = |topk: u32, s: u64| -> Result<()> {
                KernelLaunch::new(g, hist)
                    .grid([nb, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(ids_d)
                    .arg_ptr(hist_d)
                    .arg_u32(n as u32)
                    .arg_u32(EXPERTS as u32)
                    .launch(s)?;
                KernelLaunch::new(g, scan)
                    .grid([1, 1, 1])
                    .block([EXPERTS.div_ceil(32) as u32 * 32, 1, 1])
                    .arg_ptr(hist_d)
                    .arg_ptr(m[0])
                    .arg_u32(nb)
                    .arg_u32(EXPERTS as u32)
                    .launch(s)?;
                KernelLaunch::new(g, scatter)
                    .grid([nb, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(ids_d)
                    .arg_ptr(hist_d)
                    .arg_ptr(m[0])
                    .arg_ptr(m[1])
                    .arg_ptr(m[2])
                    .arg_ptr(m[3])
                    .arg_u32(n as u32)
                    .arg_u32(EXPERTS as u32)
                    .arg_u32(topk)
                    .launch(s)
            };
            let tag = format!("sort R={rows} skew={skew}");
            run_old(s)?;
            run_new(TOP_K as u32, s)?;
            let got_o: Vec<Vec<u8>> = (0..4)
                .map(|i| read(g, s, o[i], sizes[i]))
                .collect::<Result<_>>()?;
            let got_m: Vec<Vec<u8>> = (0..4)
                .map(|i| read(g, s, m[i], sizes[i]))
                .collect::<Result<_>>()?;
            same(
                &format!("{tag} offsets: one-block vs host"),
                &bytes(&want[0]),
                &got_o[0],
            )?;
            for (i, what) in [
                "offsets",
                "sorted_token_ids",
                "sorted_expert_ids",
                "token_to_perm",
            ]
            .iter()
            .enumerate()
            {
                same(
                    &format!("{tag} {what}: multi-block vs host stable"),
                    &bytes(&want[i]),
                    &got_m[i],
                )?;
            }
            // 2026-10-14: The one-block order is the atomics', so its tables are checked as a
            // permutation: every slot lands once, inside its expert's range, with its token.
            let (ot, oe, op) = (ints(&got_o[1]), ints(&got_o[2]), ints(&got_o[3]));
            let mut seen = vec![false; n];
            for (i, &p) in op.iter().enumerate() {
                let p = p as usize;
                let e = ids[i] as usize;
                ensure!(
                    p < n && !seen[p],
                    "{tag}: one-block token_to_perm[{i}] = {p} repeats"
                );
                seen[p] = true;
                ensure!(
                    oe[p] == e as i32 && ot[p] == (i / TOP_K) as i32,
                    "{tag}: one-block row {p} is not slot {i}'s"
                );
                ensure!(
                    (want[0][e]..want[0][e + 1]).contains(&(p as i32)),
                    "{tag}: one-block row {p} outside expert {e}"
                );
            }
            // 2026-10-14: Downstream bytes through the indexed combine (rows <= 1000: the host
            // routed rows of 8192 are 0.5 GB).
            if rows <= 1000 {
                let w = rng.vec(n, 1.5);
                let shared = vec![0u8; rows * HIDDEN * 2];
                let (w_d, sh_d) = (up_f32(g, &w)?, up(g, &shared)?);
                let ob = rows * HIDDEN * 2;
                let mut outs = Vec::new();
                for (tok, exp, ttp) in [(&ot, &oe, &got_o[3]), (&want[1], &want[2], &got_m[3])] {
                    let eo = up(g, &routed_rows(tok, exp))?;
                    let ttp_d = up(g, ttp)?;
                    let out = zeros(g, ob)?;
                    KernelLaunch::new(g, combine)
                        .grid([rows as u32, (HIDDEN / 256) as u32, 1])
                        .block([256, 1, 1])
                        .arg_ptr(eo)
                        .arg_ptr(ttp_d)
                        .arg_ptr(w_d)
                        .arg_ptr(sh_d)
                        .arg_ptr(out)
                        .arg_u32(HIDDEN as u32)
                        .arg_u32(TOP_K as u32)
                        .launch(s)?;
                    outs.push(read(g, s, out, ob)?);
                    for p in [eo, ttp_d, out] {
                        g.free(p)?;
                    }
                }
                same(
                    &format!("{tag} combined output: one-block vs multi-block order"),
                    &outs[0],
                    &outs[1],
                )?;
                for p in [w_d, sh_d] {
                    g.free(p)?;
                }
            }
            // 2026-10-14: Negative controls.
            if rows > 1 {
                run_new(4, s)?;
                if read(g, s, m[1], n * 4)? == bytes(&want[1]) {
                    bail!("{tag}: negative control (scatter at top_k 4) matched");
                }
                println!("{tag}: negative control (scatter at top_k 4) caught: differs");
                // 2026-10-14: Two rows of one expert are always two tokens (a row's ids are
                // distinct); a spread draw of 16 rows may have no expert with two.
                let off = &want[0];
                match (0..EXPERTS).find(|&e| off[e + 1] - off[e] >= 2) {
                    Some(e) => {
                        let mut flip = want[1].clone();
                        flip.swap(off[e] as usize, off[e] as usize + 1);
                        if same("control", &bytes(&flip), &got_m[1]).is_ok() {
                            bail!(
                                "{tag}: negative control (two rows of expert {e} swapped) matched"
                            );
                        }
                        println!(
                            "{tag}: negative control (rows of expert {e} swapped) caught: differs"
                        );
                    }
                    None => ensure!(!skew, "{tag}: a skewed draw has an expert with two rows"),
                }
            }
            let copies = if rows > 16 { 8 } else { MOE_LAYERS };
            report(
                &format!("sort mb skew={skew}"),
                rows,
                tg(g, s, copies, &run_old)?,
                tg(g, s, copies, &|s| run_new(TOP_K as u32, s))?,
            );
            for p in o.into_iter().chain(m).chain([ids_d, hist_d]) {
                g.free(p)?;
            }
        }
    }
    scan_widest(g, s, scan, rng)
}

/// 2026-10-14: `moe_sort_mb_scan` at its widest block (`MOE_SORT_MB_MAX_EXPERTS` = 1024 experts,
/// 1024 threads, which launches only under its launch bound) over 40 blocks of drawn counts:
/// `blk_hist` becomes each block's exclusive prefix inside its expert, and `expert_offsets` the
/// exclusive prefix of the expert totals plus the grand total, as the host computes them.
fn scan_widest(g: &dyn GpuBackend, s: u64, scan: KernelHandle, rng: &mut Lcg) -> Result<()> {
    const E: usize = 1024;
    const NB: usize = 40;
    let hist: Vec<i32> = (0..NB * E)
        .map(|_| ((rng.next() + 1.0) * 8.0) as i32)
        .collect();
    let (mut want_h, mut want_o) = (vec![0i32; NB * E], vec![0i32; E + 1]);
    for e in 0..E {
        let mut run = 0;
        for b in 0..NB {
            want_h[b * E + e] = run;
            run += hist[b * E + e];
        }
        want_o[e + 1] = want_o[e] + run;
    }
    let hist_d = up(g, &bytes(&hist))?;
    let off_d = up(g, &[0xA5u8; (E + 1) * 4])?;
    KernelLaunch::new(g, scan)
        .grid([1, 1, 1])
        .block([E as u32, 1, 1])
        .arg_ptr(hist_d)
        .arg_ptr(off_d)
        .arg_u32(NB as u32)
        .arg_u32(E as u32)
        .launch(s)?;
    let got_h = ints(&read(g, s, hist_d, NB * E * 4)?);
    let got_o = ints(&read(g, s, off_d, (E + 1) * 4)?);
    for p in [hist_d, off_d] {
        g.free(p)?;
    }
    ensure!(
        got_h == want_h,
        "sort mb scan at {E} experts: blk_hist differs from the host"
    );
    ensure!(
        got_o == want_o,
        "sort mb scan at {E} experts: expert_offsets differ from the host"
    );
    println!("sort mb scan at {E} experts x {NB} blocks: byte-equal to the host");
    Ok(())
}
