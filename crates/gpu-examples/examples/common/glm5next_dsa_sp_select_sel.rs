// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The selected-pools payload (`METRALE_GLM_DSA_SP_SELECT_SEL`) in the SP select gate,
//! on the 32k prompt's full chunks, from inputs where about 3% of the keys are invalid and about
//! 1% of the query rows masked, so the fold has entries to drop.
//!
//! 1. Bytes: per chunk, the unsplit token rows; the raw selected pools of the unsplit sub-pass
//!    plan (`row_passes`); each rank's `select_passes_folded` over its two slabs' sub-passes
//!    (`sp_select::run::slab_passes`, 2026-10-17) into a sentinel
//!    selected region (no foreign row written); every folded entry is its raw entry or -1, and
//!    the drops are counted; the ranks' folded rows put together (the gathers' byte copy), then
//!    `expand_folded` over all rows: 0 token rows differ from the unsplit rows.
//! 2. Negative control: `expand_folded` over the raw (unfolded) pools differs from the unsplit
//!    rows somewhere in the prompt (it expands pools the candidacy test drops).
//! 3. Timing (median of 7, spread max - min): the unsplit pass, each rank's token slabs and
//!    folded slabs (slowest rank), the fold launch alone at the largest slab, and the
//!    replicated folded expansion of all rows; the two payloads' net per chunk with the
//!    gathers at the sweep's fit, and the 32k projection over 11 layers.
//!
//! Owner: model-arch examples.
//! Invariants: errors unless every byte check and the control hold.

use super::*;
use metrale_model_arch::glm5next_dsa::select::{
    expand_folded, launch_fold, row_passes, select_passes_folded,
};

/// 2026-10-15: `inp` with about 3% of the `CEILING` keys invalid and about 1% of the
/// `MAX_ROWS` query rows masked (fixed patterns, the same every run).
fn sparse_inputs(gpu: &dyn GpuBackend, inp: &DsaSelectInputs) -> Result<DsaSelectInputs> {
    let mut r = Rng(0x2545_f491_4f6c_dd1d);
    let valid: Vec<u8> = (0..CEILING).map(|_| u8::from(r.unit() > -0.94)).collect();
    let mask: Vec<u8> = (0..MAX_ROWS).map(|i| u8::from(i % 97 != 50)).collect();
    let mut s = *inp;
    s.rows.valid = up(gpu, &valid)?;
    s.q_mask = up(gpu, &mask)?;
    Ok(s)
}

fn sel_bytes(gpu: &dyn GpuBackend, s: &DsaSelectScratch, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(s.selected(), &mut b)?;
    Ok(b)
}

fn i32s(b: &[u8]) -> impl Iterator<Item = i32> + '_ {
    b.chunks(4)
        .map(|t| i32::from_le_bytes([t[0], t[1], t[2], t[3]]))
}

/// 2026-10-15: Rows of width `w` bytes that differ.
fn differ(a: &[u8], b: &[u8], w: usize) -> usize {
    a.chunks(w).zip(b.chunks(w)).filter(|(x, y)| x != y).count()
}

/// 2026-10-15: One chunk; returns (net ms of the token payload, net ms of the selected-pools
/// payload, both with the gathers exposed, and the selected-pools net with them hidden), and
/// adds the control's differing rows and the fold's drops to `ctl`.
#[allow(clippy::too_many_arguments)]
fn sel_chunk(
    gpu: &dyn GpuBackend,
    kern: &Glm5NextDsaKernels,
    c: &Glm5NextDsaConfig,
    inp: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    k: usize,
    pools: usize,
    ctl: &mut (usize, usize),
) -> Result<(f64, f64, f64)> {
    let seq = pools * c.index_kpool;
    let pos: Vec<u8> = (0..k)
        .flat_map(|r| ((seq - k + r) as i32).to_le_bytes())
        .collect();
    gpu.copy_h2d(&pos, inp.q_pos)?;
    let geom = DsaSelectGeometry::plan(c, seq, k)?;
    let (rb, sk) = (c.out_width() * 4, geom.select_k * 4);
    let per = scratch.rows_per_pass(&geom);

    gpu.memset(scratch.tokens(), 0x11, k * rb)?;
    select_tokens_in_passes(gpu, kern, c, &geom, inp, scratch, 0)?;
    let want = tokens(gpu, scratch, k * rb)?;

    let mut raw = vec![0u8; k * sk];
    for (i, (r0, rows)) in row_passes(&[(0, k)], per)?.into_iter().enumerate() {
        select_tokens_rows(
            gpu,
            kern,
            c,
            &geom,
            inp,
            scratch,
            &[(r0, r0 + rows)],
            i == 0,
            0,
        )?;
        raw[r0 * sk..(r0 + rows) * sk].copy_from_slice(&sel_bytes(gpu, scratch, rows * sk)?);
    }

    let mut folded = vec![SENTINEL; k * sk];
    let mut foreign = 0;
    for rank in 0..WORLD {
        let own = owned(k, WORLD, rank)?;
        gpu.memset(scratch.selected(), SENTINEL, k * sk)?;
        let (a, b) = (slab_passes(own[0], per)?, slab_passes(own[1], per)?);
        select_passes_folded(gpu, kern, c, &geom, inp, scratch, &a, true, 0)?;
        select_passes_folded(gpu, kern, c, &geom, inp, scratch, &b, false, 0)?;
        let got = sel_bytes(gpu, scratch, k * sk)?;
        for (r, row) in got.chunks(sk).enumerate() {
            if own.iter().any(|&(lo, hi)| (lo..hi).contains(&r)) {
                folded[r * sk..(r + 1) * sk].copy_from_slice(row);
            } else if row.iter().any(|&x| x != SENTINEL) {
                foreign += 1;
            }
        }
    }
    let (mut dropped, mut broken) = (0usize, 0usize);
    for (f, r) in i32s(&folded).zip(i32s(&raw)) {
        dropped += usize::from(f == -1 && r >= 0);
        broken += usize::from(f != r && f != -1);
    }

    let expand = |sel: &[u8]| -> Result<Vec<u8>> {
        gpu.copy_h2d(sel, scratch.selected())?;
        gpu.memset(scratch.tokens(), 0x11, k * rb)?;
        expand_folded(gpu, kern, c, &geom, inp, scratch, 0)?;
        tokens(gpu, scratch, k * rb)
    };
    let bad = differ(&expand(&folded)?, &want, rb);
    let ctl_rows = differ(&expand(&raw)?, &want, rb);
    ctl.0 += ctl_rows;
    ctl.1 += dropped;

    let unsplit = time_ms(gpu, &|| {
        select_tokens_in_passes(gpu, kern, c, &geom, inp, scratch, 0).map(|_| ())
    })?;
    let slowest = |folded: bool| -> Result<(f64, f64)> {
        let mut m = (0.0f64, 0.0f64);
        for rank in 0..WORLD {
            let own = owned(k, WORLD, rank)?;
            let t = time_ms(gpu, &|| {
                for (i, slab) in own.iter().enumerate() {
                    let sub = slab_passes(*slab, per)?;
                    if folded {
                        select_passes_folded(gpu, kern, c, &geom, inp, scratch, &sub, i == 0, 0)?;
                    } else {
                        select_tokens_passes(gpu, kern, c, &geom, inp, scratch, &sub, i == 0, 0)?;
                    }
                }
                Ok(())
            })?;
            m = if t.0 > m.0 { t } else { m };
        }
        Ok(m)
    };
    let (tok, sel) = (slowest(false)?, slowest(true)?);
    let slab = (0..WORLD)
        .flat_map(|r| owned(k, WORLD, r).unwrap_or_default())
        .map(|(lo, hi)| hi - lo)
        .max()
        .unwrap_or(0);
    let fold_geom = DsaSelectGeometry {
        q_rows: slab.min(per),
        ..geom
    };
    let fold = time_ms(gpu, &|| launch_fold(gpu, kern, &fold_geom, inp, scratch, 0))?;
    let exp = time_ms(gpu, &|| expand_folded(gpu, kern, c, &geom, inp, scratch, 0))?;
    let xag = |row: usize| 2.0 * (XAG_LAT_MS + XAG_MS_PER_MB * (slab * row) as f64 / 1e6);
    let net_tok = tok.0 + xag(rb) - unsplit.0;
    let net_sel = sel.0 + xag(sk) + exp.0 - unsplit.0;
    let hid_sel = sel.0 + exp.0 - unsplit.0;
    println!(
        "sel k={k:5} P={pools:5} select_k={}: token rows differ {bad}, foreign selected rows \
         {foreign}, folded entries dropped {dropped}, broken {broken}, control rows differ \
         {ctl_rows}/{k}",
        geom.select_k
    );
    println!(
        "  unsplit {:.3} (spread {:.3}) | token slabs {:.3} ({:.3}) + xAG {:.3} [{:.2} MB/peer] \
         -> net {net_tok:+.3} | folded slabs {:.3} ({:.3}) + xAG {:.3} [{:.2} MB/peer] + expand \
         all {:.3} ({:.3}) -> net {net_sel:+.3}, gathers hidden {hid_sel:+.3} | fold launch \
         alone {:.4} ms ({} rows, spread {:.4})",
        unsplit.0,
        unsplit.1,
        tok.0,
        tok.1,
        xag(rb),
        (slab * rb) as f64 / 1e6,
        sel.0,
        sel.1,
        xag(sk),
        (slab * sk) as f64 / 1e6,
        exp.0,
        exp.1,
        fold.0,
        fold_geom.q_rows,
        fold.1,
    );
    ensure!(
        bad == 0 && foreign == 0 && broken == 0,
        "k {k} P {pools}: the folded selection's token rows differ from the unsplit ones"
    );
    Ok((net_tok, net_sel, hid_sel))
}

/// 2026-10-15: The payload gate over the 32k chunks on `scratch`, labelled `name`.
pub fn sel_gate(
    gpu: &dyn GpuBackend,
    kern: &Glm5NextDsaKernels,
    c: &Glm5NextDsaConfig,
    inp: &DsaSelectInputs,
    scratch: &DsaSelectScratch,
    name: &str,
) -> Result<()> {
    let sparse = sparse_inputs(gpu, inp)?;
    let mut ctl = (0, 0);
    let mut sum = (0.0, 0.0, 0.0);
    for (k, p) in C32K {
        let (t, s, h) = sel_chunk(gpu, kern, c, &sparse, scratch, k, p, &mut ctl)?;
        sum = (sum.0 + t, sum.1 + s, sum.2 + h);
    }
    println!(
        "c32k payload projection at {name} (11 layers, every full chunk split): token payload \
         {:+.1} ms, selected-pools payload {:+.1} ms (gathers exposed), {:+.1} ms (hidden); \
         control rows {} over {} dropped entries",
        sum.0 * DSA_LAYERS,
        sum.1 * DSA_LAYERS,
        sum.2 * DSA_LAYERS,
        ctl.0,
        ctl.1
    );
    ensure!(
        ctl.0 > 0 && ctl.1 > 0,
        "{name}: the unfolded control expanded the same rows (dropped {})",
        ctl.1
    );
    Ok(())
}
