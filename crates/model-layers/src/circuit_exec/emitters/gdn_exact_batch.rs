// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The GatedDeltaNet conv and recurrence of a batched MTP verify under the exact
//! chain, one launch schedule per run of the row table, mirroring
//! `qwen3_ssm/trait_decode_batched_conv_gdn_multi.rs` (`decode_batched_conv_gdn_multi`):
//! - a contiguous run (the layer's `gdn_verify_run_batched` verdict under the exact chain: the
//!   carry engages through the exact twins and the run's sequences sit on consecutive conv
//!   slots) is `gdn_carry_conv_f32` then `gdn_exact_carry{k}` (`_lazy` from
//!   `GDN_CARRY_LAZY_MIN_SEQS` sequences) over the run's slice of the WY, slot and engaged
//!   tables (`decode_batched_conv_gdn_multi_exact_carry`);
//! - a fragmented carried run folds its pending rows with the model directory's exact fold
//!   (`gdn_exact_carry_flush`, `carry.rs` `carry_flush_kernel`) and the conv fold, then runs each
//!   sequence's chain alone (`gdn_conv_chain_f32`, `gdn_exact_chain{k}`), as the single-sequence
//!   exact verify does;
//! - a run of a verify that does not carry runs each sequence's chain alone.
//!
//! The output norm is not here: legacy norms each carried run and each sequence alone, and the
//! norm is row-local, so the plan's one `gated_rms_norm_f32_input_strided` over every row writes
//! the same bits.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - As `gdn_verify_batch.rs`'s: every launch is the one the plan's run selector names, and a
//!   contiguous run's launch fails rather than read a sequence off its consecutive conv slot.
//! - The conv rows and the recurrence rows are FP32, at the strides
//!   [`super::super::gdn_verify_batch::setup`] checks.

use anyhow::{Result, bail, ensure};
use metrale_circuit::KernelId;
use metrale_circuit::runs::Times;

use super::super::super::compile::{Cx, OpEmitter};
use super::super::gdn_verify_batch::{Common, Dims, Run, fold_run, setup};
use super::{chain_ready, inline_steps};
use crate::layers::ops;

/// 2026-10-03: `gdn_verify_runs_exact`: the conv, snapshot, L2 norm and recurrence of every run
/// under the exact chain.
pub(crate) struct GdnVerifyRunsExact;

impl OpEmitter for GdnVerifyRunsExact {
    fn id(&self) -> &'static str {
        "gdn_verify_runs_exact"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        let (d, common, edges) = setup(cx, self.id())?;
        chain_ready(cx, self.id())?;
        let runs = cx.g.group.runs.clone();
        let (mut seq, mut row) = (0usize, 0usize);
        for r in &runs {
            let (k, n) = (r.run.k as usize, r.run.n as usize);
            let at = Run {
                seq,
                n,
                k,
                qkvz: edges.qkvz.offset(row * common.qkvz_stride as usize * 2),
                qkv: edges.qkv.offset(row * d.conv_dim as usize * 4),
                decay: edges.decay.offset(row * edges.gb_stride as usize * 4),
                out: edges.out.offset(row * edges.out_stride as usize * 4),
            };
            if r.run.contiguous {
                carried_run(cx, &d, &at, &common, &r.launches)?;
            } else {
                alone_run(cx, &d, &at, &common, &r.launches)?;
            }
            seq += n;
            row += k * n;
        }
        Ok(())
    }
}

/// 2026-10-03: `decode_batched_conv_gdn_multi_exact_carry` over one contiguous run.
fn carried_run(
    cx: &mut Cx<'_>,
    d: &Dims,
    at: &Run,
    c: &Common,
    launches: &[(KernelId, Times)],
) -> Result<()> {
    let [(conv_k, Times::Once), (chain_k, Times::Once)] = launches else {
        bail!("a carried exact run launches its FP32 conv and chain once each: {launches:?}");
    };
    let lazy = at.n >= ops::GDN_CARRY_LAZY_MIN_SEQS;
    let twin = format!("gdn_exact_carry{}{}", at.k, if lazy { "_lazy" } else { "" });
    ensure!(
        at.n >= 2 && conv_k.func == "gdn_carry_conv_f32" && chain_k.func == twin,
        "run {}x{}: the plan launches {conv_k} and {chain_k}, not the exact twins its width takes",
        at.k,
        at.n
    );
    let (conv_h, chain_h) = (cx.run_handle(conv_k)?, cx.run_handle(chain_k)?);
    let (layer, n, k, seq) = (c.layer, at.n, at.k, at.seq);
    let (b, conv_bytes, w, d_conv) = (c.carry, c.conv_bytes, c.conv_w, c.d_conv);
    let (qkvz, qkv, qkvz_stride, conv_dim) = (at.qkvz, at.qkv, c.qkvz_stride, d.conv_dim);
    let (qk_ch, kd) = ((d.key * 2) as u32, d.kd);
    cx.push_kernel(
        conv_k,
        Box::new(move |e| {
            let base = e.gdn_state(layer, seq)?.conv;
            for i in 1..n {
                ensure!(
                    e.gdn_state(layer, seq + i)?.conv == base.offset(i * conv_bytes),
                    "layer {layer}: sequence {} of a contiguous run is not {i} conv slots on",
                    seq + i
                );
            }
            ops::gdn_carry_conv_f32(
                e.gpu,
                conv_h,
                base,
                qkvz,
                &w,
                qkv,
                b.conv_stash,
                b.slot_tab.offset(seq * 4),
                b.pend,
                b.conv_seq_elems as u32,
                k as u32,
                conv_dim,
                d_conv,
                qk_ch,
                kd,
                1e-6,
                qkvz_stride,
                conv_dim,
                n as u32,
                lazy,
                e.stream,
            )
        }),
    )?;
    let h_table = c.layer_tables.offset(seq * 8);
    let (q, kk, v) = (qkv, qkv.offset(d.key * 4), qkv.offset(d.key * 8));
    let (gate, beta) = (at.decay, at.decay.offset(d.nv as usize * 4));
    let (out, nk, nv, value) = (at.out, d.nk, d.nv, d.nv * d.vd);
    cx.push_kernel(
        chain_k,
        Box::new(move |e| {
            ops::gdn_exact_carry(
                e.gpu,
                chain_h,
                h_table,
                q,
                kk,
                v,
                gate,
                beta,
                out,
                b.stash,
                b.slot_tab.offset(seq * 4),
                b.pend,
                b.seq_floats as u32,
                n as u32,
                nk,
                nv,
                kd,
                [conv_dim, conv_dim, nv * 2, value],
                b.flag.offset(seq * 4),
                e.stream,
            )
        }),
    )
}

/// 2026-10-03: Each sequence of a run alone (`decode_batched_conv_gdn_exact_chain` per
/// sequence), after the exact fold of its pending rows when the verify carries.
fn alone_run(
    cx: &mut Cx<'_>,
    d: &Dims,
    at: &Run,
    c: &Common,
    launches: &[(KernelId, Times)],
) -> Result<()> {
    let (fold, conv_k, chain_k) = match launches {
        [
            (flush_k, Times::Once),
            (conv_flush_k, Times::Once),
            (conv_k, Times::PerSeq),
            (chain_k, Times::PerSeq),
        ] if flush_k.func == "gdn_exact_carry_flush"
            && conv_flush_k.func == "gdn_carry_conv_flush" =>
        {
            (Some((flush_k, conv_flush_k)), conv_k, chain_k)
        }
        [(conv_k, Times::PerSeq), (chain_k, Times::PerSeq)] => (None, conv_k, chain_k),
        _ => bail!(
            "a fragmented exact run runs each sequence's chain alone, after the exact fold if it \
             carries: {launches:?}"
        ),
    };
    ensure!(
        conv_k.func == "gdn_conv_chain_f32"
            && chain_k.func == format!("gdn_exact_chain{}", at.k)
            && (2..=4).contains(&at.k),
        "run {}x{}!: the plan launches {conv_k} and {chain_k} for each sequence",
        at.k,
        at.n
    );
    let (conv_h, chain_h) = (cx.run_handle(conv_k)?, cx.run_handle(chain_k)?);
    let (n, seq, nv, conv_dim) = (at.n, at.seq, d.nv, d.conv_dim);
    if let Some((flush_k, conv_flush_k)) = fold {
        fold_run(cx, (flush_k, conv_flush_k), c, (seq, n), (nv, conv_dim))?;
    }
    let (layer, k, w, d_conv) = (c.layer, at.k, c.conv_w, c.d_conv);
    let (qk_ch, nk, kd, value) = ((d.key * 2) as u32, d.nk, d.kd, d.nv * d.vd);
    let qkvz_stride = c.qkvz_stride;
    for i in 0..n {
        let s = seq + i;
        let row0 = i * k;
        let x = at.qkvz.offset(row0 * qkvz_stride as usize * 2);
        let y = at.qkv.offset(row0 * conv_dim as usize * 4);
        cx.push_kernel(
            conv_k,
            Box::new(move |e| {
                let st = e.gdn_state(layer, s)?;
                let inter = inline_steps(layer, "conv", st.conv_steps, st.conv, k)?;
                ops::gdn_conv_chain_f32(
                    e.gpu,
                    conv_h,
                    st.conv,
                    x,
                    &w,
                    y,
                    inter,
                    k as u32,
                    conv_dim,
                    d_conv,
                    qk_ch,
                    kd,
                    1e-6,
                    qkvz_stride,
                    conv_dim,
                    e.stream,
                )
            }),
        )?;
        let (q, kk, v) = (y, y.offset(d.key * 4), y.offset(d.key * 8));
        let gate = at.decay.offset(row0 * nv as usize * 2 * 4);
        let beta = gate.offset(nv as usize * 4);
        let out = at.out.offset(row0 * value as usize * 4);
        cx.push_kernel(
            chain_k,
            Box::new(move |e| {
                let st = e.gdn_state(layer, s)?;
                let inter = inline_steps(layer, "h", st.h_steps, st.h, k)?;
                ops::gdn_exact_chain(
                    e.gpu,
                    chain_h,
                    st.h,
                    q,
                    kk,
                    v,
                    gate,
                    beta,
                    out,
                    inter,
                    nk,
                    nv,
                    kd,
                    [conv_dim, conv_dim, nv * 2, value],
                    e.stream,
                )
            }),
        )?;
    }
    Ok(())
}
