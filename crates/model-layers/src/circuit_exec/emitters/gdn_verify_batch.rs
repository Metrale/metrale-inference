// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The GatedDeltaNet conv and recurrence of a batched MTP verify, one launch
//! schedule per run of the row table, mirroring `qwen3_ssm/trait_decode_batched/conv_gdn_route.rs`
//! with the carried state (`carry.rs`, `trait_decode_batched_conv_gdn_multi.rs`):
//! - a contiguous run is `gdn_carry_conv` then `gdn_carry_wy{k}` (`_lazy` from
//!   `GDN_CARRY_LAZY_MIN_SEQS` sequences), over the run's slice of the WY, slot and engaged
//!   tables;
//! - a fragmented run folds its pending rows (`gdn_carry_flush`, `gdn_carry_conv_flush`), then
//!   runs each sequence alone as the single-sequence verify does: the conv per row with a copy of
//!   the window after each row but the last, then one `gated_delta_rule_wy{k}` over its own
//!   state and rollback slots.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Every launch is the one the plan's run selector names ([`Cx::run_handle`] refuses any
//!   other); the counts are the plan's.
//! - A contiguous run's sequences sit at consecutive conv slots; the launch checks it and fails
//!   rather than read another sequence's state.

use anyhow::{Context, Result, bail, ensure};
use metrale_circuit::runs::Times;

use super::super::bindings::{MixerFacts, WeightSlot};
use super::super::compile::{Cx, OpEmitter};
use super::super::program::MAX_VERIFY_STEPS;
use super::{dense, dim};
use crate::layer::{GdnCarryBinding, VERIFY_WY_LAYER_STRIDE_BYTES};
use crate::layers::ops;

/// 2026-09-30: `gdn_verify_runs`: the conv, snapshot, L2 norm and recurrence of every run.
pub(crate) struct GdnVerifyRuns;

struct Dims {
    nk: u32,
    kd: u32,
    nv: u32,
    vd: u32,
    conv_dim: u32,
    key: usize,
}

impl OpEmitter for GdnVerifyRuns {
    fn id(&self) -> &'static str {
        "gdn_verify_runs"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(
            self.id(),
            &[
                "conv1d_update",
                "state_snapshot",
                "l2_norm",
                "gdn_recurrence",
            ],
        )?;
        let table = cx
            .table()
            .context("a per-run group outside a batched verify")?;
        ensure!(
            table.runs.len() == cx.g.group.runs.len(),
            "the plan resolved {} runs for a table of {}",
            cx.g.group.runs.len(),
            table.runs.len()
        );
        let d = Dims {
            nk: dim(cx, "lin_k_heads")?,
            kd: dim(cx, "lin_k_dim")?,
            nv: dim(cx, "lin_v_heads")?,
            vd: dim(cx, "lin_v_dim")?,
            conv_dim: 0,
            key: 0,
        };
        let key = (d.nk * d.kd) as usize;
        let d = Dims {
            conv_dim: (key * 2) as u32 + d.nv * d.vd,
            key,
            ..d
        };
        let layer =
            cx.g.node(0)
                .layer
                .context("a GDN node outside the layers")?;
        let (carry, conv_bytes) = match cx.layer(0)?.mixer {
            MixerFacts::Gdn(g) => (
                g.carry
                    .context("the batched verify runs carried; the layer has no carry binding")?,
                usize::try_from(g.conv_state_bytes)?,
            ),
            MixerFacts::Attention(_) => bail!("`{}` is not a GDN layer", cx.g.node(0).id),
        };
        let ssm_idx = cx.g.circuit.layer_kinds[..layer]
            .iter()
            .filter(|k| **k == metrale_circuit::LayerKind::LinearAttention)
            .count();
        let tables = cx.fixed.verify_wy_tables;
        ensure!(!tables.is_null(), "the batched verify needs the WY tables");
        let layer_tables = tables.offset(ssm_idx * VERIFY_WY_LAYER_STRIDE_BYTES);
        let conv_w = dense(cx.weight(0, WeightSlot::GdnConv1d)?, "conv1d")?;
        let d_conv = u32::try_from(cx.config.linear_conv_kernel_dim)?;
        let (qkvz, qkvz_stride) = cx.strided(cx.g.input(0, 0)?)?;
        let (qkv, qkv_stride) = cx.strided(cx.g.output(2, 0)?)?;
        let (decay, gb_stride) = cx.strided(cx.g.input(3, 1)?)?;
        let (beta, _) = cx.strided(cx.g.input(3, 2)?)?;
        let (out, out_stride) = cx.strided(cx.g.output(3, 0)?)?;
        ensure!(
            qkvz_stride == d.conv_dim + d.nv * d.vd
                && qkv_stride == d.conv_dim
                && out_stride == d.nv * d.vd
                && gb_stride == d.nv * 2
                && beta == decay.offset(d.nv as usize * 4),
            "the batched verify's GDN rows are not laid out as the kernels read them (the \
             BA-gates group packs `[decay | beta]` rows) \
             (qkv {qkv_stride}, core {out_stride}, gates {gb_stride})"
        );
        let runs = cx.g.group.runs.clone();
        let (mut seq, mut row) = (0usize, 0usize);
        for r in &runs {
            let (k, n) = (r.run.k as usize, r.run.n as usize);
            let at = Run {
                seq,
                n,
                k,
                qkvz: qkvz.offset(row * qkvz_stride as usize * 2),
                qkv: qkv.offset(row * d.conv_dim as usize * 2),
                decay: decay.offset(row * gb_stride as usize * 4),
                out: out.offset(row * out_stride as usize * 2),
            };
            let common = Common {
                layer,
                carry,
                conv_bytes,
                layer_tables,
                conv_w,
                d_conv,
                qkvz_stride,
            };
            if r.run.contiguous {
                contiguous_run(cx, &d, &at, &common, &r.launches)?;
            } else {
                fragmented_run(cx, &d, &at, &common, &r.launches)?;
            }
            seq += n;
            row += k * n;
        }
        Ok(())
    }
}

/// 2026-09-30: Where one run's rows and sequences start.
struct Run {
    seq: usize,
    n: usize,
    k: usize,
    qkvz: metrale_gpu_runtime::gpu::DevicePtr,
    qkv: metrale_gpu_runtime::gpu::DevicePtr,
    decay: metrale_gpu_runtime::gpu::DevicePtr,
    out: metrale_gpu_runtime::gpu::DevicePtr,
}

/// 2026-09-30: What every run of the layer shares.
struct Common {
    layer: usize,
    carry: GdnCarryBinding,
    conv_bytes: usize,
    layer_tables: metrale_gpu_runtime::gpu::DevicePtr,
    conv_w: crate::weight_map::DenseWeight,
    d_conv: u32,
    qkvz_stride: u32,
}

fn contiguous_run(
    cx: &mut Cx<'_>,
    d: &Dims,
    at: &Run,
    c: &Common,
    launches: &[(metrale_circuit::KernelId, Times)],
) -> Result<()> {
    let [(conv_k, Times::Once), (wy_k, Times::Once)] = launches else {
        bail!("a contiguous run launches its carried conv and WY once each: {launches:?}");
    };
    ensure!(
        conv_k.func == "gdn_carry_conv"
            && wy_k.func
                == format!(
                    "gdn_carry_wy{}{}",
                    at.k,
                    if at.n >= ops::GDN_CARRY_LAZY_MIN_SEQS {
                        "_lazy"
                    } else {
                        ""
                    }
                ),
        "run {}x{}: the plan launches {conv_k} and {wy_k}, not the carried kernels its width takes",
        at.k,
        at.n
    );
    let (conv_h, wy_h) = (cx.run_handle(conv_k)?, cx.run_handle(wy_k)?);
    let (layer, n, k, seq) = (c.layer, at.n, at.k, at.seq);
    let (b, conv_bytes, w, d_conv) = (c.carry, c.conv_bytes, c.conv_w, c.d_conv);
    let (qkvz, qkv, qkvz_stride, conv_dim) = (at.qkvz, at.qkv, c.qkvz_stride, d.conv_dim);
    let (qk_ch, kd) = ((d.key * 2) as u32, d.kd);
    let lazy = n >= ops::GDN_CARRY_LAZY_MIN_SEQS;
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
            ops::gdn_carry_conv(
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
                qkvz_stride,
                conv_dim,
                1e-6,
                n as u32,
                (conv_bytes / 4) as u32,
                k as u32 * qkvz_stride,
                k as u32 * conv_dim,
                lazy,
                e.stream,
            )
        }),
    )?;
    let h_table = c.layer_tables.offset(seq * 8);
    let (q, kk, v) = (qkv, qkv.offset(d.key * 2), qkv.offset(d.key * 4));
    let (gate, beta) = (at.decay, at.decay.offset(d.nv as usize * 4));
    let (out, nk, nv) = (at.out, d.nk, d.nv);
    cx.push_kernel(
        wy_k,
        Box::new(move |e| {
            ops::gdn_carry_wy(
                e.gpu,
                wy_h,
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
                conv_dim,
                conv_dim,
                nv * 2,
                kd,
                b.flag.offset(seq * 4),
                e.stream,
            )
        }),
    )
}

fn fragmented_run(
    cx: &mut Cx<'_>,
    d: &Dims,
    at: &Run,
    c: &Common,
    launches: &[(metrale_circuit::KernelId, Times)],
) -> Result<()> {
    // 2026-09-30: A carried verify folds the run's pending rows first; one that does not carry
    // has none.
    let (fold, conv_k, wy_k) = match launches {
        [
            (flush_k, Times::Once),
            (conv_flush_k, Times::Once),
            (conv_k, Times::PerRow),
            (wy_k, Times::PerSeq),
        ] if flush_k.func == "gdn_carry_flush" && conv_flush_k.func == "gdn_carry_conv_flush" => {
            (Some((flush_k, conv_flush_k)), conv_k, wy_k)
        }
        [(conv_k, Times::PerRow), (wy_k, Times::PerSeq)] => (None, conv_k, wy_k),
        _ => bail!(
            "a fragmented run runs each sequence alone, after a fold if it carries: {launches:?}"
        ),
    };
    ensure!(
        conv_k.func == "causal_conv1d_update_l2norm"
            && wy_k.func == format!("gated_delta_rule_wy{}", at.k),
        "run {}x{}!: the plan launches {conv_k} and {wy_k} for each sequence",
        at.k,
        at.n
    );
    ensure!(
        at.k <= MAX_VERIFY_STEPS + 1,
        "a {}-row sequence has no rollback slots here",
        at.k
    );
    let (conv_h, wy_h) = (cx.run_handle(conv_k)?, cx.run_handle(wy_k)?);
    let (n, seq, nv, conv_dim) = (at.n, at.seq, d.nv, d.conv_dim);
    if let Some((flush_k, conv_flush_k)) = fold {
        fold_run(cx, (flush_k, conv_flush_k), c, (seq, n), (nv, conv_dim))?;
    }
    let (layer, k, w, d_conv) = (c.layer, at.k, c.conv_w, c.d_conv);
    let (qk_ch, kd, conv_bytes) = ((d.key * 2) as u32, d.kd, c.conv_bytes);
    let (nk, vd) = (d.nk, d.vd);
    for i in 0..n {
        let s = seq + i;
        let row0 = i * k;
        for t in 0..k {
            let x = at.qkvz.offset((row0 + t) * c.qkvz_stride as usize * 2);
            let y = at.qkv.offset((row0 + t) * conv_dim as usize * 2);
            cx.push_kernel(
                conv_k,
                Box::new(move |e| {
                    let st = e.gdn_state(layer, s)?;
                    ops::conv1d_update_l2norm(
                        e.gpu, conv_h, st.conv, x, &w, y, conv_dim, d_conv, 1, qk_ch, kd, 1e-6,
                        e.stream,
                    )
                }),
            )?;
            if t + 1 < k {
                cx.push_copy(Box::new(move |e| {
                    let st = e.gdn_state(layer, s)?;
                    let to = st.conv_steps[t];
                    ensure!(!to.is_null(), "layer {layer} has no conv rollback slot {t}");
                    e.gpu.copy_d2d_async(st.conv, to, conv_bytes, e.stream)
                }))?;
            }
        }
        let qkv = at.qkv.offset(row0 * conv_dim as usize * 2);
        let (q, kk, v) = (qkv, qkv.offset(d.key * 2), qkv.offset(d.key * 4));
        let gate = at.decay.offset(row0 * nv as usize * 2 * 4);
        let beta = gate.offset(nv as usize * 4);
        let out = at.out.offset(row0 * (nv * vd) as usize * 2);
        cx.push_kernel(
            wy_k,
            Box::new(move |e| {
                let st = e.gdn_state(layer, s)?;
                let hi = st.h_steps;
                ensure!(
                    hi[..k - 1].iter().all(|p| !p.is_null()),
                    "layer {layer} lacks the h rollback slots of a {k}-row verify"
                );
                let (st_stride, gb) = (conv_dim, nv * 2);
                match k {
                    2 => ops::gdn_decode_wy2(
                        e.gpu, wy_h, st.h, q, kk, v, gate, beta, out, hi[0], 1, nk, nv, kd, vd,
                        st_stride, st_stride, gb, false, e.stream,
                    ),
                    3 => ops::gdn_decode_wy3(
                        e.gpu, wy_h, st.h, q, kk, v, gate, beta, out, hi[0], hi[1], 1, nk, nv, kd,
                        vd, st_stride, st_stride, gb, false, e.stream,
                    ),
                    4 => ops::gdn_decode_wy4(
                        e.gpu, wy_h, st.h, q, kk, v, gate, beta, out, hi[0], hi[1], hi[2], 1, nk,
                        nv, kd, vd, st_stride, st_stride, gb, false, e.stream,
                    ),
                    other => bail!("no per-sequence WY kernel for {other} rows"),
                }
            }),
        )?;
    }
    Ok(())
}

/// 2026-09-30: Fold the pending rows of the run's `n` sequences from `seq` into their h state and
/// conv window (`carry_flush_run`), before each sequence runs alone and reads them.
fn fold_run(
    cx: &mut Cx<'_>,
    (flush_k, conv_flush_k): (&metrale_circuit::KernelId, &metrale_circuit::KernelId),
    c: &Common,
    (seq, n): (usize, usize),
    (nv, conv_dim): (u32, u32),
) -> Result<()> {
    let (flush_h, conv_flush_h) = (cx.run_handle(flush_k)?, cx.run_handle(conv_flush_k)?);
    let b = c.carry;
    let h_table = c.layer_tables.offset(seq * 8);
    cx.push_kernel(
        flush_k,
        Box::new(move |e| {
            ops::gdn_carry_flush(
                e.gpu,
                flush_h,
                h_table,
                0,
                b.stash,
                0,
                b.slot_tab.offset(seq * 4),
                b.pend,
                0,
                b.seq_floats as u32,
                n as u32,
                nv,
                1,
                e.stream,
            )
        }),
    )?;
    let window_elems = (c.conv_bytes / 4) as u32;
    cx.push_kernel(
        conv_flush_k,
        Box::new(move |e| {
            ops::gdn_carry_conv_flush(
                e.gpu,
                conv_flush_h,
                b.conv_tab.offset(seq * 8),
                0,
                b.conv_stash,
                0,
                b.slot_tab.offset(seq * 4),
                b.pend,
                0,
                b.conv_seq_elems as u32,
                n as u32,
                conv_dim,
                window_elems / conv_dim,
                1,
                e.stream,
            )
        }),
    )
}
