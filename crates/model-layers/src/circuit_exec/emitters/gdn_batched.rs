// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The GatedDeltaNet emitters of the batched multi-sequence arm, mirroring
//! `qwen3_ssm/trait_decode_multi_seq/ssm_batched_recurrent.rs` without `--gdn-fused-norm`: the
//! strided conv, recurrence and gated norm, one launch each for every row.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Row `i`'s state is at row 0's plus `i` slots (`GdnFacts::h_slot_bytes`,
//!   `GdnFacts::conv_state_bytes`), the layout the strided kernels address. A step whose slots
//!   are not contiguous runs the `gdn_state_slots_fragmented` route's program instead
//!   (`super::super::routes`). A launch that finds them out of place fails; it never runs.
//! - The strides passed are the edges' row strides in elements, as the legacy site passes
//!   `qkvz_size`, `conv_dim` and `value_dim`.

use anyhow::{Result, ensure};

use super::super::bindings::{MixerFacts, WeightSlot};
use super::super::compile::{Cx, OpEmitter};
use super::super::program::{GdnState, StepEnv};
use super::{dense, dim, expect_kernel, rows};
use crate::layers::ops;

/// 2026-09-30: The layer's state pitch: bytes between two slots' h states and conv windows.
fn pitch(cx: &Cx<'_>) -> Result<(usize, usize)> {
    match cx.layer(0)?.mixer {
        MixerFacts::Gdn(g) => Ok((
            usize::try_from(g.h_slot_bytes)?,
            usize::try_from(g.conv_state_bytes)?,
        )),
        MixerFacts::Attention(_) => anyhow::bail!(
            "`{}` reads GDN state facts from an attention layer",
            cx.g.node(0).id
        ),
    }
}

/// 2026-09-30: Row 0's state of `layer`, after checking every one of `n` rows sits `i` slots
/// past it.
pub(super) fn contiguous_base(
    e: &StepEnv<'_>,
    layer: usize,
    n: usize,
    (h_pitch, conv_pitch): (usize, usize),
) -> Result<GdnState> {
    let base = e.gdn_state(layer, 0)?;
    for i in 1..n {
        let s = e.gdn_state(layer, i)?;
        ensure!(
            s.h == base.h.offset(i * h_pitch) && s.conv == base.conv.offset(i * conv_pitch),
            "layer {layer}: row {i}'s GDN state is not {i} slots past row 0's; the batched arm \
             needs contiguous slots (the fragmented-slots route runs otherwise)"
        );
    }
    Ok(base)
}

fn gdn_layer(cx: &Cx<'_>) -> Result<usize> {
    cx.g.node(0)
        .layer
        .ok_or_else(|| anyhow::anyhow!("a GDN node outside the layers"))
}

/// 2026-09-30: `conv1d_update_l2norm_strided`: the conv1d step with SiLU and the Q/K L2 norm for
/// every row, reading `qkvz` rows at their stride and writing FP32 `[Q | K | V]` rows.
pub(crate) struct Conv1dUpdateL2normStrided;

impl OpEmitter for Conv1dUpdateL2normStrided {
    fn id(&self) -> &'static str {
        "conv1d_update_l2norm_strided"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["conv1d_update", "state_snapshot", "l2_norm"])?;
        expect_kernel(cx, 0, "causal_conv1d_update_l2norm_f32_strided")?;
        ensure!(
            cx.g.group.copies.is_none(),
            "the batched conv takes no rollback snapshots"
        );
        let (nk, kd, nv, vd) = (
            dim(cx, "lin_k_heads")?,
            dim(cx, "lin_k_dim")?,
            dim(cx, "lin_v_heads")?,
            dim(cx, "lin_v_dim")?,
        );
        let key = nk * kd;
        let conv_dim = key * 2 + nv * vd;
        let w = dense(cx.weight(0, WeightSlot::GdnConv1d)?, "conv1d")?;
        let d_conv = u32::try_from(cx.config.linear_conv_kernel_dim)?;
        let (qkvz, in_stride) = cx.strided(cx.g.input(0, 0)?)?;
        let (out, out_stride) = cx.strided(cx.g.output(2, 0)?)?;
        ensure!(
            out_stride == conv_dim,
            "the conv output rows are {out_stride} wide, the kernel writes {conv_dim}"
        );
        let (layer, n, p, k) = (gdn_layer(cx)?, rows(cx)?, pitch(cx)?, cx.handle(0)?);
        cx.push(
            0,
            Box::new(move |e| {
                let st = contiguous_base(e, layer, n as usize, p)?;
                ops::conv1d_update_l2norm_strided(
                    e.gpu,
                    k,
                    st.conv,
                    qkvz,
                    &w,
                    out,
                    conv_dim,
                    d_conv,
                    n,
                    key * 2,
                    kd,
                    1e-6,
                    in_stride,
                    out_stride,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-09-30: `gdn_decode_strided`: the FP32 gated delta rule for every row in one launch over
/// the contiguous h slots (`ops::gdn_decode_f32_strided`, with its L2 occupancy cap).
pub(crate) struct GdnDecodeStrided;

impl OpEmitter for GdnDecodeStrided {
    fn id(&self) -> &'static str {
        "gdn_decode_strided"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["gdn_recurrence"])?;
        expect_kernel(cx, 0, "gated_delta_rule_decode_f32_strided")?;
        let (nk, kd, nv, vd) = (
            dim(cx, "lin_k_heads")?,
            dim(cx, "lin_k_dim")?,
            dim(cx, "lin_v_heads")?,
            dim(cx, "lin_v_dim")?,
        );
        let key = (nk * kd) as usize * 4;
        let (qkv, qk_stride) = cx.strided(cx.g.input(0, 0)?)?;
        let (q, kk, v) = (qkv, qkv.offset(key), qkv.offset(key * 2));
        let (decay, gb_stride) = cx.strided(cx.g.input(0, 1)?)?;
        let (beta, beta_stride) = cx.strided(cx.g.input(0, 2)?)?;
        ensure!(
            beta == decay.offset(nv as usize * 4) && beta_stride == gb_stride,
            "decay and beta do not share rows"
        );
        let (out, out_stride) = cx.strided(cx.g.output(0, 0)?)?;
        let (layer, n, p, k) = (gdn_layer(cx)?, rows(cx)?, pitch(cx)?, cx.handle(0)?);
        cx.push(
            0,
            Box::new(move |e| {
                let st = contiguous_base(e, layer, n as usize, p)?;
                ops::gdn_decode_f32_strided(
                    e.gpu, k, st.h, q, kk, v, decay, beta, out, n, nk, nv, kd, vd, qk_stride,
                    qk_stride, gb_stride, out_stride, e.stream,
                )
            }),
        )
    }
}

/// 2026-09-30: `gated_rms_norm_strided`: the gated RMS norm of every row's FP32 recurrence
/// output, one launch, each operand at its row stride.
pub(crate) struct GatedRmsNormStrided;

impl OpEmitter for GatedRmsNormStrided {
    fn id(&self) -> &'static str {
        "gated_rms_norm_strided"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["gated_rms_norm"])?;
        expect_kernel(cx, 0, "gated_rms_norm_f32_input_strided")?;
        let (nk, kd, nv, vd) = (
            dim(cx, "lin_k_heads")?,
            dim(cx, "lin_k_dim")?,
            dim(cx, "lin_v_heads")?,
            dim(cx, "lin_v_dim")?,
        );
        let w = dense(cx.weight(0, WeightSlot::GdnNorm)?, "gdn norm")?;
        let eps = cx.config.rms_norm_eps as f32;
        let (core, in_stride) = cx.strided(cx.g.input(0, 0)?)?;
        let (qkvz, gate_stride) = cx.strided(cx.g.input(0, 1)?)?;
        let z = qkvz.offset(((nk * kd * 2 + nv * vd) * 2) as usize);
        let (out, out_stride) = cx.strided(cx.g.output(0, 0)?)?;
        let (n, k) = (rows(cx)?, cx.handle(0)?);
        cx.push(
            0,
            Box::new(move |e| {
                ops::gated_rms_norm_strided(
                    e.gpu,
                    k,
                    core,
                    z,
                    &w,
                    out,
                    nv,
                    n,
                    vd,
                    vd,
                    eps,
                    vd,
                    in_stride,
                    gate_stride,
                    out_stride,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-10-03: This module's emitters, for the registry in `mod.rs`.
pub(super) static ALL: &[&dyn OpEmitter] = &[
    &Conv1dUpdateL2normStrided,
    &GdnDecodeStrided,
    &GatedRmsNormStrided,
];
