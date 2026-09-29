// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The embedding, norm and residual-add emitters. Legacy sites mirrored:
//! `qwen3_ssm/trait_decode.rs` and `qwen3_attention/trait_impl/decode_inner.rs` (input norm,
//! post-mixer add+norm, FFN add) and `model-engine impl_a3_norm.rs` (final norm).
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - The residual stream is updated in place: a group's stream input and output are one buffer,
//!   or the group is refused.

use anyhow::{Result, ensure};

use super::super::compile::{Cx, OpEmitter};
use super::{attn_facts, dense, dim, norm_slot, per_row, rows};
use crate::layers::ops;

/// 2026-09-28: The group's stream input `(i, j)` and stream output `(i2, j2)` are one buffer.
fn in_place(cx: &Cx<'_>, (i, j): (usize, usize), (i2, j2): (usize, usize)) -> Result<()> {
    let a = cx.ptr(cx.g.input(i, j)?)?;
    let b = cx.ptr(cx.g.output(i2, j2)?)?;
    ensure!(
        a == b,
        "group {}: the residual stream must be updated in place",
        cx.g.index
    );
    Ok(())
}

/// 2026-09-28: `embed_copy`: the dispatch prologue embeds the token into `hidden` before the
/// program runs (`decode_dispatch_with`), so the group launches nothing; its output must be
/// that buffer.
pub(crate) struct EmbedCopy;

impl OpEmitter for EmbedCopy {
    fn id(&self) -> &'static str {
        "embed_copy"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["embed"])?;
        // 2026-09-29: The draft head's embedding is not the stream: `forward_one` copies it
        // into the draft embedding buffer.
        let want = if cx.mode == metrale_circuit::Mode::Draft {
            cx.draft_fixed()?.embed
        } else {
            cx.fixed.hidden
        };
        ensure!(
            cx.ptr(cx.g.output(0, 0)?)? == want,
            "the embedding must land in the buffer the host copies it to"
        );
        Ok(())
    }
}

/// 2026-09-28: `rms_norm_residual`: a mixer's input norm, which also copies the stream into
/// `residual`.
pub(crate) struct RmsNormResidual;

impl OpEmitter for RmsNormResidual {
    fn id(&self) -> &'static str {
        "rms_norm_residual"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["rms_norm"])?;
        let k = cx.handle(0)?;
        let (x, xn) = (cx.ptr(cx.g.input(0, 0)?)?, cx.ptr(cx.g.output(0, 0)?)?);
        let w = dense(cx.weight(0, norm_slot(cx, 0)?)?, "input norm")?;
        let (res, n, h) = (cx.fixed.residual, rows(cx)?, dim(cx, "hidden")?);
        let eps = cx.config.rms_norm_eps as f32;
        cx.push(
            0,
            Box::new(move |e| {
                ops::rms_norm_residual(e.gpu, k, x, &w, xn, res, n, h, eps, e.stream)
            }),
        )
    }
}

/// 2026-09-28: The launch shared by the two add+norm emitters: `hidden += src`, then the
/// norm of member 1 into its output, copying the stream into `residual`.
fn add_norm(cx: &mut Cx<'_>, exact: bool) -> Result<()> {
    in_place(cx, (0, 0), (0, 0))?;
    let k = cx.handle(0)?;
    let x = cx.ptr(cx.g.input(0, 0)?)?;
    let src = cx.ptr(cx.g.input(0, 1)?)?;
    let xn = cx.ptr(cx.g.output(1, 0)?)?;
    let w = dense(cx.weight(1, norm_slot(cx, 1)?)?, "add+norm weight")?;
    let (res, n, h) = (cx.fixed.residual, rows(cx)?, dim(cx, "hidden")?);
    let eps = cx.config.rms_norm_eps as f32;
    cx.push(
        0,
        Box::new(move |e| {
            if exact {
                ops::residual_add_rms_norm_exact(e.gpu, k, x, src, &w, xn, res, n, h, eps, e.stream)
            } else {
                ops::residual_add_rms_norm(e.gpu, k, x, src, &w, xn, res, n, h, eps, e.stream)
            }
        }),
    )
}

/// 2026-09-28: `residual_add_rms_norm`: the mixer's residual add and the FFN's input norm.
pub(crate) struct ResidualAddRmsNorm;

impl OpEmitter for ResidualAddRmsNorm {
    fn id(&self) -> &'static str {
        "residual_add_rms_norm"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["residual_add", "rms_norm"])?;
        add_norm(cx, false)
    }
}

/// 2026-09-28: `residual_add_rms_norm_exact`: layer i's FFN add and layer i + 1's input norm,
/// with the bytes of the unfused `bf16_residual_add` + `rms_norm_residual`.
pub(crate) struct ResidualAddRmsNormExact;

impl OpEmitter for ResidualAddRmsNormExact {
    fn id(&self) -> &'static str {
        "residual_add_rms_norm_exact"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["residual_add", "rms_norm"])?;
        add_norm(cx, true)
    }
}

/// 2026-09-28: `residual_add`: the FFN's residual add where no norm follows in its launch; over
/// all rows at once, or one launch per row (the GDN layers' 2- and 3-row FFN,
/// `trait_decode_multi_seq.rs`).
pub(crate) struct ResidualAdd;

impl OpEmitter for ResidualAdd {
    fn id(&self) -> &'static str {
        "residual_add"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["residual_add"])?;
        in_place(cx, (0, 0), (0, 0))?;
        let k = cx.handle(0)?;
        let h = dim(cx, "hidden")?;
        let (reps, x_edge, src_edge) = (cx.reps(), cx.g.input(0, 0)?, cx.g.input(0, 1)?);
        let count = if reps == 1 {
            rows(cx)? * h
        } else {
            per_row(cx)?;
            h
        };
        for i in 0..reps {
            let (x, src) = (cx.row_ptr(x_edge, i)?, cx.row_ptr(src_edge, i)?);
            cx.push(
                0,
                Box::new(move |e| ops::residual_add(e.gpu, k, x, src, count, e.stream)),
            )?;
        }
        Ok(())
    }
}

/// 2026-09-28: `rms_norm`: the final norm (a row per token) or a per-head Q/K norm (a row per
/// head, as `attention_forward` launches it).
pub(crate) struct RmsNorm;

impl OpEmitter for RmsNorm {
    fn id(&self) -> &'static str {
        "rms_norm"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        let k = cx.handle(0)?;
        let eps = cx.config.rms_norm_eps as f32;
        let (x, y) = (cx.ptr(cx.g.input(0, 0)?)?, cx.ptr(cx.g.output(0, 0)?)?);
        let op = cx.g.node(0).op.name();
        let (w, norm_rows, cols) = if op == "final_norm" {
            cx.g.expect_ops(self.id(), &["final_norm"])?;
            // 2026-09-29: The draft head's own final norm in a draft plan.
            let w = if cx.mode == metrale_circuit::Mode::Draft {
                dense(
                    cx.weight(0, super::super::bindings::WeightSlot::FinalNorm)?,
                    "final norm",
                )?
            } else {
                ensure!(
                    !cx.config.final_norm_identity,
                    "a checkpoint without a final norm copies instead"
                );
                cx.head.final_norm
            };
            (w, rows(cx)?, dim(cx, "hidden")?)
        } else if op == "rms_norm" {
            // 2026-09-29: A plain row norm (the draft head's input norms).
            let w = dense(cx.weight(0, norm_slot(cx, 0)?)?, "norm")?;
            (w, rows(cx)?, dim(cx, "hidden")?)
        } else {
            cx.g.expect_ops(self.id(), &["qk_norm"])?;
            let a = attn_facts(cx, 0)?;
            let heads = if cx.g.node(0).local == "q_norm" {
                a.num_q_heads
            } else {
                a.num_kv_heads
            };
            let width = cx.g.circuit.edges[cx.g.output(0, 0)?].dim_value;
            ensure!(
                width == u64::from(heads * a.head_dim),
                "`{}`: {heads} heads of {} do not fill its {width}-wide edge",
                cx.g.node(0).id,
                a.head_dim
            );
            let w = dense(cx.weight(0, norm_slot(cx, 0)?)?, "q/k norm")?;
            (w, rows(cx)? * heads, a.head_dim)
        };
        cx.push(
            0,
            Box::new(move |e| ops::rms_norm(e.gpu, k, x, &w, y, norm_rows, cols, eps, e.stream)),
        )
    }
}
