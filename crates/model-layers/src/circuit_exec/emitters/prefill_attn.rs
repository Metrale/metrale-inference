// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The attention emitters of the prefill modes (LIFECYCLE-DESIGN.md 15.4). Each mirrors
//! the legacy prefill call site it names: the same `ops::*` function, the same arguments, the
//! row count from the step (`StepEnv::prefill`).
//!
//! - `prefill_attn_input_norm`: the full-attention layer's input norm with its residual copy
//!   (`qwen3_attention/trait_impl/prefill_inner.rs:86`).
//! - `prefill_attention`: the whole attention sublayer from the Q/K/V projections to the O
//!   projection, one group, on the contiguous route (`prefill`, `cache_skip.rs`) or the paged
//!   route (`prefill_chunk`, `paged.rs`) (`prefill_attn_core.rs`).
//! - `prefill_attn_add_post_norm`: the attention residual add fused with the FFN's input norm
//!   (`prefill_inner.rs:302`).
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Every launch reads its row count from the step at run time; a plan's rows are only its
//!   bucket's top.
//! - The buffers are the ones the legacy layer uses: the stream edges are the model's hidden
//!   buffer, the normed rows `norm_output`, the residual copy `Fixed::residual`.

use anyhow::Result;

use super::super::bindings::WeightSlot;
use super::super::compile::{Cx, OpEmitter};
use super::{dense, dim, expect_kernel, norm_slot};
use crate::layers::ops;

#[path = "prefill_attn_core.rs"]
mod attn_core;
#[path = "prefill_attn_route.rs"]
mod attn_route;

/// 2026-10-03: `prefill_attn_input_norm`: `ops::rms_norm_residual` over the pass's rows, the
/// normed rows into the input norm's output edge (`norm_output`) and the input into the residual
/// copy (`prefill_inner.rs:85-97`).
pub(crate) struct PrefillAttnInputNorm;

impl OpEmitter for PrefillAttnInputNorm {
    fn id(&self) -> &'static str {
        "prefill_attn_input_norm"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["rms_norm"])?;
        expect_kernel(cx, 0, "rms_norm_residual")?;
        let w = dense(cx.weight(0, norm_slot(cx, 0)?)?, "input norm")?;
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        let y = cx.ptr(cx.g.output(0, 0)?)?;
        let residual = cx.fixed.residual;
        let (k, h) = (cx.handle(0)?, dim(cx, "hidden")?);
        let eps = cx.config.rms_norm_eps as f32;
        cx.push(
            0,
            Box::new(move |e| {
                let t = e.prefill()?.tokens;
                ops::rms_norm_residual(e.gpu, k, x, &w, y, residual, t, h, eps, e.stream)
            }),
        )
    }
}

/// 2026-10-03: `prefill_attn_add_post_norm`: `ops::residual_add_rms_norm` over the pass's rows:
/// the O projection's rows added into the hidden stream, the sum normed by the FFN's input norm
/// into its output edge (`norm_output`) and copied to the residual (`prefill_inner.rs:302-314`).
pub(crate) struct PrefillAttnAddPostNorm;

impl OpEmitter for PrefillAttnAddPostNorm {
    fn id(&self) -> &'static str {
        "prefill_attn_add_post_norm"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["residual_add", "rms_norm"])?;
        expect_kernel(cx, 0, "residual_add_rms_norm")?;
        let w = dense(cx.weight(1, WeightSlot::PostNorm)?, "post-attention norm")?;
        let hidden = cx.ptr(cx.g.input(0, 0)?)?;
        let src = cx.ptr(cx.g.input(0, 1)?)?;
        let out = cx.ptr(cx.g.output(1, 0)?)?;
        let residual = cx.fixed.residual;
        let (k, h) = (cx.handle(0)?, dim(cx, "hidden")?);
        let eps = cx.config.rms_norm_eps as f32;
        cx.push(
            0,
            Box::new(move |e| {
                let t = e.prefill()?.tokens;
                ops::residual_add_rms_norm(
                    e.gpu, k, hidden, src, &w, out, residual, t, h, eps, e.stream,
                )
            }),
        )
    }
}

/// 2026-10-03: This module's emitters, for the registry in `mod.rs`.
pub(super) static ALL: &[&dyn OpEmitter] = &[
    &PrefillAttnInputNorm,
    &attn_core::PrefillAttention,
    &PrefillAttnAddPostNorm,
];
