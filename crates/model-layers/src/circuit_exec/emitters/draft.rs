// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The MTP draft head's emitters, mirroring `mtp_head/forward.rs` (`forward_one`):
//! the concat of the normed embedding and target hidden, the BF16 GEMVs (`MtpHead::gemv`, a
//! BF16 head), and plain RoPE at the draft position.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - The draft head serves one row; every emitter here refuses more.

use anyhow::{Result, ensure};
use metrale_circuit::planner::Layout;
use metrale_circuit::{LinearRole, OpKind};

use super::super::bindings::WeightSlot;
use super::super::compile::{Cx, GroupRef, OpEmitter};
use super::batched::width;
use super::{attn_facts, dense, dim, expect_kernel, one_row};
use crate::layers::ops;

/// 2026-09-29: `concat`: the normed embedding and the normed target hidden side by side.
pub(crate) struct Concat;

impl OpEmitter for Concat {
    fn id(&self) -> &'static str {
        "concat"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["concat"])?;
        expect_kernel(cx, 0, "bf16_concat")?;
        one_row(cx, "bf16_concat")?;
        let (a, b) = (cx.ptr(cx.g.input(0, 0)?)?, cx.ptr(cx.g.input(0, 1)?)?);
        let y = cx.ptr(cx.g.output(0, 0)?)?;
        let (k, h) = (cx.handle(0)?, dim(cx, "hidden")?);
        cx.push(
            0,
            Box::new(move |e| ops::bf16_concat(e.gpu, k, a, b, y, h, e.stream)),
        )
    }
}

/// 2026-09-29: `dense_gemv`: one BF16 projection of one row, or the FFN's gate and up as two.
pub(crate) struct DenseGemv;

impl OpEmitter for DenseGemv {
    fn id(&self) -> &'static str {
        "dense_gemv"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear"])?;
        one_row(cx, "dense_gemv_bf16")?;
        let OpKind::Linear(role) = cx.g.node(0).op else {
            anyhow::bail!("`{}` is not a projection", cx.g.node(0).id);
        };
        let (inp, out) = (cx.g.input(0, 0)?, cx.g.output(0, 0)?);
        let x = cx.ptr(inp)?;
        let k_dim = width(cx, inp)?;
        let launches: Vec<_> = if role == LinearRole::GateUp {
            let inter = width(cx, out)? / 2;
            let base = cx.ptr(out)?;
            vec![
                (WeightSlot::FfnGate, base, inter),
                (WeightSlot::FfnUp, base.offset(inter as usize * 2), inter),
            ]
        } else {
            vec![(WeightSlot::Linear(role), cx.ptr(out)?, width(cx, out)?)]
        };
        ensure!(
            launches.len() == cx.g.group.kernels.len(),
            "`dense_gemv` launches one kernel per projection"
        );
        for (i, (slot, y, n)) in launches.into_iter().enumerate() {
            expect_kernel(cx, i, "dense_gemv_bf16")?;
            let w = dense(cx.weight(0, slot)?, role.name())?;
            let k = cx.handle(i)?;
            cx.push(
                i,
                Box::new(move |e| ops::dense_gemv(e.gpu, k, x, &w, y, n, k_dim, e.stream)),
            )?;
        }
        Ok(())
    }
}

/// 2026-09-29: `rope`: plain RoPE in place on one row's Q and K at the step's position.
pub(crate) struct Rope;

impl OpEmitter for Rope {
    fn id(&self) -> &'static str {
        "rope"
    }

    fn constrain(&self, g: &GroupRef<'_>, layout: &mut Layout) -> Result<()> {
        layout.aliases.push((g.output(0, 0)?, g.input(0, 0)?));
        layout.aliases.push((g.output(0, 1)?, g.input(0, 1)?));
        Ok(())
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["rope"])?;
        expect_kernel(cx, 0, "rope_forward")?;
        one_row(cx, "rope_forward at one position")?;
        let a = attn_facts(cx, 0)?;
        ensure!(!a.rope.mrope_interleaved, "the draft head runs plain RoPE");
        let (q, kk) = (cx.ptr(cx.g.input(0, 0)?)?, cx.ptr(cx.g.input(0, 1)?)?);
        let positions = cx.meta()?.positions;
        let k = cx.handle(0)?;
        let (nq, nkv, hd, rd, theta) = (
            a.num_q_heads,
            a.num_kv_heads,
            a.head_dim,
            a.rope.rotary_dim,
            a.rope.theta,
        );
        cx.push(
            0,
            Box::new(move |e| {
                ops::rope(
                    e.gpu, k, q, kk, positions, 1, nq, nkv, hd, rd, theta, e.stream,
                )
            }),
        )
    }
}
