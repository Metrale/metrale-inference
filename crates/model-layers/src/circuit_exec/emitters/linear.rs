// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The projection and activation emitters. Legacy sites mirrored: `ssm_forward.rs`
//! (qkvz, out_proj), `qwen3_attention/decode/attention_forward/q_proj.rs`,
//! `attention_forward_kv.rs`, `attention_forward_oproj.rs`, `dense_ffn_decode.rs` (gate+up,
//! split SiLU, down) and `model-engine impl_a3_lm_head.rs` (the BF16 head).
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - A projection's N and K are its output and input edges' widths, so a binding whose weight
//!   has another shape is caught by the kernel's own geometry, never by a wrong buffer size.
//! - The gate+up edge holds the gate rows then the up rows: gate at its base, up
//!   `rows * inter` elements after it; `silu_mul` reads it the same way.

use anyhow::{Context, Result, ensure};
use metrale_circuit::planner::Layout;
use metrale_circuit::{LinearRole, OpKind};

use super::super::bindings::{MixerFacts, WeightSlot};
use super::super::compile::{Cx, GroupRef, OpEmitter};
use super::{attn_facts, dense, dim, expect_kernel, nvfp4, one_row, rows};
use crate::layers::ops;

fn role(cx: &Cx<'_>, i: usize) -> Result<LinearRole> {
    match cx.g.node(i).op {
        OpKind::Linear(r) => Ok(r),
        _ => anyhow::bail!("`{}` is not a projection", cx.g.node(i).id),
    }
}

fn width(cx: &Cx<'_>, edge: usize) -> Result<u32> {
    u32::try_from(cx.g.circuit.edges[edge].dim_value).context("edge width overflows u32")
}

/// 2026-09-28: `w4a16_decode_gemv`: one NVFP4 projection of one row through the single-warp
/// GEMV (`ops::w4a16_decode_gemv` with `gemv_sw` on).
pub(crate) struct W4a16DecodeGemv;

impl OpEmitter for W4a16DecodeGemv {
    fn id(&self) -> &'static str {
        "w4a16_decode_gemv"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear"])?;
        expect_kernel(cx, 0, "w4a16_gemv_sw")?;
        one_row(cx, "w4a16_gemv_sw")?;
        let r = role(cx, 0)?;
        if r == LinearRole::Qkvz {
            let MixerFacts::Gdn(f) = cx.layer(0)?.mixer else {
                anyhow::bail!("a qkvz projection on a non-GDN layer");
            };
            ensure!(
                f.qkvz_deinterleaved,
                "this layer's qkvz weight is not in the deinterleaved order the plan assumes"
            );
        }
        let w = nvfp4(cx.weight(0, WeightSlot::Linear(r))?, r.name())?;
        let (inp, out) = (cx.g.input(0, 0)?, cx.g.output(0, 0)?);
        let (n, k_dim) = (width(cx, out)?, width(cx, inp)?);
        let (x, y, k) = (cx.ptr(inp)?, cx.ptr(out)?, cx.handle(0)?);
        cx.push(
            0,
            Box::new(move |e| ops::w4a16_gemv_sw(e.gpu, k, x, &w, y, n, k_dim, e.stream)),
        )
    }
}

/// 2026-09-28: `w4a16_gemv_dual`: two NVFP4 projections of one input in one launch: the FFN's
/// gate and up (one `gate_up` node), or attention's K and V (two nodes).
pub(crate) struct W4a16GemvDual;

impl OpEmitter for W4a16GemvDual {
    fn id(&self) -> &'static str {
        "w4a16_gemv_dual"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        one_row(cx, "w4a16_gemv_dual")?;
        let func =
            cx.g.group
                .kernels
                .first()
                .context("no kernel")?
                .func
                .clone();
        let k = cx.handle(0)?;
        let h = dim(cx, "hidden")?;
        let (w1, y1, w2, y2, n) = if cx.g.group.nodes.len() == 1 {
            cx.g.expect_ops(self.id(), &["linear:gate_up"])?;
            let gate = nvfp4(cx.weight(0, WeightSlot::FfnGate)?, "gate")?;
            let up = nvfp4(cx.weight(0, WeightSlot::FfnUp)?, "up")?;
            let gu = cx.g.output(0, 0)?;
            let inter = width(cx, gu)? / 2;
            let base = cx.ptr(gu)?;
            let up_at = base.offset(rows(cx)? as usize * inter as usize * 2);
            (gate, base, up, up_at, inter)
        } else {
            cx.g.expect_ops(self.id(), &["linear:k", "linear:v"])?;
            let kw = nvfp4(cx.weight(0, WeightSlot::Linear(LinearRole::K))?, "k")?;
            let vw = nvfp4(cx.weight(1, WeightSlot::Linear(LinearRole::V))?, "v")?;
            let (ko, vo) = (cx.g.output(0, 0)?, cx.g.output(1, 0)?);
            ensure!(width(cx, ko)? == width(cx, vo)?, "K and V widths differ");
            (kw, cx.ptr(ko)?, vw, cx.ptr(vo)?, width(cx, ko)?)
        };
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        let sw = match func.as_str() {
            "w4a16_gemv_dual_sw" => true,
            "w4a16_gemv_dual" => false,
            other => anyhow::bail!("`w4a16_gemv_dual` cannot launch `{other}`"),
        };
        cx.push(
            0,
            Box::new(move |e| {
                if sw {
                    ops::w4a16_gemv_dual_sw(e.gpu, k, x, &w1, y1, &w2, y2, n, h, e.stream)
                } else {
                    ops::w4a16_gemv_dual(e.gpu, k, x, &w1, y1, &w2, y2, n, h, e.stream)
                }
            }),
        )
    }
}

/// 2026-09-28: `w4a16_gemv_qg`: the gated Q projection, written as `[Q_all | Gate_all]`.
pub(crate) struct W4a16GemvQg;

impl OpEmitter for W4a16GemvQg {
    fn id(&self) -> &'static str {
        "w4a16_gemv_qg"
    }

    fn constrain(&self, g: &GroupRef<'_>, layout: &mut Layout) -> Result<()> {
        layout.packs.push(vec![g.output(1, 0)?, g.output(1, 1)?]);
        Ok(())
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear:q", "split"])?;
        expect_kernel(cx, 0, "w4a16_gemv_qg")?;
        one_row(cx, "w4a16_gemv_qg")?;
        let a = attn_facts(cx, 0)?;
        ensure!(
            a.gated,
            "`w4a16_gemv_qg` writes a gated Q; this layer is ungated"
        );
        let w = nvfp4(cx.weight(0, WeightSlot::Linear(LinearRole::Q))?, "q")?;
        let q = cx.ptr(cx.g.output(1, 0)?)?;
        let gate = cx.ptr(cx.g.output(1, 1)?)?;
        let (nq, hd) = (a.num_q_heads, a.head_dim);
        ensure!(
            gate == q.offset((nq * hd) as usize * 2),
            "Q and its gate are not packed"
        );
        let (x, k, h) = (
            cx.ptr(cx.g.input(0, 0)?)?,
            cx.handle(0)?,
            dim(cx, "hidden")?,
        );
        cx.push(
            0,
            Box::new(move |e| {
                ops::w4a16_gemv_qg(e.gpu, k, x, &w, q, nq * hd * 2, h, nq, hd, e.stream)
            }),
        )
    }
}

/// 2026-09-28: `silu_mul`: `silu(gate) * up` over the gate+up edge (split SiLU).
pub(crate) struct SiluMul;

impl OpEmitter for SiluMul {
    fn id(&self) -> &'static str {
        "silu_mul"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["silu_mul"])?;
        let act = cx.g.output(0, 0)?;
        let inter = width(cx, act)?;
        let gu = cx.ptr(cx.g.input(0, 0)?)?;
        let n = rows(cx)?;
        let up = gu.offset(n as usize * inter as usize * 2);
        let (y, k) = (cx.ptr(act)?, cx.handle(0)?);
        cx.push(
            0,
            Box::new(move |e| ops::silu_mul(e.gpu, k, gu, up, y, n * inter, e.stream)),
        )
    }
}

/// 2026-09-28: `lm_head`: the BF16 vocabulary projection into the logits buffer: the GEMV at
/// one row, and at more rows the batched GEMV up to the head's band
/// (`HeadBinding::batchm_max_rows`), else the GEMM (`lm_head_batched.rs`).
pub(crate) struct LmHead;

impl OpEmitter for LmHead {
    fn id(&self) -> &'static str {
        "lm_head"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["lm_head"])?;
        let w = dense(cx.head.lm_head, "lm_head")?;
        let (inp, out) = (cx.g.input(0, 0)?, cx.g.output(0, 0)?);
        ensure!(
            cx.ptr(out)? == cx.fixed.logits,
            "logits must land in the logits buffer"
        );
        let (v, h, m) = (width(cx, out)?, width(cx, inp)?, rows(cx)?);
        let (x, y, k) = (cx.ptr(inp)?, cx.fixed.logits, cx.handle(0)?);
        let func = cx.g.group.kernels[0].func.as_str();
        let batchm = m <= cx.head.batchm_max_rows && h.is_multiple_of(8);
        match func {
            "dense_gemv_bf16" => {
                one_row(cx, "dense_gemv_bf16")?;
                cx.push(
                    0,
                    Box::new(move |e| ops::dense_gemv(e.gpu, k, x, &w, y, v, h, e.stream)),
                )
            }
            "dense_gemv_bf16_batchm" => {
                ensure!(
                    batchm,
                    "the head serves {m} rows with the GEMM, not the batched GEMV"
                );
                cx.push(
                    0,
                    Box::new(move |e| {
                        ops::dense_gemv_batchm(e.gpu, k, x, &w, y, m, v, h, v, e.stream)
                    }),
                )
            }
            "dense_gemm_bf16" => {
                ensure!(
                    !batchm,
                    "the head serves {m} rows with the batched GEMV, not the GEMM"
                );
                cx.push(
                    0,
                    Box::new(move |e| ops::dense_gemm(e.gpu, k, x, &w, y, m, v, h, e.stream)),
                )
            }
            other => anyhow::bail!("`lm_head` cannot launch `{other}`"),
        }
    }
}
