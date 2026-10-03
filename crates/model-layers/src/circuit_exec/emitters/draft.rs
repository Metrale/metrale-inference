// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The MTP draft head's emitters, mirroring `mtp_head/forward.rs` (`forward_one`):
//! the concat of the normed embedding and target hidden, the BF16 GEMVs (`MtpHead::gemv`, a
//! BF16 head), and plain RoPE at the draft position. 2026-09-30: And the n-row draft's
//! (`forward_batch_position`): the concat per row, the tensor-core GEMVs (`gemm_rows`) and the
//! LM head (`lm_head_rows_arm`).
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - `dense_gemv` and `rope` serve one row. The n-row emitters launch what the head's own
//!   dispatch would at that width, checked at compile time against the plan's kernel.

use anyhow::{Context, Result, ensure};
use metrale_circuit::planner::Layout;
use metrale_circuit::{LinearRole, OpKind};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::bindings::WeightSlot;
use super::super::compile::{Cx, GroupRef, OpEmitter};
use super::batched::width;
use super::{attn_facts, dense, dim, expect_kernel, nvfp4, one_row, per_row, rows};
use crate::layers::ops;

/// 2026-09-29: `concat`: the normed embedding and the normed target hidden side by side.
/// 2026-09-30: One launch per row, as `forward_batch_position` concatenates its n rows.
pub(crate) struct Concat;

impl OpEmitter for Concat {
    fn id(&self) -> &'static str {
        "concat"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["concat"])?;
        expect_kernel(cx, 0, "bf16_concat")?;
        per_row(cx)?;
        let (ea, eb, ey) = (cx.g.input(0, 0)?, cx.g.input(0, 1)?, cx.g.output(0, 0)?);
        let (k, h) = (cx.handle(0)?, dim(cx, "hidden")?);
        for i in 0..cx.reps() {
            let (a, b, y) = (cx.row_ptr(ea, i)?, cx.row_ptr(eb, i)?, cx.row_ptr(ey, i)?);
            cx.push(
                0,
                Box::new(move |e| ops::bf16_concat(e.gpu, k, a, b, y, h, e.stream)),
            )?;
        }
        Ok(())
    }
}

/// 2026-09-30: The launches of a BF16 projection node: `(weight slot, output, N)` each; the
/// FFN's gate and up are two, gate rows then up rows (`rows * inter` elements after them).
fn projections(cx: &Cx<'_>, role: LinearRole) -> Result<Vec<(WeightSlot, DevicePtr, u32)>> {
    let out = cx.g.output(0, 0)?;
    Ok(if role == LinearRole::GateUp {
        let inter = width(cx, out)? / 2;
        let base = cx.ptr(out)?;
        vec![
            (WeightSlot::FfnGate, base, inter),
            (
                WeightSlot::FfnUp,
                base.offset(cx.rows as usize * inter as usize * 2),
                inter,
            ),
        ]
    } else {
        vec![(WeightSlot::Linear(role), cx.ptr(out)?, width(cx, out)?)]
    })
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
        let inp = cx.g.input(0, 0)?;
        let x = cx.ptr(inp)?;
        let k_dim = width(cx, inp)?;
        let launches = projections(cx, role)?;
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

/// 2026-09-30: `dense_gemv_tc`: one BF16 projection of the n rows on a tensor-core GEMV entry,
/// or the FFN's gate and up as two (`MtpHead::gemm_rows`, whose first arm is
/// `dense_gemv_tc::try_dense_gemv_tc`). The entry each launch takes is the router's
/// (`dense_gemv_tc::kernel_for`), which must be the plan's: with `METRALE_NO_MTP_TC` set or an
/// unrouted shape, the head runs another arm and the plan is refused.
pub(crate) struct DenseGemvTc;

impl OpEmitter for DenseGemvTc {
    fn id(&self) -> &'static str {
        "dense_gemv_tc"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear"])?;
        let OpKind::Linear(role) = cx.g.node(0).op else {
            anyhow::bail!("`{}` is not a projection", cx.g.node(0).id);
        };
        let inp = cx.g.input(0, 0)?;
        let (x, x_stride) = cx.strided(inp)?;
        let (k_dim, m) = (width(cx, inp)?, rows(cx)?);
        ensure!(
            x_stride == k_dim,
            "the projection's input rows are not contiguous"
        );
        let launches = projections(cx, role)?;
        ensure!(
            launches.len() == cx.g.group.kernels.len(),
            "`dense_gemv_tc` launches one kernel per projection"
        );
        for (i, (slot, y, n)) in launches.into_iter().enumerate() {
            let w = dense(cx.weight(0, slot)?, role.name())?;
            let k = cx.handle(i)?;
            let (routed, grid_x) = ops::dense_gemv_tc::kernel_for(cx.gpu, m, n, k_dim)
                .with_context(|| {
                    format!(
                        "the head's {m}x{n}x{k_dim} {} projection takes no tensor-core GEMV \
                         (METRALE_NO_MTP_TC, or K not a multiple of 64)",
                        role.name()
                    )
                })?;
            ensure!(
                routed.0 == k.0,
                "the head's {m}x{n}x{k_dim} {} projection routes to another tensor-core entry \
                 than the plan's `{}`",
                role.name(),
                cx.g.group.kernels[i].func
            );
            cx.push(
                i,
                Box::new(move |e| {
                    ops::dense_gemv_tc::launch(e.gpu, k, grid_x, x, &w, y, m, n, k_dim, n, e.stream)
                }),
            )?;
        }
        Ok(())
    }
}

/// 2026-09-30: `draft_lm_head_rows`: the n-row draft's NVFP4 vocabulary projection, as
/// `forward_batch_position` launches it, rows packed `DraftFixed::vocab` apart:
/// `ops::w4a16_gemv_batchm` on the head's
/// `lm_head_batch_kernel(n)`, which takes the tensor-core entry when one routes. The kernel
/// that runs must be the plan's, and a width where the head takes the tile GEMM on its twin
/// (`lm_head_rows_arm`) is refused: the plan does not state it.
pub(crate) struct DraftLmHeadRows;

impl OpEmitter for DraftLmHeadRows {
    fn id(&self) -> &'static str {
        "draft_lm_head_rows"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["lm_head"])?;
        let d = cx.draft_fixed()?;
        let facts = d
            .rows
            .as_ref()
            .context("an n-row draft plan for a head outside the batched propose")?;
        let w = nvfp4(cx.weight(0, WeightSlot::LmHead)?, "draft lm_head")?;
        let (inp, out) = (cx.g.input(0, 0)?, cx.g.output(0, 0)?);
        let (v, h, m) = (d.vocab, width(cx, inp)?, rows(cx)?);
        // 2026-10-01: The scored rows of each draft row are packed `v` apart (a head with
        // `mtp_vocab_size` scores fewer rows than the logits edge holds), as the head writes
        // them and as the host and the argmax read them.
        ensure!(
            v <= width(cx, out)?,
            "the head scores {v} rows; the logits edge holds {}",
            width(cx, out)?
        );
        ensure!(
            m <= 32,
            "w4a16_gemv_batchm caps at 32 rows; the plan has {m}"
        );
        let n = m as usize;
        let tc = crate::layers::mtp_head::tc_lm_head(cx.gpu, n, v, h);
        ensure!(
            crate::layers::mtp_head::lm_head_rows_arm(n, tc, facts.lm_head_twin)
                == crate::layers::mtp_head::LmHeadRowsArm::Gemv,
            "at {m} rows the head scores with the tile GEMM on its transposed twin; the plan \
             does not state it"
        );
        let gemv = facts
            .lm_head_gemv
            .get(n)
            .copied()
            .filter(|k| k.0 != 0)
            .with_context(|| format!("the head has no batched LM-head GEMV at {m} rows"))?;
        let runs = ops::gemv_tc::tc_kernel(cx.gpu, m, v, h).map_or(gemv, |(t, _)| t);
        ensure!(
            runs.0 == cx.handle(0)?.0,
            "at {m} rows the head's LM head launches another kernel than the plan's `{}`",
            cx.g.group.kernels[0].func
        );
        let (x, y) = (cx.ptr(inp)?, cx.ptr(out)?);
        cx.push(
            0,
            Box::new(move |e| ops::w4a16_gemv_batchm(e.gpu, gemv, x, &w, y, m, v, h, e.stream)),
        )
    }
}

/// 2026-10-03: This module's emitters, for the registry in `mod.rs`.
pub(super) static ALL: &[&dyn OpEmitter] =
    &[&Concat, &DenseGemv, &Rope, &DenseGemvTc, &DraftLmHeadRows];
