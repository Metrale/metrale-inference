// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The target vocabulary head under a fixed `--activation-quantization` for the
//! head (`lm_head_rows`): the BF16 head as `project_bf16_lm_head` runs it then, the batched GEMV
//! `dense_gemv_bf16_batchm` at every row count, one row included, in chunks of its widest launch
//! (`DENSE_GEMV_BATCHM_MAX_M` rows), so a row's logits do not depend on the batch.
//!
//! 2026-10-03: And the NVFP4 head's tile GEMM (`lm_head_nvfp4_tile`).
//!
//! 2026-10-05: And a declared NVFP4 head on the W4A16 row tiles (`lm_head_nvfp4_rows`,
//! model-engine `lm_head_nvfp4_rows.rs`): at every row count, in calls of at most 64 rows, each
//! call's entry by its own rows.
//!
//! Owner: model-layers (MoE) circuit emitters (the Qwen3.6 FP8 recipes run it first).
//! Invariants:
//! - The rules select it only under a fixed head format (`activation_quantization` in the
//!   policy, which the live policy reads from the published flag); under `adaptive` the head
//!   takes the per-width routing the `lm_head` emitter encodes.
//! - Logits land in the logits buffer, rows a full vocabulary apart.

use anyhow::{Result, ensure};

use super::super::compile::{Cx, OpEmitter};
use super::batched::width;
use super::{dense, expect_kernel, nvfp4, rows};
use crate::layers::ops;

/// 2026-10-03: `lm_head_rows`.
pub(crate) struct LmHeadRows;

impl OpEmitter for LmHeadRows {
    fn id(&self) -> &'static str {
        "lm_head_rows"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["lm_head"])?;
        let m = rows(cx)?;
        let w = dense(cx.head.lm_head, "lm_head")?;
        let (inp, out) = (cx.g.input(0, 0)?, cx.g.output(0, 0)?);
        ensure!(
            cx.ptr(out)? == cx.fixed.logits,
            "logits must land in the logits buffer"
        );
        let (v, h) = (width(cx, out)?, width(cx, inp)?);
        ensure!(
            h.is_multiple_of(8),
            "the batched GEMV reads K = {h} in 8-element loads"
        );
        let x = cx.ptr(inp)?;
        let y = cx.fixed.logits;
        let step = ops::DENSE_GEMV_BATCHM_MAX_M;
        let chunks = m.div_ceil(step);
        ensure!(
            cx.g.group.kernels.len() == 1 && cx.reps() == chunks as usize,
            "group {}: {} launches planned for {chunks} chunks of {m} rows",
            cx.g.index,
            cx.reps()
        );
        expect_kernel(cx, 0, "dense_gemv_bf16_batchm")?;
        let k = cx.handle(0)?;
        for c in 0..chunks {
            let (done, rows_c) = (c * step, (m - c * step).min(step));
            let (xc, yc) = (
                x.offset(done as usize * h as usize * 2),
                y.offset(done as usize * v as usize * 2),
            );
            cx.push(
                0,
                Box::new(move |e| {
                    ops::dense_gemv_batchm(e.gpu, k, xc, &w, yc, rows_c, v, h, v, e.stream)
                }),
            )?;
        }
        Ok(())
    }
}

/// 2026-10-03: `lm_head_nvfp4_tile`: the NVFP4 head's tile GEMM over its transposed twin
/// (`w4a16::w4a16_gemm_t`, `ops::w4a16_gemm_n128_ldb`), which the decode heads take at every
/// row count under the row-invariant tiers (`lm_head_project_batched`), one decode row under a
/// fixed head format too, and the verify heads under the row-invariant tiers
/// (`lm_head_batched_wide`).
pub(crate) struct LmHeadNvfp4Tile;

impl OpEmitter for LmHeadNvfp4Tile {
    fn id(&self) -> &'static str {
        "lm_head_nvfp4_tile"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["lm_head"])?;
        expect_kernel(cx, 0, "w4a16_gemm_t")?;
        let m = rows(cx)?;
        let (twin, ldb) = cx
            .head
            .nvfp4_twin
            .ok_or_else(|| anyhow::anyhow!("the plan runs the NVFP4 head; the model binds none"))?;
        let (inp, out) = (cx.g.input(0, 0)?, cx.g.output(0, 0)?);
        ensure!(
            cx.ptr(out)? == cx.fixed.logits,
            "logits must land in the logits buffer"
        );
        let (v, h) = (width(cx, out)?, width(cx, inp)?);
        let (x, y, k) = (cx.ptr(inp)?, cx.fixed.logits, cx.handle(0)?);
        cx.push(
            0,
            Box::new(move |e| {
                ops::w4a16_gemm_n128_ldb(e.gpu, k, x, &twin, y, m, v, h, ldb, e.stream)
            }),
        )
    }
}

/// 2026-10-05: The row-tile entry `ops::w4a16_tc_rows` launches for `m` rows.
fn w4a16_rows_entry(m: u32) -> &'static str {
    match m {
        0..=16 => "w4a16_tc_rows_16",
        17..=32 => "w4a16_tc_rows_32",
        _ => "w4a16_tc_rows_64",
    }
}

/// 2026-10-05: `lm_head_nvfp4_rows`: the head `install_declared_lm_head_w4a16_rows` installs,
/// as `lm_head_nvfp4_rows_run` runs it.
pub(crate) struct LmHeadNvfp4Rows;

impl OpEmitter for LmHeadNvfp4Rows {
    fn id(&self) -> &'static str {
        "lm_head_nvfp4_rows"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["lm_head"])?;
        ensure!(
            cx.head.nvfp4_rows,
            "the plan runs the declared NVFP4 head's row tiles; the model did not install them"
        );
        let w = nvfp4(cx.head.lm_head, "lm_head")?;
        let m = rows(cx)?;
        let (inp, out) = (cx.g.input(0, 0)?, cx.g.output(0, 0)?);
        ensure!(
            cx.ptr(out)? == cx.fixed.logits,
            "logits must land in the logits buffer"
        );
        let (v, h) = (width(cx, out)?, width(cx, inp)?);
        ensure!(
            ops::w4a16_tc_rows_shape_ok(1, v, h, h, v),
            "the row tiles refuse a {v} x {h} head"
        );
        let (x, y) = (cx.ptr(inp)?, cx.fixed.logits);
        let step = ops::W4A16_TC_ROWS_MAX_M;
        let chunks: Vec<u32> = (0..m.div_ceil(step))
            .map(|c| (m - c * step).min(step))
            .collect();
        ensure!(
            cx.g.group.kernels.len() == chunks.len(),
            "group {}: the plan lists {} kernels for {} calls of {m} rows",
            cx.g.index,
            cx.g.group.kernels.len(),
            chunks.len()
        );
        for (i, &rows_i) in chunks.iter().enumerate() {
            expect_kernel(cx, i, w4a16_rows_entry(rows_i))?;
            let done = i * step as usize;
            let (xi, yi) = (
                x.offset(done * h as usize * 2),
                y.offset(done * v as usize * 2),
            );
            cx.push(
                i,
                Box::new(move |e| {
                    ops::w4a16_tc_rows(e.gpu, xi, &w, yi, rows_i, v, h, h, v, e.stream)
                }),
            )?;
        }
        Ok(())
    }
}

/// 2026-10-03: The head emitters of this module.
pub(super) static ALL: &[&dyn OpEmitter] = &[&LmHeadRows, &LmHeadNvfp4Tile, &LmHeadNvfp4Rows];
