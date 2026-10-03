// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The collective emitters (LIFECYCLE-DESIGN.md 15.10, TP/EP): `all_reduce` sums a
//! row-parallel projection's partial output (or the vocabulary-parallel head's logits) over the
//! ranks, `ep_reduce` the routed MoE sum, in place, through the model's communicator
//! (`CommBackend::all_reduce_async`, as legacy's reduce sites call it); each is queued as the
//! group's copy-engine transfer. `lm_head_vocab_shard` runs the head over this rank's slice.
//!
//! Owner: model-layers circuit executor (FEATURES workstream).
//! Invariants:
//! - The sum is in place: the reduced edge is the partial edge's buffer.
//! - A plan with a cross-rank sum and no communicator refuses to compile.

use anyhow::{Context, Result, ensure};
use metrale_circuit::Format;
use metrale_circuit::planner::Layout;

use super::super::compile::{Cx, GroupRef, OpEmitter};

/// 2026-10-03: Every emitter of this module.
pub(super) static ALL: &[&dyn OpEmitter] = &[
    &Reduce("all_reduce", "all_reduce"),
    &Reduce("ep_reduce", "ep_reduce"),
    &AllReduceSkipped,
    &LmHeadVocabShard,
];

/// 2026-10-03: An in-place BF16 sum over the communicator of `rows x dim`: `all_reduce` after a
/// row-parallel projection (its output aliases its input), or the head's (no output: the
/// logits buffer is summed where it is); `ep_reduce` of the routed MoE sum.
pub(crate) struct Reduce(&'static str, &'static str);

impl OpEmitter for Reduce {
    fn id(&self) -> &'static str {
        self.0
    }

    fn constrain(&self, g: &GroupRef<'_>, layout: &mut Layout) -> Result<()> {
        g.expect_ops(self.0, &[self.1])?;
        if !g.node(0).outputs.is_empty() {
            layout.aliases.push((g.output(0, 0)?, g.input(0, 0)?));
        }
        Ok(())
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.0, &[self.1])?;
        let comm = comm(cx)?;
        let x = cx.g.input(0, 0)?;
        let e = &cx.g.circuit.edges[x];
        ensure!(
            e.format == Format::Bf16,
            "`{}` is {}; the communicator sums BF16",
            e.id,
            e.format
        );
        let bytes = usize::try_from(cx.rows * e.dim_value * 2)?;
        let p = cx.ptr(x)?;
        cx.push_copy(Box::new(move |env| {
            comm.all_reduce_async(p.0, bytes, env.stream)
        }))
    }
}

fn comm(cx: &Cx<'_>) -> Result<std::sync::Arc<dyn metrale_comm::CommBackend>> {
    cx.fixed
        .comm
        .clone()
        .context("a plan with a cross-rank sum without the model's communicator")
}

/// 2026-10-03: `all_reduce_skipped`: the head's reduce at widths where every rank runs the whole
/// head; nothing to launch.
pub(crate) struct AllReduceSkipped;

impl OpEmitter for AllReduceSkipped {
    fn id(&self) -> &'static str {
        "all_reduce_skipped"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["all_reduce"])?;
        ensure!(
            cx.g.node(0).outputs.is_empty(),
            "only the head's in-place reduce may be skipped"
        );
        Ok(())
    }
}

/// 2026-10-03: `lm_head_vocab_shard`: the BF16 head over this rank's slice of the vocabulary into
/// zeroed logits (one memset, then the slice's GEMV at one row or batched GEMV with full-vocab
/// rows), as legacy's `lm_head` / batched head under `lmhead_vocab_shard`.
pub(crate) struct LmHeadVocabShard;

impl OpEmitter for LmHeadVocabShard {
    fn id(&self) -> &'static str {
        "lm_head_vocab_shard"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["lm_head"])?;
        let n = cx.g.node(0);
        let param = |k: &str| -> Result<u32> {
            n.params
                .get(k)
                .with_context(|| format!("`{}` states no `{k}`", n.id))?
                .parse()
                .with_context(|| format!("`{}`'s `{k}`", n.id))
        };
        let (begin, len) = (param("vocab_begin")?, param("vocab_len")?);
        let w = super::dense(cx.head.lm_head, "the vocabulary-parallel head")?;
        let (h, v) = (super::dim(cx, "hidden")?, super::dim(cx, "vocab")?);
        ensure!(
            begin + len <= v,
            "the vocabulary slice passes the vocabulary"
        );
        let (x, logits) = (cx.ptr(cx.g.input(0, 0)?)?, cx.ptr(cx.g.output(0, 0)?)?);
        let rows = super::rows(cx)?;
        let zero = rows as usize * v as usize * 2;
        cx.push_copy(Box::new(move |e| {
            e.gpu.memset_async(logits, 0, zero, e.stream)
        }))?;
        let slice = crate::weight_map::DenseWeight {
            weight: w.weight.offset(begin as usize * h as usize * 2),
        };
        let out = logits.offset(begin as usize * 2);
        let k = cx.handle(0)?;
        if rows == 1 {
            super::expect_kernel(cx, 0, "dense_gemv_bf16")?;
            cx.push(
                0,
                Box::new(move |e| {
                    crate::layers::ops::dense_gemv(e.gpu, k, x, &slice, out, len, h, e.stream)
                }),
            )
        } else {
            super::expect_kernel(cx, 0, "dense_gemv_bf16_batchm")?;
            cx.push(
                0,
                Box::new(move |e| {
                    crate::layers::ops::dense_gemv_batchm(
                        e.gpu, k, x, &slice, out, rows, len, h, v, e.stream,
                    )
                }),
            )
        }
    }
}

#[cfg(test)]
#[path = "collective_tests.rs"]
mod collective_tests;
