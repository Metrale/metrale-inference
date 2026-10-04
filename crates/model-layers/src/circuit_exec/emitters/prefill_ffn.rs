// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The dense FFN, embedding and head emitters of the prefill modes
//! (LIFECYCLE-DESIGN.md 15.4). Each mirrors the legacy prefill call site it names: the same
//! `ops::*` function, the same arguments, the row count from the step (`StepEnv::prefill`).
//!
//! - `prefill_ffn_mmq`: the NVFP4 MMQ arm of `DenseFfnLayer::prefill_nvfp4`
//!   (`dense_ffn_prefill_nvfp4.rs`): gate and up into `expert_gate_out` / `expert_up_out`, the
//!   fused SiLU·mul quantize into `ffn_act_q8`, down into `moe_output`, down's scale by the
//!   pipelined GEMM or by `nvfp4_scale_bf16`.
//! - `prefill_residual_add`: the FFN's residual add over the pass's `T * hidden` elements.
//! - `prefill_final_norm`: the head's final norm of the pass's last row into `norm_output`
//!   (`finalize_last.rs`, `prefill_a.rs`, `impl_a3_norm.rs`).
//! - `prefill_lm_head`: the BF16 head on that one row into the logits buffer
//!   (`impl_a3_lm_head.rs` `lm_head`).
//!
//! - `prefill_embed` (2026-10-04): the pass's token embedding gathered from the staged ids.
//!
//! The ids' upload, the vision splices and the sampling of a prefill pass stay on the host
//! driver: the driver stages the chunk's ids before the program and splices vision rows after
//! its embedding segment (`impl_circuit_prefill.rs`), and the scheduler samples the returned
//! logits (`host_sampling`, which launches nothing). A model with an n-gram embedding, an
//! embedding scale or an embedding overlay is refused (`impl_circuit.rs` `circuit_head`).
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - The MMQ tile a group launches is the one the legacy ladder picks for the pass's rows
//!   (`legacy_tile`), checked at compile time for the bucket's top and at run time for `T`.
//! - Every buffer is the one the legacy layer uses; a placement that disagrees refuses the build.

use anyhow::{Result, bail, ensure};
use metrale_circuit::{LinearRole, OpKind};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::bindings::{BoundWeight, WeightSlot};
use super::super::compile::{Cx, OpEmitter};
use super::batched::width;
use super::{dense, dim, expect_kernel, nvfp4};
use crate::layers::ops;

/// 2026-10-03: The MMQ GEMM the legacy prefill launches for `m` rows and its tile: mmq16, mmq32
/// and mmq64 up to their tile, the pipelined GEMM above (`dense_ffn_prefill_nvfp4.rs:73-91`;
/// the 128 tile runs `metrale_nvfp4_gemm_pipe` when K % 256 == 0, `ops/nvfp4_mmq.rs`
/// `nvfp4_mmq_gemm_tiled`).
fn legacy_tile(m: u32) -> (&'static str, u32) {
    match m {
        0..=16 => ("metrale_nvfp4_mmq16_nc", 16),
        17..=32 => ("metrale_nvfp4_mmq32_nc", 32),
        33..=64 => ("metrale_nvfp4_mmq64_nc", 64),
        _ => ("metrale_nvfp4_gemm_pipe", 128),
    }
}

/// 2026-10-03: The plan's GEMM kernel is the legacy ladder's for the bucket's top, and the
/// shapes keep the legacy launch on it: N a multiple of 128 (the no-column-tail kernel), and
/// for the pipelined GEMM K a multiple of 256.
fn plan_tile(cx: &Cx<'_>, func: &str, n: u32, k: u32) -> Result<u32> {
    let rows = u32::try_from(cx.rows)?;
    let (want, tile) = legacy_tile(rows);
    ensure!(
        func == want,
        "the plan launches `{func}` at {rows} rows; the legacy prefill launches `{want}`"
    );
    ensure!(
        n.is_multiple_of(128),
        "N = {n} would take the column-tail MMQ kernel"
    );
    ensure!(
        tile != 128 || k.is_multiple_of(256),
        "K = {k} keeps the 128 tile off the pipelined GEMM"
    );
    Ok(tile)
}

/// 2026-10-03: The pass's rows, checked against the tile the program was compiled for.
fn pass_rows(e: &super::super::program::StepEnv<'_>, func: &'static str) -> Result<u32> {
    let m = e.prefill()?.tokens;
    ensure!(
        m >= 1 && legacy_tile(m).0 == func,
        "a {m}-row pass reached a program compiled for `{func}`"
    );
    Ok(m)
}

/// 2026-10-03: Member `i`'s MMQ repack in `slot` (the same cell legacy's
/// `ensure_nvfp4_mmq_weight` returns, bound by `DenseFfnLayer::circuit_bind`).
fn mmq_weight(cx: &Cx<'_>, i: usize, slot: WeightSlot) -> Result<DevicePtr> {
    match cx.weight(i, slot)? {
        BoundWeight::Mmq(p) => Ok(p),
        other => bail!(
            "{slot:?}: expected an MMQ repack, the layer holds {}",
            other.family()
        ),
    }
}

/// 2026-10-03: The edge's placed buffer is `want`, the buffer the legacy layer uses.
fn placed_at(cx: &Cx<'_>, edge: usize, want: DevicePtr, what: &str) -> Result<DevicePtr> {
    let got = cx.ptr(edge)?;
    ensure!(
        got == want,
        "`{}` is placed apart from the legacy {what}",
        cx.g.circuit.edges[edge].id
    );
    Ok(got)
}

/// 2026-10-03: `prefill_ffn_mmq`: the FFN's gate and up, or its activation and down, under the
/// NVFP4 MMQ arm of the legacy prefill.
pub(crate) struct PrefillFfnMmq;

impl PrefillFfnMmq {
    /// 2026-10-03: `lin` is the gate|up projection's member: 0, or 1 behind the declared
    /// circuit's act_quant, which the MMQ quantizer is.
    fn gate_up(cx: &mut Cx<'_>, lin: usize) -> Result<()> {
        ensure!(
            cx.g.group.kernels.len() == 3 && cx.g.group.kernels[2] == cx.g.group.kernels[1],
            "the gate and up GEMMs take one kernel"
        );
        let func = legacy_tile(u32::try_from(cx.rows)?).0;
        let gu = cx.g.output(lin, 0)?;
        let (h, inter) = (dim(cx, "hidden")?, width(cx, gu)? / 2);
        let tile = plan_tile(cx, &cx.g.group.kernels[1].func, inter, h)?;
        let arena = cx.arena()?;
        let (gate, up) = (
            mmq_weight(cx, lin, WeightSlot::FfnGateMmq)?,
            mmq_weight(cx, lin, WeightSlot::FfnUpMmq)?,
        );
        let g = placed_at(cx, gu, arena.expert_gate_out(), "gate output")?;
        let u = arena.expert_up_out();
        let x = placed_at(cx, cx.g.input(0, 0)?, arena.norm_output(), "FFN input")?;
        let q8 = arena.ffn_act_q8();
        let kq = cx.handle(0)?;
        cx.push(
            0,
            Box::new(move |e| {
                let m = pass_rows(e, func)?;
                ops::nvfp4_mmq_quantize_act(e.gpu, kq, x, q8, m, h, e.stream)
            }),
        )?;
        for (i, (w, y)) in [(gate, g), (up, u)].into_iter().enumerate() {
            let k = cx.handle(1 + i)?;
            cx.push(
                1 + i,
                Box::new(move |e| {
                    let m = pass_rows(e, func)?;
                    let applied = ops::nvfp4_mmq_gemm_tiled(
                        e.gpu, k, k, tile, q8, w, y, m, inter, h, None, e.stream,
                    )?;
                    ensure!(!applied, "the gate/up MMQ applied a scale it was not given");
                    Ok(())
                }),
            )?;
        }
        Ok(())
    }

    /// 2026-10-03: `lin` is the down projection's member: 1, or 2 behind the act_quant.
    fn act_down(cx: &mut Cx<'_>, lin: usize) -> Result<()> {
        let func = legacy_tile(u32::try_from(cx.rows)?).0;
        let gu = cx.g.input(0, 0)?;
        let (h, inter) = (dim(cx, "hidden")?, width(cx, gu)? / 2);
        let tile = plan_tile(cx, &cx.g.group.kernels[1].func, h, inter)?;
        let pipe = tile == 128;
        ensure!(
            cx.g.group.kernels.len() == if pipe { 2 } else { 3 },
            "the pipelined down GEMM applies its scale; the tiles need `nvfp4_scale_bf16`"
        );
        let gate_s = nvfp4(cx.weight(0, WeightSlot::FfnGate)?, "gate")?.weight_scale_2;
        let up_s = nvfp4(cx.weight(0, WeightSlot::FfnUp)?, "up")?.weight_scale_2;
        let down_s = nvfp4(
            cx.weight(lin, WeightSlot::Linear(LinearRole::Down))?,
            "down",
        )?
        .weight_scale_2;
        let down = mmq_weight(cx, lin, WeightSlot::FfnDownMmq)?;
        let arena = cx.arena()?;
        let g = placed_at(cx, gu, arena.expert_gate_out(), "gate output")?;
        let u = arena.expert_up_out();
        let y = placed_at(cx, cx.g.output(lin, 0)?, arena.moe_output(), "FFN output")?;
        let q8 = arena.ffn_act_q8();
        let ks = cx.handle(0)?;
        cx.push(
            0,
            Box::new(move |e| {
                let m = pass_rows(e, func)?;
                ops::nvfp4_silu_mul_quant(e.gpu, ks, g, u, q8, gate_s, up_s, m, inter, e.stream)
            }),
        )?;
        let kg = cx.handle(1)?;
        cx.push(
            1,
            Box::new(move |e| {
                let m = pass_rows(e, func)?;
                let applied = ops::nvfp4_mmq_gemm_tiled(
                    e.gpu,
                    kg,
                    kg,
                    tile,
                    q8,
                    down,
                    y,
                    m,
                    h,
                    inter,
                    Some(down_s),
                    e.stream,
                )?;
                ensure!(
                    applied == pipe,
                    "the down GEMM's scale was not applied as planned"
                );
                Ok(())
            }),
        )?;
        if !pipe {
            expect_kernel(cx, 2, "metrale_nvfp4_scale_bf16")?;
            let kc = cx.handle(2)?;
            cx.push(
                2,
                Box::new(move |e| {
                    let m = pass_rows(e, func)?;
                    ops::nvfp4_scale_bf16(e.gpu, kc, y, down_s, m * h, e.stream)
                }),
            )?;
        }
        Ok(())
    }
}

impl OpEmitter for PrefillFfnMmq {
    fn id(&self) -> &'static str {
        "prefill_ffn_mmq"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        let lin = cx.g.group.nodes.len() - 1;
        match cx.g.node(0).op {
            OpKind::Linear(_) | OpKind::ActQuant(_) => {
                let ops: &[&str] = if lin == 0 {
                    &["linear:gate_up"]
                } else {
                    &["act_quant:nvfp4/g16", "linear:gate_up"]
                };
                cx.g.expect_ops(self.id(), ops)?;
                expect_kernel(cx, 0, "metrale_nvfp4_quantize_bf16")?;
                Self::gate_up(cx, lin)
            }
            _ => {
                let ops: &[&str] = if lin == 1 {
                    &["silu_mul", "linear:down"]
                } else {
                    &["silu_mul", "act_quant:nvfp4/g16", "linear:down"]
                };
                cx.g.expect_ops(self.id(), ops)?;
                expect_kernel(cx, 0, "metrale_nvfp4_silu_mul_quant")?;
                Self::act_down(cx, lin)
            }
        }
    }
}

/// 2026-10-03: `prefill_residual_add`: the stream plus the FFN output over the pass's
/// `T * hidden` elements, in place (`qwen3_ssm/trait_prefill.rs`,
/// `qwen3_attention/trait_impl/prefill_inner/ffn_residual.rs`).
pub(crate) struct PrefillResidualAdd;

impl OpEmitter for PrefillResidualAdd {
    fn id(&self) -> &'static str {
        "prefill_residual_add"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["residual_add"])?;
        expect_kernel(cx, 0, "bf16_residual_add")?;
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        ensure!(
            x == cx.ptr(cx.g.output(0, 0)?)? && x == cx.fixed.hidden,
            "group {}: the residual stream must be updated in place in `hidden`",
            cx.g.index
        );
        let src = placed_at(
            cx,
            cx.g.input(0, 1)?,
            cx.arena()?.moe_output(),
            "FFN output",
        )?;
        let (k, h) = (cx.handle(0)?, dim(cx, "hidden")?);
        cx.push(
            0,
            Box::new(move |e| {
                let m = e.prefill()?.tokens;
                ops::residual_add(e.gpu, k, x, src, m * h, e.stream)
            }),
        )
    }
}

/// 2026-10-03: `prefill_final_norm`: the final RMSNorm of the pass's last row only, into
/// `norm_output` (`finalize_last.rs`: `hidden + (T - 1) * hidden * 2`, one row).
pub(crate) struct PrefillFinalNorm;

impl OpEmitter for PrefillFinalNorm {
    fn id(&self) -> &'static str {
        "prefill_final_norm"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["final_norm"])?;
        expect_kernel(cx, 0, "rms_norm")?;
        ensure!(
            !cx.config.final_norm_identity,
            "a checkpoint without a final norm copies instead"
        );
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        ensure!(
            x == cx.fixed.hidden,
            "the final norm reads the residual stream"
        );
        let y = placed_at(
            cx,
            cx.g.output(0, 0)?,
            cx.arena()?.norm_output(),
            "norm output",
        )?;
        let (k, w, h) = (cx.handle(0)?, cx.head.final_norm, dim(cx, "hidden")?);
        let eps = cx.config.rms_norm_eps as f32;
        cx.push(
            0,
            Box::new(move |e| {
                let m = e.prefill()?.tokens;
                ensure!(m >= 1, "a prefill pass of no rows has no last row");
                let last = x.offset((m as usize - 1) * h as usize * 2);
                ops::rms_norm(e.gpu, k, last, &w, y, 1, h, eps, e.stream)
            }),
        )
    }
}

/// 2026-10-03: `prefill_lm_head`: the BF16 head on the normed last row into the logits buffer
/// (`impl_a3_lm_head.rs` `lm_head`: `dense_gemv`, one row).
pub(crate) struct PrefillLmHead;

impl OpEmitter for PrefillLmHead {
    fn id(&self) -> &'static str {
        "prefill_lm_head"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["lm_head"])?;
        expect_kernel(cx, 0, "dense_gemv_bf16")?;
        let w = dense(cx.head.lm_head, "lm_head")?;
        let (inp, out) = (cx.g.input(0, 0)?, cx.g.output(0, 0)?);
        ensure!(
            cx.ptr(out)? == cx.fixed.logits,
            "logits must land in the logits buffer"
        );
        let x = placed_at(cx, inp, cx.arena()?.norm_output(), "norm output")?;
        let (v, h) = (width(cx, out)?, dim(cx, "hidden")?);
        let (y, k) = (cx.fixed.logits, cx.handle(0)?);
        cx.push(
            0,
            Box::new(move |e| ops::dense_gemv(e.gpu, k, x, &w, y, v, h, e.stream)),
        )
    }
}

/// 2026-10-04: `prefill_embed`: the pass's token embedding, one `batched_embed` gather of `T`
/// rows from the ids the driver staged in the arena's `token_ids` (from row
/// `PrefillStep::ids_row0`) into the residual stream (`prefill_b/embed_chunk.rs`,
/// `prefill_b/proc_range.rs`, `impl_ngram.rs` `embed_tokens_fused`).
pub(crate) struct PrefillEmbed;

impl OpEmitter for PrefillEmbed {
    fn id(&self) -> &'static str {
        "prefill_embed"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["embed"])?;
        expect_kernel(cx, 0, "batched_embed")?;
        let y = cx.ptr(cx.g.output(0, 0)?)?;
        ensure!(
            y == cx.fixed.hidden,
            "the prefill embedding writes the residual stream"
        );
        let (ids, k) = (cx.arena()?.token_ids(), cx.handle(0)?);
        let (table, h) = (cx.head.embed.weight, dim(cx, "hidden")?);
        cx.push(
            0,
            Box::new(move |e| {
                let s = e.prefill()?;
                let row0 = ids.offset(s.ids_row0 as usize * 4);
                ops::batched_embed(e.gpu, k, row0, table, y, s.tokens, h, e.stream)
            }),
        )
    }
}

/// 2026-10-03: This module's emitters, for the registry in `mod.rs`.
pub(super) static ALL: &[&dyn OpEmitter] = &[
    &PrefillEmbed,
    &PrefillFfnMmq,
    &PrefillResidualAdd,
    &PrefillFinalNorm,
    &PrefillLmHead,
];

#[cfg(test)]
#[path = "prefill_ffn_tests.rs"]
mod prefill_ffn_tests;
