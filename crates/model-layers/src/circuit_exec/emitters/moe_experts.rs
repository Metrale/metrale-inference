// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The MoE expert steps besides the FP8 W8A8 one (`moe.rs`):
//! - `moe_grouped_experts`: the grouped NVFP4 decode's expert step
//!   (`forward_nvfp4_grouped_decode.rs`, the lean or the row-major tensor-core pair) and the
//!   grouped BF16 one (`forward_bf16_grouped_decode.rs`): gate+up writing each SiLU product into
//!   the arena, then down. One group, two launches; the shared expert rides both.
//! - `moe_fused_bf16`: `MoeLayer::forward`'s one-row BF16 path (`moe_shared_expert_fused_bf16`):
//!   gate+up of the top-k and the shared expert, then SiLU and down.
//! - `moe_topk`: that path's top-k (`moe_topk_softmax`).
//!
//! Owner: model-layers (MoE) circuit emitters.
//! Invariants:
//! - Each group's kernels are checked against the ones the layer's own dispatch launches
//!   (`MoeBinding::kernels`); a plan that disagrees is refused.
//! - The grouped step's SiLU products go where legacy puts them: the routed rows by sorted
//!   position in the arena's `expert_gate_out`, the shared rows by token in its `logits`
//!   (`MoeScratch`), each row in the FP32 row space as the BF16 hi + lo pair. Only the step's
//!   own down kernel reads them, before the head writes the logits.
//! - The one-row path's gate+up writes the gate block, then the up block, of its `egu` edge (and
//!   of `sgu` for the shared expert), which the down kernel reads back the same way.

use anyhow::{Result, bail, ensure};

use super::super::compile::{Cx, OpEmitter};
use super::moe::{binding, bound, same_kernel};
use super::{dim, one_row, rows};
use crate::layers::moe::{ExpertKind, MoeExperts};
use crate::layers::ops;

/// 2026-10-05: The gate+up and down entries the layer's dispatch launches for its experts' kind.
fn step_kernels(kind: ExpertKind) -> Result<[&'static str; 2]> {
    Ok(match kind {
        ExpertKind::Nvfp4Lean => [
            "moe_expert_gate_up_act_nvfp4_grouped_tc_lean",
            "moe_expert_down_act_nvfp4_grouped_tc_lean",
        ],
        ExpertKind::Nvfp4TensorCore => [
            "moe_expert_gate_up_act_nvfp4_grouped_tc",
            "moe_expert_down_act_nvfp4_grouped_tc",
        ],
        ExpertKind::Bf16 => [
            "moe_expert_gate_up_act_bf16_grouped_tc",
            "moe_expert_down_act_bf16_grouped_tc",
        ],
        ExpertKind::Fp8 => bail!("the grouped NVFP4/BF16 expert step over FP8 experts"),
    })
}

/// 2026-10-05: `moe_grouped_experts`.
pub(crate) struct MoeGroupedExperts;

impl OpEmitter for MoeGroupedExperts {
    fn id(&self) -> &'static str {
        "moe_grouped_experts"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(
            self.id(),
            &[
                "expert_gate_up",
                "silu_mul",
                "expert_down",
                "linear:shared_gate_up",
                "silu_mul",
                "linear:shared_down",
            ],
        )?;
        let (b, s) = bound(cx, 0)?;
        let want = step_kernels(b.facts.kind)?;
        for (k, func) in want.iter().enumerate() {
            let got = &cx.g.group.kernels[k].func;
            ensure!(
                got == func,
                "group {}: the plan's expert step runs `{got}`; the experts' tables are {:?}, \
                 whose dispatch launches `{func}`",
                cx.g.index,
                b.facts.kind
            );
        }
        same_kernel(cx, 0, b.kernels.gate_up, "expert gate+up")?;
        same_kernel(cx, 1, b.kernels.down, "expert down")?;
        let x_edge = cx.g.input(0, 0)?;
        ensure!(
            cx.g.input(3, 0)? == x_edge,
            "the shared gate+up does not read the routed experts' input"
        );
        let f = b.facts;
        let m = rows(cx)?;
        let (sort, cap) = b.sort_out(&s, m);
        let x = cx.ptr(x_edge)?;
        let (edown, sdown) = (cx.ptr(cx.g.output(2, 0)?)?, cx.ptr(cx.g.output(5, 0)?)?);
        let (act, sh_act) = (s.routed_act, s.shared_act);
        let (kg, kd) = (cx.handle(0)?, cx.handle(1)?);
        let (gg, dg) = (b.kernels.gate_up_geometry, b.kernels.down_geometry);
        match b.experts {
            MoeExperts::Nvfp4 {
                gate,
                up,
                down,
                shared: [sg, su, sd],
            } => {
                cx.push(
                    0,
                    Box::new(move |e| {
                        ops::moe_expert_gate_up_act_nvfp4_grouped(
                            e.gpu,
                            kg,
                            gg,
                            x,
                            gate,
                            up,
                            act,
                            sort.expert_offsets,
                            sort.sorted_token_ids,
                            sort.active_experts,
                            sort.active_count,
                            &sg,
                            &su,
                            sh_act,
                            f.inter,
                            f.hidden,
                            cap,
                            m,
                            e.stream,
                        )
                    }),
                )?;
                cx.push(
                    1,
                    Box::new(move |e| {
                        ops::moe_expert_down_act_nvfp4_grouped(
                            e.gpu,
                            kd,
                            dg,
                            act,
                            down,
                            edown,
                            sort.expert_offsets,
                            sort.active_experts,
                            sort.active_count,
                            sh_act,
                            &sd,
                            sdown,
                            f.hidden,
                            f.inter,
                            cap,
                            m,
                            e.stream,
                        )
                    }),
                )
            }
            MoeExperts::Bf16 {
                gate,
                up,
                down,
                shared: [sg, su, sd],
            } => {
                cx.push(
                    0,
                    Box::new(move |e| {
                        ops::moe_expert_gate_up_act_bf16_grouped(
                            e.gpu,
                            kg,
                            x,
                            gate,
                            up,
                            act,
                            sort.expert_offsets,
                            sort.sorted_token_ids,
                            sort.active_experts,
                            sort.active_count,
                            sg.weight,
                            su.weight,
                            sh_act,
                            f.inter,
                            f.hidden,
                            cap,
                            m,
                            e.stream,
                        )
                    }),
                )?;
                cx.push(
                    1,
                    Box::new(move |e| {
                        ops::moe_expert_down_act_bf16_grouped(
                            e.gpu,
                            kd,
                            act,
                            down,
                            edown,
                            sort.expert_offsets,
                            sort.active_experts,
                            sort.active_count,
                            sh_act,
                            sd.weight,
                            sdown,
                            f.hidden,
                            f.inter,
                            cap,
                            m,
                            e.stream,
                        )
                    }),
                )
            }
            MoeExperts::Fp8 { .. } => {
                bail!("the grouped NVFP4/BF16 expert step over a layer whose experts are FP8")
            }
        }
    }
}

/// 2026-10-05: `moe_topk`: `MoeLayer::forward`'s one-row top-k (`decode_topk_scored` without a
/// correction bias) into `topk_w` / `topk_id`.
pub(crate) struct MoeTopk;

impl OpEmitter for MoeTopk {
    fn id(&self) -> &'static str {
        "moe_topk"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["top_k"])?;
        one_row(cx, "moe_topk_softmax")?;
        let b = binding(cx, 0)?;
        same_kernel(cx, 0, b.kernels.topk_one_row, "one-row top-k")?;
        let f = b.facts;
        let logits = cx.ptr(cx.g.input(0, 0)?)?;
        let (weights, ids) = (cx.ptr(cx.g.output(0, 0)?)?, cx.ptr(cx.g.output(0, 1)?)?);
        let k = cx.handle(0)?;
        cx.push(
            0,
            Box::new(move |e| {
                ops::moe_topk_softmax(
                    e.gpu,
                    k,
                    logits,
                    ids,
                    weights,
                    f.num_experts,
                    f.top_k,
                    f.norm_topk_prob,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-10-05: `moe_fused_bf16`: the one-row BF16 experts (`MoeLayer::forward`,
/// `set_bf16_experts`), in its two groups.
pub(crate) struct MoeFusedBf16;

impl OpEmitter for MoeFusedBf16 {
    fn id(&self) -> &'static str {
        "moe_fused_bf16"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        one_row(cx, "the fused BF16 expert kernels")?;
        let b = binding(cx, 0)?;
        let MoeExperts::Bf16 {
            gate,
            up,
            down,
            shared: [sg, su, sd],
        } = b.experts
        else {
            bail!("the fused BF16 expert kernels over a layer whose experts are not BF16");
        };
        let f = b.facts;
        ensure!(
            dim(cx, "shared_inter")? == f.inter,
            "the fused BF16 kernels run the shared expert at the routed width"
        );
        let (up_off, sh_up_off) = (
            f.top_k as usize * f.inter as usize * 2,
            f.inter as usize * 2,
        );
        let k = cx.handle(0)?;
        if cx.g.group.nodes.len() == 2 {
            cx.g.expect_ops(self.id(), &["expert_gate_up", "linear:shared_gate_up"])?;
            same_kernel(cx, 0, b.kernels.fused_gate_up_bf16, "fused gate+up")?;
            let x_edge = cx.g.input(0, 0)?;
            ensure!(
                cx.g.input(1, 0)? == x_edge,
                "the shared gate+up does not read the routed experts' input"
            );
            let (x, ids) = (cx.ptr(x_edge)?, cx.ptr(cx.g.input(0, 1)?)?);
            let (egu, sgu) = (cx.ptr(cx.g.output(0, 0)?)?, cx.ptr(cx.g.output(1, 0)?)?);
            return cx.push(
                0,
                Box::new(move |e| {
                    ops::moe_expert_gate_up_shared_bf16(
                        e.gpu,
                        k,
                        x,
                        gate,
                        egu,
                        up,
                        egu.offset(up_off),
                        ids,
                        sg.weight,
                        sgu,
                        su.weight,
                        sgu.offset(sh_up_off),
                        f.inter,
                        f.hidden,
                        f.top_k,
                        e.stream,
                    )
                }),
            );
        }
        cx.g.expect_ops(
            self.id(),
            &["silu_mul", "expert_down", "silu_mul", "linear:shared_down"],
        )?;
        same_kernel(cx, 0, b.kernels.fused_down_bf16, "fused SiLU + down")?;
        let (egu, ids, sgu) = (
            cx.ptr(cx.g.input(0, 0)?)?,
            cx.ptr(cx.g.input(1, 1)?)?,
            cx.ptr(cx.g.input(2, 0)?)?,
        );
        let (edown, sdown) = (cx.ptr(cx.g.output(1, 0)?)?, cx.ptr(cx.g.output(3, 0)?)?);
        cx.push(
            0,
            Box::new(move |e| {
                ops::moe_expert_silu_down_shared_bf16(
                    e.gpu,
                    k,
                    egu,
                    egu.offset(up_off),
                    down,
                    edown,
                    ids,
                    sgu,
                    sgu.offset(sh_up_off),
                    sd.weight,
                    sdown,
                    f.hidden,
                    f.inter,
                    f.top_k,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-10-05: The emitters of this module.
pub(super) static ALL: &[&dyn OpEmitter] = &[&MoeGroupedExperts, &MoeTopk, &MoeFusedBf16];
