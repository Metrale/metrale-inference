// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The MoE FFN's emitters: the grouped FP8 decode of `MoeLayer`
//! (`forward_fp8_grouped_decode_routed`), one group per step of it. `moe_router` (the per-row
//! router GEMV, or the batched router GEMM of the drafter's n-row propose), `moe_grouped_route`
//! (top-k and the slot sort), `moe_grouped_fp8_w8a8` (the W8A8 expert step: quantize the input, gate+up with a re-quantized
//! SiLU product, down) and `moe_blend` (the shared-expert gate and the weighted sum).
//!
//! Owner: model-layers (MoE) circuit emitters.
//! Invariants:
//! - Each group's kernel is checked at compile time against the one the layer's own dispatch
//!   launches for this width (`MoeBinding::kernels`, `MoeFacts`): the per-row router everywhere
//!   but a drafter step of two or more rows, which routes batched as `forward_batch_ffn.rs` does;
//!   the W8A8 expert step, which the layer runs exactly when the serve published FP8 expert
//!   activations (its W8A16 step, under a `moe:bf16` activation override, has no rule yet). A
//!   plan that disagrees is refused.
//! - The sort writes where legacy's does (`grouped_sort_out` over the arena's `gate_logits`); the
//!   routed rows of `eact`/`edown` and the W8A8 products are by sorted position, the shared rows
//!   by token, as the kernels lay them out.
//! - Every width is admitted by `MoeFacts::check_rows`, the conditions legacy checks.

use anyhow::{Context, Result, bail, ensure};
use metrale_circuit::{Format, Mode, Scale};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::compile::{Cx, OpEmitter};
use super::{expect_kernel, rows};
use crate::layers::moe::{MoeBinding, MoeScratch, router_block_rows};
use crate::layers::ops;

/// 2026-10-03: Member `i`'s layer's MoE binding.
fn binding(cx: &Cx<'_>, i: usize) -> Result<MoeBinding> {
    cx.layer(i)?
        .moe
        .clone()
        .with_context(|| format!("`{}`: its layer binds no MoE", cx.g.node(i).id))
}

/// 2026-10-03: Member `i`'s layer's MoE binding and the arena scratch, after checking that the
/// plan's rows are a width the grouped decode takes.
fn bound(cx: &Cx<'_>, i: usize) -> Result<(MoeBinding, MoeScratch)> {
    let b = binding(cx, i)?;
    let s = cx
        .fixed
        .moe
        .context("a MoE plan without the MoE arena scratch")?;
    b.facts.check_rows(cx.rows, &s)?;
    Ok((b, s))
}

/// 2026-10-03: The drafter's propose of two or more rows routes batched
/// (`MoeLayer::forward_fp8_grouped_decode`, `mtp_head/forward_batch_ffn.rs`); every other step,
/// one drafter row included (`MoeLayer::forward`), routes per row.
fn batched_routing(cx: &Cx<'_>) -> bool {
    cx.mode == Mode::Draft && cx.rows >= 2
}

/// 2026-10-03: Refuse a plan kernel `k` that is not `want`, the handle legacy launches.
fn same_kernel(cx: &Cx<'_>, k: usize, want: KernelHandle, what: &str) -> Result<()> {
    let h = cx.handle(k)?;
    ensure!(
        want.0 != 0 && h.0 == want.0,
        "group {}: the plan's {what} kernel `{}` is not the one the MoE layer launches here",
        cx.g.index,
        cx.g.group.kernels[k]
    );
    Ok(())
}

/// 2026-10-03: `moe_router`: the router logits `[rows, experts]` BF16.
pub(crate) struct MoeRouter;

impl OpEmitter for MoeRouter {
    fn id(&self) -> &'static str {
        "moe_router"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["router"])?;
        let (b, _) = bound(cx, 0)?;
        let f = b.facts;
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        let y = cx.ptr(cx.g.output(0, 0)?)?;
        let m = rows(cx)?;
        let w = b.router;
        let k = cx.handle(0)?;
        if batched_routing(cx) {
            expect_kernel(cx, 0, "moe_router_gemm_bf16")?;
            same_kernel(cx, 0, b.kernels.router_gemm, "router GEMM")?;
            return cx.push(
                0,
                Box::new(move |e| {
                    ops::moe_router_gemm(e.gpu, k, x, &w, y, m, f.num_experts, f.hidden, e.stream)
                }),
            );
        }
        same_kernel(cx, 0, b.kernels.router_rows, "per-row router")?;
        let blocks = router_block_rows(m);
        cx.push(
            0,
            Box::new(move |e| {
                ops::dense_gemv_batchm_split(
                    e.gpu,
                    k,
                    x,
                    &w,
                    y,
                    m,
                    blocks,
                    f.num_experts,
                    f.hidden,
                    f.num_experts,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-10-03: `moe_grouped_route`: top-k over the router logits into `topk_w` / `topk_id`
/// (`[rows * top_k]` FP32 and u32), then the slot sort into the arena scratch.
pub(crate) struct MoeGroupedRoute;

impl OpEmitter for MoeGroupedRoute {
    fn id(&self) -> &'static str {
        "moe_grouped_route"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["top_k"])?;
        let (b, s) = bound(cx, 0)?;
        let f = b.facts;
        expect_kernel(cx, 1, "moe_fp8_grouped_sort")?;
        let topk_want = if batched_routing(cx) {
            b.kernels.topk_batched
        } else {
            b.kernels.topk_rows
        };
        same_kernel(cx, 0, topk_want, "top-k")?;
        same_kernel(cx, 1, b.kernels.sort, "sort")?;
        let logits = cx.ptr(cx.g.input(0, 0)?)?;
        let weights = cx.ptr(cx.g.output(0, 0)?)?;
        let ids = cx.ptr(cx.g.output(0, 1)?)?;
        let m = rows(cx)?;
        let (sort, _) = b.sort_out(&s, m);
        let (kt, ks) = (cx.handle(0)?, cx.handle(1)?);
        cx.push(
            0,
            Box::new(move |e| {
                ops::moe_topk_softmax_batched(
                    e.gpu,
                    kt,
                    logits,
                    ids,
                    weights,
                    f.num_experts,
                    f.top_k,
                    f.norm_topk_prob,
                    m,
                    e.stream,
                )
            }),
        )?;
        cx.push(
            1,
            Box::new(move |e| {
                ops::moe_fp8_grouped_sort(
                    e.gpu,
                    ks,
                    sort,
                    ids,
                    m * f.top_k,
                    f.num_experts,
                    f.top_k,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-10-03: An E4M3 edge with one FP32 scale per row and 128 columns, packed as its bytes
/// count it (`Format::bytes`): `[rows, width]` E4M3 at the base, the scales `[rows, width / 128]`
/// after them. The W8A8 kernels take the two halves as separate pointers.
fn fp8_g128(cx: &Cx<'_>, edge: usize) -> Result<(DevicePtr, DevicePtr)> {
    let e = &cx.g.circuit.edges[edge];
    ensure!(
        cx.edge_format(edge)
            == Format::Fp8E4m3 {
                scale: Scale::Group(128)
            },
        "`{}` is stored as {}, not E4M3 with 128-column scales",
        e.id,
        cx.edge_format(edge).name()
    );
    let mut dims = cx.g.circuit.dims.clone();
    dims.insert("n".into(), cx.rows);
    let rows = e
        .rows
        .eval(&dims)
        .map_err(|err| anyhow::anyhow!("`{}` has no row count: {err}", e.id))?;
    let base = cx.ptr(edge)?;
    Ok((base, base.offset(usize::try_from(rows * e.dim_value)?)))
}

/// 2026-10-03: `moe_grouped_fp8_w8a8`: the W8A8 expert step
/// (`MoeLayer::run_fp8_grouped_w8a8`). The gate+up group quantizes the layer input into
/// `xn_quant` and writes the re-quantized SiLU products `eact_quant` (by sorted position) and
/// `sact_quant` (by token); the down group reads them.
pub(crate) struct MoeGroupedFp8W8a8;

impl OpEmitter for MoeGroupedFp8W8a8 {
    fn id(&self) -> &'static str {
        "moe_grouped_fp8_w8a8"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        let (b, s) = bound(cx, 0)?;
        ensure!(
            b.facts.w8a8,
            "the MoE layer runs W8A16 experts (no FP8 expert activations published, or the W8A8 \
             kernels did not resolve); this W8A8 plan does not describe it"
        );
        let f = b.facts;
        let m = rows(cx)?;
        let (sort, cap) = b.sort_out(&s, m);
        let rows_of = ops::Fp8GroupedW8a8Rows {
            expert_offsets: sort.expert_offsets,
            sorted_token_ids: sort.sorted_token_ids,
            active_experts: sort.active_experts,
            active_count: sort.active_count,
            cap,
            num_tokens: m,
        };
        if cx.g.group.nodes.len() == 7 {
            cx.g.expect_ops(
                self.id(),
                &[
                    "act_quant",
                    "expert_gate_up",
                    "silu_mul",
                    "act_quant",
                    "linear:shared_gate_up",
                    "silu_mul",
                    "act_quant",
                ],
            )?;
            same_kernel(cx, 0, b.kernels.quant_w8a8, "W8A8 input quantizer")?;
            same_kernel(cx, 1, b.kernels.gate_up_w8a8, "W8A8 gate+up")?;
            let x = cx.ptr(cx.g.input(0, 0)?)?;
            let xq_edge = cx.g.output(0, 0)?;
            ensure!(
                cx.g.input(1, 0)? == xq_edge && cx.g.input(4, 0)? == xq_edge,
                "the routed and shared gate+up do not read the quantized input"
            );
            let (xq, xs) = fp8_g128(cx, xq_edge)?;
            let act = fp8_g128(cx, cx.g.output(3, 0)?)?;
            let sh_act = fp8_g128(cx, cx.g.output(6, 0)?)?;
            let (kq, kg) = (cx.handle(0)?, cx.handle(1)?);
            cx.push(
                0,
                Box::new(move |e| {
                    ops::moe_act_quant_e4m3(e.gpu, kq, x, xq, xs, m, f.hidden, e.stream)
                }),
            )?;
            let (gate, up, sh) = (b.gate, b.up, b.shared);
            return cx.push(
                1,
                Box::new(move |e| {
                    ops::moe_expert_gate_up_act_fp8_grouped_tc_w8a8(
                        e.gpu,
                        kg,
                        xq,
                        xs,
                        (gate.weights, gate.scales),
                        (up.weights, up.scales),
                        act,
                        &rows_of,
                        &sh.gate_proj,
                        &sh.up_proj,
                        sh_act,
                        f.inter,
                        f.hidden,
                        e.stream,
                    )
                }),
            );
        }
        cx.g.expect_ops(self.id(), &["expert_down", "linear:shared_down"])?;
        same_kernel(cx, 0, b.kernels.down_w8a8, "W8A8 down")?;
        let act = fp8_g128(cx, cx.g.input(0, 0)?)?;
        let sh_act = fp8_g128(cx, cx.g.input(1, 0)?)?;
        let out = cx.ptr(cx.g.output(0, 0)?)?;
        let sh_out = cx.ptr(cx.g.output(1, 0)?)?;
        let (down, sh) = (b.down, b.shared);
        let k = cx.handle(0)?;
        cx.push(
            0,
            Box::new(move |e| {
                ops::moe_expert_down_act_fp8_grouped_tc_w8a8(
                    e.gpu,
                    k,
                    act,
                    (down.weights, down.scales),
                    out,
                    &rows_of,
                    sh_act,
                    &sh.down_proj,
                    sh_out,
                    f.hidden,
                    f.inter,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-10-03: `moe_blend`: `f = Σ_k w_k · edown[perm(t, k)] + sigmoid(xn · seg) · sdown`, one
/// BF16 rounding (`moe_weighted_sum_blend_fp8_grouped`), the shared-expert gate fused in.
pub(crate) struct MoeBlend;

impl OpEmitter for MoeBlend {
    fn id(&self) -> &'static str {
        "moe_blend"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear:shared_gate", "blend"])?;
        let (b, s) = bound(cx, 0)?;
        expect_kernel(cx, 0, "moe_weighted_sum_blend_fp8_grouped")?;
        same_kernel(cx, 0, b.kernels.blend, "blend")?;
        let f = b.facts;
        let m = rows(cx)?;
        let (sort, _) = b.sort_out(&s, m);
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        let blend = cx.g.node(1);
        let [edown, topk_w, sdown, sg] = blend.inputs[..] else {
            bail!("`{}` reads {} edges, not 4", blend.id, blend.inputs.len());
        };
        ensure!(
            sg == cx.g.output(0, 0)?,
            "the blend does not read the shared gate it fuses"
        );
        let (edown, topk_w, sdown) = (cx.ptr(edown)?, cx.ptr(topk_w)?, cx.ptr(sdown)?);
        let out = cx.ptr(cx.g.output(1, 0)?)?;
        let gate = b.shared_gate;
        let k = cx.handle(0)?;
        cx.push(
            0,
            Box::new(move |e| {
                ops::moe_weighted_sum_blend_fp8_grouped(
                    e.gpu,
                    k,
                    out,
                    edown,
                    topk_w,
                    sort.token_to_perm,
                    sdown,
                    x,
                    gate.weight,
                    f.hidden,
                    f.top_k,
                    f.hidden,
                    m,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-10-03: The MoE emitters (LIFECYCLE-DESIGN.md 15.11).
pub(super) static ALL: &[&dyn OpEmitter] =
    &[&MoeRouter, &MoeGroupedRoute, &MoeGroupedFp8W8a8, &MoeBlend];
