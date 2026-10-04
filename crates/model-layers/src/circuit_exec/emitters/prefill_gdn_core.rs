// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `prefill_gdn_core`: the GatedDeltaNet block body of a prefill pass between its two
//! projections, as one group (`qwen3_ssm/trait_prefill_block.rs:127-327`): the BA GEMM with the
//! gates, the conv with its state write, the L2 norm of the Q and K heads, the recurrence and
//! the gated output norm. Its intermediates live where the legacy layer keeps them: the gates
//! in `ssm_gates`, the conv rows in `ssm_qkvz`, the recurrence output in `attn_output`.
//!
//! The recurrence arm is the plan's: the chunked FLA trio (`gdn_exact_replay = off`) or the
//! register-resident kernel (`on`, a pass after a prefix-cache restore), as
//! `qwen3_ssm/trait_prefill_recur.rs:96-262` picks them.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - The group's kernels are `[ba, conv, conv state, l2, <recurrence...>, gated norm]`; the
//!   recurrence is three kernels (FLA) or one (register-resident).
//! - The BA-gates kernel the engine picks for the pass's rows is the plan's, checked on every
//!   launch: the pick depends on the rows, which a bucket bounds but the build does not know.

use anyhow::{Result, bail, ensure};
use metrale_circuit::LinearRole;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::super::bindings::WeightSlot;
use super::super::super::compile::{Cx, OpEmitter};
use super::super::{dense, expect_kernel};
use super::{GdnDims, layer_of, present, push_bundle};
use crate::layers::ops;

/// 2026-10-03: The group's ops, in pattern order.
const OPS: [&str; 7] = [
    "linear:ba",
    "gdn_gates",
    "conv1d_update",
    "state_snapshot",
    "l2_norm",
    "gdn_recurrence",
    "gated_rms_norm",
];

/// 2026-10-03: The FLA arm's chunk (`trait_prefill_recur.rs:184`, the kernels' `CHUNK`).
const FLA_CHUNK: u32 = 64;

pub(crate) struct PrefillGdnCore;

impl OpEmitter for PrefillGdnCore {
    fn id(&self) -> &'static str {
        "prefill_gdn_core"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &OPS)?;
        let fla = match cx.g.group.kernels.len() {
            8 => true,
            6 => false,
            n => bail!(
                "`{}` takes 8 kernels (FLA) or 6 (replay), not {n}",
                self.id()
            ),
        };
        let c = cx.config;
        let d = GdnDims::of(c);
        ensure!(
            d.kd == 128 && d.vd == 128,
            "both prefill recurrence arms need 128-dim heads (k {}, v {})",
            d.kd,
            d.vd
        );
        let layer = layer_of(cx, 5)?;
        let arena = cx.arena()?;
        let (gates, conv_out, gdn_out) = (arena.ssm_gates(), arena.ssm_qkvz(), arena.attn_output());
        let fla_scratch = arena.gdn_fla_scratch();
        let xn = cx.ptr(cx.g.input(0, 0)?)?;
        let qkvz = cx.ptr(cx.g.input(2, 0)?)?;
        let gated = cx.ptr(cx.g.output(6, 0)?)?;
        ensure!(
            cx.ptr(cx.g.input(6, 1)?)? == qkvz,
            "the gated norm reads Z from the qkvz projection's rows"
        );
        ensure!(
            gated == conv_out,
            "the gated norm writes over the conv rows (`ssm_qkvz`), as the legacy layer does"
        );
        emit_ba_gates(cx, &d, xn, gates)?;

        // 2026-10-03: The conv over `[Q | K | V]` of each `qkvz_size`-wide row, its window in the
        // sequence's conv state (`trait_prefill_recur.rs:397-412`).
        expect_kernel(cx, 1, "causal_conv1d_update_prefill_tp")?;
        expect_kernel(cx, 2, "causal_conv1d_prefill_state")?;
        let conv_w = dense(cx.weight(2, WeightSlot::GdnConv1d)?, "conv1d")?;
        let conv_k = cx
            .gpu
            .kernel("causal_conv1d", "causal_conv1d_update_prefill")?;
        let tp_k = present(
            crate::layers::try_kernel(cx.gpu, "causal_conv1d", "causal_conv1d_update_prefill_tp"),
            "causal_conv1d_update_prefill_tp",
        )?;
        let (conv_dim, d_conv) = (d.conv_dim() as u32, c.linear_conv_kernel_dim as u32);
        let qkvz_size = c.ssm_qkvz_size() as u32;
        push_bundle(
            cx,
            1,
            1,
            Box::new(move |e| {
                let t = e.prefill()?.tokens;
                let st = e.gdn_state(layer, 0)?;
                ops::conv1d_update_prefill(
                    e.gpu,
                    conv_k,
                    tp_k,
                    st.conv,
                    qkvz,
                    &conv_w,
                    DevicePtr::NULL,
                    conv_out,
                    conv_dim,
                    d_conv,
                    t,
                    qkvz_size,
                    conv_dim,
                    e.stream,
                )
            }),
        )?;

        // 2026-10-03: L2 norm of the Q and K heads, in place (`trait_prefill_block.rs:232-242`).
        expect_kernel(cx, 3, "l2_norm_bf16")?;
        let l2_k = cx.handle(3)?;
        let (qk_heads, kd) = ((d.nk * 2) as u32, d.kd as u32);
        cx.push(
            3,
            Box::new(move |e| {
                let t = e.prefill()?.tokens;
                ops::l2_norm(
                    e.gpu, l2_k, conv_out, qk_heads, kd, 1e-6, t, conv_dim, e.stream,
                )
            }),
        )?;

        let rec = Recurrence {
            layer,
            q: conv_out,
            k: conv_out.offset(d.key_dim() * 2),
            v: conv_out.offset(d.key_dim() * 2 * 2),
            gates,
            beta: gates.offset(d.nv * 4),
            out: gdn_out,
            dims: [d.nk as u32, d.nv as u32, d.kd as u32, d.vd as u32],
            conv_dim,
        };
        let gated_at = if fla {
            emit_fla(cx, rec, fla_scratch)?;
            7
        } else {
            emit_regresident(cx, rec)?;
            5
        };

        // 2026-10-03: The gated output norm over the recurrence output and Z, into the conv rows
        // (`trait_prefill_block.rs:303-317`).
        expect_kernel(cx, gated_at, "gated_rms_norm_prefill")?;
        ensure!(
            !c.gdn_norm_sigmoid,
            "a sigmoid-gated GDN norm launches `gated_norm_sigmoid`, which no prefill rule names"
        );
        let norm_k = cx.handle(gated_at)?;
        let norm = dense(cx.weight(6, WeightSlot::GdnNorm)?, "gdn norm")?;
        let z = qkvz.offset((d.key_dim() * 2 + d.value_dim()) * 2);
        let (nv, vd, value_dim) = (d.nv as u32, d.vd as u32, d.value_dim() as u32);
        let eps = c.rms_norm_eps as f32;
        cx.push(
            gated_at,
            Box::new(move |e| {
                let t = e.prefill()?.tokens;
                ops::gated_rms_norm_prefill(
                    e.gpu, norm_k, gdn_out, z, &norm, gated, nv, vd, eps, t, value_dim, qkvz_size,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-10-03: The BA GEMM with the gates over the pass's rows into `ssm_gates`, per token
/// `decay[nv]` then `beta[nv]` (`trait_prefill_block.rs:131-155`). The engine's function picks
/// the base kernel or its twin by the rows; the plan's kernel must be that pick.
fn emit_ba_gates(cx: &mut Cx<'_>, d: &GdnDims, xn: DevicePtr, gates: DevicePtr) -> Result<()> {
    let twin = cx.g.group.kernels[0].func == "dense_gemm_ba_gates_prefill_hopper";
    if !twin {
        expect_kernel(cx, 0, "dense_gemm_ba_gates_prefill")?;
    }
    let c = cx.config;
    let ba = dense(cx.weight(0, WeightSlot::Linear(LinearRole::Ba))?, "ba")?;
    let a_log = dense(cx.weight(1, WeightSlot::GdnALog)?, "A_log")?.weight;
    let dt_bias = dense(cx.weight(1, WeightSlot::GdnDtBias)?, "dt_bias")?.weight;
    let base_k = cx
        .gpu
        .kernel("ssm_preprocess", "dense_gemm_ba_gates_prefill")?;
    let twin_k = crate::layers::try_target_kernel(
        cx.gpu,
        "ssm_ba_gates_hopper",
        "dense_gemm_ba_gates_prefill_hopper",
    );
    if twin {
        present(twin_k, "dense_gemm_ba_gates_prefill_hopper")?;
    }
    let (ba_size, h) = (c.ssm_ba_size() as u32, c.hidden_size as u32);
    let (gate_stride, nv, vpg) = ((d.nv * 2) as u32, d.nv as u32, (d.nv / d.nk) as u32);
    let sm = ops::ba_gates_sm_count(cx.gpu);
    cx.push(
        0,
        Box::new(move |e| {
            let t = e.prefill()?.tokens;
            let reject = ops::ssm_ba_gates_hopper_reject(
                ops::ssm_ba_gates_hopper_enabled(),
                twin_k.0 != 0,
                t,
                ba_size,
                h,
                h,
                sm,
            );
            ensure!(
                twin == reject.is_none(),
                "the plan runs the {} BA-gates kernel at {t} rows where the engine picks the other \
                 ({})",
                if twin { "twin" } else { "base" },
                reject.unwrap_or("the twin's guard passes")
            );
            ops::dense_gemm_ba_gates_prefill(
                e.gpu,
                base_k,
                twin_k,
                xn,
                &ba,
                a_log,
                dt_bias,
                gates,
                t,
                ba_size,
                h,
                h,
                gate_stride,
                nv,
                vpg,
                e.stream,
            )
        }),
    )
}

/// 2026-10-03: The recurrence's operands: the layer whose h state it updates, the L2-normed
/// conv rows, the gates, and its output.
#[derive(Clone, Copy)]
struct Recurrence {
    layer: usize,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    gates: DevicePtr,
    beta: DevicePtr,
    out: DevicePtr,
    /// 2026-10-03: `[nk, nv, kd, vd]`.
    dims: [u32; 4],
    conv_dim: u32,
}

/// 2026-10-03: The chunked FLA arm, `ops::gdn_prefill_fla`'s three launches: `recompute_wu`,
/// the pipe state spine, and the 8-warp `chunk_fwd_o` twin (`trait_prefill_recur.rs:157-221`).
/// Every other spine and twin is refused: their handles go in as 0, and the switches
/// (`policy::PREFILL_ENV_SWITCHES`) and the target default that would pick them refuse the build.
fn emit_fla(cx: &mut Cx<'_>, r: Recurrence, scratch: DevicePtr) -> Result<()> {
    ensure!(
        !cfg!(metrale_scale),
        "a metrale_scale build runs the split4 recurrence, which no prefill rule names"
    );
    ensure!(
        !ops::target_defaults::resolved().gdn_prefill_tc.value,
        "the target runs the tensor-core FLA spine (`gdn_prefill_tc`), which no prefill rule names"
    );
    ensure!(
        ops::gdn_scalar_spine() == ops::GdnScalarSpine::Pipe,
        "the FLA state spine is not the pipe spine the rules name"
    );
    ensure!(
        scratch.0 != 0,
        "the FLA scratch (`gdn_fla_scratch`) is not allocated"
    );
    ensure!(
        cx.gpu.has_module("gdn_chunk_fwd_o_mma8"),
        "the 8-warp chunk_fwd_o twin is not compiled for this target"
    );
    expect_kernel(cx, 4, "gated_delta_rule_recompute_wu")?;
    expect_kernel(cx, 5, ops::GDN_SCALAR_SPINE_PIPE)?;
    expect_kernel(cx, 6, "gated_delta_rule_chunk_fwd_o_mma8")?;
    let (wu, spine) = (cx.handle(4)?, cx.handle(5)?);
    // 2026-10-03: The parent `chunk_fwd_o`: the twin replaces it inside the call
    // (`ops::gdn_fwd_o_mma8`), which needs the parent's handle to recognise the pick.
    let fwd_o = present(
        crate::layers::try_kernel(
            cx.gpu,
            "gated_delta_rule_fla",
            "gated_delta_rule_chunk_fwd_o",
        ),
        "gated_delta_rule_chunk_fwd_o",
    )?;
    let none = KernelHandle(0);
    let [nk, nv, kd, vd] = r.dims;
    push_bundle(
        cx,
        4,
        2,
        Box::new(move |e| {
            let t = e.prefill()?.tokens;
            let h = e.gdn_state(r.layer, 0)?.h;
            let nt = t.div_ceil(FLA_CHUNK);
            let (n, nvu, kdu, vdu) = (nt as usize, nv as usize, kd as usize, vd as usize);
            let w_out = scratch;
            let u_out = w_out.offset(n * nvu * 64 * kdu * 2);
            let s_out = u_out.offset(n * nvu * 64 * vdu * 2);
            let uc_out = s_out.offset(n * nvu * kdu * vdu * 2);
            let gc_out = uc_out.offset(n * nvu * 64 * vdu * 2);
            ops::gdn_prefill_fla(
                e.gpu,
                wu,
                none,
                none,
                none,
                none,
                none,
                spine,
                none,
                fwd_o,
                h,
                r.q,
                r.k,
                r.v,
                r.gates,
                r.beta,
                r.out,
                w_out,
                u_out,
                s_out,
                uc_out,
                gc_out,
                1,
                t,
                nt,
                nk,
                nv,
                kd,
                vd,
                r.conv_dim,
                r.conv_dim,
                nv * 2,
                false,
                DevicePtr::NULL,
                DevicePtr::NULL,
                false,
                false,
                e.stream,
            )
        }),
    )
}

/// 2026-10-03: The exact-replay arm, the register-resident recurrence
/// (`trait_prefill_recur.rs:222-262`), which legacy takes after a restore when
/// `METRALE_NO_GDN_REGRESIDENT` is unset. With it set legacy runs WY4, which no prefill rule
/// names.
fn emit_regresident(cx: &mut Cx<'_>, r: Recurrence) -> Result<()> {
    ensure!(
        ops::ModelLevers::get().gdn_regresident,
        "METRALE_NO_GDN_REGRESIDENT=1 replays with WY4, which no prefill rule names"
    );
    expect_kernel(cx, 4, "gated_delta_rule_prefill_regresident")?;
    let k = cx.handle(4)?;
    let [nk, nv, kd, vd] = r.dims;
    cx.push(
        4,
        Box::new(move |e| {
            let t = e.prefill()?.tokens;
            let h = e.gdn_state(r.layer, 0)?.h;
            ops::gdn_prefill_regresident(
                e.gpu,
                k,
                h,
                r.q,
                r.k,
                r.v,
                r.gates,
                r.beta,
                r.out,
                1,
                t,
                nk,
                nv,
                kd,
                vd,
                r.conv_dim,
                r.conv_dim,
                nv * 2,
                e.stream,
            )
        }),
    )
}
