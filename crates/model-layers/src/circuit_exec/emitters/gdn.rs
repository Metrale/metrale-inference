// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The GatedDeltaNet emitters, mirroring `qwen3_ssm/ssm_forward.rs` on its FP32
//! arm (`conv1d_l2norm_f32_k`, `gdn_f32_k` and `gated_rms_norm_f32_k` resolved, the fused
//! output norm off).
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - The qkvz edge is `[Q | K | V | Z]` BF16 (`key_dim`, `key_dim`, `value_dim`, `value_dim`);
//!   the conv reads its first `2 * key_dim + value_dim` channels and the output norm its Z.
//! - The conv output is `[Q | K | V]` FP32; the recurrence reads each part at its offset.
//! - The recurrent and conv state are the sequence's, read from the step
//!   ([`super::super::program::StepEnv::gdn_state`]), never baked at compile time.

use anyhow::{Context, Result, ensure};
use metrale_circuit::Mode;

use super::super::bindings::WeightSlot;
use super::super::compile::{Cx, OpEmitter};
use super::super::program::MAX_VERIFY_STEPS;
use super::{dense, dim, per_row, rows};
use crate::layers::ops;

/// 2026-09-28: The GDN dims every emitter here reads.
struct Dims {
    nk: u32,
    kd: u32,
    nv: u32,
    vd: u32,
}

impl Dims {
    fn of(cx: &Cx<'_>) -> Result<Self> {
        Ok(Dims {
            nk: dim(cx, "lin_k_heads")?,
            kd: dim(cx, "lin_k_dim")?,
            nv: dim(cx, "lin_v_heads")?,
            vd: dim(cx, "lin_v_dim")?,
        })
    }

    fn key(&self) -> usize {
        (self.nk * self.kd) as usize
    }

    fn value(&self) -> usize {
        (self.nv * self.vd) as usize
    }
}

fn layer_index(cx: &Cx<'_>, i: usize) -> Result<usize> {
    cx.g.node(i).layer.context("a GDN node outside the layers")
}

/// 2026-09-28: `dense_gemv_ba_gates`: the BF16 `in_proj_ba` GEMV with the decay and beta gates.
pub(crate) struct DenseGemvBaGates;

impl OpEmitter for DenseGemvBaGates {
    fn id(&self) -> &'static str {
        "dense_gemv_ba_gates"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear:ba", "gdn_gates"])?;
        per_row(cx)?;
        let d = Dims::of(cx)?;
        let ba = dense(
            cx.weight(0, WeightSlot::Linear(metrale_circuit::LinearRole::Ba))?,
            "ba",
        )?;
        let a_log = dense(cx.weight(1, WeightSlot::GdnALog)?, "A_log")?.weight;
        let dt_bias = dense(cx.weight(1, WeightSlot::GdnDtBias)?, "dt_bias")?.weight;
        let ba_size = u32::try_from(cx.g.circuit.edges[cx.g.output(0, 0)?].dim_value)?;
        let (k, h, vpg) = (cx.handle(0)?, dim(cx, "hidden")?, d.nv / d.nk);
        for i in 0..cx.reps() {
            let x = cx.row_ptr(cx.g.input(0, 0)?, i)?;
            let decay = cx.row_ptr(cx.g.output(1, 0)?, i)?;
            let beta = cx.row_ptr(cx.g.output(1, 1)?, i)?;
            cx.push(
                0,
                Box::new(move |e| {
                    ops::dense_gemv_ba_gates(
                        e.gpu, k, x, &ba, a_log, dt_bias, decay, beta, ba_size, h, vpg, e.stream,
                    )
                }),
            )?;
        }
        Ok(())
    }
}

/// 2026-09-28: `conv1d_update_l2norm`: the conv1d step with SiLU, then the per-head L2 norm of
/// Q and K, into the conv output (FP32 in decode, BF16 in the verify). 2026-09-29: A verify
/// then copies the window after each row but the last into the step's rollback slots (the
/// `state_snapshot` member); other modes take no snapshot.
pub(crate) struct Conv1dUpdateL2norm;

impl OpEmitter for Conv1dUpdateL2norm {
    fn id(&self) -> &'static str {
        "conv1d_update_l2norm"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["conv1d_update", "state_snapshot", "l2_norm"])?;
        per_row(cx)?;
        let d = Dims::of(cx)?;
        let w = dense(cx.weight(0, WeightSlot::GdnConv1d)?, "conv1d")?;
        let layer = layer_index(cx, 0)?;
        let conv_dim = (d.key() * 2 + d.value()) as u32;
        let d_conv = u32::try_from(cx.config.linear_conv_kernel_dim)?;
        let (qk, kd, k) = ((d.key() * 2) as u32, d.kd, cx.handle(0)?);
        // 2026-09-29: The FP32 window `[conv_dim, d_conv]` (`SsmLayerState::conv_state`).
        let window = conv_dim as usize * d_conv as usize * 4;
        let snapshots = cx.g.group.copy_count(cx.rows) as usize;
        ensure!(
            snapshots == 0 || (cx.mode == Mode::Verify && snapshots < MAX_VERIFY_STEPS + 1),
            "{snapshots} conv snapshots in a {:?} plan",
            cx.mode
        );
        for i in 0..cx.reps() {
            let qkvz = cx.row_ptr(cx.g.input(0, 0)?, i)?;
            let out = cx.row_ptr(cx.g.output(2, 0)?, i)?;
            let s = cx.state_row(i);
            cx.push(
                0,
                Box::new(move |e| {
                    let st = e.gdn_state(layer, s)?;
                    ops::conv1d_update_l2norm(
                        e.gpu, k, st.conv, qkvz, &w, out, conv_dim, d_conv, 1, qk, kd, 1e-6,
                        e.stream,
                    )
                }),
            )?;
            if i < snapshots {
                cx.push_copy(Box::new(move |e| {
                    let st = e.gdn_state(layer, s)?;
                    let to = st.conv_steps[i];
                    ensure!(!to.is_null(), "layer {layer} has no conv rollback slot {i}");
                    e.gpu.copy_d2d_async(st.conv, to, window, e.stream)
                }))?;
            }
        }
        Ok(())
    }
}

/// 2026-09-28: `gdn_decode`: the FP32 gated delta rule over the sequence's h state.
pub(crate) struct GdnDecode;

impl OpEmitter for GdnDecode {
    fn id(&self) -> &'static str {
        "gdn_decode"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["gdn_recurrence"])?;
        per_row(cx)?;
        let d = Dims::of(cx)?;
        let layer = layer_index(cx, 0)?;
        let k = cx.handle(0)?;
        let (nk, nv, kd, vd) = (d.nk, d.nv, d.kd, d.vd);
        for i in 0..cx.reps() {
            let qkv = cx.row_ptr(cx.g.input(0, 0)?, i)?;
            let (q, kk, v) = (qkv, qkv.offset(d.key() * 4), qkv.offset(d.key() * 2 * 4));
            let decay = cx.row_ptr(cx.g.input(0, 1)?, i)?;
            let beta = cx.row_ptr(cx.g.input(0, 2)?, i)?;
            let out = cx.row_ptr(cx.g.output(0, 0)?, i)?;
            let s = cx.state_row(i);
            cx.push(
                0,
                Box::new(move |e| {
                    let st = e.gdn_state(layer, s)?;
                    ops::gdn_decode(
                        e.gpu, k, st.h, q, kk, v, decay, beta, out, 1, nk, nv, kd, vd, e.stream,
                    )
                }),
            )?;
        }
        Ok(())
    }
}

/// 2026-09-28: `gated_rms_norm`: the RMS norm of the FP32 recurrence output gated by Z, one row
/// per value head.
pub(crate) struct GatedRmsNorm;

impl OpEmitter for GatedRmsNorm {
    fn id(&self) -> &'static str {
        "gated_rms_norm"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["gated_rms_norm"])?;
        let d = Dims::of(cx)?;
        let w = dense(cx.weight(0, WeightSlot::GdnNorm)?, "gdn norm")?;
        let (k, nv, vd) = (cx.handle(0)?, d.nv, d.vd);
        let eps = cx.config.rms_norm_eps as f32;
        let z_at = (d.key() * 2 + d.value()) * 2;
        if cx.g.group.kernels[0].func == "gated_rms_norm_prefill" {
            // 2026-09-29: Every (head, row) pair in one launch, rows at their strides
            // (`trait_decode_batched/gates_norm.rs` `batched_gated_norm`).
            let (core, in_stride) = cx.strided(cx.g.input(0, 0)?)?;
            let (qkvz, gate_stride) = cx.strided(cx.g.input(0, 1)?)?;
            let z = qkvz.offset(z_at);
            let out = cx.ptr(cx.g.output(0, 0)?)?;
            let n = rows(cx)?;
            return cx.push(
                0,
                Box::new(move |e| {
                    ops::gated_rms_norm_prefill(
                        e.gpu,
                        k,
                        core,
                        z,
                        &w,
                        out,
                        nv,
                        vd,
                        eps,
                        n,
                        in_stride,
                        gate_stride,
                        e.stream,
                    )
                }),
            );
        }
        per_row(cx)?;
        for i in 0..cx.reps() {
            let core = cx.row_ptr(cx.g.input(0, 0)?, i)?;
            let z = cx.row_ptr(cx.g.input(0, 1)?, i)?.offset(z_at);
            let out = cx.row_ptr(cx.g.output(0, 0)?, i)?;
            cx.push(
                0,
                Box::new(move |e| {
                    ops::gated_rms_norm(e.gpu, k, core, z, &w, out, nv, vd, vd, eps, vd, e.stream)
                }),
            )?;
        }
        Ok(())
    }
}

/// 2026-10-03: This module's emitters, for the registry in `mod.rs`.
pub(super) static ALL: &[&dyn OpEmitter] = &[
    &DenseGemvBaGates,
    &Conv1dUpdateL2norm,
    &GdnDecode,
    &GatedRmsNorm,
];
