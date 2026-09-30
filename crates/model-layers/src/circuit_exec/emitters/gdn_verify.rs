// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The GatedDeltaNet emitters of the MTP verify, mirroring
//! `qwen3_ssm/trait_decode_batched/gates_norm.rs` (the batched BA gates and output norm) and
//! `qwen3_ssm/trait_decode_batched_conv_gdn.rs` (the WY recurrence over `K` rows of one
//! sequence).
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Each row's decay and beta share its row (`[decay | beta]`, a row pack), the layout the
//!   batched gate kernel writes and the WY kernel reads through one base and a stride.
//! - The WY kernel updates the sequence's h state in place and writes the h state after each
//!   row but the last into the step's rollback slots; it reads them from the step
//!   (`StepEnv::gdn_state`), never baked at compile time.

use anyhow::{Result, bail, ensure};
use metrale_circuit::LinearRole;
use metrale_circuit::planner::{Layout, RowPack};
use metrale_gpu_runtime::gpu::KernelHandle;

use super::super::bindings::WeightSlot;
use super::super::compile::{Cx, GroupRef, OpEmitter};
use super::{dense, dim, expect_kernel, rows};
use crate::layers::ops;

fn gdn_dims(cx: &Cx<'_>) -> Result<(u32, u32, u32, u32)> {
    Ok((
        dim(cx, "lin_k_heads")?,
        dim(cx, "lin_k_dim")?,
        dim(cx, "lin_v_heads")?,
        dim(cx, "lin_v_dim")?,
    ))
}

/// 2026-09-29: `dense_gemm_ba_gates`: the BF16 `in_proj_ba` GEMM over every row with the decay
/// and beta gates, one launch. At the verify's row counts the batched kernel's Hopper twin
/// declines (it needs two rows per SM), so the base kernel runs.
pub(crate) struct DenseGemmBaGates;

impl OpEmitter for DenseGemmBaGates {
    fn id(&self) -> &'static str {
        "dense_gemm_ba_gates"
    }

    fn constrain(&self, g: &GroupRef<'_>, layout: &mut Layout) -> Result<()> {
        layout.row_packs.push(RowPack {
            members: vec![g.output(1, 0)?, g.output(1, 1)?],
            over: None,
        });
        Ok(())
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear:ba", "gdn_gates"])?;
        expect_kernel(cx, 0, "dense_gemm_ba_gates_prefill")?;
        let (nk, _, nv, _) = gdn_dims(cx)?;
        let ba = dense(cx.weight(0, WeightSlot::Linear(LinearRole::Ba))?, "ba")?;
        let a_log = dense(cx.weight(1, WeightSlot::GdnALog)?, "A_log")?.weight;
        let dt_bias = dense(cx.weight(1, WeightSlot::GdnDtBias)?, "dt_bias")?.weight;
        let ba_size = u32::try_from(cx.g.circuit.edges[cx.g.output(0, 0)?].dim_value)?;
        let (decay, stride) = cx.strided(cx.g.output(1, 0)?)?;
        let (beta, _) = cx.strided(cx.g.output(1, 1)?)?;
        ensure!(
            beta == decay.offset(nv as usize * 4) && stride == nv * 2,
            "decay and beta do not share rows"
        );
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        let (k, m, h, vpg) = (cx.handle(0)?, rows(cx)?, dim(cx, "hidden")?, nv / nk);
        cx.push(
            0,
            Box::new(move |e| {
                ops::dense_gemm_ba_gates_prefill(
                    e.gpu,
                    k,
                    KernelHandle(0),
                    x,
                    &ba,
                    a_log,
                    dt_bias,
                    decay,
                    m,
                    ba_size,
                    h,
                    h,
                    stride,
                    nv,
                    vpg,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-09-29: `gdn_decode_wy`: the WY-chunkwise gated delta rule over the `K` verify rows of
/// one sequence (`gated_delta_rule_wy2`/`wy3`/`wy4`), one launch.
pub(crate) struct GdnDecodeWy;

impl OpEmitter for GdnDecodeWy {
    fn id(&self) -> &'static str {
        "gdn_decode_wy"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["gdn_recurrence"])?;
        let func = cx.g.group.kernels[0].func.clone();
        let rows = cx.rows as usize;
        ensure!(
            func == format!("gated_delta_rule_wy{rows}"),
            "`{func}` does not serve {rows} verify rows"
        );
        let (nk, kd, nv, vd) = gdn_dims(cx)?;
        let layer =
            cx.g.node(0)
                .layer
                .ok_or_else(|| anyhow::anyhow!("a GDN node outside the layers"))?;
        let (qkv, qk_stride) = cx.strided(cx.g.input(0, 0)?)?;
        let key = (nk * kd) as usize * 2;
        let (q, kk, v) = (qkv, qkv.offset(key), qkv.offset(key * 2));
        let (gate, gb_stride) = cx.strided(cx.g.input(0, 1)?)?;
        let (beta, _) = cx.strided(cx.g.input(0, 2)?)?;
        let out = cx.ptr(cx.g.output(0, 0)?)?;
        let k = cx.handle(0)?;
        cx.push(
            0,
            Box::new(move |e| {
                let st = e.gdn_state(layer, 0)?;
                let hi = st.h_steps;
                ensure!(
                    hi[..rows - 1].iter().all(|p| !p.is_null()),
                    "layer {layer} lacks the h rollback slots of a {rows}-row verify"
                );
                match rows {
                    2 => ops::gdn_decode_wy2(
                        e.gpu, k, st.h, q, kk, v, gate, beta, out, hi[0], 1, nk, nv, kd, vd,
                        qk_stride, qk_stride, gb_stride, false, e.stream,
                    ),
                    3 => ops::gdn_decode_wy3(
                        e.gpu, k, st.h, q, kk, v, gate, beta, out, hi[0], hi[1], 1, nk, nv, kd, vd,
                        qk_stride, qk_stride, gb_stride, false, e.stream,
                    ),
                    4 => ops::gdn_decode_wy4(
                        e.gpu, k, st.h, q, kk, v, gate, beta, out, hi[0], hi[1], hi[2], 1, nk, nv,
                        kd, vd, qk_stride, qk_stride, gb_stride, false, e.stream,
                    ),
                    other => bail!("no WY kernel for {other} rows"),
                }
            }),
        )
    }
}
