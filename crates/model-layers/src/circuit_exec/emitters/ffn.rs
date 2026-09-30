// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The dense FFN's multi-row emitters: `dense_ffn_km` (gate and up as two batched
//! GEMVs, `dense_ffn_decode_batch.rs` `forward_km`) and `dense_ffn_mmq` (the NVFP4 MMQ arm of
//! `dense_ffn_prefill_nvfp4.rs`, which the multi-sequence decode takes through
//! `forward_prefill`).
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - The gate+up edge holds the gate rows then the up rows (gate at its base, up `rows * inter`
//!   elements after it), as `gate_out` and `up_out` are two buffers in legacy.
//! - Under the MMQ arm the gate and up rows hold the GEMM outputs without `weight_scale_2`; the
//!   fused SiLU·mul quantize applies both scales, and down's is applied by the pipelined GEMM
//!   or by `nvfp4_scale_bf16`, as legacy does.
//! - The quantized activation lives in `Fixed::ffn_act_q8`, the buffer legacy uses; it is
//!   written and read within one group. (2026-09-30) Under the declared circuit the group
//!   also holds the act_quant node before the projection: the MMQ quantizer is that node.

use anyhow::{Result, bail, ensure};
use metrale_circuit::{LinearRole, OpKind};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::bindings::{BoundWeight, WeightSlot};
use super::super::compile::{Cx, OpEmitter};
use super::batched::{batchm_runs, width};
use super::{dim, nvfp4, rows};
use crate::layers::ops;

/// 2026-09-28: A gate+up edge: `(gate rows, up rows, inter)`.
fn gate_up_rows(cx: &Cx<'_>, edge: usize) -> Result<(DevicePtr, DevicePtr, u32)> {
    let inter = width(cx, edge)? / 2;
    let base = cx.ptr(edge)?;
    Ok((
        base,
        base.offset(cx.rows as usize * inter as usize * 2),
        inter,
    ))
}

/// 2026-09-28: `dense_ffn_km`: gate and up through the tensor-core batched GEMV, two launches
/// on the same input.
pub(crate) struct DenseFfnKm;

impl OpEmitter for DenseFfnKm {
    fn id(&self) -> &'static str {
        "dense_ffn_km"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear:gate_up"])?;
        let gate = nvfp4(cx.weight(0, WeightSlot::FfnGate)?, "gate")?;
        let up = nvfp4(cx.weight(0, WeightSlot::FfnUp)?, "up")?;
        let (g, u, inter) = gate_up_rows(cx, cx.g.output(0, 0)?)?;
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        let (m, h) = (rows(cx)?, dim(cx, "hidden")?);
        for (i, (w, y)) in [(gate, g), (up, u)].into_iter().enumerate() {
            let k = cx.handle(i)?;
            batchm_runs(cx, k, &w, [m, inter, h])?;
            cx.push(
                i,
                Box::new(move |e| {
                    ops::w4a16_gemv_batchm(e.gpu, k, x, &w, y, m, inter, h, e.stream)
                }),
            )?;
        }
        Ok(())
    }
}

/// 2026-09-28: The MMQ tile a kernel serves: 16, 32, 64, or 128 for the pipelined GEMM.
fn mmq_tile(func: &str) -> Result<u32> {
    Ok(match func {
        "metrale_nvfp4_mmq16_nc" => 16,
        "metrale_nvfp4_mmq32_nc" => 32,
        "metrale_nvfp4_mmq64_nc" => 64,
        "metrale_nvfp4_gemm_pipe" => 128,
        other => bail!("`dense_ffn_mmq` cannot launch `{other}` as its GEMM"),
    })
}

/// 2026-09-28: Member `i`'s MMQ repack in `slot`.
fn mmq_weight(cx: &Cx<'_>, i: usize, slot: WeightSlot) -> Result<DevicePtr> {
    match cx.weight(i, slot)? {
        BoundWeight::Mmq(p) => Ok(p),
        other => bail!(
            "{slot:?}: expected an MMQ repack, the layer holds {}",
            other.family()
        ),
    }
}

/// 2026-09-28: `dense_ffn_mmq`: the FFN's gate and up (quantize the input, then one MMQ GEMM
/// each), or its activation and down (the fused SiLU·mul quantize, the MMQ GEMM, and the
/// down scale when the GEMM did not apply it).
pub(crate) struct DenseFfnMmq;

impl DenseFfnMmq {
    /// 2026-09-30: `lin` is the projection's member: 0, or 1 behind the declared circuit's
    /// act_quant, which the MMQ quantizer replaces (its own block layout stays in `q8`).
    fn gate_up(cx: &mut Cx<'_>, lin: usize) -> Result<()> {
        let tile = mmq_tile(&cx.g.group.kernels[1].func)?;
        ensure!(
            cx.g.group.kernels.len() == 3 && cx.g.group.kernels[2] == cx.g.group.kernels[1],
            "the gate and up GEMMs take one kernel"
        );
        let (gate, up) = (
            mmq_weight(cx, lin, WeightSlot::FfnGateMmq)?,
            mmq_weight(cx, lin, WeightSlot::FfnUpMmq)?,
        );
        let (g, u, inter) = gate_up_rows(cx, cx.g.output(lin, 0)?)?;
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        let (m, h, q8) = (rows(cx)?, dim(cx, "hidden")?, cx.fixed.ffn_act_q8);
        tile_fits(tile, m, inter, h)?;
        let kq = cx.handle(0)?;
        cx.push(
            0,
            Box::new(move |e| ops::nvfp4_mmq_quantize_act(e.gpu, kq, x, q8, m, h, e.stream)),
        )?;
        for (i, (w, y)) in [(gate, g), (up, u)].into_iter().enumerate() {
            let k = cx.handle(1 + i)?;
            cx.push(
                1 + i,
                Box::new(move |e| {
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

    /// 2026-09-30: `lin` is the down projection's member: 1, or 2 behind the act_quant.
    fn act_down(cx: &mut Cx<'_>, lin: usize) -> Result<()> {
        let tile = mmq_tile(&cx.g.group.kernels[1].func)?;
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
        let (g, u, inter) = gate_up_rows(cx, cx.g.input(0, 0)?)?;
        let y = cx.ptr(cx.g.output(lin, 0)?)?;
        let (m, h, q8) = (rows(cx)?, dim(cx, "hidden")?, cx.fixed.ffn_act_q8);
        tile_fits(tile, m, h, inter)?;
        let ks = cx.handle(0)?;
        cx.push(
            0,
            Box::new(move |e| {
                ops::nvfp4_silu_mul_quant(e.gpu, ks, g, u, q8, gate_s, up_s, m, inter, e.stream)
            }),
        )?;
        let kg = cx.handle(1)?;
        cx.push(
            1,
            Box::new(move |e| {
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
            super::expect_kernel(cx, 2, "metrale_nvfp4_scale_bf16")?;
            let kc = cx.handle(2)?;
            cx.push(
                2,
                Box::new(move |e| ops::nvfp4_scale_bf16(e.gpu, kc, y, down_s, m * h, e.stream)),
            )?;
        }
        Ok(())
    }
}

/// 2026-09-28: The tile serves `m` rows in one M tile, and the plan's kernel is the one
/// `nvfp4_mmq_gemm_tiled` launches for `n x k`: the no-column-tail kernel (N a multiple of 128),
/// and for the 128 tile the pipelined GEMM (K a multiple of 256).
fn tile_fits(tile: u32, m: u32, n: u32, k: u32) -> Result<()> {
    ensure!(m <= tile, "{m} rows exceed the {tile}-row MMQ tile");
    ensure!(
        n.is_multiple_of(128),
        "N = {n} would take the column-tail MMQ kernel"
    );
    ensure!(
        tile != 128 || k.is_multiple_of(256),
        "K = {k} keeps the 128 tile off the pipelined GEMM"
    );
    Ok(())
}

impl OpEmitter for DenseFfnMmq {
    fn id(&self) -> &'static str {
        "dense_ffn_mmq"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        let first = cx.g.node(0).op;
        match (cx.g.group.nodes.len(), first) {
            (1 | 2, OpKind::Linear(_) | OpKind::ActQuant(_)) => {
                let lin = cx.g.group.nodes.len() - 1;
                let ops: &[&str] = if lin == 0 {
                    &["linear:gate_up"]
                } else {
                    &["act_quant:nvfp4/g16", "linear:gate_up"]
                };
                cx.g.expect_ops(self.id(), ops)?;
                super::expect_kernel(cx, 0, "metrale_nvfp4_quantize_bf16")?;
                Self::gate_up(cx, lin)
            }
            _ => {
                let lin = cx.g.group.nodes.len() - 1;
                let ops: &[&str] = if lin == 1 {
                    &["silu_mul", "linear:down"]
                } else {
                    &["silu_mul", "act_quant:nvfp4/g16", "linear:down"]
                };
                cx.g.expect_ops(self.id(), ops)?;
                super::expect_kernel(cx, 0, "metrale_nvfp4_silu_mul_quant")?;
                Self::act_down(cx, lin)
            }
        }
    }
}
