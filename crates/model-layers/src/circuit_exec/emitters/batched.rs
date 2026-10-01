// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The multi-row NVFP4 projection emitters, mirroring the legacy multi-sequence
//! sites: `qwen3_ssm/trait_decode_multi_seq/ssm_batched_proj.rs` and `qwen3_ssm/kernel_select.rs`
//! (qkvz, out_proj), `qwen3_attention/trait_impl/multi_seq/qkv/batch.rs`, `qkv.rs`
//! (`wide_verify_gemm`) and `attn/o_proj.rs` (q, k, v, o), and `dense_ffn_decode_batch.rs`
//! (`forward_k2`, `forward_km` down).
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Every launch reads and writes contiguous rows, except the gated Q of `w4a16_gemv_qg_batch`,
//!   which writes `[Q_i | Gate_i]` rows (a row pack).
//! - A GEMV the legacy launcher would reroute (to the tensor-core kernel, to W4A4) is checked
//!   at compile time against the plan's kernel: the plan either names the kernel the launcher
//!   will run, or the program is refused.
//! - The tile GEMMs read the transposed twin (`WeightSlot::Transposed`); the GEMVs the base
//!   weight.

use anyhow::{Context, Result, ensure};
use metrale_circuit::planner::{Layout, RowPack};
use metrale_circuit::{LinearRole, OpKind};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::bindings::{MixerFacts, WeightSlot};
use super::super::compile::{Cx, GroupRef, OpEmitter};
use super::{attn_facts, dim, nvfp4, rows};
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

fn role(cx: &Cx<'_>, i: usize) -> Result<LinearRole> {
    match cx.g.node(i).op {
        OpKind::Linear(r) => Ok(r),
        _ => anyhow::bail!("`{}` is not a projection", cx.g.node(i).id),
    }
}

pub(super) fn width(cx: &Cx<'_>, edge: usize) -> Result<u32> {
    u32::try_from(cx.g.circuit.edges[edge].dim_value).context("edge width overflows u32")
}

/// 2026-09-28: Member `i`'s projection weight: the base weight, or its transposed twin.
fn weight(cx: &Cx<'_>, i: usize, transposed: bool) -> Result<QuantizedWeight> {
    let r = role(cx, i)?;
    if r == LinearRole::Qkvz {
        let MixerFacts::Gdn(f) = cx.layer(i)?.mixer else {
            anyhow::bail!("a qkvz projection on a non-GDN layer");
        };
        ensure!(
            f.qkvz_deinterleaved,
            "this layer's qkvz weight is not in the deinterleaved order the plan assumes"
        );
    }
    let slot = if transposed {
        WeightSlot::Transposed(r)
    } else {
        WeightSlot::Linear(r)
    };
    nvfp4(cx.weight(i, slot)?, r.name())
}

/// 2026-09-28: Member `i`'s `(input, output, N, K)`.
fn io(cx: &Cx<'_>, i: usize) -> Result<(DevicePtr, DevicePtr, u32, u32)> {
    let (inp, out) = (cx.g.input(i, 0)?, cx.g.output(i, 0)?);
    Ok((cx.ptr(inp)?, cx.ptr(out)?, width(cx, out)?, width(cx, inp)?))
}

/// 2026-09-28: `ops::w4a16_gemv_batchm` with `handle` launches exactly `handle`: the shape is
/// not routed to W4A4 and the tensor-core route picks `handle` itself.
pub(super) fn batchm_runs(
    cx: &Cx<'_>,
    handle: KernelHandle,
    w: &QuantizedWeight,
    [m, n, k]: [u32; 3],
) -> Result<()> {
    ensure!(
        !ops::w4a4_proj::routes_w4a4(cx.gpu, w, m, n, k),
        "the projection at {m}x{n}x{k} routes to W4A4; the plan launches W4A16"
    );
    let tc = ops::gemv_tc::tc_kernel(cx.gpu, m, n, k);
    ensure!(
        tc.is_some_and(|(h, _)| h.0 == handle.0),
        "the batched GEMV at {m}x{n}x{k} does not route to the plan's tensor-core kernel"
    );
    Ok(())
}

/// 2026-09-28: `w4a16_gemv_batchm`: one projection of every row through the tensor-core GEMV
/// (`nvfp4_proj_small_m` on its W4A16 path).
pub(crate) struct W4a16GemvBatchm;

impl OpEmitter for W4a16GemvBatchm {
    fn id(&self) -> &'static str {
        "w4a16_gemv_batchm"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear"])?;
        let w = weight(cx, 0, false)?;
        let (x, y, n, k_dim) = io(cx, 0)?;
        let (k, m) = (cx.handle(0)?, rows(cx)?);
        batchm_runs(cx, k, &w, [m, n, k_dim])?;
        cx.push(
            0,
            Box::new(move |e| ops::w4a16_gemv_batchm(e.gpu, k, x, &w, y, m, n, k_dim, e.stream)),
        )
    }
}

/// 2026-09-28: The fixed-M launchers take their CUDA-core kernel unless the opt-in wide rows
/// reroute them (`gemv_tc::tc_fixed_m`). 2026-09-29: Returns the plan's row count, which the
/// kernel (`<stem>2` or `<stem>3`) must serve.
fn fixed_m(cx: &Cx<'_>, stem: &str) -> Result<u32> {
    let func = &cx.g.group.kernels[0].func;
    let m = match func.strip_prefix(stem) {
        Some("2") => 2,
        Some("3") => 3,
        _ => anyhow::bail!("`{func}` is not `{stem}2` or `{stem}3`"),
    };
    ensure!(
        cx.rows == u64::from(m),
        "`{func}` serves {m} rows, not {}",
        cx.rows
    );
    ensure!(
        !ops::gemv_tc::wide_rows_enabled(),
        "{func}: METRALE_W4A16_TC_WIDE reroutes it to the tensor-core kernel"
    );
    Ok(m)
}

/// 2026-09-28: `w4a16_gemv_batch`: one projection of 2 (or 3) rows.
pub(crate) struct W4a16GemvBatch;

impl OpEmitter for W4a16GemvBatch {
    fn id(&self) -> &'static str {
        "w4a16_gemv_batch"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear"])?;
        let m = fixed_m(cx, "w4a16_gemv_batch")?;
        let w = weight(cx, 0, false)?;
        let (x, y, n, k_dim) = io(cx, 0)?;
        let k = cx.handle(0)?;
        cx.push(
            0,
            Box::new(move |e| {
                if m == 2 {
                    ops::w4a16_gemv_batch2(e.gpu, k, x, &w, y, n, k_dim, e.stream)
                } else {
                    ops::w4a16_gemv_batch3(e.gpu, k, x, &w, y, n, k_dim, e.stream)
                }
            }),
        )
    }
}

/// 2026-09-28: `w4a16_gemv_dual_batch`: two projections of one 2-row input in one launch: the
/// FFN's gate and up (one `gate_up` node, gate rows then up rows) or attention's K and V.
pub(crate) struct W4a16GemvDualBatch;

impl OpEmitter for W4a16GemvDualBatch {
    fn id(&self) -> &'static str {
        "w4a16_gemv_dual_batch"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        let m = fixed_m(cx, "w4a16_gemv_dual_batch")?;
        let (k, h) = (cx.handle(0)?, dim(cx, "hidden")?);
        let (w1, y1, w2, y2, n) = if cx.g.group.nodes.len() == 1 {
            cx.g.expect_ops(self.id(), &["linear:gate_up"])?;
            let gate = nvfp4(cx.weight(0, WeightSlot::FfnGate)?, "gate")?;
            let up = nvfp4(cx.weight(0, WeightSlot::FfnUp)?, "up")?;
            let gu = cx.g.output(0, 0)?;
            let inter = width(cx, gu)? / 2;
            let base = cx.ptr(gu)?;
            let up_at = base.offset(cx.rows as usize * inter as usize * 2);
            (gate, base, up, up_at, inter)
        } else {
            cx.g.expect_ops(self.id(), &["linear:k", "linear:v"])?;
            let kw = nvfp4(cx.weight(0, WeightSlot::Linear(LinearRole::K))?, "k")?;
            let vw = nvfp4(cx.weight(1, WeightSlot::Linear(LinearRole::V))?, "v")?;
            let (ko, vo) = (cx.g.output(0, 0)?, cx.g.output(1, 0)?);
            ensure!(width(cx, ko)? == width(cx, vo)?, "K and V widths differ");
            (kw, cx.ptr(ko)?, vw, cx.ptr(vo)?, width(cx, ko)?)
        };
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        cx.push(
            0,
            Box::new(move |e| {
                if m == 2 {
                    ops::w4a16_gemv_dual_batch2(e.gpu, k, x, &w1, y1, &w2, y2, n, h, e.stream)
                } else {
                    ops::w4a16_gemv_dual_batch3(e.gpu, k, x, &w1, y1, &w2, y2, n, h, e.stream)
                }
            }),
        )
    }
}

/// 2026-09-28: `w4a16_gemv_qg_batch`: the gated Q projection of 2 rows, written as
/// `[Q_i | Gate_i]` rows.
pub(crate) struct W4a16GemvQgBatch;

impl OpEmitter for W4a16GemvQgBatch {
    fn id(&self) -> &'static str {
        "w4a16_gemv_qg_batch"
    }

    fn constrain(&self, g: &GroupRef<'_>, layout: &mut Layout) -> Result<()> {
        layout.row_packs.push(RowPack {
            members: vec![g.output(1, 0)?, g.output(1, 1)?],
            over: None,
        });
        Ok(())
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["linear:q", "split"])?;
        let m = fixed_m(cx, "w4a16_gemv_qg_batch")?;
        let a = attn_facts(cx, 0)?;
        ensure!(
            a.gated,
            "`w4a16_gemv_qg_batch2` writes a gated Q; this layer is ungated"
        );
        let w = nvfp4(cx.weight(0, WeightSlot::Linear(LinearRole::Q))?, "q")?;
        let (nq, hd) = (a.num_q_heads, a.head_dim);
        let (q, stride) = cx.strided(cx.g.output(1, 0)?)?;
        let (gate, _) = cx.strided(cx.g.output(1, 1)?)?;
        ensure!(
            gate == q.offset((nq * hd) as usize * 2) && stride == nq * hd * 2,
            "Q and its gate do not share rows"
        );
        let (x, k, h) = (
            cx.ptr(cx.g.input(0, 0)?)?,
            cx.handle(0)?,
            dim(cx, "hidden")?,
        );
        cx.push(
            0,
            Box::new(move |e| {
                if m == 2 {
                    ops::w4a16_gemv_qg_batch2(e.gpu, k, x, &w, q, nq * hd * 2, h, nq, hd, e.stream)
                } else {
                    ops::w4a16_gemv_qg_batch3(e.gpu, k, x, &w, q, nq * hd * 2, h, nq, hd, e.stream)
                }
            }),
        )
    }
}

/// 2026-09-28: Which tile-GEMM launcher a kernel takes.
#[derive(Clone, Copy)]
enum Tile {
    /// 2026-09-28: `ops::w4a16_gemm_n128`: the 128-wide N tile (`t_p3`, `t_k64_p3`).
    N128,
    /// 2026-09-28: `ops::w4a16_gemm`: the 64-wide N tile (`t_k64_n64_p3`).
    N64,
    /// 2026-09-28: `ops::w4a16_gemm_n128_m128`: the 128-row M tile (`t_m128`).
    M128,
}

/// 2026-09-28: One tile GEMM over the transposed twin, for every row.
fn tile_gemm(cx: &mut Cx<'_>, id: &str, tile: Tile, funcs: &[&str]) -> Result<()> {
    cx.g.expect_ops(id, &["linear"])?;
    let func = cx.g.group.kernels[0].func.clone();
    ensure!(
        funcs.contains(&func.as_str()),
        "`{id}` cannot launch `{func}`"
    );
    let w = weight(cx, 0, true)?;
    let (x, y, n, k_dim) = io(cx, 0)?;
    let (k, m) = (cx.handle(0)?, rows(cx)?);
    cx.push(
        0,
        Box::new(move |e| match tile {
            Tile::N128 => ops::w4a16_gemm_n128(e.gpu, k, x, &w, y, m, n, k_dim, e.stream),
            Tile::N64 => ops::w4a16_gemm(e.gpu, k, x, &w, y, m, n, k_dim, e.stream),
            Tile::M128 => ops::w4a16_gemm_n128_m128(e.gpu, k, x, &w, y, m, n, k_dim, e.stream),
        }),
    )
}

/// 2026-09-28: `w4a16_gemm_n128`: the 128-wide N tile GEMM (`ms_proj_gemm`'s `deep_k_gemm`
/// arm, `wide_verify_gemm`'s small-M arm).
pub(crate) struct W4a16GemmN128;

impl OpEmitter for W4a16GemmN128 {
    fn id(&self) -> &'static str {
        "w4a16_gemm_n128"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        tile_gemm(
            cx,
            self.id(),
            Tile::N128,
            &["w4a16_gemm_t_p3", "w4a16_gemm_t_k64_p3"],
        )
    }
}

/// 2026-09-28: `w4a16_gemm`: the 64-wide N tile GEMM (`k64_n64_wins`).
pub(crate) struct W4a16Gemm;

impl OpEmitter for W4a16Gemm {
    fn id(&self) -> &'static str {
        "w4a16_gemm"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        tile_gemm(cx, self.id(), Tile::N64, &["w4a16_gemm_t_k64_n64_p3"])
    }
}

/// 2026-09-28: `w4a16_gemm_n128_m128`: the 128-row M tile GEMM.
pub(crate) struct W4a16GemmN128M128;

impl OpEmitter for W4a16GemmN128M128 {
    fn id(&self) -> &'static str {
        "w4a16_gemm_n128_m128"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        tile_gemm(cx, self.id(), Tile::M128, &["w4a16_gemm_t_m128"])
    }
}
