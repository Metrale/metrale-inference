// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The attention emitters, mirroring `qwen3_attention/decode/attention_forward.rs`
//! on a BF16 KV cache: interleaved MRoPE in place on Q and K, the `reshape_and_cache` write, the
//! non-split paged decode kernel and the sigmoid output gate.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Positions, KV slots, sequence lengths and block tables are read on the device from the
//!   step's metadata (`Fixed::meta`); the only per-step host value is the block-table width.
//! - The KV pools are the layer's own (`attn_layer_idx`), fixed after boot.

use anyhow::{Context, Result, ensure};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_circuit::planner::Layout;

use super::super::compile::{Cx, GroupRef, OpEmitter};
use super::{attn_facts, dim, one_row, rows};
use crate::layers::ops;

/// 2026-09-28: The KV pools of member `i`'s layer.
fn pools(
    cx: &Cx<'_>,
    i: usize,
) -> Result<(
    metrale_gpu_runtime::gpu::DevicePtr,
    metrale_gpu_runtime::gpu::DevicePtr,
)> {
    let a = attn_facts(cx, i)?;
    ensure!(
        a.kv_dtype == KvCacheDtype::Bf16,
        "`{}`: the plan writes a BF16 cache; this layer's is {:?}",
        cx.g.node(i).id,
        a.kv_dtype
    );
    let k = *cx
        .fixed
        .k_pools
        .get(a.attn_layer_idx)
        .with_context(|| format!("no K pool for attention layer {}", a.attn_layer_idx))?;
    let v = *cx
        .fixed
        .v_pools
        .get(a.attn_layer_idx)
        .with_context(|| format!("no V pool for attention layer {}", a.attn_layer_idx))?;
    Ok((k, v))
}

/// 2026-09-28: The circuit's attention dims agree with the layer's, so every edge the plan
/// sized holds what the layer's kernels write.
fn same_dims(cx: &Cx<'_>, i: usize) -> Result<()> {
    let a = attn_facts(cx, i)?;
    ensure!(
        (a.num_q_heads, a.num_kv_heads, a.head_dim)
            == (
                dim(cx, "q_heads")?,
                dim(cx, "kv_heads")?,
                dim(cx, "head_dim")?
            ),
        "`{}`: the layer's heads differ from the circuit's dims",
        cx.g.node(i).id
    );
    Ok(())
}

/// 2026-09-28: `rope_mrope`: interleaved MRoPE, in place on Q and K.
pub(crate) struct RopeMrope;

impl OpEmitter for RopeMrope {
    fn id(&self) -> &'static str {
        "rope_mrope"
    }

    fn constrain(&self, g: &GroupRef<'_>, layout: &mut Layout) -> Result<()> {
        layout.aliases.push((g.output(0, 0)?, g.input(0, 0)?));
        layout.aliases.push((g.output(0, 1)?, g.input(0, 1)?));
        Ok(())
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["rope"])?;
        one_row(cx, "rope_forward_mrope_interleaved at one position")?;
        same_dims(cx, 0)?;
        let a = attn_facts(cx, 0)?;
        ensure!(
            a.rope.mrope_interleaved,
            "the layer does not run interleaved MRoPE"
        );
        let (q, kk) = (cx.ptr(cx.g.input(0, 0)?)?, cx.ptr(cx.g.input(0, 1)?)?);
        let m = cx.fixed.meta;
        let (k, n) = (cx.handle(0)?, rows(cx)?);
        let (nq, nkv, hd, rd, theta) = (
            a.num_q_heads,
            a.num_kv_heads,
            a.head_dim,
            a.rope.rotary_dim,
            a.rope.theta,
        );
        cx.push(
            0,
            Box::new(move |e| {
                ops::rope_mrope_interleaved(
                    e.gpu,
                    k,
                    q,
                    kk,
                    m.positions,
                    m.positions_h,
                    m.positions_w,
                    n,
                    nq,
                    nkv,
                    hd,
                    rd,
                    theta,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-09-28: `kv_write`: K and V into the layer's BF16 paged cache at the step's slots.
pub(crate) struct KvWrite;

impl OpEmitter for KvWrite {
    fn id(&self) -> &'static str {
        "kv_write"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["kv_write"])?;
        same_dims(cx, 0)?;
        let a = attn_facts(cx, 0)?;
        let (kp, vp) = pools(cx, 0)?;
        let (key, value) = (cx.ptr(cx.g.input(0, 0)?)?, cx.ptr(cx.g.input(0, 1)?)?);
        let (slot, bs, stride) = (
            cx.fixed.meta.slot,
            cx.fixed.block_size,
            cx.fixed.cache_stride,
        );
        let (k, n, nkv, hd) = (cx.handle(0)?, rows(cx)?, a.num_kv_heads, a.head_dim);
        cx.push(
            0,
            Box::new(move |e| {
                ops::reshape_and_cache(
                    e.gpu,
                    k,
                    key,
                    value,
                    kp,
                    vp,
                    slot,
                    n,
                    nkv,
                    hd,
                    bs,
                    nkv * hd,
                    nkv * hd,
                    stride,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-09-28: `paged_decode`: the non-split BF16 paged decode attention.
pub(crate) struct PagedDecode;

impl OpEmitter for PagedDecode {
    fn id(&self) -> &'static str {
        "paged_decode"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["paged_attention"])?;
        same_dims(cx, 0)?;
        let a = attn_facts(cx, 0)?;
        ensure!(
            a.paged_decode_plain,
            "this layer's decode routing leaves the plain paged kernel (split-K, GQA packing or \
             a 512-wide head); the plan does not model that"
        );
        let (kp, vp) = pools(cx, 0)?;
        let q = cx.ptr(cx.g.input(0, 0)?)?;
        let out = cx.ptr(cx.g.output(0, 0)?)?;
        let m = cx.fixed.meta;
        let (k, n, bs) = (cx.handle(0)?, rows(cx)?, cx.fixed.block_size);
        let (nq, nkv, hd, scale, window) = (
            a.num_q_heads,
            a.num_kv_heads,
            a.head_dim,
            a.softmax_scale,
            a.sliding_window,
        );
        cx.push(
            0,
            Box::new(move |e| {
                ops::paged_decode_attn_bf16(
                    e.gpu,
                    k,
                    q,
                    kp,
                    vp,
                    out,
                    m.block_table,
                    m.seq_len,
                    e.max_blocks_per_seq,
                    n,
                    nq,
                    nkv,
                    hd,
                    bs,
                    scale,
                    nq * hd,
                    window,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-09-28: `sigmoid_gate_mul`: the attention output times the sigmoid of its gate.
pub(crate) struct SigmoidGateMul;

impl OpEmitter for SigmoidGateMul {
    fn id(&self) -> &'static str {
        "sigmoid_gate_mul"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["sigmoid_gate_mul"])?;
        same_dims(cx, 0)?;
        let a = attn_facts(cx, 0)?;
        let (x, gate) = (cx.ptr(cx.g.input(0, 0)?)?, cx.ptr(cx.g.input(0, 1)?)?);
        let y = cx.ptr(cx.g.output(0, 0)?)?;
        let (k, count) = (cx.handle(0)?, rows(cx)? * a.num_q_heads * a.head_dim);
        cx.push(
            0,
            Box::new(move |e| ops::sigmoid_gate_mul(e.gpu, k, x, gate, y, count, e.stream)),
        )
    }
}
