// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The attention emitters, mirroring `qwen3_attention/decode/attention_forward.rs`
//! (one row) and `qwen3_attention/trait_impl/multi_seq/{qkv,attn,attn/o_proj}.rs` (one row per
//! sequence) on a BF16 KV cache: the Q/gate split, the per-head Q/K norms, RoPE in place on Q
//! and K, the `reshape_and_cache` write, the non-split paged decode kernel and the sigmoid
//! output gate.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Positions, KV slots, sequence lengths and block tables are read on the device from the
//!   step's metadata (`Cx::meta`); the only per-step host value is the block-table width.
//! - The KV pools are the layer's own (`attn_layer_idx`), fixed after boot.
//! - Q and its gate share each row (`[Q_i | Gate_i]`, a row pack); every kernel reading them
//!   takes their row stride, as the legacy multi-sequence kernels take `per_seq_qkv`.

use anyhow::{Context, Result, ensure};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_circuit::planner::{Layout, RowPack};

use super::super::compile::{Cx, GroupRef, OpEmitter};
use super::{attn_facts, dense, dim, expect_kernel, norm_slot, one_row, rows};
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
        let m = cx.meta();
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
        let (key, key_stride) = cx.strided(cx.g.input(0, 0)?)?;
        let (value, value_stride) = cx.strided(cx.g.input(0, 1)?)?;
        let (slot, bs, stride) = (cx.meta().slot, cx.fixed.block_size, cx.fixed.cache_stride);
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
                    key_stride,
                    value_stride,
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
            a.paged_decode_plain(cx.rows),
            "this layer's decode routing at {} rows leaves the plain paged kernel (split-K, GQA \
             packing or a 512-wide head); the plan does not model that",
            cx.rows
        );
        let (kp, vp) = pools(cx, 0)?;
        let (q, q_stride) = cx.strided(cx.g.input(0, 0)?)?;
        let out = cx.ptr(cx.g.output(0, 0)?)?;
        let m = cx.meta();
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
                    q_stride,
                    window,
                    e.stream,
                )
            }),
        )
    }
}

/// 2026-09-28: `sigmoid_gate_mul`: the attention output times the sigmoid of its gate; the
/// batched kernel reads each row's gate at its row stride.
pub(crate) struct SigmoidGateMul;

impl OpEmitter for SigmoidGateMul {
    fn id(&self) -> &'static str {
        "sigmoid_gate_mul"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["sigmoid_gate_mul"])?;
        same_dims(cx, 0)?;
        let a = attn_facts(cx, 0)?;
        let x = cx.ptr(cx.g.input(0, 0)?)?;
        let (gate, gate_stride) = cx.strided(cx.g.input(0, 1)?)?;
        let y = cx.ptr(cx.g.output(0, 0)?)?;
        let (k, n, dim_q) = (cx.handle(0)?, rows(cx)?, a.num_q_heads * a.head_dim);
        if cx.g.group.kernels[0].func == "sigmoid_gate_mul_batched" {
            return cx.push(
                0,
                Box::new(move |e| {
                    ops::sigmoid_gate_mul_batched(
                        e.gpu,
                        k,
                        x,
                        gate,
                        y,
                        dim_q,
                        gate_stride,
                        n,
                        e.stream,
                    )
                }),
            );
        }
        expect_kernel(cx, 0, "sigmoid_gate_mul")?;
        ensure!(
            n == 1 || gate_stride == dim_q,
            "the gate's rows are not contiguous"
        );
        let count = n * dim_q;
        cx.push(
            0,
            Box::new(move |e| ops::sigmoid_gate_mul(e.gpu, k, x, gate, y, count, e.stream)),
        )
    }
}

/// 2026-09-28: `deinterleave_qg`: the gated Q projection's per-head `[q_h | g_h]` rows split in
/// place into `[Q_i | Gate_i]`, one launch for all rows (`multi_seq/qkv/batch.rs`). Q and the
/// gate are a row pack laid over the projection's output.
pub(crate) struct DeinterleaveQg;

impl OpEmitter for DeinterleaveQg {
    fn id(&self) -> &'static str {
        "deinterleave_qg"
    }

    fn constrain(&self, g: &GroupRef<'_>, layout: &mut Layout) -> Result<()> {
        layout.row_packs.push(RowPack {
            members: vec![g.output(0, 0)?, g.output(0, 1)?],
            over: Some(g.input(0, 0)?),
        });
        Ok(())
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["split"])?;
        expect_kernel(cx, 0, "deinterleave_qg")?;
        same_dims(cx, 0)?;
        let a = attn_facts(cx, 0)?;
        ensure!(
            a.gated,
            "`deinterleave_qg` splits a gated Q; this layer is ungated"
        );
        let (qg, stride) = cx.strided(cx.g.input(0, 0)?)?;
        let (q, _) = cx.strided(cx.g.output(0, 0)?)?;
        let (gate, _) = cx.strided(cx.g.output(0, 1)?)?;
        let (nq, hd) = (a.num_q_heads, a.head_dim);
        ensure!(
            q == qg && gate == qg.offset((nq * hd) as usize * 2) && stride == nq * hd * 2,
            "Q and its gate are not laid over the projection's rows"
        );
        let (k, n) = (cx.handle(0)?, rows(cx)?);
        cx.push(
            0,
            Box::new(move |e| ops::deinterleave_qg(e.gpu, k, qg, n, nq, hd, stride, e.stream)),
        )
    }
}

/// 2026-09-28: `rms_norm_strided`: a per-head Q or K norm over every row in one launch, in
/// place (`multi_seq/qkv.rs` `ms_qkv_norms`); the kernel is bit-identical to `rms_norm` with a
/// row per head.
pub(crate) struct RmsNormStrided;

impl OpEmitter for RmsNormStrided {
    fn id(&self) -> &'static str {
        "rms_norm_strided"
    }

    fn constrain(&self, g: &GroupRef<'_>, layout: &mut Layout) -> Result<()> {
        layout.aliases.push((g.output(0, 0)?, g.input(0, 0)?));
        Ok(())
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["qk_norm"])?;
        expect_kernel(cx, 0, "rms_norm_strided")?;
        same_dims(cx, 0)?;
        let a = attn_facts(cx, 0)?;
        let heads = if cx.g.node(0).local == "q_norm" {
            a.num_q_heads
        } else {
            a.num_kv_heads
        };
        let (x, stride) = cx.strided(cx.g.input(0, 0)?)?;
        let (y, _) = cx.strided(cx.g.output(0, 0)?)?;
        ensure!(x == y, "the per-head norm runs in place");
        let w = dense(cx.weight(0, norm_slot(cx, 0)?)?, "q/k norm")?;
        let (k, n, hd, eps) = (
            cx.handle(0)?,
            rows(cx)?,
            a.head_dim,
            cx.config.rms_norm_eps as f32,
        );
        cx.push(
            0,
            Box::new(move |e| {
                ops::rms_norm_strided(e.gpu, k, x, &w, y, heads, n, hd, eps, stride, e.stream)
            }),
        )
    }
}

/// 2026-09-28: `rope_strided`: plain RoPE in place on every row's Q and K at the row's position
/// (`multi_seq/attn.rs` `ms_phase_rope`).
pub(crate) struct RopeStrided;

impl OpEmitter for RopeStrided {
    fn id(&self) -> &'static str {
        "rope_strided"
    }

    fn constrain(&self, g: &GroupRef<'_>, layout: &mut Layout) -> Result<()> {
        layout.aliases.push((g.output(0, 0)?, g.input(0, 0)?));
        layout.aliases.push((g.output(0, 1)?, g.input(0, 1)?));
        Ok(())
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &["rope"])?;
        expect_kernel(cx, 0, "rope_forward_strided")?;
        same_dims(cx, 0)?;
        let a = attn_facts(cx, 0)?;
        let (q, q_stride) = cx.strided(cx.g.input(0, 0)?)?;
        let (kk, k_stride) = cx.strided(cx.g.input(0, 1)?)?;
        ensure!(a.rope.rotary_dim > 0, "a zero rotary dim");
        let positions = cx.meta().positions;
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
                ops::rope_strided(
                    e.gpu, k, q, kk, positions, n, nq, nkv, hd, rd, theta, q_stride, k_stride,
                    e.stream,
                )
            }),
        )
    }
}
