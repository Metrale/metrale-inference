// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `prefill_attention`: the gated full-attention sublayer of a prefill pass, from
//! the Q/K/V projections to the O projection, as one group, mirroring the legacy routes on a
//! BF16 KV cache:
//!
//! - `prefill` (offset 0), the contiguous route `prefill_attention_with_cache_skip`
//!   (`qwen3_attention/prefill/cache_skip.rs:171-410`): Q, K, V; `deinterleave_qg_split_qnorm`;
//!   the K norm; plain RoPE (`cache_skip_norm_rope.rs:308`); the K/V write from the KV floor
//!   (`cache_skip.rs:263-285`); `attn_prefill_fa128` (`cache_skip_attn_gates.rs:116`); the
//!   sigmoid gate; O.
//! - `prefill_chunk` (offset above 0), the paged route `prefill_attention_paged`
//!   (`paged.rs:108-476`): the V region zeroed; Q, K, V; the split and norms (`paged/norms.rs`);
//!   interleaved MRoPE (`paged/rope_cache.rs:121`); the K/V write from `min(floor, T)`
//!   (`rope_cache.rs:162-179`); `attn_prefill_paged` below 256 rows, `attn_prefill_fa128_paged`
//!   from 256 (`paged_attn.rs:143, 379, 419`); the sigmoid gate; O.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Q goes to `qkv_output`, K to `ssm_qkvz`, V right after K's T rows, the split Q to
//!   `ssm_deinterleaved`, the attention output to `attn_output`, O to the output edge
//!   (`norm_output`): the legacy route's buffers. V's offset is the pass's row count, so it is
//!   computed at run time.
//! - The group's kernels are exactly the route's launches for its arm, in order; any other list
//!   is refused at build. A pass whose rows another arm serves is refused at run time.

use anyhow::{Context, Result, bail, ensure};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_circuit::{LinearRole, Mode};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::super::bindings::WeightSlot;
use super::super::super::compile::{Cx, OpEmitter};
use super::super::{attn_facts, dense, dim, nvfp4};
use super::attn_route::{self, Attn, Proj, expected};
use crate::layers::ops;

/// 2026-10-03: The member index of each node of the group's pattern.
const Q: usize = 0;
const K: usize = 2;
const V: usize = 3;
const Q_NORM: usize = 4;
const K_NORM: usize = 5;
const O: usize = 10;

/// 2026-10-03: The ops of the group, in pattern order.
const OPS: [&str; 11] = [
    "linear:q",
    "split",
    "linear:k",
    "linear:v",
    "qk_norm",
    "qk_norm",
    "rope",
    "kv_write",
    "paged_attention",
    "sigmoid_gate_mul",
    "linear:o",
];

/// 2026-10-03: See the module header.
pub(crate) struct PrefillAttention;

impl OpEmitter for PrefillAttention {
    fn id(&self) -> &'static str {
        "prefill_attention"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        cx.g.expect_ops(self.id(), &OPS)?;
        let paged = match cx.mode {
            Mode::Prefill => false,
            Mode::PrefillChunk => true,
            other => bail!(
                "`prefill_attention` serves the prefill modes, not {}",
                other.name()
            ),
        };
        let a = attn_facts(cx, Q)?;
        ensure!(
            (a.num_q_heads, a.num_kv_heads, a.head_dim)
                == (
                    dim(cx, "q_heads")?,
                    dim(cx, "kv_heads")?,
                    dim(cx, "head_dim")?
                ),
            "the layer's heads differ from the circuit's dims"
        );
        ensure!(
            a.gated,
            "the prefill attention rules model a gated Q; this layer is ungated"
        );
        ensure!(
            a.kv_dtype == KvCacheDtype::Bf16,
            "the prefill attention rules write a BF16 cache; this layer's is {:?}",
            a.kv_dtype
        );
        let kernels = &cx.g.group.kernels;
        let proj = Proj::of(kernels.first().context("no kernels")?)?;
        let per = proj.kernels().len();
        let attn = match kernels
            .get(3 * per + 4)
            .map(|k| (k.module.as_str(), k.func.as_str()))
        {
            Some(("attn_prefill_fa128", "attn_prefill_fa128")) if !paged => Attn::Contiguous,
            Some(("prefill_paged", "attn_prefill_paged")) if paged => Attn::PagedSmall,
            Some(("attn_prefill_fa128", "attn_prefill_fa128_paged")) if paged => Attn::PagedFa128,
            other => bail!(
                "no {} attention kernel at the route's position: {other:?}",
                cx.mode.name()
            ),
        };
        let mrope = paged && a.rope.mrope_interleaved;
        let rope = if mrope {
            ("rope_mrope_interleaved", "rope_forward_mrope_interleaved")
        } else {
            ("rope", "rope_forward")
        };
        let got: Vec<(String, String)> = kernels
            .iter()
            .map(|k| (k.module.clone(), k.func.clone()))
            .collect();
        ensure!(
            got == expected(proj, rope, attn),
            "the group's kernels are not the {} route's for its arm: {got:?}",
            cx.mode.name()
        );
        attn_route::check_levers(proj)?;
        let fa = ops::AttnFa128Kernels::resolve(cx.gpu);
        ensure!(
            attn == Attn::PagedSmall
                || (a.head_dim == 256
                    && a.sliding_window == 0
                    && a.num_q_heads.is_multiple_of(a.num_kv_heads)),
            "the 128-row attention twin does not apply to this layer's heads or window"
        );

        // 2026-10-03: Geometry and weights.
        let (nq, nkv, hd) = (a.num_q_heads, a.num_kv_heads, a.head_dim);
        let h = dim(cx, "hidden")?;
        let (q_dim, kv_dim) = (nq * hd, nkv * hd);
        let q_proj_dim = q_dim * 2;
        ensure!(
            super::super::batched::width(cx, cx.g.output(Q, 0)?)? == q_proj_dim,
            "the Q projection's width is not the gated [Q | gate]"
        );
        ensure!(
            q_dim % 32 == 0 && h % 32 == 0,
            "the FP8 projection arm needs K % 32 == 0"
        );
        let twin = |cx: &Cx<'_>, i: usize, r: LinearRole| {
            nvfp4(cx.weight(i, WeightSlot::Transposed(r))?, r.name())
        };
        let (wq, wk, wv, wo) = (
            twin(cx, Q, LinearRole::Q)?,
            twin(cx, K, LinearRole::K)?,
            twin(cx, V, LinearRole::V)?,
            twin(cx, O, LinearRole::O)?,
        );
        let q_norm = dense(cx.weight(Q_NORM, WeightSlot::QNorm)?, "q norm")?;
        let k_norm = dense(cx.weight(K_NORM, WeightSlot::KNorm)?, "k norm")?;
        ensure!(!q_norm.weight.is_null(), "the gated Q has no per-head norm");
        ensure!(!k_norm.weight.is_null(), "K has no per-head norm");
        let eps = cx.config.rms_norm_eps as f32;
        let (rd, theta, scale, window) = (
            a.rope.rotary_dim,
            a.rope.theta,
            a.softmax_scale,
            a.sliding_window,
        );

        // 2026-10-03: Buffers.
        let arena = cx.arena()?;
        let normed = cx.ptr(cx.g.input(Q, 0)?)?;
        let o_out = cx.ptr(cx.g.output(O, 0)?)?;
        ensure!(
            normed == arena.norm_output() && o_out == arena.norm_output(),
            "the attention reads its input from and writes O to `norm_output`, as the legacy \
             route does"
        );
        let (qg, k_buf, q_buf, attn_out) = (
            arena.qkv_output(),
            arena.ssm_qkvz(),
            arena.ssm_deinterleaved(),
            arena.attn_output(),
        );
        let kp = *cx
            .fixed
            .k_pools
            .get(a.attn_layer_idx)
            .with_context(|| format!("no K pool for attention layer {}", a.attn_layer_idx))?;
        let vp = *cx
            .fixed
            .v_pools
            .get(a.attn_layer_idx)
            .with_context(|| format!("no V pool for attention layer {}", a.attn_layer_idx))?;
        let (bs, cache_stride) = (cx.fixed.block_size, cx.fixed.cache_stride);
        let v_at = move |t: u32| k_buf.offset(t as usize * kv_dim as usize * 2);

        // 2026-10-03: The paged route zeroes V's region before the projections (`paged.rs:109`).
        if paged {
            cx.push_copy(Box::new(move |e| {
                let t = e.prefill()?.tokens;
                e.gpu
                    .memset_async(v_at(t), 0, t as usize * kv_dim as usize * 2, e.stream)
            }))?;
        }

        // 2026-10-03: Q, K, V (`cache_skip_qkv.rs:69-127`, `paged_qkv.rs:52-111`).
        let mut k = 0;
        for (w, out, n) in [
            (wq, None, q_proj_dim),
            (wk, Some(false), kv_dim),
            (wv, Some(true), kv_dim),
        ] {
            let dst = move |t: u32| match out {
                None => qg,
                Some(false) => k_buf,
                Some(true) => v_at(t),
            };
            push_proj(cx, &mut k, proj, w, normed, dst, [n, h])?;
        }

        // 2026-10-03: The Q split with its norm, then the K norm (`cache_skip_norm_rope.rs:67,
        // 178`, `paged/norms.rs:39, 104`).
        let kh = cx.handle(k)?;
        cx.push(
            k,
            Box::new(move |e| {
                let t = e.prefill()?.tokens;
                ops::deinterleave_qg_split_qnorm(
                    e.gpu,
                    kh,
                    qg,
                    q_buf,
                    q_norm.weight,
                    t,
                    nq,
                    hd,
                    q_proj_dim,
                    eps,
                    e.stream,
                )
            }),
        )?;
        k += 1;
        let kh = cx.handle(k)?;
        cx.push(
            k,
            Box::new(move |e| {
                let t = e.prefill()?.tokens;
                ops::rms_norm(e.gpu, kh, k_buf, &k_norm, k_buf, nkv * t, hd, eps, e.stream)
            }),
        )?;
        k += 1;

        // 2026-10-03: RoPE in place on Q and K.
        let kh = cx.handle(k)?;
        cx.push(
            k,
            Box::new(move |e| {
                let s = e.prefill()?;
                let m = s.meta()?;
                if mrope {
                    ops::rope_mrope_interleaved(
                        e.gpu,
                        kh,
                        q_buf,
                        k_buf,
                        m.positions,
                        m.positions_h,
                        m.positions_w,
                        s.tokens,
                        nq,
                        nkv,
                        hd,
                        rd,
                        theta,
                        e.stream,
                    )
                } else {
                    ops::rope(
                        e.gpu,
                        kh,
                        q_buf,
                        k_buf,
                        m.positions,
                        s.tokens,
                        nq,
                        nkv,
                        hd,
                        rd,
                        theta,
                        e.stream,
                    )
                }
            }),
        )?;
        k += 1;

        // 2026-10-03: K and V into the cache from the floor (the contiguous route writes from
        // `kv_write_start`, the paged one from `min(floor, T)`: the same rows).
        let kh = cx.handle(k)?;
        cx.push(
            k,
            Box::new(move |e| {
                let s = e.prefill()?;
                let from = s.kv_write_floor.min(s.tokens);
                if from >= s.tokens {
                    return Ok(());
                }
                let off = from as usize * kv_dim as usize * 2;
                ops::reshape_and_cache(
                    e.gpu,
                    kh,
                    k_buf.offset(off),
                    v_at(s.tokens).offset(off),
                    kp,
                    vp,
                    s.meta()?.slot.offset(from as usize * 8),
                    s.tokens - from,
                    nkv,
                    hd,
                    bs,
                    kv_dim,
                    kv_dim,
                    cache_stride,
                    e.stream,
                )
            }),
        )?;
        k += 1;

        // 2026-10-03: Attention.
        let kh = cx.handle(k)?;
        cx.push(
            k,
            Box::new(move |e| {
                let s = e.prefill()?;
                ensure!(
                    attn.serves(s.tokens),
                    "a {}-row pass does not take the {attn:?} attention this plan was built for",
                    s.tokens
                );
                let ran = match attn {
                    Attn::Contiguous => fa.contiguous(
                        e.gpu,
                        q_buf,
                        k_buf,
                        v_at(s.tokens),
                        attn_out,
                        s.tokens,
                        1,
                        nq,
                        nkv,
                        hd,
                        scale,
                        true,
                        window,
                        e.stream,
                    )?,
                    Attn::PagedSmall => {
                        ops::prefill_attention_paged(
                            e.gpu,
                            kh,
                            q_buf,
                            kp,
                            vp,
                            attn_out,
                            s.meta()?.block_table,
                            s.tokens,
                            s.start + s.tokens,
                            s.start,
                            nq,
                            nkv,
                            hd,
                            bs,
                            window,
                            scale,
                            e.stream,
                        )?;
                        true
                    }
                    Attn::PagedFa128 => fa.paged(
                        e.gpu,
                        q_buf,
                        kp,
                        vp,
                        attn_out,
                        s.meta()?.block_table,
                        s.tokens,
                        s.start + s.tokens,
                        s.start,
                        nq,
                        nkv,
                        hd,
                        bs,
                        window,
                        scale,
                        e.stream,
                    )?,
                };
                ensure!(ran, "the 128-row attention twin declined the pass");
                Ok(())
            }),
        )?;
        k += 1;

        // 2026-10-03: The output gate (`cache_skip_attn_gates.rs:242-252`, `paged.rs:429`).
        let kh = cx.handle(k)?;
        let gate = qg.offset(q_dim as usize * 2);
        cx.push(
            k,
            Box::new(move |e| {
                let t = e.prefill()?.tokens;
                ops::sigmoid_gate_mul_batched(
                    e.gpu, kh, attn_out, gate, attn_out, q_dim, q_proj_dim, t, e.stream,
                )
            }),
        )?;
        k += 1;

        // 2026-10-03: O (`paged_oproj.rs:199-223`).
        push_proj(cx, &mut k, proj, wo, attn_out, move |_| o_out, [h, q_dim])
    }
}

/// 2026-10-03: Queue one projection's launches from kernel `*k` on: the arm's GEMM (or the FP8
/// arm's three kernels, which `ops::w4a16_t_via_fp8_ldmab` issues together from the first
/// launch; the other two entries carry no work of their own).
fn push_proj(
    cx: &mut Cx<'_>,
    k: &mut usize,
    proj: Proj,
    w: crate::weight_map::QuantizedWeight,
    x: DevicePtr,
    dst: impl Fn(u32) -> DevicePtr + Send + Sync + 'static,
    [n, kdim]: [u32; 2],
) -> Result<()> {
    let handle: KernelHandle = cx.handle(*k)?;
    cx.push_bundle(
        *k,
        proj.kernels().len(),
        Box::new(move |e| {
            let t = e.prefill()?.tokens;
            attn_route::run(e.gpu, proj, handle, x, &w, dst(t), [t, n, kdim], e.stream)
        }),
    )?;
    *k += proj.kernels().len();
    Ok(())
}
