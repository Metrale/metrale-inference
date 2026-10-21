// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-21: The GLM-5.3 decode step's chain of kernels for one KDA layer and one MoE layer at
//! M = 2 rows (a speculative verify of one sequence) on one rank of the TP=3 layout, launched the
//! way the serve launches them (grid, block, shared memory and argument order of the model-arch
//! launchers; `ops::w4a16_gemv_pipe` for the W4A16 projections), for `pdl_chain_gate` and
//! `pdl_chain_bench`.
//!
//! KDA layer: hc_pre_norm_c1 -> w4a16_gemv_pipe_m2 (q|k|v|f_a|b|g_a) -> kda_pack_qkv_bf16 ->
//! kda_gates_w8a8_k128_rows -> [conv snapshot copy] -> causal_conv1d_window_advance ->
//! causal_conv1d_update_l2norm_tokens -> causal_conv1d_window_advance ->
//! kda_recurrent_decode_bf16_seq_reg_snap -> kda_o_norm_gated_bf16 -> w4a16_gemv_pipe_m2 (o) ->
//! [all-reduce] -> bf16_add_rank_sum -> glm5next_hc_post.
//! MoE layer: hc_pre_norm_c1 -> dense_gemv_bf16_batchm_few_fp32out (router) ->
//! glm5next_router_topk_rank -> [zero expert_out] -> w4a4_quant_rows_static ->
//! glm5next_moe_row_union -> w4a4_gemv_mx8_moe_union_sweep (gate|up) -> glm5next_swiglu_clamp ->
//! w4a4_quant_rows_static_k64 -> w4a4_gemv_mx8_moe_union_sweep_k64 (down) -> w4a16_gemv_pipe_m2
//! (shared gate|up) -> glm5next_swiglu_clamp -> w4a16_gemv_pipe_m2 (shared down) ->
//! glm5next_moe_combine -> [all-reduce] -> bf16_add_rank_sum -> glm5next_hc_post.
//!
//! The all-reduce is not run (one GPU): a 16-byte memset node stands in for the NCCL kernel. Like
//! it, it is not a programmatic predecessor, so the chain breaks there as the serve's does.
//!
//! Owner: gpu-examples.
//! Invariants: launches only; every buffer is `Bufs`'s.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops::{W4a16Seg, w4a16_gemv_pipe};
use metrale_model_layers::weight_map::QuantizedWeight;

use super::bufs::*;

#[path = "pdl_chain_kern.rs"]
mod kern;
pub use kern::{Kern, Variant};

fn pre_norm(g: &dyn GpuBackend, k: &Kern, b: &Bufs, site: &Site, s: u64) -> Result<()> {
    KernelLaunch::new(g, k.pre)
        .grid([T, MIX, 1])
        .block([256, 1, 1])
        .arg_ptr(b.s.streams)
        .arg_ptr(site.hc_fn)
        .arg_ptr(site.scale)
        .arg_ptr(site.base)
        .arg_ptr(site.mix)
        .arg_ptr(site.done)
        .arg_ptr(site.norm_w)
        .arg_ptr(b.s.y)
        .arg_ptr(site.post)
        .arg_ptr(site.comb)
        .arg_ptr(b.s.normed)
        .arg_u32(H)
        .arg_u32(HC)
        .arg_u32(20)
        .arg_f32(1e-5)
        .arg_f32(1e-6)
        .arg_f32(1e-5)
        .launch(s)
}

/// 2026-10-21: The all-reduce stand-in, the rank sum and the mHC post-mix of `block_out`.
fn reduce_post(
    g: &dyn GpuBackend,
    k: &Kern,
    b: &Bufs,
    site: &Site,
    block_out: DevicePtr,
    s: u64,
) -> Result<()> {
    g.memset_async(b.s.ar_mark, 0, 16, s)?;
    let n = T * H;
    KernelLaunch::new(g, k.rank_sum)
        .grid([div_ceil(n, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(b.s.peers)
        .arg_u64(n as u64)
        .arg_i32(3)
        .arg_i32(0)
        .arg_i32(n as i32)
        .launch(s)?;
    KernelLaunch::new(g, k.post)
        .grid([T, H.div_ceil(256), 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(b.s.streams)
        .arg_ptr(site.post)
        .arg_ptr(site.comb)
        .arg_ptr(b.s.streams)
        .arg_u32(H)
        .arg_u32(HC)
        .launch(s)
}

fn pipe(
    g: &dyn GpuBackend,
    k: &Kern,
    input: DevicePtr,
    ws: &[QuantizedWeight],
    outs: &[DevicePtr],
    ns: &[u32],
    kk: u32,
    s: u64,
) -> Result<()> {
    let segs: Vec<W4a16Seg> = ws
        .iter()
        .zip(outs)
        .zip(ns)
        .map(|((&weight, &output), &n)| W4a16Seg { weight, output, n })
        .collect();
    w4a16_gemv_pipe(g, input, &segs, T, kk, k.sms, s)
}

/// 2026-10-21: One KDA layer (attention site, mixer, reduce, post-mix).
pub fn kda_layer(g: &dyn GpuBackend, k: &Kern, b: &Bufs, l: &Layer, s: u64) -> Result<()> {
    let x = &b.s;
    pre_norm(g, k, b, &l.attn, s)?;
    let part = (T * QKV * 2) as usize;
    let outs = [
        x.qkv_parts,
        x.qkv_parts.offset(part),
        x.qkv_parts.offset(2 * part),
        x.lowrank,
        x.beta_bf16,
        x.lowrank_g,
    ];
    pipe(g, k, x.normed, &l.six, &outs, &SIX, H, s)?;
    KernelLaunch::new(g, k.pack)
        .grid([div_ceil(QKV, 256), T, 1])
        .block([256, 1, 1])
        .arg_ptr(outs[0])
        .arg_ptr(outs[1])
        .arg_ptr(outs[2])
        .arg_ptr(x.qkv_proj)
        .arg_u32(T)
        .arg_u32(QKV)
        .launch(s)?;
    KernelLaunch::new(g, k.gates)
        .grid([(2 * QKV / 16).div_ceil(8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x.lowrank)
        .arg_ptr(x.lowrank_g)
        .arg_ptr(l.fb)
        .arg_ptr(l.fb_s)
        .arg_ptr(l.gb)
        .arg_ptr(l.gb_s)
        .arg_ptr(x.g_raw)
        .arg_ptr(x.out_gate)
        .arg_ptr(l.dt_bias)
        .arg_ptr(l.a_log)
        .arg_ptr(x.gate)
        .arg_ptr(x.beta_bf16)
        .arg_ptr(x.beta)
        .arg_u32(T)
        .arg_u32(QKV)
        .arg_u32(D)
        .arg_u32(HEADS)
        .arg_f32(-5.0)
        .launch(s)?;
    g.copy_d2d_async(l.conv, l.conv_int, CONV_BYTES, s)?;
    let adv = |state: DevicePtr, t: u32| {
        KernelLaunch::new(g, k.adv)
            .grid([div_ceil(CD, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(state)
            .arg_ptr(x.qkv_proj)
            .arg_u32(CD)
            .arg_u32(4)
            .arg_u32(t)
            .launch(s)
    };
    adv(l.conv_int, 1)?;
    KernelLaunch::new(g, k.upd)
        .grid([div_ceil(CD, 256), T, 1])
        .block([256, 1, 1])
        .arg_ptr(l.conv)
        .arg_ptr(x.qkv_proj)
        .arg_ptr(l.conv_w)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(x.conv_out)
        .arg_u32(CD)
        .arg_u32(4)
        .arg_u32(QK_CH)
        .arg_u32(D)
        .arg_f32(1e-6)
        .launch(s)?;
    adv(l.conv, T)?;
    KernelLaunch::new(g, k.rec)
        .grid([HEADS, D / 32, 1])
        .block([32, 1, 1])
        .shared_mem(3 * D * 4)
        .arg_ptr(x.conv_out)
        .arg_ptr(x.conv_out.offset((QKV * 2) as usize))
        .arg_ptr(x.conv_out.offset((QKV * 4) as usize))
        .arg_ptr(x.gate)
        .arg_ptr(x.beta)
        .arg_ptr(l.rec)
        .arg_ptr(x.core)
        .arg_u32(HEADS)
        .arg_f32(1.0 / (D as f32).sqrt())
        .arg_u32(T)
        .arg_u32(CD)
        .arg_u32(QKV)
        .arg_u32(HEADS)
        .arg_u32(QKV)
        .arg_ptr(l.h_int)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(1)
        .launch(s)?;
    KernelLaunch::new(g, k.onorm)
        .grid([T * HEADS, 1, 1])
        .block([D, 1, 1])
        .arg_ptr(x.core)
        .arg_ptr(x.out_gate)
        .arg_ptr(l.o_norm_w)
        .arg_ptr(x.o_norm_out)
        .arg_u32(D)
        .arg_f32(1e-6)
        .launch(s)?;
    pipe(g, k, x.o_norm_out, &[l.o], &[x.final_out], &[H], QKV, s)?;
    reduce_post(g, k, b, &l.attn, x.final_out, s)
}

/// 2026-10-21: A union sweep: activations `a` (Aq, As, Ag), the union tables, then each
/// projection's (table, output); a single projection passes its pair twice.
#[allow(clippy::too_many_arguments)]
fn sweep(
    g: &dyn GpuBackend,
    k: &Kern,
    h: KernelHandle,
    b: &Bufs,
    a: [DevicePtr; 3],
    p: [(&Table, DevicePtr); 2],
    nproj: u32,
    n: u32,
    kk: u32,
    act_div: u32,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, h)
        .grid([k.sms, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a[0])
        .arg_ptr(a[1])
        .arg_ptr(a[2])
        .arg_ptr(b.s.u_eid)
        .arg_ptr(b.s.u_slot)
        .arg_ptr(p[0].0.packed)
        .arg_ptr(p[0].0.scale)
        .arg_ptr(p[0].0.scale2)
        .arg_ptr(p[0].1)
        .arg_ptr(p[1].0.packed)
        .arg_ptr(p[1].0.scale)
        .arg_ptr(p[1].0.scale2)
        .arg_ptr(p[1].1)
        .arg_u32(nproj)
        .arg_u32(n)
        .arg_u32(kk)
        .arg_u32(T)
        .arg_u32(TOP_K)
        .arg_u32(act_div)
        .arg_u32(EXPERTS)
        .launch(s)
}

fn swiglu(
    g: &dyn GpuBackend,
    k: &Kern,
    gate: DevicePtr,
    up: DevicePtr,
    out: DevicePtr,
    n: u32,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, k.swiglu)
        .grid([div_ceil(n, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out)
        .arg_u32(n)
        .arg_f32(10.0)
        .launch(s)
}

fn quant(
    g: &dyn GpuBackend,
    h: KernelHandle,
    rows: u32,
    x: DevicePtr,
    a: [DevicePtr; 3],
    kk: u32,
    gs: f32,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, h)
        .grid([rows, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(a[0])
        .arg_ptr(a[1])
        .arg_ptr(a[2])
        .arg_u32(kk)
        .arg_f32(gs)
        .launch(s)
}

/// 2026-10-21: One MoE layer (FFN site, routed and shared experts, reduce, post-mix).
pub fn moe_layer(g: &dyn GpuBackend, k: &Kern, b: &Bufs, l: &Layer, s: u64) -> Result<()> {
    let x = &b.s;
    pre_norm(g, k, b, &l.ffn, s)?;
    KernelLaunch::new(g, k.few32)
        .grid([div_ceil(EXPERTS, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x.normed)
        .arg_ptr(l.router)
        .arg_ptr(x.logits)
        .arg_u32(T)
        .arg_u32(EXPERTS)
        .arg_u32(H)
        .arg_u32(EXPERTS)
        .launch(s)?;
    KernelLaunch::new(g, k.topk)
        .grid([T, 1, 1])
        .block([EXPERTS.div_ceil(32) * 32, 1, 1])
        .arg_ptr(x.logits)
        .arg_ptr(l.router_bias)
        .arg_ptr(x.ids)
        .arg_ptr(x.wts)
        .arg_u32(EXPERTS)
        .arg_u32(TOP_K)
        .arg_u32(1)
        .arg_f32(2.5)
        .arg_u32(1)
        .arg_u32(0)
        .launch(s)?;
    g.memset_async(x.expert_out, 0, (SLOTS * H * 2) as usize, s)?;
    let a1 = [x.aq, x.as_, x.ag];
    quant(g, k.quant, T, x.normed, a1, H, 3.0 / 2688.0, s)?;
    KernelLaunch::new(g, k.union)
        .grid([1, 1, 1])
        .block([SLOTS, 1, 1])
        .arg_ptr(x.ids)
        .arg_ptr(x.u_eid)
        .arg_ptr(x.u_slot)
        .arg_u32(T)
        .arg_u32(TOP_K)
        .launch(s)?;
    sweep(
        g,
        k,
        k.sweep,
        b,
        a1,
        [(&b.gate_t, x.a_gate), (&b.up_t, x.a_up)],
        2,
        MI,
        H,
        TOP_K,
        s,
    )?;
    swiglu(g, k, x.a_gate, x.a_up, x.a_act, SLOTS * MI, s)?;
    let a2 = [x.aq2, x.as2, x.ag2];
    quant(g, k.quant64, SLOTS, x.a_act, a2, MI, 100.0 / 2688.0, s)?;
    sweep(
        g,
        k,
        k.sweep64,
        b,
        a2,
        [(&b.down_t, x.expert_out), (&b.down_t, x.expert_out)],
        1,
        H,
        MI,
        1,
        s,
    )?;
    pipe(
        g,
        k,
        x.normed,
        &l.sh_gu,
        &[x.sg, x.su],
        &[SHARED, SHARED],
        H,
        s,
    )?;
    swiglu(g, k, x.sg, x.su, x.sact, T * SHARED, s)?;
    pipe(g, k, x.sact, &[l.sh_down], &[x.shared_out], &[H], SHARED, s)?;
    KernelLaunch::new(g, k.combine)
        .grid([T, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x.expert_out)
        .arg_ptr(x.wts)
        .arg_ptr(x.shared_out)
        .arg_ptr(x.moe_out)
        .arg_u32(H)
        .arg_u32(TOP_K)
        .launch(s)?;
    reduce_post(g, k, b, &l.ffn, x.moe_out, s)
}
