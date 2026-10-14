// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the Qwen3.8-Flash-Next QSA indexer kernels
//! (`kernels/gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu`): block-key pooling,
//! query prep, block scoring, the selected-token K/V gather, and the per-row
//! prefill selection passes. The gather writes a contiguous scratch that the
//! paged decode attention reads through an identity block table.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-25: Pool `n_new` newly complete blocks starting at `first_block`: the
/// mean of `ratio` raw keys, then RMSNorm scaled by `1 + w`, then rope at the
/// block's first token position. Returns without launching when `n_new == 0`.
///
/// 2026-10-13: `raw_keys` holds exactly those blocks' keys from its row 0 (row
/// `i * ratio + r` is position `(first_block + i) * ratio + r`), so a caller pools out
/// of a slab-sized scratch rather than a sequence-long buffer (L-K9a).
#[allow(clippy::too_many_arguments)]
pub fn qsa_block_pool(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    raw_keys: DevicePtr,
    k_norm_w: DevicePtr,
    block_keys: DevicePtr,
    first_block: u32,
    n_new: u32,
    ratio: u32,
    hd: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    if n_new == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([n_new, 1, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(raw_keys)
        .arg_ptr(k_norm_w)
        .arg_ptr(block_keys)
        .arg_u32(first_block)
        .arg_u32(ratio)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_f32(theta)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-10-13: One decode token's indexer ingest at the position `pos_dev[0]` (u32 on
/// the device): `raw_tail[pos % ratio] = key` (`raw_tail` is a `[ratio, hd]` ring), and
/// the block it completes, if any, pooled from the ring as [`qsa_block_pool`] pools it. The position is read on the device, so a captured graph
/// replays the step at every later position.
#[allow(clippy::too_many_arguments)]
pub fn qsa_ingest_step(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    key: DevicePtr,
    raw_tail: DevicePtr,
    k_norm_w: DevicePtr,
    block_keys: DevicePtr,
    pos_dev: DevicePtr,
    ratio: u32,
    hd: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(key)
        .arg_ptr(raw_tail)
        .arg_ptr(k_norm_w)
        .arg_ptr(block_keys)
        .arg_ptr(pos_dev)
        .arg_u32(ratio)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_f32(theta)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: One decode query: per head, RMSNorm scaled by `1 + w`, then
/// partial rope at `pos`; FP32 output.
#[allow(clippy::too_many_arguments)]
pub fn qsa_qprep(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_in: DevicePtr,
    q_norm_w: DevicePtr,
    q_out: DevicePtr,
    n_heads: u32,
    hd: u32,
    rot: u32,
    pos: u32,
    theta: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_heads, 1, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(q_in)
        .arg_ptr(q_norm_w)
        .arg_ptr(q_out)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_u32(pos)
        .arg_f32(theta)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: `scores[b] = sum_h relu(q_h . k_b) / sqrt(hd)` for `n_blocks`
/// blocks.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    block_keys: DevicePtr,
    scores: DevicePtr,
    n_blocks: u32,
    n_heads: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_blocks, 1, 1])
        .block([hd, 1, 1])
        .shared_mem(32 * 4)
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .launch(stream)
}

/// 2026-09-25: Copy the selected tokens' K/V rows into contiguous NHD scratch:
/// slot `i` holds position `sel[i]`.
#[allow(clippy::too_many_arguments)]
pub fn qsa_gather(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    block_table: DevicePtr,
    sel: DevicePtr,
    k_out: DevicePtr,
    v_out: DevicePtr,
    n_sel: u32,
    block_size: u32,
    nkv: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_sel, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(block_table)
        .arg_ptr(sel)
        .arg_ptr(k_out)
        .arg_ptr(v_out)
        .arg_u32(block_size)
        .arg_u32(nkv)
        .arg_u32(hd)
        .launch(stream)
}

/// 2026-09-25: [`qsa_qprep`] for `rows` prefill rows at positions
/// `first_pos + row`, reading q from rows of the indexer projection `qk`.
#[allow(clippy::too_many_arguments)]
pub fn qsa_qprep_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qk: DevicePtr,
    q_norm_w: DevicePtr,
    q_out: DevicePtr,
    rows: u32,
    first_pos: u32,
    qkw: u32,
    n_heads: u32,
    hd: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, n_heads, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(qk)
        .arg_ptr(q_norm_w)
        .arg_ptr(q_out)
        .arg_u32(first_pos)
        .arg_u32(qkw)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_f32(theta)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: Tensor-core [`qsa_score_rows`]: one block of 8 warps covers 16
/// rows by 64 blocks instead of one block per (row, block). q is split into
/// BF16 high and low halves for the MMA, so the scores are not bit-identical to
/// [`qsa_score_rows`]; they feed a top-k.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_rows_tc(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    block_keys: DevicePtr,
    scores: DevicePtr,
    rows: u32,
    n_blocks_max: u32,
    first_pos: u32,
    score_stride: u32,
    ratio: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows.div_ceil(16), n_blocks_max.div_ceil(64), 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(first_pos)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(n_blocks_max)
        .launch(stream)
}

/// 2026-09-25: Per-row block scores for prefill rows; blocks at or past a row's
/// complete-block count score `-inf`.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    block_keys: DevicePtr,
    scores: DevicePtr,
    rows: u32,
    n_blocks_max: u32,
    first_pos: u32,
    score_stride: u32,
    ratio: u32,
    n_heads: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, n_blocks_max, 1])
        .block([hd, 1, 1])
        .shared_mem(32 * 4)
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(first_pos)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .launch(stream)
}

/// 2026-10-14: Shared memory of `qsa_prefill_attn_tc` (`QPA_SMEM_BYTES` in
/// kernels/hopper/common/qsa_prefill_attn_tc_hopper.cu): Q tile, two K/V stages of 64 keys,
/// the S and P tiles and two 16-float row vectors.
const QSA_ATTN_TC_SMEM: u32 = 16 * 264 * 2 + 2 * 2 * 64 * 264 * 2 + 16 * 68 * 4 + 16 * 72 * 2 + 128;
/// 2026-10-14: The head dimension `qsa_prefill_attn_tc` is built for (`QPA_HD`).
const QSA_ATTN_TC_HD: u32 = 256;

/// 2026-10-14: Which selective prefill attention kernel runs.
#[derive(Clone, Copy, Debug)]
pub enum QsaPrefillAttn {
    /// 2026-10-14: `qsa_prefill_attn`: one CTA per (row, query head), CUDA cores.
    Base(KernelHandle),
    /// 2026-10-14: `qsa_prefill_attn_tc` (Hopper): one CTA per (row, KV head), the KV head's
    /// query heads stacked into one m16n8k16 tile, K/V gathered once per KV head.
    Tc(KernelHandle),
}

impl QsaPrefillAttn {
    /// 2026-10-14: The tensor-core kernel's handle: zero where the target does not compile
    /// kernels/hopper/common/qsa_prefill_attn_tc_hopper.cu, or when `METRALE_QSA_ATTN_TC` is
    /// `"0"` (read once per process), so [`QsaPrefillAttn::select`] keeps the base kernel.
    pub fn resolve_tc(gpu: &dyn GpuBackend) -> KernelHandle {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("METRALE_QSA_ATTN_TC").as_deref() == Ok("0")) {
            return KernelHandle(0);
        }
        crate::layers::try_target_kernel(gpu, "qsa_prefill_attn_tc_hopper", "qsa_prefill_attn_tc")
    }

    /// 2026-10-14: The tensor-core kernel when `tc` is resolved (non-zero) and admits the shape
    /// (`hd == 256`, `1 <= nq / nkv <= 16`, `nq % nkv == 0`), else the base kernel.
    pub fn select(base: KernelHandle, tc: KernelHandle, nq: u32, nkv: u32, hd: u32) -> Self {
        let admitted = tc.0 != 0
            && hd == QSA_ATTN_TC_HD
            && nkv > 0
            && nq.is_multiple_of(nkv)
            && (1..=16).contains(&(nq / nkv));
        if admitted {
            Self::Tc(tc)
        } else {
            Self::Base(base)
        }
    }
}

/// 2026-09-25: Per (row, q head), attention over exactly the selected blocks in
/// `lists` plus the incomplete tail, read from the paged cache; the result
/// overwrites that row's context in `attn_out`. 2026-10-14: [`QsaPrefillAttn::Tc`] computes the
/// same attention per (row, KV head) on tensor cores.
#[allow(clippy::too_many_arguments)]
pub fn qsa_prefill_attn(
    gpu: &dyn GpuBackend,
    kernel: QsaPrefillAttn,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    block_table: DevicePtr,
    lists: DevicePtr,
    attn_out: DevicePtr,
    rows: u32,
    first_pos: u32,
    topk: u32,
    ratio: u32,
    block_size: u32,
    nq: u32,
    nkv: u32,
    hd: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: Base: per-warp `[hd]` partials for 8 warps, then 8 `m` and 8 `l`.
    let (handle, grid_y, smem) = match kernel {
        QsaPrefillAttn::Base(h) => (h, nq, (8 * hd + 16) * 4),
        QsaPrefillAttn::Tc(h) => (h, nkv, QSA_ATTN_TC_SMEM),
    };
    KernelLaunch::new(gpu, handle)
        .grid([rows, grid_y, 1])
        .block([256, 1, 1])
        .shared_mem(smem)
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(block_table)
        .arg_ptr(lists)
        .arg_ptr(attn_out)
        .arg_u32(first_pos)
        .arg_u32(topk)
        .arg_u32(ratio)
        .arg_u32(block_size)
        .arg_u32(nq)
        .arg_u32(nkv)
        .arg_u32(hd)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}

#[cfg(test)]
#[path = "qsa_ingest_tests.rs"]
mod qsa_ingest_tests;
