// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `w4a16_gemv_pipe_m<M>` (`kernels/gb10/common/w4a16_gemv_pipe.cu`):
//! `w4a16_gemv_seg_m<M>` (R = 1) with each warp's first weight loads issued before the input is
//! staged, on a grid capped to the resident blocks when the launch is a few waves.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`w4a16_gemv_pipe`] launches only what [`w4a16_gemv_seg_rows_ok`] accepts at a row count
//!   the module has an entry for ([`w4a16_gemv_pipe_entry`]); the shared-memory layout is
//!   `w4a16_gemv_seg`'s ([`w4a16_gemv_seg_smem`]).
//! - Every output byte equals `w4a16_gemv`'s for that input row alone, at every grid (the
//!   kernel's invariant; GPU proof: the `glm5next_w4a16_gemv_pipe_bench` example).

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{
    W4A16_GEMV_SEG_MAX, W4a16Seg, w4a16_gemv_seg_rows_ok, w4a16_gemv_seg_smem, w4a16_gemv_seg_tiles,
};

/// 2026-10-21: The kernel module.
pub const W4A16_GEMV_PIPE_MODULE: &str = "w4a16_gemv_pipe";

/// 2026-10-21: Warps per block (`PIPE_WARPS`), one output row at a time each.
pub const W4A16_GEMV_PIPE_WARPS: u32 = 8;

/// 2026-10-21: Resident blocks per SM the entries are compiled for (`__launch_bounds__(256, 4)`,
/// 64 registers).
pub const W4A16_GEMV_PIPE_BLOCKS_PER_SM: u32 = 4;

/// 2026-10-21: Shared memory per SM and the per-block reservation on GB10 (sm_121: 100 KB per
/// SM, 1 KB reserved per resident block): the blocks per SM a launch's dynamic shared memory
/// leaves room for.
pub const W4A16_GEMV_PIPE_SMEM_PER_SM: u32 = 100 * 1024;
pub const W4A16_GEMV_PIPE_SMEM_RESERVED: u32 = 1024;

/// 2026-10-21: A launch is capped to one resident wave when it needs at most this many waves;
/// wider launches stay one output per warp. GB10, M = 2: shared down and KDA o (2.67 waves)
/// 10.5 -> 9.9 us and 32.2 -> 31.1 us capped; the KDA six-group (5.7 waves) 84.8 uncapped vs
/// 86.0 capped (the block scheduler balances the last rows, a static stride does not).
pub const W4A16_GEMV_PIPE_CAP_WAVES: u32 = 3;

/// 2026-10-21: The entry of `m` input rows; `None` for a row count the module lacks.
pub fn w4a16_gemv_pipe_entry(m: u32) -> Option<&'static str> {
    match m {
        1 => Some("w4a16_gemv_pipe_m1"),
        2 => Some("w4a16_gemv_pipe_m2"),
        _ => None,
    }
}

/// 2026-10-21: The kernel's contract at `m` rows, without a GPU: an entry for `m` and
/// [`w4a16_gemv_seg_rows_ok`].
pub fn w4a16_gemv_pipe_ok(k: u32, ns: &[u32], m: u32) -> bool {
    w4a16_gemv_pipe_entry(m).is_some() && w4a16_gemv_seg_rows_ok(k, ns, m)
}

/// 2026-10-21: Blocks of a launch over `tiles` one-row outputs: one output per warp, capped at
/// `max_blocks` (the warps then stride over the rest), at least one.
pub fn w4a16_gemv_pipe_blocks(tiles: u64, max_blocks: u32) -> u32 {
    let need = tiles.div_ceil(u64::from(W4A16_GEMV_PIPE_WARPS));
    need.min(u64::from(max_blocks)).max(1) as u32
}

/// 2026-10-21: The block cap of a launch over segments `ns`, `k`, `m` rows on `sms` SMs: one
/// resident wave when [`W4A16_GEMV_PIPE_BLOCKS_PER_SM`] blocks fit an SM's shared memory and
/// the launch needs at most [`W4A16_GEMV_PIPE_CAP_WAVES`] waves of them, else uncapped.
pub fn w4a16_gemv_pipe_cap(ns: &[u32], k: u32, m: u32, sms: u32) -> u32 {
    let per_block = w4a16_gemv_seg_smem(k, m) + W4A16_GEMV_PIPE_SMEM_RESERVED;
    let fits = W4A16_GEMV_PIPE_BLOCKS_PER_SM * per_block <= W4A16_GEMV_PIPE_SMEM_PER_SM;
    let wave = sms.saturating_mul(W4A16_GEMV_PIPE_BLOCKS_PER_SM);
    let need = w4a16_gemv_pipe_blocks(w4a16_gemv_seg_tiles(ns, 1), u32::MAX);
    if fits && wave > 0 && need <= wave.saturating_mul(W4A16_GEMV_PIPE_CAP_WAVES) {
        wave
    } else {
        u32::MAX
    }
}

/// 2026-10-23: The input a `w4a16_gemv_pipe` instantiation stages: its leading kernel
/// parameters (`w4a16_gemv_pipe.cuh`: the stage functor's fields), pushed before the segments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PipeInput {
    /// 2026-10-23: `w4a16_gemv_pipe_m<M>`: `m` contiguous BF16 rows of K.
    Rows(DevicePtr),
    /// 2026-10-23: An instantiation whose stage computes the rows from two `m`-row BF16 inputs
    /// of K and an F32 (`glm5next_swiglu_down_pipe_m<M>`: gate, up, limit).
    Pair(DevicePtr, DevicePtr, f32),
}

/// 2026-10-21: One launch of `segs` (1..=6) over `m` contiguous input rows of K on at most
/// `max_blocks` blocks; segment i's output is `m` contiguous rows of `n_i`. Refuses a shape
/// [`w4a16_gemv_pipe_ok`] refuses or `max_blocks == 0`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_pipe_at(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    segs: &[W4a16Seg],
    m: u32,
    k: u32,
    max_blocks: u32,
    stream: u64,
) -> Result<()> {
    let entry = w4a16_gemv_pipe_entry(m).context("w4a16_gemv_pipe: no entry")?;
    let kernel = gpu
        .op_cache()
        .kernel(gpu, W4A16_GEMV_PIPE_MODULE, entry)
        .with_context(|| format!("w4a16_gemv_pipe: {W4A16_GEMV_PIPE_MODULE}::{entry}"))?;
    w4a16_gemv_pipe_staged(
        gpu,
        kernel,
        PipeInput::Rows(input),
        segs,
        m,
        k,
        max_blocks,
        stream,
    )
}

/// 2026-10-23: One launch of the `w4a16_gemv_pipe` instantiation `kernel` (an entry of `m`
/// rows) staging `input`, as [`w4a16_gemv_pipe_at`] otherwise.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_pipe_staged(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: PipeInput,
    segs: &[W4a16Seg],
    m: u32,
    k: u32,
    max_blocks: u32,
    stream: u64,
) -> Result<()> {
    let ns: Vec<u32> = segs.iter().map(|g| g.n).collect();
    anyhow::ensure!(
        w4a16_gemv_pipe_ok(k, &ns, m) && max_blocks > 0,
        "w4a16_gemv_pipe: k={k} n={ns:?} rows={m} max_blocks={max_blocks} outside the kernel's \
         contract"
    );
    let blocks = w4a16_gemv_pipe_blocks(w4a16_gemv_seg_tiles(&ns, 1), max_blocks);
    let launch = KernelLaunch::new(gpu, kernel)
        .grid([blocks, 1, 1])
        .block([W4A16_GEMV_PIPE_WARPS * 32, 1, 1])
        .shared_mem(w4a16_gemv_seg_smem(k, m));
    let mut launch = match input {
        PipeInput::Rows(a) => launch.arg_ptr(a),
        PipeInput::Pair(a, b, x) => launch.arg_ptr(a).arg_ptr(b).arg_f32(x),
    };
    for i in 0..W4A16_GEMV_SEG_MAX {
        launch = match segs.get(i) {
            Some(g) => launch
                .arg_ptr(g.weight.weight)
                .arg_ptr(g.weight.weight_scale)
                .arg_f32(g.weight.weight_scale_2)
                .arg_ptr(g.output)
                .arg_u32(g.n),
            None => launch
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(DevicePtr::NULL)
                .arg_f32(0.0)
                .arg_ptr(DevicePtr::NULL)
                .arg_u32(0),
        };
    }
    launch.arg_u32(k).launch(stream)
}

/// 2026-10-21: [`w4a16_gemv_pipe_at`] at the cap of [`w4a16_gemv_pipe_cap`] on `sms` SMs.
pub fn w4a16_gemv_pipe(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    segs: &[W4a16Seg],
    m: u32,
    k: u32,
    sms: u32,
    stream: u64,
) -> Result<()> {
    let ns: Vec<u32> = segs.iter().map(|g| g.n).collect();
    let cap = w4a16_gemv_pipe_cap(&ns, k, m, sms);
    w4a16_gemv_pipe_at(gpu, input, segs, m, k, cap, stream)
}

#[cfg(test)]
#[path = "w4a16_gemv_pipe_tests.rs"]
mod tests;
