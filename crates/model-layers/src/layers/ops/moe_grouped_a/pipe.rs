// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Launchers for the pipelined Hopper twins of `moe_w4a16_grouped_gemm_ptrtable`
//! (kernels/hopper/common/moe_w4a16_grouped_pipe_hopper.cu): the per-expert grid
//! (`moe_w4a16_grouped_gemm_ptrtable_pipe`, moved here from `moe_grouped_a.rs`) and the
//! persistent grid (`moe_w4a16_grouped_gemm_ptrtable_persist`), which finds its tiles on the
//! device and so needs no host tile count.
//!
//! Owner: model-layers ops.
//! Invariants: both write the base kernel's bytes for every shape `moe_w4a16_pipe_admits`
//! admits (`ops/moe_w4a16_pipe_tests.rs`).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::*;
use super::launch_w4a16_ptrtable;

/// 2026-10-13: Threads and dynamic shared memory of `moe_w4a16_grouped_gemm_ptrtable_pipe`
/// (kernels/hopper/common/moe_w4a16_grouped_pipe_hopper.cu: PIPE_THREADS, PIPE_SMEM_BYTES; the
/// kernel traps on less shared memory).
pub const MOE_W4A16_PIPE_LAUNCH: (u32, u32) = (256, 73_728);
/// 2026-10-13: The K step and N tile the pipelined twin requires (its PIPE_KS and PIPE_NT).
pub const MOE_W4A16_PIPE_K_STEP: u32 = 64;
/// 2026-10-14: The pipelined twins' M tile (PIPE_MT).
const PIPE_M_TILE: u32 = 64;
/// 2026-10-14: Resident CTAs per SM the persistent grid is sized for. sm_90a ptxas: 78
/// registers x 256 threads (3 CTAs fit in the 64K-register file) and 73,728 + 4 x (E + 1) bytes
/// of shared memory (3 fit in 227 KiB at E = 512).
pub const MOE_W4A16_PERSIST_CTAS_PER_SM: u32 = 3;

/// 2026-10-13: Whether the pipelined twin computes an `n_out` x `k` projection: N a multiple of
/// the N tile it publishes and K of its K step. Other shapes take the base kernel.
pub fn moe_w4a16_pipe_admits(gpu: &dyn GpuBackend, pipe: KernelHandle, n_out: u32, k: u32) -> bool {
    pipe.0 != 0
        && k.is_multiple_of(MOE_W4A16_PIPE_K_STEP)
        && gpu
            .kernel_n_tile(pipe)
            .is_ok_and(|t| n_out.is_multiple_of(t))
}

/// 2026-10-13: `moe_w4a16_grouped_gemm_ptrtable`'s product through the pipelined Hopper twin, its
/// output byte-identical (grid y counts the same 64-row tiles). Launch only shapes
/// `moe_w4a16_pipe_admits` admits.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_grouped_gemm_ptrtable_pipe(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        moe_w4a16_pipe_admits(gpu, kernel, n_out, k),
        "moe_w4a16_grouped_gemm_ptrtable_pipe: N={n_out} K={k} not admitted"
    );
    let tables = [a, b_packed_ptrs, b_scale_ptrs, scale2_vals, c];
    let rows = [expert_offsets, sorted_token_ids];
    let grid = [n_tile_blocks(gpu, kernel, n_out)?, max_m_tiles, num_experts];
    let (threads, smem) = MOE_W4A16_PIPE_LAUNCH;
    let dims = [num_experts, n_out, k];
    launch_w4a16_ptrtable(
        gpu,
        kernel,
        (grid, threads, smem),
        tables,
        rows,
        dims,
        stream,
    )
}

/// 2026-10-14: The persistent grid's CTA count for `total_rows` expert-sorted rows. Any count
/// of one or more computes every tile (each CTA strides over the units); this one is never more
/// than the work units can be (each nonempty expert adds at most one partial M tile, so the
/// tiles are at most ceil(rows / 64) + min(E, rows)), and never more than
/// `MOE_W4A16_PERSIST_CTAS_PER_SM` per SM. Zero rows: zero CTAs.
pub fn moe_w4a16_persist_ctas(
    sm_count: u32,
    n_tiles: u32,
    num_experts: u32,
    total_rows: u32,
) -> u32 {
    let tiles = total_rows.div_ceil(PIPE_M_TILE) + num_experts.min(total_rows);
    tiles
        .saturating_mul(n_tiles)
        .min(MOE_W4A16_PERSIST_CTAS_PER_SM * sm_count)
}

/// 2026-10-14: [`moe_w4a16_grouped_gemm_ptrtable_pipe`]'s bytes from the persistent grid: no
/// tile count, only `total_rows`, the expert-sorted row count (an upper bound on
/// `expert_offsets[E]` serves too). Launch only shapes `moe_w4a16_pipe_admits` admits.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_grouped_gemm_ptrtable_persist(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    total_rows: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        moe_w4a16_pipe_admits(gpu, kernel, n_out, k),
        "moe_w4a16_grouped_gemm_ptrtable_persist: N={n_out} K={k} not admitted"
    );
    let n_tiles = n_tile_blocks(gpu, kernel, n_out)?;
    let ctas = moe_w4a16_persist_ctas(gpu.sm_count()?, n_tiles, num_experts, total_rows);
    if ctas == 0 {
        return Ok(());
    }
    let tables = [a, b_packed_ptrs, b_scale_ptrs, scale2_vals, c];
    let rows = [expert_offsets, sorted_token_ids];
    let (threads, smem) = MOE_W4A16_PIPE_LAUNCH;
    let smem = smem + (num_experts + 1) * 4;
    let dims = [num_experts, n_out, k];
    launch_w4a16_ptrtable(
        gpu,
        kernel,
        ([ctas, 1, 1], threads, smem),
        tables,
        rows,
        dims,
        stream,
    )
}
