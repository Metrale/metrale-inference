// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Launcher for `moe_w4a16_grouped_gemm_ptrtable_wgmma`
//! (kernels/hopper/common/moe_w4a16_grouped_wgmma_hopper.cu), the warpgroup-MMA twin of
//! `moe_w4a16_grouped_gemm_ptrtable`: a persistent grid of one CTA per SM over (expert, 128-row
//! tile, 128-column tile) units, which finds its tiles on the device, so it needs no host tile
//! count.
//!
//! Owner: model-layers ops.
//! Invariants: it launches only shapes `moe_w4a16_wgmma_admits` admits, the kernel's traps
//! (K % 64, N % 128, dynamic shared memory) cannot fire on them, and its grid never exceeds the
//! SM count. Its bytes against the base and pipelined kernels are the GPU test's question
//! (`ops/moe_w4a16_wgmma_tests.rs`); the MoE layer runs it only under
//! METRALE_MOE_PREFILL_WGMMA=1, and only for calls `moe_w4a16_wgmma_routes` accepts at
//! the recipe's METRALE_MOE_PREFILL_WGMMA_MIN_ROWS.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::*;
use super::launch_w4a16_ptrtable;

/// 2026-10-14: Threads and fixed dynamic shared memory of the wgmma twin (its WG_THREADS and
/// WG_SMEM_BYTES: 1,024 alignment slack + 8 stages x (16,384 B tokens + 4,096 B weights + 512 B
/// scales) + the 34,816-byte output staging tile). The tile scan adds (E + 1) x 4 bytes.
pub const MOE_W4A16_WGMMA_LAUNCH: (u32, u32) = (256, 203_776);
/// 2026-10-14: The wgmma twin's token rows per unit (WG_MT), the N tile it publishes and
/// requires (WG_NT, `<entry>_n_tile`), and its K step (WG_KS).
pub const MOE_W4A16_WGMMA_M_TILE: u32 = 128;
const WGMMA_N_TILE: u32 = 128;
const WGMMA_K_STEP: u32 = 64;
/// 2026-10-14: The most dynamic shared memory one CTA may opt in to on sm_90 (227 KiB); the
/// kernel is compiled only for that class.
pub const MOE_W4A16_WGMMA_SMEM_OPTIN: u32 = 232_448;

/// 2026-10-14: The environment variable holding the fewest expert-sorted rows (tokens x top-k)
/// a routed prefill call must have for the MoE layer to route it to the wgmma twin; smaller calls
/// keep the persistent pipelined twin. It is a measured per-class crossover (on the H100, 512
/// experts top-10, the twin lost at 300 tokens and won at 4,096, evidence wgmma-01), so it comes
/// from the recipe, never from code.
pub const MOE_W4A16_WGMMA_MIN_ROWS_ENV: &str = "METRALE_MOE_PREFILL_WGMMA_MIN_ROWS";

/// 2026-10-14: The routed-rows minimum from `raw`, the value of `MOE_W4A16_WGMMA_MIN_ROWS_ENV`.
/// Errors when it is unset, not a whole number, or zero: METRALE_MOE_PREFILL_WGMMA=1 without a
/// measured minimum is a recipe error, not a default.
pub fn moe_w4a16_wgmma_min_rows(raw: Option<&str>) -> Result<u32> {
    let raw = raw.ok_or_else(|| {
        anyhow::anyhow!(
            "METRALE_MOE_PREFILL_WGMMA=1 needs {MOE_W4A16_WGMMA_MIN_ROWS_ENV}, the measured \
             routed-rows crossover of the wgmma twin"
        )
    })?;
    match raw.parse::<u32>() {
        Ok(rows) if rows > 0 => Ok(rows),
        _ => anyhow::bail!(
            "{MOE_W4A16_WGMMA_MIN_ROWS_ENV}={raw:?}: expected a routed-row count above zero"
        ),
    }
}

/// 2026-10-14: Whether a routed prefill call of `total_rows` expert-sorted rows goes to the
/// wgmma twin (when the lever loaded it and it admits the shape): at least `min_rows`. The twin
/// writes the persistent twin's bytes at every row count (wgmma-01, class B), so the route never
/// changes a token's result, whatever the prefill chunking (`ops/moe_w4a16_wgmma_chunk_tests.rs`).
pub fn moe_w4a16_wgmma_routes(total_rows: u32, min_rows: u32) -> bool {
    total_rows >= min_rows
}

/// 2026-10-14: The wgmma twin's dynamic shared memory for `num_experts` experts; `None` past the
/// u32 range.
pub fn moe_w4a16_wgmma_smem(num_experts: u32) -> Option<u32> {
    num_experts
        .checked_add(1)?
        .checked_mul(4)?
        .checked_add(MOE_W4A16_WGMMA_LAUNCH.1)
}

/// 2026-10-14: Whether the wgmma twin computes an `n_out` x `k` projection over `num_experts`
/// experts: resolved, publishing the 128-column N tile it was built with, N a multiple of it, K
/// of its K step, and the scan's shared memory within the opt-in maximum. Other shapes keep the
/// pipelined or base kernel.
pub fn moe_w4a16_wgmma_admits(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    n_out: u32,
    k: u32,
    num_experts: u32,
) -> bool {
    kernel.0 != 0
        && k.is_multiple_of(WGMMA_K_STEP)
        && n_out.is_multiple_of(WGMMA_N_TILE)
        && gpu.kernel_n_tile(kernel).is_ok_and(|t| t == WGMMA_N_TILE)
        && moe_w4a16_wgmma_smem(num_experts).is_some_and(|s| s <= MOE_W4A16_WGMMA_SMEM_OPTIN)
}

/// 2026-10-14: The persistent grid's CTA count for `total_rows` expert-sorted rows: one per SM,
/// and never more than the work units can be (each nonempty expert adds at most one partial
/// 128-row tile, so the tiles are at most ceil(rows / 128) + min(E, rows)). Zero rows: zero
/// CTAs. Any count of one or more computes every unit (each CTA strides over them).
pub fn moe_w4a16_wgmma_ctas(sm_count: u32, n_tiles: u32, num_experts: u32, total_rows: u32) -> u32 {
    let tiles = total_rows.div_ceil(MOE_W4A16_WGMMA_M_TILE) + num_experts.min(total_rows);
    tiles.saturating_mul(n_tiles).min(sm_count)
}

/// 2026-10-14: `moe_w4a16_grouped_gemm_ptrtable`'s product through the wgmma twin. Like the
/// persistent pipelined twin it takes `total_rows`, the expert-sorted row count (an upper bound
/// on `expert_offsets[E]` serves too), not a tile count. Errors on a shape
/// `moe_w4a16_wgmma_admits` rejects.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_grouped_gemm_ptrtable_wgmma(
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
        moe_w4a16_wgmma_admits(gpu, kernel, n_out, k, num_experts),
        "moe_w4a16_grouped_gemm_ptrtable_wgmma: N={n_out} K={k} E={num_experts} not admitted"
    );
    let ctas = moe_w4a16_wgmma_ctas(
        gpu.sm_count()?,
        n_out / WGMMA_N_TILE,
        num_experts,
        total_rows,
    );
    if ctas == 0 {
        return Ok(());
    }
    let smem = moe_w4a16_wgmma_smem(num_experts).ok_or_else(|| {
        anyhow::anyhow!("moe_w4a16 wgmma: shared memory overflows at E={num_experts}")
    })?;
    let tables = [a, b_packed_ptrs, b_scale_ptrs, scale2_vals, c];
    let rows = [expert_offsets, sorted_token_ids];
    let dims = [num_experts, n_out, k];
    launch_w4a16_ptrtable(
        gpu,
        kernel,
        ([ctas, 1, 1], MOE_W4A16_WGMMA_LAUNCH.0, smem),
        tables,
        rows,
        dims,
        stream,
    )
}
