// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The deterministic per-row top-k over a score matrix
//! (`dsa_topk_pools`, `kernels/gb10/common/dsa_indexer.cu`), shared by the GLM-5.3 DSA
//! selector and the Qwen3.8-Flash-Next QSA prefill selection.
//!
//! The kernel orders by score descending, then index ascending: a total order over unique
//! indices, so its top `select_k` equals a whole-row sort's prefix whatever the tile width.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`dsa_topk_pools`] launches only with `select_k <= tile <= topk_tile()` and `tile` a
//!   power of two, so its shared memory stays within [`TOPK_SMEM_CEILING`].

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-25: Shared-memory ceiling the top-k select is budgeted against; the same value
/// as `SMEM_CEILING` in `model-arch/examples/dsa_indexer_microtest.rs`.
pub const TOPK_SMEM_CEILING: usize = 49_152;

/// 2026-10-13: Threads per `dsa_topk_pools` block.
const TOPK_BLOCK: u32 = 256;

/// 2026-09-25: Tile width `dsa_topk_pools` walks the pool axis in.
///
/// The block holds two tiles, the running best list and the candidate tile, of `[f32, i32]`
/// pairs: `16 · T` bytes. This is the largest power of two `T` that fits
/// [`TOPK_SMEM_CEILING`]: 2,048.
pub fn topk_tile() -> usize {
    let mut t = 2usize;
    while t * 2 * 16 <= TOPK_SMEM_CEILING {
        t *= 2;
    }
    t
}

/// 2026-09-25: Shared memory one `dsa_topk_pools` block needs for a tile of `t` pools.
pub fn topk_smem_for_tile(t: usize) -> usize {
    t * 2 * 8
}

/// 2026-10-13: One `dsa_topk_pools` pass: for each of `rows` rows of `scores` (`[rows, n]`
/// f32, row stride `n`), write the indices of its `select_k` best entries, best first, to
/// `selected` (`[rows, select_k]` i32). `geom` is the kernel's optional device geometry
/// (`DevicePtr::NULL` for none). Enqueued on `stream`; nothing synchronises.
#[allow(clippy::too_many_arguments)]
pub fn dsa_topk_pools(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    scores: DevicePtr,
    selected: DevicePtr,
    rows: u32,
    n: u32,
    tile: u32,
    select_k: u32,
    geom: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        tile.is_power_of_two() && select_k <= tile && tile as usize <= topk_tile(),
        "dsa_topk_pools: tile {tile} must be a power of two in [select_k {select_k}, {}]",
        topk_tile()
    );
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([TOPK_BLOCK, 1, 1])
        .shared_mem(topk_smem_for_tile(tile as usize) as u32)
        .arg_ptr(scores)
        .arg_ptr(selected)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(tile)
        .arg_u32(select_k)
        .arg_ptr(geom)
        .launch(stream)
}

#[cfg(test)]
#[path = "pool_topk_tests.rs"]
mod tests;
