// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `w4a16_gemv_seg_r<R>` (`kernels/gb10/common/w4a16_gemv_seg.cu`): the one-row NVFP4
//! W4A16 GEMV over up to six weight segments that share the input row, in one persistent launch
//! whose warps each own a tile of R output rows.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`w4a16_gemv_seg`] launches only when [`w4a16_gemv_seg_shape_ok`] holds.
//! - Every segment's output bytes are `w4a16_gemv`'s (`ops::w4a16_gemv`) on that weight, for
//!   every R and grid (the kernel's invariant; GPU proof: the `glm5next_w4a16_gemv_seg_bench`
//!   example of model-arch).

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::W4a16Seg;

/// 2026-10-10: The kernel module.
pub const W4A16_GEMV_SEG_MODULE: &str = "w4a16_gemv_seg";

/// 2026-10-10: Weight segments one launch takes (`SEG1_ENTRY`'s parameter sets).
pub const W4A16_GEMV_SEG_MAX: usize = 6;

/// 2026-10-10: Threads per block (`SEG1_THREADS`): eight warps, one tile each at a time.
pub const W4A16_GEMV_SEG_WARPS: u32 = 8;

/// 2026-10-10: The widest K the permuted input fits in 48 KB of dynamic shared memory.
pub const W4A16_GEMV_SEG_MAX_K: u32 = 24576;

/// 2026-10-10: The tile heights (rows per warp) the module has entries for.
pub const W4A16_GEMV_SEG_TILES: [u32; 3] = [1, 2, 4];

/// 2026-10-10: The entry point of tile height `r`; `None` for a height the module lacks.
pub fn w4a16_gemv_seg_entry(r: u32) -> Option<&'static str> {
    match r {
        1 => Some("w4a16_gemv_seg_r1"),
        2 => Some("w4a16_gemv_seg_r2"),
        4 => Some("w4a16_gemv_seg_r4"),
        _ => None,
    }
}

/// 2026-10-10: The kernel's contract, without a GPU: 1..=6 segments of positive width, K a
/// positive multiple of 32 up to [`W4A16_GEMV_SEG_MAX_K`].
pub fn w4a16_gemv_seg_shape_ok(k: u32, ns: &[u32]) -> bool {
    (1..=W4A16_GEMV_SEG_MAX).contains(&ns.len())
        && ns.iter().all(|&n| n > 0)
        && k > 0
        && k.is_multiple_of(32)
        && k <= W4A16_GEMV_SEG_MAX_K
}

/// 2026-10-10: Dynamic shared memory of a launch over K: the permuted input, 4 KB per 2048 of K
/// (one block of 128 chunks of 16 values), rounded up.
pub fn w4a16_gemv_seg_smem(k: u32) -> u32 {
    k.div_ceil(2048) * 4096
}

/// 2026-10-10: Warp tiles of a launch of tile height `r` over segments `ns`.
pub fn w4a16_gemv_seg_tiles(ns: &[u32], r: u32) -> u64 {
    ns.iter().map(|&n| u64::from(n.div_ceil(r))).sum()
}

/// 2026-10-10: Blocks of a launch: enough for one tile per warp, capped at `max_blocks` (the
/// persistent grid; the warps stride over the rest).
pub fn w4a16_gemv_seg_blocks(tiles: u64, max_blocks: u32) -> u32 {
    let need = tiles.div_ceil(u64::from(W4A16_GEMV_SEG_WARPS));
    need.min(u64::from(max_blocks)).max(1) as u32
}

/// 2026-10-10: The tile height and block cap of a launch over `k` on `sms` SMs: one output row
/// per warp, one tile per warp (no grid striding), at every K. GB10 (48 SMs), interleaved A/B of
/// 5 rounds on an idle GPU (`glm5next_w4a16_gemv_seg_bench --sweep`), medians: R = 1 uncapped is
/// fastest or within 1% on every GLM-5.3 shape and group (KDA six-group 88.2 us vs 90.0 at R = 2,
/// 90.8 at R = 4; shared down 9.72 vs 9.98 / 10.31; o_absorb K 11264 114.5 vs 116.7 / 118.5);
/// caps of 2 or 4 blocks per SM never won. `k` and `sms` stay arguments so a measured
/// device- or shape-specific tile can enter here without touching callers.
pub fn w4a16_gemv_seg_policy(k: u32, sms: u32) -> (u32, u32) {
    let _ = (k, sms);
    (1, u32::MAX)
}

/// 2026-10-10: One launch of `segs` (1..=6) over one row of `input` with tile height `r` on at
/// most `max_blocks` blocks. Refuses a shape [`w4a16_gemv_seg_shape_ok`] refuses, a tile height
/// the module lacks, or `max_blocks == 0`.
pub fn w4a16_gemv_seg(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    segs: &[W4a16Seg],
    k: u32,
    r: u32,
    max_blocks: u32,
    stream: u64,
) -> Result<()> {
    let ns: Vec<u32> = segs.iter().map(|g| g.n).collect();
    anyhow::ensure!(
        w4a16_gemv_seg_shape_ok(k, &ns) && max_blocks > 0,
        "w4a16_gemv_seg: k={k} n={ns:?} max_blocks={max_blocks} outside the kernel's contract"
    );
    let entry = w4a16_gemv_seg_entry(r)
        .with_context(|| format!("w4a16_gemv_seg: no entry for tile height {r}"))?;
    let kernel = gpu.op_cache().kernel(gpu, W4A16_GEMV_SEG_MODULE, entry)?;
    let blocks = w4a16_gemv_seg_blocks(w4a16_gemv_seg_tiles(&ns, r), max_blocks);
    let mut launch = KernelLaunch::new(gpu, kernel)
        .grid([blocks, 1, 1])
        .block([W4A16_GEMV_SEG_WARPS * 32, 1, 1])
        .shared_mem(w4a16_gemv_seg_smem(k))
        .arg_ptr(input);
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

#[cfg(test)]
#[path = "w4a16_gemv_seg_tests.rs"]
mod tests;
