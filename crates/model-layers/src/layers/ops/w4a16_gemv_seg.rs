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
//! - 2026-10-11: The M-row points (`w4a16_gemv_seg_m<M>`, M = 2..=4, [`W4a16GemvSegPoint::Rows`])
//!   write, for each row m, the bytes the one-row kernel writes for row m alone; the hoist point
//!   ([`W4a16GemvSegPoint::Hoist`]) writes the `r1` bytes.

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

/// 2026-10-11: An entry point of the module: a tile height R over one row (`r<R>`), R = 1 with
/// the step-0 weight loads hoisted above the input staging (`r1h`), or M input rows at R = 1
/// (`m<M>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W4a16GemvSegPoint {
    Tile(u32),
    Hoist,
    Rows(u32),
}

impl W4a16GemvSegPoint {
    /// 2026-10-11: The entry point's name; `None` for a point the module lacks.
    pub fn entry(self) -> Option<&'static str> {
        match self {
            Self::Tile(r) => w4a16_gemv_seg_entry(r),
            Self::Hoist => Some("w4a16_gemv_seg_r1h"),
            Self::Rows(2) => Some("w4a16_gemv_seg_m2"),
            Self::Rows(3) => Some("w4a16_gemv_seg_m3"),
            Self::Rows(4) => Some("w4a16_gemv_seg_m4"),
            Self::Rows(_) => None,
        }
    }

    /// 2026-10-11: Input rows per launch.
    pub fn rows(self) -> u32 {
        match self {
            Self::Rows(m) => m,
            Self::Tile(_) | Self::Hoist => 1,
        }
    }

    /// 2026-10-11: Output rows per warp tile.
    pub fn tile(self) -> u32 {
        match self {
            Self::Tile(r) => r,
            Self::Hoist | Self::Rows(_) => 1,
        }
    }
}

/// 2026-10-11: The widest M-row point (`SEG1_ROWS_ENTRY`).
pub const W4A16_GEMV_SEG_MAX_ROWS: u32 = 4;

/// 2026-10-11: The dynamic shared memory a launch may ask for: the M staged input rows. 96 KB
/// fits GB10's 99 KB per-block opt-in (the runtime opts in above 48 KB).
pub const W4A16_GEMV_SEG_MAX_SMEM: u32 = 96 * 1024;

/// 2026-10-10: The kernel's contract, without a GPU: 1..=6 segments of positive width, K a
/// positive multiple of 32 up to [`W4A16_GEMV_SEG_MAX_K`].
pub fn w4a16_gemv_seg_shape_ok(k: u32, ns: &[u32]) -> bool {
    (1..=W4A16_GEMV_SEG_MAX).contains(&ns.len())
        && ns.iter().all(|&n| n > 0)
        && k > 0
        && k.is_multiple_of(32)
        && k <= W4A16_GEMV_SEG_MAX_K
}

/// 2026-10-11: [`w4a16_gemv_seg_shape_ok`] for `m` input rows: 1..=[`W4A16_GEMV_SEG_MAX_ROWS`]
/// and the staged rows within [`W4A16_GEMV_SEG_MAX_SMEM`] (K 11264 takes M = 4 at exactly 96 KB).
pub fn w4a16_gemv_seg_rows_ok(k: u32, ns: &[u32], m: u32) -> bool {
    w4a16_gemv_seg_shape_ok(k, ns)
        && (1..=W4A16_GEMV_SEG_MAX_ROWS).contains(&m)
        && w4a16_gemv_seg_smem(k, m) <= W4A16_GEMV_SEG_MAX_SMEM
}

/// 2026-10-10: Dynamic shared memory of a launch over K: the permuted input, 4 KB per 2048 of K
/// (one block of 128 chunks of 16 values), rounded up. 2026-10-11: Per input row, `m` rows.
pub fn w4a16_gemv_seg_smem(k: u32, m: u32) -> u32 {
    k.div_ceil(2048) * 4096 * m
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

/// 2026-10-11: The widest K a LONE projection (one segment) runs on this kernel instead of
/// `w4a16_gemv`. Same A/B as [`w4a16_gemv_seg_policy`]: one segment wins below K 2048 (shared down
/// K 768: 12.0 -> 9.7 us; DSA q_absorb K 1536: 42.3 -> 40.3) and does not from 2048 up (KDA o
/// K 2816: 30.9 -> 32.5; KDA q K 4096: 29.7 -> 30.3; o_absorb K 11264: 118.4 -> 118.6). Groups
/// of two or more segments win at every measured K.
pub const W4A16_GEMV_SEG_LONE_MAX_K: u32 = 2047;

/// 2026-10-11: Whether a one-segment launch over `k` takes this kernel ([`W4A16_GEMV_SEG_LONE_MAX_K`]).
pub fn w4a16_gemv_seg_lone(k: u32) -> bool {
    k <= W4A16_GEMV_SEG_LONE_MAX_K
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
    w4a16_gemv_seg_point(
        gpu,
        W4a16GemvSegPoint::Tile(r),
        input,
        segs,
        k,
        max_blocks,
        stream,
    )
}

/// 2026-10-11: One launch of `segs` (1..=6) at `point` on at most `max_blocks` blocks: `input`
/// holds `point.rows()` contiguous rows of K, segment i's output `point.rows()` contiguous rows
/// of `n_i`. Refuses a point the module lacks, a shape [`w4a16_gemv_seg_rows_ok`] refuses, or
/// `max_blocks == 0`.
pub fn w4a16_gemv_seg_point(
    gpu: &dyn GpuBackend,
    point: W4a16GemvSegPoint,
    input: DevicePtr,
    segs: &[W4a16Seg],
    k: u32,
    max_blocks: u32,
    stream: u64,
) -> Result<()> {
    let ns: Vec<u32> = segs.iter().map(|g| g.n).collect();
    let m = point.rows();
    anyhow::ensure!(
        w4a16_gemv_seg_rows_ok(k, &ns, m) && max_blocks > 0,
        "w4a16_gemv_seg: k={k} n={ns:?} rows={m} max_blocks={max_blocks} outside the kernel's \
         contract"
    );
    let entry = point
        .entry()
        .with_context(|| format!("w4a16_gemv_seg: no entry for {point:?}"))?;
    let kernel = gpu.op_cache().kernel(gpu, W4A16_GEMV_SEG_MODULE, entry)?;
    let blocks = w4a16_gemv_seg_blocks(w4a16_gemv_seg_tiles(&ns, point.tile()), max_blocks);
    let mut launch = KernelLaunch::new(gpu, kernel)
        .grid([blocks, 1, 1])
        .block([W4A16_GEMV_SEG_WARPS * 32, 1, 1])
        .shared_mem(w4a16_gemv_seg_smem(k, m))
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
