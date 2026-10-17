// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `w4a16_tc_rows_seg_<R>_k<S>` (`kernels/gb10/common/w4a16_tc_rows_seg.cu`): the
//! NVFP4 W4A16 row tiles of `w4a16_tc_rows` over up to three weight segments that share the
//! input (one launch), with a K-split of S whose blocks merge in one thread-block cluster.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`w4a16_tc_rows_seg`] launches only when [`w4a16_tc_rows_seg_shape_ok`] holds.
//! - S = 1: each segment's output bytes are `w4a16_tc_rows`'s. S > 1: other bits, which depend
//!   on K and S alone; for one S a row's bits do not depend on M or on the entry point.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{W4A16_TC_ROWS_COLS, W4A16_TC_ROWS_MAX_M};
use crate::weight_map::QuantizedWeight;

/// 2026-10-09: The kernel module.
pub const W4A16_TC_ROWS_SEG_MODULE: &str = "w4a16_tc_rows_seg";

/// 2026-10-09: Weight segments one launch takes.
pub const W4A16_TC_ROWS_SEG_MAX: usize = 3;

/// 2026-10-09: The K-splits the module has entries for (`W4A16_SEG_SPLITS`).
pub const W4A16_TC_ROWS_SPLITS: [u32; 3] = [1, 2, 4];

/// 2026-10-09: The K unit a split divides (`TR_SPLIT_K`, `tc_rows.cuh`).
pub const W4A16_TC_ROWS_SPLIT_K: u32 = 256;

/// 2026-10-09: One weight segment: `output [m, n] = input [m, k] x W^T`, `output` contiguous.
#[derive(Clone, Copy, Debug)]
pub struct W4a16Seg {
    pub weight: QuantizedWeight,
    pub output: DevicePtr,
    pub n: u32,
}

/// 2026-10-09: Column tiles (blocks per split) a launch over segments of widths `ns` has.
pub fn w4a16_tc_rows_seg_tiles(ns: &[u32]) -> u64 {
    ns.iter()
        .map(|&n| u64::from(n.div_ceil(W4A16_TC_ROWS_COLS)))
        .sum()
}

/// 2026-10-09: The entry point for `m` rows and split `s`: the row tiles of
/// `w4a16_tc_rows_entry(m)`. `None` for a split the module has no entry for.
pub fn w4a16_tc_rows_seg_entry(m: u32, s: u32) -> Option<&'static str> {
    let rows = if m <= 16 {
        0
    } else if m <= 32 {
        1
    } else {
        2
    };
    const NAMES: [[&str; 3]; 3] = [
        [
            "w4a16_tc_rows_seg_16_k1",
            "w4a16_tc_rows_seg_16_k2",
            "w4a16_tc_rows_seg_16_k4",
        ],
        [
            "w4a16_tc_rows_seg_32_k1",
            "w4a16_tc_rows_seg_32_k2",
            "w4a16_tc_rows_seg_32_k4",
        ],
        [
            "w4a16_tc_rows_seg_64_k1",
            "w4a16_tc_rows_seg_64_k2",
            "w4a16_tc_rows_seg_64_k4",
        ],
    ];
    let si = W4A16_TC_ROWS_SPLITS.iter().position(|&x| x == s)?;
    Some(NAMES[rows][si])
}

/// 2026-10-09: Every entry point of the module, for a load-time lookup.
pub fn w4a16_tc_rows_seg_entries() -> impl Iterator<Item = &'static str> {
    [16u32, 32, 64].into_iter().flat_map(|m| {
        W4A16_TC_ROWS_SPLITS
            .into_iter()
            .filter_map(move |s| w4a16_tc_rows_seg_entry(m, s))
    })
}

/// 2026-10-09: The kernel's shape contract, without a GPU: 1..=64 rows, 1..=3 segments of
/// positive width, K a positive multiple of the split unit, a split the module has with at least
/// one unit per split, an A pitch that keeps rows 16-byte aligned and covers K, and a grid that
/// fits a launch.
pub fn w4a16_tc_rows_seg_shape_ok(m: u32, k: u32, lda: u32, ns: &[u32], s: u32) -> bool {
    (1..=W4A16_TC_ROWS_MAX_M).contains(&m)
        && (1..=W4A16_TC_ROWS_SEG_MAX).contains(&ns.len())
        && ns.iter().all(|&n| n > 0)
        && k > 0
        && k.is_multiple_of(W4A16_TC_ROWS_SPLIT_K)
        && W4A16_TC_ROWS_SPLITS.contains(&s)
        && s <= k / W4A16_TC_ROWS_SPLIT_K
        && lda >= k
        && lda.is_multiple_of(8)
        && w4a16_tc_rows_seg_tiles(ns) <= u64::from(i32::MAX as u32)
}

/// 2026-10-09: The K-split for a launch of `tiles` column tiles over `k` on `sms` SMs: the
/// widest split (of 1, 2, 4) that keeps at most one block per SM (`tiles * s <= sms`) and at
/// most one per split unit of K; 1 when none does. A function of the shape and the device alone,
/// never of the row count, so a CUDA graph replays one choice.
///
/// GB10 (48 SMs), measured 2026-10-09 at 4..16 rows (cold weights, graph replay): the KDA
/// f_a+g_a+b group (5 tiles) and a 512- or 768-wide projection (8, 12 tiles) are fastest at 4,
/// the shared gate+up group (24 tiles) at 2, and the 2816-wide KDA q/k/v (44 tiles) unsplit (2 was
/// 1-7% slower). A split of 8 won only on 1- and 2-tile shapes, by 0.5 us, and lost 20% on the
/// 5-tile group, so the module stops at 4.
pub fn w4a16_tc_rows_split_for(tiles: u64, k: u32, sms: u32) -> u32 {
    let units = k / W4A16_TC_ROWS_SPLIT_K;
    W4A16_TC_ROWS_SPLITS
        .into_iter()
        .rev()
        .find(|&s| s <= units && tiles * u64::from(s) <= u64::from(sms))
        .unwrap_or(1)
}

/// 2026-10-09: One launch of `segs` (1..=3) over `m` rows of `input` (pitch `lda`) with K-split
/// `s`. Refuses a shape [`w4a16_tc_rows_seg_shape_ok`] refuses.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_tc_rows_seg(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    segs: &[W4a16Seg],
    m: u32,
    k: u32,
    lda: u32,
    s: u32,
    stream: u64,
) -> Result<()> {
    let ns: Vec<u32> = segs.iter().map(|g| g.n).collect();
    anyhow::ensure!(
        w4a16_tc_rows_seg_shape_ok(m, k, lda, &ns, s),
        "w4a16_tc_rows_seg: m={m} k={k} lda={lda} n={ns:?} split={s} outside the kernel's contract"
    );
    let entry = w4a16_tc_rows_seg_entry(m, s).context("w4a16_tc_rows_seg: no entry")?;
    let kernel = gpu
        .op_cache()
        .kernel(gpu, W4A16_TC_ROWS_SEG_MODULE, entry)?;
    let mut launch = KernelLaunch::new(gpu, kernel)
        .grid([w4a16_tc_rows_seg_tiles(&ns) as u32, s, 1])
        .block([128, 1, 1])
        .arg_ptr(input);
    for i in 0..W4A16_TC_ROWS_SEG_MAX {
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
    launch.arg_u32(m).arg_u32(k).arg_u32(lda).launch(stream)
}

#[cfg(test)]
#[path = "w4a16_tc_rows_seg_tests.rs"]
mod tests;
