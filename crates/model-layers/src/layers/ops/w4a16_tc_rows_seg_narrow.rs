// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `w4a16_tc_rows_seg_<R>_k1_w<NW>` (`kernels/gb10/common/w4a16_tc_rows_seg.cu`): the
//! unsplit segmented row tiles of 1..=16 rows on blocks of 1, 2 or 4 warps (16 NW columns), with
//! an 8-row tile for 1..=8 rows.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`w4a16_tc_rows_seg_narrow`] launches only when [`w4a16_tc_rows_seg_shape_ok`] holds for
//!   split 1, `m` <= [`W4A16_TC_ROWS_SEG_NARROW_MAX_M`] and `nw` is one of
//!   [`W4A16_TC_ROWS_SEG_WARPS`].
//! - Every output byte equals `w4a16_tc_rows_seg` at split 1 (the kernel's invariant: the row
//!   tile count and the warps per block change no output's MMA sequence).
//! - [`w4a16_tc_rows_seg_warps_for`] is a function of the segment widths, the fill target and the
//!   SM count, never of the row count, so a CUDA graph replays one choice per shape.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::w4a16_tc_rows_seg::w4a16_tc_rows_seg_args;
use super::{W4A16_TC_ROWS_SEG_MODULE, W4a16Seg, w4a16_tc_rows_seg_shape_ok};

/// 2026-10-16: The widest launch the narrow entries take.
pub const W4A16_TC_ROWS_SEG_NARROW_MAX_M: u32 = 16;

/// 2026-10-16: Warps per block the narrow entries have, widest first.
pub const W4A16_TC_ROWS_SEG_WARPS: [u32; 3] = [4, 2, 1];

/// 2026-10-16: The entry for `m` rows on `nw`-warp blocks: the 8-row tile at 1..=8 rows, the
/// 16-row tile at 9..=16 (at 4 warps `w4a16_tc_rows_seg_16_k1` itself). `None` outside them.
pub fn w4a16_tc_rows_seg_narrow_entry(m: u32, nw: u32) -> Option<&'static str> {
    let wi = W4A16_TC_ROWS_SEG_WARPS.iter().position(|&x| x == nw)?;
    const NAMES: [[&str; 3]; 2] = [
        [
            "w4a16_tc_rows_seg_8_k1_w4",
            "w4a16_tc_rows_seg_8_k1_w2",
            "w4a16_tc_rows_seg_8_k1_w1",
        ],
        [
            "w4a16_tc_rows_seg_16_k1",
            "w4a16_tc_rows_seg_16_k1_w2",
            "w4a16_tc_rows_seg_16_k1_w1",
        ],
    ];
    match m {
        1..=8 => Some(NAMES[0][wi]),
        9..=W4A16_TC_ROWS_SEG_NARROW_MAX_M => Some(NAMES[1][wi]),
        _ => None,
    }
}

/// 2026-10-16: Every narrow entry, for a load-time lookup.
pub fn w4a16_tc_rows_seg_narrow_entries() -> impl Iterator<Item = &'static str> {
    [8u32, 16].into_iter().flat_map(|m| {
        W4A16_TC_ROWS_SEG_WARPS
            .into_iter()
            .filter_map(move |nw| w4a16_tc_rows_seg_narrow_entry(m, nw))
    })
}

/// 2026-10-16: Blocks a launch over segments of widths `ns` has on `nw`-warp blocks.
pub fn w4a16_tc_rows_seg_blocks(ns: &[u32], nw: u32) -> u64 {
    ns.iter().map(|&n| u64::from(n.div_ceil(16 * nw))).sum()
}

/// 2026-10-16: The warps per block for segments `ns`: the widest of
/// [`W4A16_TC_ROWS_SEG_WARPS`] whose launch has at least `fill` blocks per SM, else 1.
pub fn w4a16_tc_rows_seg_warps_for(ns: &[u32], fill: u32, sms: u32) -> u32 {
    let want = u64::from(fill) * u64::from(sms);
    W4A16_TC_ROWS_SEG_WARPS
        .into_iter()
        .find(|&nw| w4a16_tc_rows_seg_blocks(ns, nw) >= want)
        .unwrap_or(1)
}

/// 2026-10-16: One unsplit launch of `segs` (1..=3) over `m` (1..=16) rows of `input` (pitch
/// `lda`) on `nw`-warp blocks. Refuses what the module has no entry for.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_tc_rows_seg_narrow(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    segs: &[W4a16Seg],
    m: u32,
    k: u32,
    lda: u32,
    nw: u32,
    stream: u64,
) -> Result<()> {
    let ns: Vec<u32> = segs.iter().map(|g| g.n).collect();
    anyhow::ensure!(
        w4a16_tc_rows_seg_shape_ok(m, k, lda, &ns, 1),
        "w4a16_tc_rows_seg_narrow: m={m} k={k} lda={lda} n={ns:?} outside the kernel's contract"
    );
    let entry = w4a16_tc_rows_seg_narrow_entry(m, nw)
        .with_context(|| format!("w4a16_tc_rows_seg_narrow: no entry for m={m} warps={nw}"))?;
    let kernel = gpu
        .op_cache()
        .kernel(gpu, W4A16_TC_ROWS_SEG_MODULE, entry)?;
    let launch = KernelLaunch::new(gpu, kernel)
        .grid([w4a16_tc_rows_seg_blocks(&ns, nw) as u32, 1, 1])
        .block([32 * nw, 1, 1])
        .arg_ptr(input);
    w4a16_tc_rows_seg_args(launch, segs, m, k, lda).launch(stream)
}

#[cfg(test)]
#[path = "w4a16_tc_rows_seg_narrow_tests.rs"]
mod tests;
