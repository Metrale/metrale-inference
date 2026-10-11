// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: `w4a16_tc_rows_{16,32,64}` (`kernels/gb10/common/w4a16_tc_rows.cu`): the NVFP4
//! W4A16 projection of 1..=64 rows with the rows as the tensor-core MMA's N columns, the NVFP4
//! point of the row-tile family whose FP8 point is `w8a16_tc_rows`. A row's bits do not depend
//! on the row count or the entry point, so callers chunk wider row counts in 64-row calls.
//!
//! Owner: model-layers ops.
//! Invariants: [`w4a16_tc_rows`] launches only when [`w4a16_tc_rows_shape_ok`] holds.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::weight_map::QuantizedWeight;

/// 2026-10-02: Output columns per CTA (`TR_COLS`).
pub const W4A16_TC_ROWS_COLS: u32 = 64;

/// 2026-10-05: Output columns per CTA of the `_w2` entry points (2 warps).
pub const W4A16_TC_ROWS_COLS_NARROW: u32 = 32;

/// 2026-10-05: The N tile of an `n`-column launch on a device of `sm_count` SMs: the 64-column
/// tile when its grid fills one wave, else the 32-column tile (twice the CTAs). The tile changes
/// no output bit (`tc_rows.cuh`), so this is speed only; a class's `sm_count` is its
/// `[hardware] sm_count` (`metrale_kernels::TARGET_SM_COUNT`).
pub fn w4a16_tc_rows_cols(n: u32, sm_count: u32) -> u32 {
    if n.div_ceil(W4A16_TC_ROWS_COLS) >= sm_count {
        W4A16_TC_ROWS_COLS
    } else {
        W4A16_TC_ROWS_COLS_NARROW
    }
}

/// 2026-10-02: Widest row count of one launch (`8 * NT` of `w4a16_tc_rows_64`).
pub const W4A16_TC_ROWS_MAX_M: u32 = 64;

/// 2026-10-02: The kernel module.
pub const W4A16_TC_ROWS_MODULE: &str = "w4a16_tc_rows";

/// 2026-10-05: `METRALE_W4A16_TC_ROWS_COLS` = `64` or `32` pins the N tile for an A/B on one
/// binary; unset or any other value leaves the sm_count rule. Read once per process.
fn forced_cols() -> &'static Option<u32> {
    static FORCED: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    FORCED.get_or_init(|| {
        metrale_config::levers::var("METRALE_W4A16_TC_ROWS_COLS")
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|c| [W4A16_TC_ROWS_COLS, W4A16_TC_ROWS_COLS_NARROW].contains(c))
    })
}

/// 2026-10-09: The entry for `m` rows: the row tier (16, 32, 64), the N tile (`wide`: 64 columns,
/// else the `_w2` 32-column twin) and, on the 64-column tile, the class's load-ahead `pf` (the
/// widest compiled `_pf` point at or below it: the 16-row tier has `_pf2` only). Every choice
/// gives the same bits.
pub fn w4a16_tc_rows_entry(m: u32, wide: bool, pf: u32) -> &'static str {
    match (m, wide, pf) {
        (..=16, false, _) => "w4a16_tc_rows_16_w2",
        (..=32, false, _) => "w4a16_tc_rows_32_w2",
        (_, false, _) => "w4a16_tc_rows_64_w2",
        (..=16, true, 0..=1) => "w4a16_tc_rows_16",
        (..=16, true, _) => "w4a16_tc_rows_16_pf2",
        (..=32, true, 0..=1) => "w4a16_tc_rows_32",
        (..=32, true, 2) => "w4a16_tc_rows_32_pf2",
        (..=32, true, _) => "w4a16_tc_rows_32_pf3",
        (_, true, 0..=1) => "w4a16_tc_rows_64",
        (_, true, 2) => "w4a16_tc_rows_64_pf2",
        (_, true, _) => "w4a16_tc_rows_64_pf3",
    }
}

/// 2026-10-02: The kernel's shape contract, without a GPU: 1..=64 rows, any positive N (the
/// entry points are ragged: a partial last CTA loads zero weight rows and stores nothing past N),
/// K a positive multiple of 256 (whole load groups of every entry point), and an A pitch that
/// keeps rows 16-byte aligned and covers K; the C pitch covers N.
pub fn w4a16_tc_rows_shape_ok(m: u32, n: u32, k: u32, lda: u32, ldc: u32) -> bool {
    (1..=W4A16_TC_ROWS_MAX_M).contains(&m)
        && n > 0
        && k > 0
        && k.is_multiple_of(256)
        && lda >= k
        && lda.is_multiple_of(8)
        && ldc >= n
}

/// 2026-10-02: `output [m, ldc] = input [m, lda] x W^T` for the row-major NVFP4 `weight` `[n, k]`
/// (packed E2M1, E4M3 scales of 16, `weight_scale_2`). Refuses a shape
/// [`w4a16_tc_rows_shape_ok`] refuses.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_tc_rows(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        w4a16_tc_rows_shape_ok(m, n, k, lda, ldc),
        "w4a16_tc_rows: m={m} n={n} k={k} lda={lda} ldc={ldc} outside the kernel's contract"
    );
    let cols =
        forced_cols().unwrap_or_else(|| w4a16_tc_rows_cols(n, metrale_kernels::TARGET_SM_COUNT));
    let entry = w4a16_tc_rows_entry(
        m,
        cols == W4A16_TC_ROWS_COLS,
        super::target_defaults::resolved().w4a16_tc_rows_pf.value,
    );
    let kernel = gpu.op_cache().kernel(gpu, W4A16_TC_ROWS_MODULE, entry)?;
    KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(cols), 1, 1])
        .block([cols * 2, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(lda)
        .arg_u32(ldc)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CU: &str = include_str!("../../../../../kernels/gb10/common/w4a16_tc_rows.cu");
    const ROWS: &str = include_str!("../../../../../kernels/gb10/common/tc_rows.cuh");

    /// 2026-10-02: The grid and the widest entry point match the kernel (`TR_WARPS` warps of 16
    /// columns, `tr_block<Nvfp4G16, 8, ...>` row tiles of 8), or columns or rows go unwritten.
    #[test]
    fn launch_matches_the_kernel() {
        assert!(
            ROWS.contains("#define TR_WARPS 4\n")
                && ROWS.contains("#define TR_COLS (TR_WARPS * 16)\n")
        );
        assert_eq!(W4A16_TC_ROWS_COLS, 4 * 16);
        assert!(CU.contains(
            "tr_block<Nvfp4G16, 8, 1, true>(A, {packed, scale, s2}, C, M, N, K, lda, ldc, blockIdx.x);"
        ));
        assert_eq!(W4A16_TC_ROWS_MAX_M, 8 * 8);
        for entry in [
            "w4a16_tc_rows_16(",
            "w4a16_tc_rows_32(",
            "w4a16_tc_rows_64(",
        ] {
            assert!(CU.contains(entry), "{entry} missing from the kernel");
        }
        // 2026-10-05: The `_w2` twins: the same row tiles with 2 warps of 16 columns.
        assert_eq!(W4A16_TC_ROWS_COLS_NARROW, 2 * 16);
        for (entry, tiles) in [("16", "2, 2"), ("32", "4, 1"), ("64", "8, 1")] {
            assert!(CU.contains(&format!("__launch_bounds__(64) w4a16_tc_rows_{entry}_w2(")));
            assert!(CU.contains(&format!("tr_block<Nvfp4G16, {tiles}, true, 2>(")));
        }
    }

    /// 2026-10-05: The N tile fills a wave: the 64-column grid when it reaches `sm_count` CTAs,
    /// else the 32-column one. The 27B's down (N = 5120) takes 32 on a 132-SM class and 64 on a
    /// 48-SM one; gate/up (N = 17408) and the vocab take 64 on both.
    #[test]
    fn the_n_tile_fills_a_wave() {
        assert_eq!(w4a16_tc_rows_cols(5120, 132), 32);
        assert_eq!(w4a16_tc_rows_cols(5120, 48), 64);
        assert_eq!(w4a16_tc_rows_cols(17408, 132), 64);
        assert_eq!(w4a16_tc_rows_cols(248320, 132), 64);
        assert_eq!(w4a16_tc_rows_cols(64 * 131 + 1, 132), 64);
        assert_eq!(w4a16_tc_rows_cols(64 * 131, 132), 32);
    }

    /// 2026-10-02: The shape contract refuses each bound it names and admits the 35B head.
    #[test]
    fn shape_contract() {
        assert!(w4a16_tc_rows_shape_ok(1, 248320, 2048, 2048, 248320));
        assert!(w4a16_tc_rows_shape_ok(64, 248320, 2048, 2048, 248320));
        assert!(!w4a16_tc_rows_shape_ok(0, 248320, 2048, 2048, 248320));
        assert!(!w4a16_tc_rows_shape_ok(65, 248320, 2048, 2048, 248320));
        // 2026-10-02: Ragged N: the checkpoint's 248070-entry vocab.
        assert!(w4a16_tc_rows_shape_ok(8, 248070, 2048, 2048, 248070));
        assert!(!w4a16_tc_rows_shape_ok(8, 248320, 2048 + 128, 2176, 248320));
        assert!(!w4a16_tc_rows_shape_ok(8, 248320, 2048, 2044, 248320));
        assert!(!w4a16_tc_rows_shape_ok(8, 248320, 2048, 2048, 248319));
    }

    /// 2026-10-09: Every entry the launcher can name exists in the kernel file, and a `_pf` point
    /// is a 64-column (4-warp) entry of the same row tier.
    #[test]
    fn every_routed_entry_is_compiled() {
        for m in [1u32, 16, 17, 32, 33, 64] {
            for wide in [true, false] {
                for pf in 1..=3 {
                    let e = w4a16_tc_rows_entry(m, wide, pf);
                    assert!(
                        CU.contains(&format!("{e}(")) || CU.contains(&format!("({e},")),
                        "{e}"
                    );
                    let tier = if m <= 16 {
                        "16"
                    } else if m <= 32 {
                        "32"
                    } else {
                        "64"
                    };
                    assert!(
                        e.starts_with(&format!("w4a16_tc_rows_{tier}")),
                        "{e} for m={m}"
                    );
                    assert_eq!(e.ends_with("_w2"), !wide, "{e}");
                }
            }
        }
        assert_eq!(w4a16_tc_rows_entry(64, true, 1), "w4a16_tc_rows_64");
        assert_eq!(w4a16_tc_rows_entry(16, true, 3), "w4a16_tc_rows_16_pf2");
    }
}
