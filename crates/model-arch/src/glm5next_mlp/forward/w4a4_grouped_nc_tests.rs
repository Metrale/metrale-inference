// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The nc lever's launch plan (`METRALE_GLM_MOE_W4A4_NC`) on the recording mock: at
//! the m128 tile the down launch becomes its `_n` twin with the plain arguments plus the down
//! raster group (0 without one) and `nc`, over a grid `ceil(ceil(N / 64) / nc)` wide; everything
//! else (the fused or separate gate/up, the narrow tiles) is the plan without the lever, argument
//! for argument. The bytes are the GPU bench's job (`examples/glm5next_moe_epi_bench`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::super::super::w4a4_grouped_nc::{check_ready, parse_nc, readback};
use super::*;

const G: RasterGroups = RasterGroups { gu: 16, down: 4 };

/// 2026-10-15: `got` is `want` (a plain or `_r` down) as an `_n` twin: the plain entry's 13
/// arguments, then `raster` and `nc`; grid x `ceil(want's x / nc)`, the rest unchanged.
fn assert_nc(got: &MockLaunch, want: &MockLaunch, raster: u32, nc: u32) {
    assert_eq!(got.args.len(), 15);
    assert_eq!(&got.args[..13], &want.args[..13]);
    assert_eq!((u32_arg(got, 13), u32_arg(got, 14)), (raster, nc));
    let plain_x = 4096u32.div_ceil(64);
    assert_eq!(want.grid[0], plain_x);
    assert_eq!(
        (got.grid[0], got.grid[1], got.grid[2]),
        (plain_x.div_ceil(nc), want.grid[1], want.grid[2])
    );
    assert_eq!(
        (got.block, got.shared_mem, got.stream),
        (want.block, want.shared_mem, want.stream)
    );
}

/// 2026-10-15: The tp layout (704-wide slices, the `_k64` down) at the m128 tile, fused, with the
/// default down raster 4: only the down changes, to `mx_m128_k64_n` carrying group 4 and `nc`, at
/// a divisor (4), a non-divisor (3) and one past the 64 column tiles (a single CTA column).
#[test]
fn the_m128_k64_down_takes_its_nc_twin_with_the_raster_group() {
    let base = plan(cfg_tp(), M128_ROWS, true, Some(G), None);
    assert_eq!(funcs(&base), vec![SORT, TILES, QUANT, RGU128, R128_K64]);
    for nc in [1, 3, 4, 65] {
        let n = plan(cfg_tp(), M128_ROWS, true, Some(G), Some(nc));
        assert_eq!(funcs(&n), vec![SORT, TILES, QUANT, RGU128, N128_K64]);
        assert_same(&n[..4], &base[..4]);
        assert_nc(&n[4], &base[4], 4, nc);
    }
}

/// 2026-10-15: Without a down raster group the twin's raster argument is 0 (row-major), and the
/// plain-width (2048) down takes `mx_m128_n`.
#[test]
fn without_a_raster_group_the_twin_walks_row_major() {
    let base = plan(cfg_tp(), M128_ROWS, true, None, None);
    let n = plan(cfg_tp(), M128_ROWS, true, None, Some(8));
    assert_eq!(funcs(&n), vec![SORT, TILES, QUANT, GU128, N128_K64]);
    assert_same(&n[..4], &base[..4]);
    assert_nc(&n[4], &base[4], 0, 8);
    let off = RasterGroups { gu: 16, down: 0 };
    let base = plan(cfg_small(), 48, true, Some(off), None);
    let n = plan(cfg_small(), 48, true, Some(off), Some(2));
    assert_eq!(funcs(&n), vec![SORT, TILES, QUANT, RGU128, N128]);
    assert_same(&n[..4], &base[..4]);
    assert_nc(&n[4], &base[4], 0, 2);
}

/// 2026-10-15: The separate gate and up GEMMs, the SwiGLU and the quantizer keep their entries;
/// below the m128 tile nothing changes.
#[test]
fn the_separate_gate_up_and_the_narrow_tiles_keep_their_entries() {
    let base = plan(cfg_tp(), M128_ROWS, false, Some(G), None);
    let n = plan(cfg_tp(), M128_ROWS, false, Some(G), Some(4));
    assert_eq!(
        funcs(&n),
        vec![SORT, TILES, QUANT, G128, G128, SWIGLU, QUANT_K64, N128_K64]
    );
    assert_same(&n[..7], &base[..7]);
    assert_nc(&n[7], &base[7], 4, 4);
    let narrow = plan(cfg_tp(), 32, true, Some(G), None);
    assert_same(&plan(cfg_tp(), 32, true, Some(G), Some(4)), &narrow);
    assert_eq!(funcs(&narrow), vec![SORT, TILES, QUANT, GU32, G32_K64]);
}

/// 2026-10-15: The lever parses strictly; on the grouped path it needs both `_n` entries.
#[test]
fn the_nc_lever_parses_strictly_and_needs_its_entries() {
    assert_eq!(parse_nc(None), Ok(None));
    assert_eq!(parse_nc(Some("0")), Ok(None));
    assert_eq!(parse_nc(Some("1")), Ok(Some(1)));
    assert_eq!(parse_nc(Some("4096")), Ok(Some(4096)));
    for bad in ["4097", "-1", "2,4", " 4", "4 ", "x", ""] {
        let e = parse_nc(Some(bad)).unwrap_err();
        assert!(e.contains("METRALE_GLM_MOE_W4A4_NC"), "{bad}: {e}");
    }
    assert_eq!(check_ready(true, true), Ok(()));
    assert_eq!(check_ready(false, false), Ok(()));
    let e = check_ready(true, false).unwrap_err();
    assert!(e.contains("_n"), "{e}");
}

/// 2026-10-15: The readback says `active` only on the grouped path, and names `nc`.
#[test]
fn the_nc_readback_names_the_tiles_and_whether_they_run() {
    assert_eq!(
        readback(Some(4), true),
        "METRALE_GLM_MOE_W4A4_NC: active, 4 column tiles per m128 down CTA"
    );
    assert_eq!(
        readback(Some(4), false),
        "METRALE_GLM_MOE_W4A4_NC: dormant (METRALE_GLM_MOE_W4A4_GROUPED off), 4 column tiles per m128 down CTA"
    );
    assert_eq!(readback(None, true), "METRALE_GLM_MOE_W4A4_NC: off");
}
