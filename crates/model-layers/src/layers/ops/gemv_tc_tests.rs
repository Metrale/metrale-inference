// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the tensor-core routing decision of the narrow NVFP4 batched GEMV.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;

// 2026-09-25: The (N, K) projection shapes of Qwen3.8-27B (hidden 5120, intermediate 17408,
// 24 q heads and 4 kv heads of 256, 16 GDN key heads and 48 value heads of 128); each must
// route to a tensor-core entry.
const SHAPES_27B: [(u32, u32); 8] = [
    (16384, 5120),  // 2026-09-25: GDN in_proj qkvz
    (12288, 5120),  // 2026-09-25: attention q + gate
    (1024, 5120),   // 2026-09-25: attention k / v
    (5120, 6144),   // 2026-09-25: attention o / GDN out_proj
    (34816, 5120),  // 2026-09-25: FFN gate+up
    (5120, 17408),  // 2026-09-25: FFN down
    (248320, 5120), // 2026-09-25: lm_head at the checkpoint's vocab_size
    (248077, 5120), // 2026-09-25: lm_head at the tokenizer's 248077 ids (odd)
];

#[test]
fn every_27b_shape_routes_at_every_row_count() {
    for (n, k) in SHAPES_27B {
        for m in 1..=8 {
            assert_eq!(
                tc_route(m, n, k, true, true, true),
                Some(TcKind::M8),
                "m={m} n={n} k={k}"
            );
        }
        for m in 9..=16 {
            assert_eq!(
                tc_route(m, n, k, true, true, true),
                Some(TcKind::M16),
                "m={m} n={n} k={k}"
            );
        }
    }
}

#[test]
fn above_sixteen_rows_declines() {
    // 2026-09-25: tc16 stores rows below 16 only, so a 17th row would never be written.
    assert_eq!(tc_route(17, 5120, 5120, true, true, true), None);
}

#[test]
fn k_tail_declines_but_any_n_routes() {
    // 2026-09-25: The kernel walks K in 128-element blocks, so a K tail would be dropped.
    assert_eq!(tc_route(4, 5120, 5120 + 64, true, true, true), None);
    // 2026-09-25: An odd N routes; the kernel guards the partial last tile.
    assert_eq!(
        tc_route(4, 248077, 5120, true, true, true),
        Some(TcKind::M8)
    );
    assert_eq!(
        tc_route(12, 248077, 5120, true, true, true),
        Some(TcKind::M16)
    );
}

#[test]
fn degenerate_launches_decline() {
    assert_eq!(tc_route(0, 5120, 5120, true, true, true), None);
    assert_eq!(tc_route(4, 0, 5120, true, true, true), None);
    assert_eq!(tc_route(4, 5120, 0, true, true, true), None);
}

#[test]
fn kill_switch_declines() {
    assert_eq!(tc_route(4, 5120, 5120, false, true, true), None);
}

#[test]
fn missing_entries_fall_back_correctly() {
    // 2026-09-25: Without tc8, narrow rows still route to tc16, which covers M <= 16.
    assert_eq!(
        tc_route(4, 5120, 5120, true, false, true),
        Some(TcKind::M16)
    );
    assert_eq!(tc_route(12, 5120, 5120, true, true, false), None);
    assert_eq!(tc_route(4, 5120, 5120, true, false, false), None);
}

#[test]
fn grid_covers_every_column() {
    for (n, _) in SHAPES_27B {
        for kind in [TcKind::M8, TcKind::M16] {
            let ctas = n.div_ceil(kind.cols_per_cta());
            assert!(ctas * kind.cols_per_cta() >= n);
            assert!((ctas - 1) * kind.cols_per_cta() < n, "no empty CTA");
        }
    }
}

/// 2026-10-09: Every schedule point a class can declare is an entry of the kernel file with the
/// tier's MT, and the NT the launcher sizes its grid with (`w4a16_gemv_tc_nt`) is the NT the entry
/// was instantiated with: a mismatch would leave columns unwritten or write past N.
#[test]
fn every_schedule_point_is_compiled_with_the_nt_its_grid_assumes() {
    const CU: &str = include_str!("../../../../../kernels/gb10/common/w4a16_gemv_tc.cu");
    let entries: Vec<(String, u32, u32)> = CU
        .lines()
        .filter_map(|l| l.trim().strip_prefix("W4TC_ENTRY(w4a16_gemv_"))
        .map(|rest| {
            let f: Vec<&str> = rest.split(',').map(str::trim).collect();
            (
                f[0].to_string(),
                f[1].parse().unwrap(),
                f[2].parse().unwrap(),
            )
        })
        .collect();
    assert!(entries.len() >= 2, "the kernel file's entries: {entries:?}");
    let points = metrale_kernels::w4a16_gemv_tc_entries::W4A16_GEMV_TC_POINTS;
    for (tier, mt) in [(0usize, 8u32), (1, 16)] {
        for p in points[tier] {
            let (_, emt, ent) = entries
                .iter()
                .find(|(name, _, _)| name == p)
                .unwrap_or_else(|| panic!("point {p} has no W4TC_ENTRY"));
            assert_eq!(*emt, mt, "{p}: MT");
            assert_eq!(
                *ent,
                metrale_kernels::w4a16_gemv_tc_entries::w4a16_gemv_tc_nt(p),
                "{p}: NT"
            );
        }
    }
    let wide = metrale_kernels::w4a16_gemv_tc_entries::W4A16_GEMV_TC_WIDE_POINTS;
    for (tier, mt) in [(0usize, 32u32), (1, 64)] {
        for p in wide[tier] {
            let (_, emt, ent) = entries
                .iter()
                .find(|(name, _, _)| name == p)
                .unwrap_or_else(|| panic!("wide point {p} has no W4TC_ENTRY"));
            assert_eq!(*emt, mt, "{p}: MT");
            assert_eq!(
                *ent,
                metrale_kernels::w4a16_gemv_tc_entries::w4a16_gemv_tc_nt(p),
                "{p}: NT"
            );
        }
    }
    assert_eq!(
        8 * metrale_kernels::w4a16_gemv_tc_entries::w4a16_gemv_tc_nt("tc8"),
        TC8_COLS_PER_CTA
    );
    assert_eq!(
        8 * metrale_kernels::w4a16_gemv_tc_entries::w4a16_gemv_tc_nt("tc16"),
        TC16_COLS_PER_CTA
    );
}
