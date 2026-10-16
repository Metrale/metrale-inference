// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The narrow unsplit segmented row tiles: their names and tiles against the kernel
//! source, the warps rule at GLM-5.3's TP=3 groups, and the launch on a recording mock backend
//! against `w4a16_tc_rows_seg`'s.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::super::{w4a16_tc_rows_seg, w4a16_tc_rows_seg_tiles};
use super::*;
use crate::weight_map::QuantizedWeight;

const CU: &str = include_str!("../../../../../kernels/gb10/common/w4a16_tc_rows_seg.cu");

/// 2026-10-16: Each name is an entry the module defines with the tile its rows need (NT = 1 for
/// 8 rows, 2 for 16, G = 2 as the 16-row entry) on the block it is launched with; the 4-warp
/// 16-row point is the existing entry, whose (NT, G) the narrow 16-row points repeat.
#[test]
fn entries_match_the_kernel() {
    assert!(CU.contains("W4A16_SEG_SPLITS(16, 2, 2)"));
    assert!(CU.contains(
        "extern \"C\" __global__ void __launch_bounds__(32 * NW) w4a16_tc_rows_seg_##R##_k1_w##NW("
    ));
    assert!(CU.contains("w4a16_seg_block<NT, G, 1, NW>("));
    assert!(CU.contains("constexpr unsigned int COLS = 16 * NW;"));
    let mut n = 0;
    for m in [8u32, 16] {
        for nw in W4A16_TC_ROWS_SEG_WARPS {
            let name = w4a16_tc_rows_seg_narrow_entry(m, nw).unwrap();
            if m == 16 && nw == 4 {
                assert_eq!(name, "w4a16_tc_rows_seg_16_k1");
                continue;
            }
            assert_eq!(name, format!("w4a16_tc_rows_seg_{m}_k1_w{nw}"));
            let nt = m / 8;
            assert!(
                CU.contains(&format!("W4A16_SEG_NARROW_ENTRY({m}, {nt}, 2, {nw})\n")),
                "{name}"
            );
            n += 1;
        }
    }
    assert_eq!(
        CU.matches("W4A16_SEG_NARROW_ENTRY(").count(),
        n + 1,
        "an entry the host never names"
    );
    let all: std::collections::BTreeSet<_> = w4a16_tc_rows_seg_narrow_entries().collect();
    assert_eq!(all.len(), 6);
}

/// 2026-10-16: 1..=8 rows take the 8-row tile, 9..=16 the 16-row one, at every warps count;
/// 0 and 17+ rows and a warps count the module lacks have no entry.
#[test]
fn entry_by_rows_and_warps() {
    for nw in W4A16_TC_ROWS_SEG_WARPS {
        for m in 1..=8 {
            assert!(
                w4a16_tc_rows_seg_narrow_entry(m, nw)
                    .unwrap()
                    .starts_with("w4a16_tc_rows_seg_8_k1")
            );
        }
        for m in 9..=16 {
            assert!(
                w4a16_tc_rows_seg_narrow_entry(m, nw)
                    .unwrap()
                    .starts_with("w4a16_tc_rows_seg_16_k1")
            );
        }
        assert_eq!(w4a16_tc_rows_seg_narrow_entry(0, nw), None);
        assert_eq!(w4a16_tc_rows_seg_narrow_entry(17, nw), None);
    }
    for nw in [0, 3, 8] {
        assert_eq!(w4a16_tc_rows_seg_narrow_entry(4, nw), None, "{nw} warps");
    }
}

/// 2026-10-16: At 4 warps the blocks are the 64-column tiles; the rule on 48 SMs at GLM-5.3's
/// TP=3 groups (KDA q/k/v, f_a/g_a/b, o, shared gate/up and down), fill 1 and 2; more fill never
/// gets wider blocks, and a group too narrow for any width gets 1 warp.
#[test]
fn the_warps_rule_at_glm_groups() {
    let qkv: &[u32] = &[2816, 2816, 2816];
    let fgb: &[u32] = &[128, 128, 22];
    let gate_up: &[u32] = &[768, 768];
    let wide: &[u32] = &[4096];
    for ns in [qkv, fgb, gate_up, wide] {
        assert_eq!(w4a16_tc_rows_seg_blocks(ns, 4), w4a16_tc_rows_seg_tiles(ns));
    }
    assert_eq!(w4a16_tc_rows_seg_blocks(fgb, 1), 8 + 8 + 2);
    assert_eq!(w4a16_tc_rows_seg_blocks(&[22], 2), 1);
    let rule = |ns, fill| w4a16_tc_rows_seg_warps_for(ns, fill, 48);
    assert_eq!(
        [rule(qkv, 1), rule(fgb, 1), rule(gate_up, 1), rule(wide, 1)],
        [4, 1, 2, 4]
    );
    assert_eq!(
        [rule(qkv, 2), rule(fgb, 2), rule(gate_up, 2), rule(wide, 2)],
        [4, 1, 1, 2]
    );
    assert_eq!(
        [rule(qkv, 3), rule(fgb, 3), rule(gate_up, 3), rule(wide, 3)],
        [2, 1, 1, 1]
    );
    for ns in [qkv, fgb, gate_up, wide, &[22][..]] {
        let mut last = u32::MAX;
        for fill in 1..=8 {
            let nw = rule(ns, fill);
            assert!(
                nw <= last && W4A16_TC_ROWS_SEG_WARPS.contains(&nw),
                "{ns:?} fill {fill}"
            );
            last = nw;
        }
    }
}

/// 2026-10-16: A narrow launch passes exactly `w4a16_tc_rows_seg`'s arguments (one helper writes
/// both) on grid (blocks, 1, 1) and block 32 NW, on the entry of its rows and warps; a shape or
/// warps count the module refuses launches nothing.
#[test]
fn the_launch_is_the_unsplit_launch_on_narrower_blocks() {
    let w = |base: u64, s2: f32| QuantizedWeight {
        weight: DevicePtr(base),
        weight_scale: DevicePtr(base + 0x100),
        weight_scale_2: s2,
        ..QuantizedWeight::null()
    };
    let segs = [
        W4a16Seg {
            weight: w(0x1000, 0.5),
            output: DevicePtr(0x9000),
            n: 768,
        },
        W4a16Seg {
            weight: w(0x2000, 0.25),
            output: DevicePtr(0xA000),
            n: 768,
        },
    ];
    let a = DevicePtr(0x5000);
    for (m, nw, entry, blocks) in [
        (6, 2, "w4a16_tc_rows_seg_8_k1_w2", 48),
        (12, 1, "w4a16_tc_rows_seg_16_k1_w1", 96),
        (16, 4, "w4a16_tc_rows_seg_16_k1", 24),
        (1, 4, "w4a16_tc_rows_seg_8_k1_w4", 24),
    ] {
        let unsplit = MockGpuBackend::new();
        w4a16_tc_rows_seg(&unsplit, a, &segs, m, 4096, 4096, 1, 5).unwrap();
        let gpu = MockGpuBackend::new();
        w4a16_tc_rows_seg_narrow(&gpu, a, &segs, m, 4096, 4096, nw, 5).unwrap();
        let (l, u) = (gpu.launches_snapshot(), unsplit.launches_snapshot());
        assert_eq!(l.len(), 1);
        assert_eq!(
            (l[0].grid, l[0].block, l[0].stream),
            ([blocks, 1, 1], [32 * nw, 1, 1], 5)
        );
        assert_eq!(
            (&l[0].args, l[0].shared_mem),
            (&u[0].args, u[0].shared_mem),
            "m={m} warps={nw}"
        );
        assert_eq!(
            gpu.kernel_lookups_snapshot(),
            vec![(W4A16_TC_ROWS_SEG_MODULE.to_string(), entry.to_string())]
        );
    }
    let gpu = MockGpuBackend::new();
    assert!(w4a16_tc_rows_seg_narrow(&gpu, a, &segs, 17, 4096, 4096, 4, 5).is_err());
    assert!(w4a16_tc_rows_seg_narrow(&gpu, a, &segs, 0, 4096, 4096, 4, 5).is_err());
    assert!(w4a16_tc_rows_seg_narrow(&gpu, a, &segs, 8, 4096, 4096, 3, 5).is_err());
    assert!(w4a16_tc_rows_seg_narrow(&gpu, a, &segs, 8, 4000, 4096, 2, 5).is_err());
    assert!(gpu.launches_snapshot().is_empty());
}
