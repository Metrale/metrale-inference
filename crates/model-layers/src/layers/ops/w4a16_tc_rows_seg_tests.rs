// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The segmented, K-split row-tile launch: its contract against the kernel source,
//! the split rule at GLM-5.3's shapes, and the launch it makes on a recording mock backend.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

use super::*;

const CU: &str = include_str!("../../../../../kernels/gb10/common/w4a16_tc_rows_seg.cu");
const ONE: &str = include_str!("../../../../../kernels/gb10/common/w4a16_tc_rows.cu");
const ROWS: &str = include_str!("../../../../../kernels/gb10/common/tc_rows.cuh");

/// 2026-10-09: Each row entry runs `w4a16_tc_rows_<R>`'s (NT, G) and policy, which is what makes
/// S = 1 byte-identical to it; the splits and the split unit are the host's; the segment count
/// is the kernel's argument list.
#[test]
fn launch_matches_the_kernel() {
    for (r, nt, g) in [(16, 2, 2), (32, 4, 1), (64, 8, 1)] {
        assert!(
            ONE.contains(&format!(
                "tr_block<Nvfp4G16, {nt}, {g}, true>(A, {{packed, scale, s2}}, C, M, N, K, lda, ldc, blockIdx.x);"
            )),
            "w4a16_tc_rows_{r} is no longer ({nt}, {g})"
        );
        assert!(
            CU.contains(&format!("W4A16_SEG_SPLITS({r}, {nt}, {g})")),
            "seg_{r} is not ({nt}, {g})"
        );
    }
    assert!(CU.contains("tr_block_split<Nvfp4G16, NT, G, true, S, typename TrSegMerge<S>::T>("));
    assert!(ROWS.contains(
        "    tr_block_split<P, NT, G, RAGGED, 1, TrWhole>(A, W, C, M, N, K, lda, ldc, col_block, 0u);"
    ));
    for s in W4A16_TC_ROWS_SPLITS.into_iter().filter(|&s| s > 1) {
        assert!(
            CU.contains(&format!(
                "W4A16_SEG_ENTRY(R, NT, G, {s}, __cluster_dims__(1, {s}, 1))"
            )),
            "split {s}"
        );
    }
    assert!(CU.contains("W4A16_SEG_ENTRY(R, NT, G, 1, )"));
    assert!(ROWS.contains(&format!("#define TR_SPLIT_K {W4A16_TC_ROWS_SPLIT_K}\n")));
    assert_eq!(CU.matches("const unsigned char* p2,").count(), 2);
    assert_eq!(W4A16_TC_ROWS_SEG_MAX, 3);
}

/// 2026-10-09: Twelve distinct entries; each takes `w4a16_tc_rows_entry`'s row tier.
#[test]
fn entries_follow_the_row_tiers() {
    let all: std::collections::BTreeSet<_> = w4a16_tc_rows_seg_entries().collect();
    assert_eq!(all.len(), 9);
    for m in [1, 2, 16, 17, 32, 33, 64] {
        let rows = &super::super::w4a16_tc_rows_entry(m)["w4a16_tc_rows_".len()..];
        for s in W4A16_TC_ROWS_SPLITS {
            assert_eq!(
                w4a16_tc_rows_seg_entry(m, s).unwrap(),
                format!("w4a16_tc_rows_seg_{rows}_k{s}")
            );
        }
    }
    assert_eq!(w4a16_tc_rows_seg_entry(8, 3), None);
    assert_eq!(w4a16_tc_rows_seg_entry(8, 16), None);
}

/// 2026-10-09: The contract refuses each bound it names.
#[test]
fn shape_contract() {
    assert!(w4a16_tc_rows_seg_shape_ok(
        16,
        4096,
        4096,
        &[2816, 2816, 2816],
        1
    ));
    assert!(w4a16_tc_rows_seg_shape_ok(
        64,
        4096,
        4096,
        &[128, 128, 22],
        4
    ));
    assert!(
        !w4a16_tc_rows_seg_shape_ok(64, 4096, 4096, &[128], 8),
        "no split of 8"
    );
    assert!(w4a16_tc_rows_seg_shape_ok(2, 768, 768, &[4096], 2));
    assert!(
        !w4a16_tc_rows_seg_shape_ok(2, 768, 768, &[4096], 4),
        "3 units, 4 splits"
    );
    assert!(!w4a16_tc_rows_seg_shape_ok(0, 4096, 4096, &[128], 1));
    assert!(!w4a16_tc_rows_seg_shape_ok(65, 4096, 4096, &[128], 1));
    assert!(!w4a16_tc_rows_seg_shape_ok(8, 4096, 4096, &[], 1));
    assert!(!w4a16_tc_rows_seg_shape_ok(8, 4096, 4096, &[128; 4], 1));
    assert!(!w4a16_tc_rows_seg_shape_ok(8, 4096, 4096, &[128, 0], 1));
    assert!(!w4a16_tc_rows_seg_shape_ok(8, 4096 + 128, 4224, &[128], 1));
    assert!(!w4a16_tc_rows_seg_shape_ok(8, 4096, 4096, &[128], 3));
    assert!(!w4a16_tc_rows_seg_shape_ok(8, 4096, 4092, &[128], 1));
    assert!(!w4a16_tc_rows_seg_shape_ok(8, 4096, 4100, &[128], 1));
}

/// 2026-10-09: On GB10's 48 SMs the rule gives the measured best split at GLM-5.3's TP=3 shapes;
/// a split never exceeds K's units, and more tiles never get a wider split.
#[test]
fn the_split_rule_at_glm_shapes() {
    let split = |ns: &[u32], k| w4a16_tc_rows_split_for(w4a16_tc_rows_seg_tiles(ns), k, 48);
    assert_eq!(split(&[2816], 4096), 1);
    assert_eq!(split(&[2816, 2816, 2816], 4096), 1);
    assert_eq!(split(&[4096], 2816), 1);
    assert_eq!(split(&[4096], 768), 1);
    assert_eq!(split(&[768], 4096), 4);
    assert_eq!(split(&[768, 768], 4096), 2);
    assert_eq!(split(&[128], 4096), 4);
    assert_eq!(split(&[128, 128, 22], 4096), 4);
    assert_eq!(split(&[22], 4096), 4);
    assert_eq!(split(&[512], 4096), 4);
    assert_eq!(w4a16_tc_rows_split_for(1, 512, 48), 2, "two units of K");
    assert_eq!(w4a16_tc_rows_split_for(1, 256, 48), 1);
    let mut last = u32::MAX;
    for tiles in 1..200 {
        let s = w4a16_tc_rows_split_for(tiles, 4096, 48);
        assert!(s <= last && W4A16_TC_ROWS_SPLITS.contains(&s), "{tiles}");
        last = s;
    }
}

fn bytes(a: &MockArg) -> &[u8] {
    match a {
        MockArg::Bytes(b) => b,
        MockArg::Buffer(p) => panic!("expected bytes, got buffer {p}"),
    }
}

fn ptr(a: &MockArg) -> DevicePtr {
    match a {
        MockArg::Buffer(p) => *p,
        MockArg::Bytes(_) => panic!("expected a buffer"),
    }
}

/// 2026-10-09: Two segments: grid (tiles, S), the segments' pointers, scales and widths in the
/// kernel's order, the third slot empty, then M, K, lda; a refused shape launches nothing.
#[test]
fn the_launch_carries_each_segment() {
    let gpu = MockGpuBackend::new();
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
            n: 128,
        },
        W4a16Seg {
            weight: w(0x2000, 0.25),
            output: DevicePtr(0xA000),
            n: 22,
        },
    ];
    let a = DevicePtr(0x5000);
    w4a16_tc_rows_seg(&gpu, a, &segs, 9, 4096, 4096, 4, 3).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    let l = &l[0];
    assert_eq!((l.grid, l.block, l.stream), ([3, 4, 1], [128, 1, 1], 3));
    assert_eq!(l.args.len(), 19);
    assert_eq!(ptr(&l.args[0]), a);
    for (i, g) in segs.iter().enumerate() {
        let b = 1 + 5 * i;
        assert_eq!(ptr(&l.args[b]), g.weight.weight);
        assert_eq!(ptr(&l.args[b + 1]), g.weight.weight_scale);
        assert_eq!(bytes(&l.args[b + 2]), g.weight.weight_scale_2.to_le_bytes());
        assert_eq!(ptr(&l.args[b + 3]), g.output);
        assert_eq!(bytes(&l.args[b + 4]), g.n.to_le_bytes());
    }
    assert_eq!(ptr(&l.args[11]), DevicePtr::NULL);
    assert_eq!(bytes(&l.args[15]), 0u32.to_le_bytes());
    for (i, v) in [9u32, 4096, 4096].into_iter().enumerate() {
        assert_eq!(bytes(&l.args[16 + i]), v.to_le_bytes());
    }
    assert!(gpu.kernel_lookups_snapshot().contains(&(
        W4A16_TC_ROWS_SEG_MODULE.into(),
        "w4a16_tc_rows_seg_16_k4".into()
    )));
    assert!(
        w4a16_tc_rows_seg(&gpu, a, &segs, 9, 768, 768, 4, 3).is_err(),
        "3 units of K, 4 splits"
    );
    assert_eq!(gpu.launches_snapshot().len(), 1);
}
