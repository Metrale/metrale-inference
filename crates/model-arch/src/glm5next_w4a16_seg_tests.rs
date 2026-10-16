// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `METRALE_GLM_W4A16_SEG_FILL`: its parse, the launches it moves to the narrow
//! unsplit entries at GLM-5.3's TP=3 groups, and the launches it leaves alone, on a recording
//! mock backend.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_layers::weight_map::QuantizedWeight;

use super::*;

const FUSE: SegPlan = SegPlan {
    mode: SegMode::Fuse,
    sms: 48,
    gemv_rows: 0,
    one_row: false,
    fill: 0,
};
const QKV: [u32; 3] = [2816, 2816, 2816];
const FGB: [u32; 3] = [128, 128, 22];
const GATE_UP: [u32; 2] = [768, 768];

/// 2026-10-16: `0`/unset off; 1..=8 under `fuse` or `split`; anything else, or any fill with
/// the segmented mode off, refused.
#[test]
fn the_fill_lever_parses_and_needs_the_segmented_mode() {
    for mode in [SegMode::Off, SegMode::Fuse, SegMode::Split] {
        assert_eq!(parse_seg_fill(None, mode), Ok(0));
        assert_eq!(parse_seg_fill(Some("0"), mode), Ok(0));
    }
    for mode in [SegMode::Fuse, SegMode::Split] {
        assert_eq!(parse_seg_fill(Some("1"), mode), Ok(1));
        assert_eq!(parse_seg_fill(Some("8"), mode), Ok(SEG_FILL_MAX));
        for bad in ["9", "-1", "", "x", "1.5", " 1"] {
            assert!(parse_seg_fill(Some(bad), mode).is_err(), "{bad:?}");
        }
    }
    let e = parse_seg_fill(Some("2"), SegMode::Off).unwrap_err();
    assert!(e.contains("needs METRALE_GLM_W4A16_SEG"), "{e}");
}

/// 2026-10-16: No fill, 0 or 17+ rows, or a K-split launch: no narrow warps. Under a fill the
/// choice is the same at every row count 1..=16 (graph replays need one per shape) and is the
/// ops rule's: on 48 SMs fill 1 moves f_a/g_a/b to 1 warp and the shared gate/up to 2.
#[test]
fn narrow_warps_only_for_unsplit_launches_of_up_to_16_rows() {
    let fill1 = SegPlan { fill: 1, ..FUSE };
    for m in [0, 17, 64] {
        assert_eq!(fill1.narrow_warps(&QKV, m, 4096), None, "{m} rows");
    }
    for m in 1..=16 {
        assert_eq!(FUSE.narrow_warps(&GATE_UP, m, 4096), None, "no fill");
        assert_eq!(fill1.narrow_warps(&QKV, m, 4096), Some(4));
        assert_eq!(fill1.narrow_warps(&FGB, m, 4096), Some(1));
        assert_eq!(fill1.narrow_warps(&GATE_UP, m, 4096), Some(2));
        assert_eq!(fill1.narrow_warps(&[4096], m, 768), Some(4));
    }
    let fill2 = SegPlan { fill: 2, ..fill1 };
    assert_eq!(fill2.narrow_warps(&[4096], 8, 2816), Some(2));
    assert_eq!(fill2.narrow_warps(&GATE_UP, 8, 4096), Some(1));
    let split = SegPlan {
        mode: SegMode::Split,
        ..fill1
    };
    assert_eq!(split.split(&FGB, 4096), 4);
    assert_eq!(
        split.narrow_warps(&FGB, 8, 4096),
        None,
        "a K-split launch keeps its entry"
    );
    assert_eq!(
        split.narrow_warps(&QKV, 8, 4096),
        Some(4),
        "split 1 at q/k/v"
    );
}

fn segs(ns: &[u32]) -> Vec<W4a16Seg> {
    ns.iter()
        .enumerate()
        .map(|(i, &n)| W4a16Seg {
            weight: QuantizedWeight {
                weight: DevicePtr(0x1000 * (i as u64 + 1)),
                weight_scale: DevicePtr(0x1000 * (i as u64 + 1) + 0x100),
                weight_scale_2: 0.5,
                ..QuantizedWeight::null()
            },
            output: DevicePtr(0x10_0000 * (i as u64 + 1)),
            n,
        })
        .collect()
}

/// 2026-10-16: Under fill 1 a launch of 6 rows of the shared gate/up is one launch of the 8-row
/// 2-warp entry on 48 blocks of 64 threads, with exactly the arguments the fill-0 launch passes
/// (which stays the 4-warp `w4a16_tc_rows_seg_16_k1` on 24 blocks); q/k/v at 12 rows keep 4
/// warps; 17 rows, and f_a/g_a/b under `split`, launch as without the fill.
#[test]
fn the_fill_moves_only_unsplit_launches_of_up_to_16_rows() {
    let a = DevicePtr(0x5000);
    let fill1 = SegPlan { fill: 1, ..FUSE };
    let run = |plan: SegPlan, ns: &[u32], m: usize| {
        let gpu = MockGpuBackend::new();
        launch(&gpu, &plan, a, &segs(ns), m, 4096, 3).unwrap();
        (gpu.launches_snapshot(), gpu.kernel_lookups_snapshot())
    };
    let entry = |l: &[(String, String)]| l.iter().map(|(_, e)| e.clone()).collect::<Vec<_>>();
    let cases: [(SegPlan, &[u32], usize, [u32; 3], u32, &str); 5] = [
        (
            FUSE,
            &GATE_UP,
            6,
            [24, 1, 1],
            128,
            "w4a16_tc_rows_seg_16_k1",
        ),
        (
            fill1,
            &GATE_UP,
            6,
            [48, 1, 1],
            64,
            "w4a16_tc_rows_seg_8_k1_w2",
        ),
        (fill1, &QKV, 12, [132, 1, 1], 128, "w4a16_tc_rows_seg_16_k1"),
        (
            fill1,
            &GATE_UP,
            17,
            [24, 1, 1],
            128,
            "w4a16_tc_rows_seg_32_k1",
        ),
        (
            SegPlan {
                mode: SegMode::Split,
                ..fill1
            },
            &FGB,
            6,
            [5, 4, 1],
            128,
            "w4a16_tc_rows_seg_16_k4",
        ),
    ];
    for (plan, ns, m, grid, threads, name) in cases {
        let (l, k) = run(plan, ns, m);
        assert_eq!(l.len(), 1, "{ns:?} m={m}");
        assert_eq!(
            (l[0].grid, l[0].block),
            (grid, [threads, 1, 1]),
            "{ns:?} m={m}"
        );
        assert_eq!(entry(&k), vec![name.to_string()], "{ns:?} m={m}");
        let (off, _) = run(FUSE, ns, m);
        if plan.mode == SegMode::Fuse {
            assert_eq!(l[0].args, off[0].args, "{ns:?} m={m}: the same arguments");
        }
    }
}
