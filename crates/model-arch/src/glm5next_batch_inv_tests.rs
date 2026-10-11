// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `METRALE_GLM_BATCH_INVARIANT`: its parse, the levers it refuses, and the all-reduce
//! slices.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-11: The levers the lever needs (W4A4 grouped ready, nothing else on).
const OK: Levers = Levers {
    gemv_seg: false,
    gemv_rows: 0,
    dequant_gemm: false,
    w4a4_grouped_ready: true,
    dsa_batch_qidx: false,
    kda_chunked: false,
};

/// 2026-10-11: 0/unset off, 1 on, anything else refused.
#[test]
fn parses_and_refuses() {
    assert_eq!(parse(None), Ok(false));
    assert_eq!(parse(Some("0")), Ok(false));
    assert_eq!(parse(Some("1")), Ok(true));
    for bad in ["", "on", "2", "true"] {
        assert!(parse(Some(bad)).is_err(), "{bad:?}");
    }
}

/// 2026-10-11: Off, any combination passes; on, each conflicting lever is refused by name, and
/// a missing grouped W4A4 GEMM too.
#[test]
fn each_conflicting_lever_is_refused_by_name() {
    assert!(check_levers(true, OK).is_ok());
    let all_bad = Levers {
        gemv_seg: true,
        gemv_rows: 4,
        dequant_gemm: true,
        w4a4_grouped_ready: false,
        dsa_batch_qidx: true,
        kda_chunked: true,
    };
    assert!(check_levers(false, all_bad).is_ok());
    for (l, name) in [
        (
            Levers {
                gemv_seg: true,
                ..OK
            },
            "METRALE_GLM_W4A16_GEMV_SEG",
        ),
        (Levers { gemv_rows: 2, ..OK }, "METRALE_GLM_W4A16_GEMV_ROWS"),
        (
            Levers {
                dequant_gemm: true,
                ..OK
            },
            "DEQUANT_GEMM_MIN_ROWS",
        ),
        (
            Levers {
                w4a4_grouped_ready: false,
                ..OK
            },
            "METRALE_GLM_MOE_W4A4_GROUPED",
        ),
        (
            Levers {
                dsa_batch_qidx: true,
                ..OK
            },
            "METRALE_GLM_DSA_BATCH_QIDX",
        ),
        (
            Levers {
                kda_chunked: true,
                ..OK
            },
            "METRALE_GLM_KDA_CHUNK",
        ),
    ] {
        let e = check_levers(true, l).unwrap_err();
        assert!(e.contains(name), "{name}: {e}");
    }
}

/// 2026-10-11: Off, or at two ranks, one slice of every row; on at three ranks, whole-row slices
/// of the most rows that fit the one-shot buffer, covering every row once in order; a buffer
/// smaller than a row is refused.
#[test]
fn all_reduce_slices_cover_every_row_in_fitting_whole_rows() {
    let row = 4096 * 2;
    let mib = 1 << 20;
    assert_eq!(
        ar_slices(false, 3, mib, 2048, row).unwrap(),
        vec![(0, 2048)]
    );
    assert_eq!(ar_slices(true, 2, 0, 2048, row).unwrap(), vec![(0, 2048)]);
    assert_eq!(ar_slices(true, 3, mib, 16, row).unwrap(), vec![(0, 16)]);
    let s = ar_slices(true, 3, mib, 300, row).unwrap();
    assert_eq!(s, vec![(0, 128), (128, 128), (256, 44)]);
    for rows in [1usize, 127, 128, 129, 2048, 2049] {
        let s = ar_slices(true, 3, mib, rows, row).unwrap();
        let mut next = 0;
        for &(r0, n) in &s {
            assert_eq!(r0, next);
            assert!(n >= 1 && n * row <= mib);
            next += n;
        }
        assert_eq!(next, rows);
    }
    assert!(ar_slices(true, 3, 0, 4, row).is_err());
    assert!(ar_slices(true, 3, row - 2, 4, row).is_err());
}

/// 2026-10-11: The one-row W4A16 launch under the lever is the row tile (`w4a16_tc_rows_16`, the
/// policy's 4 warps) over one row, with the same A, weight, C and shape as any wider launch.
#[test]
fn one_row_runs_the_row_tile() {
    use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
    let gpu = MockGpuBackend::new();
    let w = QuantizedWeight {
        weight: DevicePtr(0x1000),
        weight_scale: DevicePtr(0x1100),
        weight_scale_2: 0.25,
        ..QuantizedWeight::null()
    };
    let (a, c) = (DevicePtr(0x5000), DevicePtr(0x9000));
    one_row(&gpu, &w, a, c, 2816, 4096, 3).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    assert_eq!(
        (l[0].grid, l[0].block, l[0].stream),
        ([44, 1, 1], [128, 1, 1], 3)
    );
    assert_eq!(l[0].args[0], MockArg::Buffer(a));
    assert_eq!(l[0].args[4], MockArg::Buffer(c));
    assert_eq!(l[0].args[5], MockArg::Bytes(1u32.to_le_bytes().to_vec()));
    assert!(
        gpu.kernel_lookups_snapshot()
            .contains(&("w4a16_tc_rows".to_string(), "w4a16_tc_rows_16".to_string()))
    );
}
