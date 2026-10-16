// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Tests of the BF16 head's register-entry bands (`ops::DenseGemvBatchmPair`, levers
//! `METRALE_LM_HEAD_BATCHM_NARROW` / `METRALE_LM_HEAD_BATCHM_WIDE`) through
//! `project_bf16_lm_head`: which entry each row count launches, with the runtime-M entry's grid
//! and arguments unchanged.
//!
//! Owner: model-engine decode.
//! Invariants: none beyond the types.

use super::{LmHeadM16Tc, project_bf16_lm_head};
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

const NARROW: u64 = 0xBF16;
const REG8: u64 = 0x1D8;
const WIDE: u64 = 0x1D16;
const FALLBACK: u64 = 0xCAFE;

fn tc_off() -> LmHeadM16Tc {
    LmHeadM16Tc {
        narrow: KernelHandle(0),
        wide: KernelHandle(0),
        enabled: false,
        n_tile: ops::DENSE_GEMM_M16_BF16_N_TILE,
    }
}

/// 2026-10-16: One head projection of `m` rows over a 67 x `k` weight; returns each launch's
/// `(func, grid, args[3..])`.
fn launches(
    m: u32,
    pair: ops::DenseGemvBatchmPair,
    batchm_max: u32,
    fixed: bool,
) -> Vec<(u64, [u32; 3], Vec<MockArg>)> {
    launches_full(m, pair, batchm_max, false, fixed)
        .into_iter()
        .map(|l| (l.0, l.1, l.2[3..].to_vec()))
        .collect()
}

/// 2026-10-16: As [`launches`] with `rows32`, returning every argument (A, C pointers first).
fn launches_full(
    m: u32,
    pair: ops::DenseGemvBatchmPair,
    batchm_max: u32,
    rows32: bool,
    fixed: bool,
) -> Vec<(u64, [u32; 3], Vec<MockArg>)> {
    let gpu = MockGpuBackend::new();
    let (n, k) = (67_u32, 64_u32);
    let input = gpu.alloc((m * k * 2) as usize).unwrap();
    let weight = DenseWeight {
        weight: gpu.alloc((n * k * 2) as usize).unwrap(),
    };
    let output = gpu.alloc((m * n * 2) as usize).unwrap();
    project_bf16_lm_head(
        &gpu,
        KernelHandle(FALLBACK),
        pair,
        input,
        &weight,
        output,
        [m, n, k],
        true,
        batchm_max,
        rows32,
        tc_off(),
        fixed,
        7,
    )
    .unwrap();
    gpu.launches_snapshot()
        .into_iter()
        .map(|l| (l.func, l.grid, l.args))
        .collect()
}

/// 2026-10-16: The entries `resolve_head_batchm` builds for the two levers.
fn pair(narrow_on: bool, wide_on: bool) -> ops::DenseGemvBatchmPair {
    ops::DenseGemvBatchmPair {
        narrow: KernelHandle(NARROW),
        reg8: KernelHandle(if narrow_on { REG8 } else { 0 }),
        wide: KernelHandle(if wide_on { WIDE } else { 0 }),
    }
}

/// 2026-10-16: Each lever moves exactly its row counts to its register entry, with the
/// runtime-M entry's grid and arguments; one row and every row count with both levers off
/// stay on the runtime-M entry; past `batchm_max` the tile GEMM keeps the rows whatever the
/// levers say.
#[test]
fn each_lever_moves_exactly_its_rows() {
    for (nb, wb) in [(false, false), (true, false), (false, true), (true, true)] {
        for m in 1..=16_u32 {
            let l = launches(m, pair(nb, wb), 16, false);
            assert_eq!(l.len(), 1);
            let want = match m {
                1 => NARROW,
                2..=8 if nb => REG8,
                9..=16 if wb => WIDE,
                _ => NARROW,
            };
            assert_eq!(l[0].0, want, "m={m} narrow={nb} wide={wb}");
            assert_eq!(l[0].1, [67_u32.div_ceil(4), 1, 1], "m={m}");
            let args: Vec<MockArg> = [m, 67, 64, 67]
                .iter()
                .map(|v| MockArg::Bytes(v.to_ne_bytes().to_vec()))
                .collect();
            assert_eq!(l[0].2, args, "m={m}");
        }
        let l = launches(9, pair(nb, wb), 8, false);
        assert_eq!(
            l[0].0, FALLBACK,
            "batchm_max=8 keeps 9 rows on the tile GEMM"
        );
    }
}

/// 2026-10-16: A fixed head's chunks pick their entry by their own row count: 40 rows run
/// 16 + 16 + 8; with only the wide lever the 8-row tail stays on the runtime-M entry, with both
/// it takes the reg8 entry.
#[test]
fn fixed_head_chunks_pick_their_entry_by_chunk_rows() {
    let funcs = |p| -> Vec<u64> { launches(40, p, 0, true).into_iter().map(|l| l.0).collect() };
    assert_eq!(funcs(pair(false, true)), vec![WIDE, WIDE, NARROW]);
    assert_eq!(funcs(pair(true, true)), vec![WIDE, WIDE, REG8]);
    assert_eq!(funcs(pair(true, false)), vec![NARROW, NARROW, REG8]);
    assert_eq!(funcs(pair(false, false)), vec![NARROW, NARROW, NARROW]);
}

fn u32_bytes(v: u32) -> MockArg {
    MockArg::Bytes(v.to_ne_bytes().to_vec())
}

/// 2026-10-16: `METRALE_DENSE_BATCHM_32` (`rows32`) runs exactly 17..=32 rows as 16-row launches
/// plus the remainder, each on the entry its own row count picks, with A and C advanced by whole
/// rows (A by K elements, C by N) and the runtime-M grid; 1..=16 rows are the same single launch
/// either way. Off, or past 32 rows, the tile GEMM keeps 17+ rows.
#[test]
fn rows32_splits_exactly_17_to_32_rows_into_whole_row_launches() {
    let p = pair(true, true);
    for m in 1..=16_u32 {
        assert_eq!(
            launches_full(m, p, 16, true, false),
            launches_full(m, p, 16, false, false),
            "m={m}"
        );
    }
    for m in 17..=32_u32 {
        let l = launches_full(m, p, 16, true, false);
        let tail = m - 16;
        assert_eq!(l.len(), 2, "m={m}");
        let tail_func = match tail {
            1 => NARROW,
            2..=8 => REG8,
            _ => WIDE,
        };
        assert_eq!((l[0].0, l[1].0), (WIDE, tail_func), "m={m}");
        let (a0, c0) = (&l[0].2[0], &l[0].2[2]);
        let (MockArg::Buffer(a0), MockArg::Buffer(c0)) = (a0, c0) else {
            panic!("expected buffers");
        };
        assert_eq!(
            l[1].2[0],
            MockArg::Buffer(a0.offset(16 * 64 * 2)),
            "m={m} A"
        );
        assert_eq!(
            l[1].2[2],
            MockArg::Buffer(c0.offset(16 * 67 * 2)),
            "m={m} C"
        );
        for (i, rows) in [(0, 16), (1, tail)] {
            assert_eq!(l[i].1, [67_u32.div_ceil(4), 1, 1], "m={m}");
            let want: Vec<MockArg> = [rows, 67, 64, 67].into_iter().map(u32_bytes).collect();
            assert_eq!(l[i].2[3..], want[..], "m={m} launch {i}");
        }
        let off = launches_full(m, p, 16, false, false);
        assert_eq!(off.len(), 1);
        assert_eq!(off[0].0, FALLBACK, "m={m}: off keeps the tile GEMM");
    }
    for m in [33_u32, 40, 64] {
        let l = launches_full(m, p, 16, true, false);
        assert_eq!(
            l.iter().map(|l| l.0).collect::<Vec<_>>(),
            vec![FALLBACK],
            "m={m}"
        );
    }
}

/// 2026-10-16: The band rule itself: `1..=band`, plus 17..=32 with `rows32`, whatever `band`.
#[test]
fn batchm_band_adds_exactly_17_to_32() {
    for band in [0_u32, 8, 16] {
        for m in 0..=40_u32 {
            let base = (1..=band).contains(&m);
            assert_eq!(
                ops::dense_batchm_band(band, false, m),
                base,
                "band={band} m={m}"
            );
            let want = base || (17..=32).contains(&m);
            assert_eq!(
                ops::dense_batchm_band(band, true, m),
                want,
                "band={band} m={m}"
            );
        }
    }
}
