// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `METRALE_GLM_MOE_QUANT_EARLY`: the lever's parse and readback, the shard
//! quantization after the last section-2 piece only, the in-place gather of the three outputs
//! on the comm stream, and the one-shot hand-off to the grouped launch.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use std::sync::Mutex;

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::*;
use crate::glm5next_sp_mhc::shards;
use crate::glm5next_sp_mhc::tests::Recorder;

const H: usize = 4096;
const X: DevicePtr = DevicePtr(0x9000_0000);
const COMM: u64 = 18;
const MAIN: u64 = 13;

fn codes() -> GateUpCodes {
    GateUpCodes {
        kern: KernelHandle(0x51),
        gs: 0.000642,
        k: H,
        out: [
            DevicePtr(0x1000_0000),
            DevicePtr(0x2000_0000),
            DevicePtr(0x3000_0000),
        ],
        rows_cap: 8208 * 8,
    }
}

fn recorder(rank: usize) -> Recorder {
    Recorder {
        rank,
        world: 3,
        ops: Mutex::new(Vec::new()),
    }
}

fn early(gpu: &MockGpuBackend, k: usize, rank: usize) -> Early {
    plan(codes(), X, k, &shards(k, 3), rank, COMM, side(gpu).unwrap()).unwrap()
}

/// 2026-10-15: Unset/0 off whatever else is set; 1 on only with the overlap and the grouped
/// GEMM on and the pipeline off; anything else refused.
#[test]
fn parses_strictly_and_names_what_it_needs() {
    for (o, p, g) in [(false, false, false), (true, true, true)] {
        assert_eq!(parse(None, o, p, g), Ok(false));
        assert_eq!(parse(Some("0"), o, p, g), Ok(false));
    }
    assert_eq!(parse(Some("1"), true, false, true), Ok(true));
    for (o, p, g, needle) in [
        (false, false, true, "METRALE_GLM_SP_OVERLAP=1"),
        (true, true, true, "METRALE_GLM_SP_PIPELINE=1"),
        (true, false, false, "METRALE_GLM_MOE_W4A4_GROUPED=1"),
    ] {
        let e = parse(Some("1"), o, p, g).unwrap_err();
        assert!(e.contains(needle), "{e}");
    }
    for bad in ["", "on", "2", "true"] {
        assert!(parse(Some(bad), true, false, true).is_err(), "{bad:?}");
    }
    assert!(readback(true).starts_with("METRALE_GLM_MOE_QUANT_EARLY=1"));
    assert!(readback(false).starts_with("METRALE_GLM_MOE_QUANT_EARLY off"));
}

/// 2026-10-15: A plan needs a rank inside shards that cover the rows, and rows the scratch
/// holds.
#[test]
fn a_plan_outside_the_shards_or_scratch_is_refused() {
    let gpu = MockGpuBackend::new();
    let s = side(&gpu).unwrap();
    assert!(plan(codes(), X, 900, &shards(900, 3), 3, COMM, s).is_err());
    assert!(plan(codes(), X, 900, &shards(899, 3), 0, COMM, s).is_err());
    let small = GateUpCodes {
        rows_cap: 899,
        ..codes()
    };
    assert!(plan(small, X, 900, &shards(900, 3), 0, COMM, s).is_err());
    assert!(plan(codes(), X, 900, &shards(900, 3), 2, COMM, s).is_ok());
}

/// 2026-10-15: On rank 1 of 3 at 8208 rows (shard 2736..5472, four pieces): the pieces before
/// the last enqueue nothing; the last quantizes exactly the shard on the side stream, at the
/// shard's offset in the input and in each output; a second end of the shard is refused.
#[test]
fn only_the_last_piece_quantizes_the_shard_on_the_side_stream() {
    let gpu = MockGpuBackend::new();
    let mut e = early(&gpu, 8208, 1);
    for b in [3420, 4104, 4788] {
        after_piece(Some(&mut e), &gpu, b, MAIN).unwrap();
    }
    assert!(gpu.launches_snapshot().is_empty());
    after_piece(Some(&mut e), &gpu, 5472, MAIN).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    let lo = 2736usize;
    let c = codes();
    assert_eq!(
        (l[0].func, l[0].grid, l[0].stream),
        (0x51, [2736, 1, 1], side(&gpu).unwrap().stream)
    );
    let want = [
        X.offset(lo * H * 2),
        c.out[0].offset(lo * H / 2),
        c.out[1].offset(lo * H / 16),
        c.out[2].offset(lo * 4),
    ];
    for (i, p) in want.into_iter().enumerate() {
        assert_eq!(l[0].args[i], MockArg::Buffer(p), "arg {i}");
    }
    assert_eq!(
        l[0].args[4],
        MockArg::Bytes((H as u32).to_le_bytes().to_vec())
    );
    assert_eq!(
        l[0].args[5],
        MockArg::Bytes(c.gs.to_bits().to_le_bytes().to_vec())
    );
    assert!(after_piece(Some(&mut e), &gpu, 5472, MAIN).is_err());
    after_piece(None, &gpu, 5472, MAIN).unwrap();
}

/// 2026-10-15: The gather is three grouped in-place all-gathers on the comm stream, codes then
/// group scales then global scales, each sending this rank's shard rows and receiving every
/// peer's at its rows; it arms one grouped launch of exactly those rows and that scratch.
#[test]
fn the_gather_moves_each_output_in_place_and_arms_one_launch() {
    let gpu = MockGpuBackend::new();
    let comm = recorder(1);
    let mut e = early(&gpu, 8208, 1);
    assert!(
        gather(Some(&mut e), &gpu, &comm).is_err(),
        "before the quant"
    );
    after_piece(Some(&mut e), &gpu, 5472, MAIN).unwrap();
    gather(Some(&mut e), &gpu, &comm).unwrap();
    let ops = std::mem::take(&mut *comm.ops.lock().unwrap());
    let c = codes();
    let mut want = Vec::new();
    for (out, rb) in c.out.into_iter().zip([H / 2, H / 16, 4]) {
        want.push(("group_start", 0, 0, 0, 0));
        for peer in [0usize, 2] {
            want.push(("send", out.0 + (2736 * rb) as u64, 2736 * rb, peer, COMM));
            want.push((
                "recv",
                out.0 + (peer * 2736 * rb) as u64,
                2736 * rb,
                peer,
                COMM,
            ));
        }
        want.push(("group_end", 0, 0, 0, 0));
    }
    assert_eq!(ops, want);
    let other = GateUpCodes { gs: 1.0, ..c };
    assert!(take(X, 8208, &other).is_err(), "another scale");
    assert_eq!(take(X, 8208, &c).unwrap(), None, "a refusal disarms");
    gather(Some(&mut e), &gpu, &comm).unwrap();
    assert!(take(X, 8207, &c).is_err(), "other rows");
    gather(Some(&mut e), &gpu, &comm).unwrap();
    assert!(take(X.offset(2), 8208, &c).is_err(), "another input");
    gather(Some(&mut e), &gpu, &comm).unwrap();
    assert_eq!(
        take(X, 8208, &c).unwrap(),
        Some(side(&gpu).unwrap().gathered)
    );
    assert_eq!(take(X, 8208, &c).unwrap(), None, "once");
    finish(Some(e)).unwrap();
}

/// 2026-10-15: Codes nothing consumed fail the MLP's finish (and a gather over them fails
/// before it enqueues anything); nothing armed, or no plan, finishes clean.
#[test]
fn unconsumed_codes_are_refused() {
    let gpu = MockGpuBackend::new();
    let comm = recorder(0);
    finish(None).unwrap();
    let mut e = early(&gpu, 900, 0);
    after_piece(Some(&mut e), &gpu, 300, MAIN).unwrap();
    gather(Some(&mut e), &gpu, &comm).unwrap();
    assert!(gather(Some(&mut e), &gpu, &comm).is_err());
    let msg = finish(Some(e)).unwrap_err().to_string();
    assert!(msg.contains("did not run the grouped W4A4 launch"), "{msg}");
    assert_eq!(take(X, 900, &codes()).unwrap(), None);
    gather(None, &gpu, &comm).unwrap();
    assert_eq!(
        comm.ops.lock().unwrap().len(),
        18,
        "one gather; the refused one enqueued nothing"
    );
}
