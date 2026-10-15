// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The gate/up codes of the grouped W4A4 launch (`w4a4_early`) on the recording
//! mock: a launch of every row through [`GateUpCodes::launch_rows`] is the grouped launch's own
//! quantization argument for argument, a shard's is it offset by the shard's first row, and a
//! grouped launch armed with gathered codes (`METRALE_GLM_MOE_QUANT_EARLY`) drops exactly its
//! quantization and keeps every other launch. The bytes are the GPU gate's job
//! (`examples/glm5next_moe_quant_early_gate.rs`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use std::sync::Mutex;

use super::super::super::w4a4_early::{GateUpCodes, gate_up_codes};
use super::*;
use crate::glm5next_moe_quant_early as early;
use crate::glm5next_sp_mhc::shards;
use crate::glm5next_sp_mhc::tests::Recorder;

const ROWS: usize = 96;

/// 2026-10-15: The codes of a `ROWS`-row grouped workspace on a fresh mock (the addresses
/// every fresh mock gives the same workspace).
fn codes_of(rows: usize) -> (MockGpuBackend, GateUpCodes) {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    let ws = Glm5NextMlpWorkspace::new_with(&gpu, &cfg_tp(), ROWS, true).unwrap();
    let w = moe_weights(usize::MAX);
    let c = gate_up_codes(&k, &cfg_tp(), &w, &ws, rows).unwrap();
    (gpu, c.expect("a grouped W4A4 launch"))
}

/// 2026-10-15: Rows the slot GEMV serves have no codes; past it, the codes are the grouped
/// launch's quantizer, gate/up scale, hidden width and scratch.
#[test]
fn the_codes_exist_exactly_where_the_grouped_launch_runs() {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    let ws = Glm5NextMlpWorkspace::new_with(&gpu, &cfg_tp(), ROWS, true).unwrap();
    let w = moe_weights(usize::MAX);
    assert!(gate_up_codes(&k, &cfg_tp(), &w, &ws, 16).unwrap().is_none());
    let c = gate_up_codes(&k, &cfg_tp(), &w, &ws, 17).unwrap().unwrap();
    assert_eq!(
        (c.kern.0, c.gs, c.k, c.out, c.rows_cap),
        (
            QUANT,
            SCALES.gate_up,
            4096,
            [ws.w4a4_aq, ws.w4a4_as, ws.w4a4_ag],
            ws.w4a4_rows
        )
    );
}

/// 2026-10-15: All rows through `launch_rows` is the grouped launch's quantization exactly;
/// each of three shards is it at the shard's rows (input, codes, group scales and globals
/// offset by the shard's first row, one block per row), and the shards cover every row once.
#[test]
fn shard_launches_are_the_whole_launch_at_their_rows() {
    let (r, l, _ws) = run(cfg_tp(), ROWS, true);
    r.unwrap();
    let whole = l.iter().find(|l| l.func == QUANT).unwrap();
    let (gpu, c) = codes_of(ROWS);
    c.launch_rows(&gpu, X, 0, ROWS, 5).unwrap();
    assert_same(&gpu.launches_snapshot(), std::slice::from_ref(whole));

    let gpu = MockGpuBackend::new();
    let s = shards(ROWS, 3);
    for &(lo, hi) in &s {
        c.launch_rows(&gpu, X, lo, hi, 7).unwrap();
    }
    let got = gpu.launches_snapshot();
    let mut next = 0;
    for (g, &(lo, hi)) in got.iter().zip(&s) {
        assert_eq!(lo, next);
        next = hi;
        let at = [
            X.offset(lo * 4096 * 2),
            c.out[0].offset(lo * 2048),
            c.out[1].offset(lo * 256),
            c.out[2].offset(lo * 4),
        ];
        for (i, p) in at.into_iter().enumerate() {
            assert_eq!(ptr_arg(g, i), p, "shard {lo} arg {i}");
        }
        assert_eq!(&g.args[4..], &whole.args[4..]);
        assert_eq!(
            (g.func, g.grid, g.block, g.stream),
            (QUANT, [(hi - lo) as u32, 1, 1], whole.block, 7)
        );
    }
    assert_eq!(next, ROWS);
    assert!(c.launch_rows(&gpu, X, 5, 5, 7).is_err());
    assert!(c.launch_rows(&gpu, X, 0, c.rows_cap + 1, 7).is_err());
}

/// 2026-10-15: Arm the grouped launch on this thread the way `forward_rows_sp` does: rank 1's
/// shard quantized after its last piece, then the gather.
fn arm(c: GateUpCodes) -> early::Early {
    let gpu = MockGpuBackend::new();
    let comm = Recorder {
        rank: 1,
        world: 3,
        ops: Mutex::new(Vec::new()),
    };
    let side = early::side(&gpu).unwrap();
    let mut e = early::plan(c, X, ROWS, &shards(ROWS, 3), 1, 18, side).unwrap();
    early::after_piece(Some(&mut e), &gpu, 64, 5).unwrap();
    early::gather(Some(&mut e), &gpu, &comm).unwrap();
    e
}

/// 2026-10-15: Armed, the grouped launch enqueues the unarmed launch list minus its gate/up
/// quantization, argument for argument, and consumes the codes; armed for other rows it is
/// refused.
#[test]
fn an_armed_launch_drops_only_its_quantization() {
    let (r, want, _ws) = run(cfg_tp(), ROWS, true);
    r.unwrap();
    let (_g, c) = codes_of(ROWS);
    let e = arm(c);
    let (r, got, _ws) = run(cfg_tp(), ROWS, true);
    r.unwrap();
    early::finish(Some(e)).unwrap();
    let q = want.iter().position(|l| l.func == QUANT).unwrap();
    let mut kept = want.clone();
    kept.remove(q);
    assert_same(&got, &kept);
    assert_eq!(funcs(&got).iter().filter(|&&f| f == QUANT).count(), 0);

    let e = arm(c);
    let (r, _, _) = run(cfg_tp(), ROWS - 1, true);
    assert!(r.is_err(), "codes of other rows");
    early::finish(Some(e)).unwrap();
}
