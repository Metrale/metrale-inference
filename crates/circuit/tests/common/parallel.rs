// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Rank 0 of a 2-rank tensor-parallel dense 27B (no draft head: TP refuses one),
//! planned at the rank's shape with its reduces (`metrale_circuit::parallel`); and the rendered
//! plans the TP goldens under kernels/circuits/plans/tp/ pin.
//!
//! Owner: metrale-circuit tests (FEATURES workstream).
//! Invariants: none beyond the types.

use metrale_circuit::parallel::{TpRank, rank_shape, with_reduces};
use metrale_circuit::{Instance, Loaded, Mode};

/// 2026-10-03: The shapes the TP goldens pin: one row, and multi-sequence widths across the
/// row tiers.
pub const SHAPES: [(Mode, u64); 4] = [
    (Mode::Decode, 1),
    (Mode::MultiSeq, 2),
    (Mode::MultiSeq, 8),
    (Mode::MultiSeq, 64),
];

/// 2026-10-03: The rank's instance and circuit.
pub fn rank() -> (Instance, Loaded) {
    let mut inst = super::instances()
        .into_iter()
        .find(|i| i.recipe == super::lora::RECIPE)
        .expect("the dense instance");
    inst.shape.dims.insert("mtp".into(), 0);
    inst.shape = rank_shape(&inst.shape, TpRank { rank: 0, world: 2 }).expect("divides");
    let mut loaded = super::load(&inst);
    loaded.circuit =
        with_reduces(&loaded.circuit, TpRank { rank: 0, world: 2 }).expect("the reduces");
    (inst, loaded)
}

/// 2026-10-03: The TP goldens: (path under kernels/circuits/plans/, rendered plan).
pub fn plans() -> Vec<(String, String)> {
    let (inst, loaded) = rank();
    let avail = super::available(&inst, &loaded.rules);
    let fams = super::families(&inst);
    SHAPES
        .into_iter()
        .map(|(mode, rows)| {
            let text = metrale_circuit::render_plan(&inst, &loaded, &avail, mode, rows, &fams)
                .unwrap_or_else(|e| panic!("TP {} n={rows}: {e}", mode.name()));
            (format!("tp/{}", inst.plan_file(mode, rows)), text)
        })
        .collect()
}
