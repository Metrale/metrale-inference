// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The prefill FFN rules of the embedded FUSIONS.toml against the emitter's
//! mirror of the legacy MMQ ladder: every rule launches the legacy tile at both ends of its rows,
//! the rules tile the rows with no gap or overlap, and every prefill rule names a registered
//! emitter.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use metrale_circuit::{Mode, Rule};

use super::legacy_tile;
use crate::circuit_exec::sources;

fn rules() -> Vec<Rule> {
    let text = sources::lookup(&sources::FUSIONS, "gb10", "FUSIONS").unwrap();
    metrale_circuit::parse_rule_set(text).unwrap().rules
}

/// 2026-10-03: The widest pass the dense 27B recipe's arena holds (`max_batch_tokens`).
const ARENA_ROWS: u64 = 8193;

#[test]
fn every_prefill_ffn_rule_launches_the_legacy_tile_at_both_ends_of_its_rows() {
    let ffn: Vec<Rule> = rules()
        .into_iter()
        .filter(|r| r.emitter == "prefill_ffn_mmq")
        .collect();
    assert!(ffn.len() >= 8, "the prefill FFN rules are missing");
    for r in &ffn {
        assert!(r.modes.contains(&Mode::Prefill) && r.modes.contains(&Mode::PrefillChunk));
        let gemm = r.kernels[1].func.as_str();
        for rows in [r.rows.0, r.rows.1.min(ARENA_ROWS)] {
            assert_eq!(
                legacy_tile(rows as u32).0,
                gemm,
                "`{}` at {rows} rows launches {gemm}",
                r.id
            );
        }
        let act_down = r.id.ends_with("_act_down");
        if act_down {
            let pipe = gemm == "metrale_nvfp4_gemm_pipe";
            assert_eq!(r.kernels.len(), if pipe { 2 } else { 3 }, "`{}`", r.id);
        }
    }
    // 2026-10-03: Each rule family (gate|up and act|down) covers 1..=1048576 once. The declared
    // circuit's act_quant twins come with its W8A8 FFN prefill rule.
    for (gate_up, a4) in [(true, false), (false, false)] {
        let mut ranges: Vec<(u64, u64)> = ffn
            .iter()
            .filter(|r| r.id.ends_with("_gate_up") == gate_up && r.id.contains("_a4_") == a4)
            .map(|r| r.rows)
            .collect();
        ranges.sort();
        assert_eq!(ranges.first().map(|r| r.0), Some(1), "{gate_up} {a4}");
        assert_eq!(
            ranges.last().map(|r| r.1),
            Some(1_048_576),
            "{gate_up} {a4}"
        );
        for w in ranges.windows(2) {
            assert_eq!(w[0].1 + 1, w[1].0, "{gate_up} {a4}: {ranges:?}");
        }
    }
}

#[test]
fn the_ladder_switches_tiles_at_the_legacy_edges() {
    let at = |m: u32| legacy_tile(m).1;
    assert_eq!(
        [at(1), at(16), at(17), at(32), at(33), at(64)],
        [16, 16, 32, 32, 64, 64]
    );
    assert_eq!([at(65), at(8193)], [128, 128]);
}

#[test]
fn every_prefill_rule_names_a_registered_emitter() {
    for r in rules()
        .iter()
        .filter(|r| r.modes.iter().any(|m| m.is_prefill()))
    {
        assert!(
            super::super::emitter(&r.emitter).is_ok(),
            "prefill rule `{}` names no emitter `{}`",
            r.id,
            r.emitter
        );
    }
}
