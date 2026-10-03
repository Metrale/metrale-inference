// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The dense circuit's state programs bound to its pool: every node lands on the
//! pool's own layer and unit, the snapshot programs convert only on the prefix side, and a
//! disagreement with the pool refuses the bind.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use metrale_circuit::{Circuit, LayerKind};

use super::*;
use crate::circuit_exec::sources;
use crate::ssm_reserve::state_formats;

fn circuit() -> Circuit {
    let inst = sources::instance(super::super::exec_fixture::RECIPE).unwrap();
    metrale_circuit::load(&inst, sources::sources(&inst).unwrap())
        .unwrap()
        .circuit
}

fn recurrent(c: &Circuit) -> Vec<usize> {
    c.layer_kinds
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == LayerKind::LinearAttention)
        .map(|(i, _)| i)
        .collect()
}

/// 2026-10-03: One h and one conv unit of the pool, read off the circuit's own declarations.
fn units(c: &Circuit, f16: bool) -> PoolUnits {
    let f = state_formats(f16);
    let unit = |v| {
        c.states
            .iter()
            .find(|s| s.verify == Some(v))
            .unwrap()
            .unit_bytes(&f)
            .unwrap() as usize
    };
    PoolUnits {
        h_stored: unit(VerifySteps::H),
        conv: unit(VerifySteps::Conv),
    }
}

#[test]
fn every_program_covers_every_recurrent_layer_in_pool_order() {
    let c = circuit();
    let layers = recurrent(&c);
    assert!(!layers.is_empty());
    let sp = StatePrograms::bind(&c, &state_formats(false), &layers, units(&c, false)).unwrap();
    for id in StateProgramId::ALL {
        let nodes = sp.nodes(id).unwrap();
        assert_eq!(nodes.len(), 2 * layers.len(), "{id:?}");
        for (l, pair) in nodes.chunks(2).enumerate() {
            let parts: Vec<_> = pair.iter().map(|n| (n.ssm_layer, n.part)).collect();
            assert!(
                parts == [(l, StatePart::H), (l, StatePart::Conv)]
                    || parts == [(l, StatePart::Conv), (l, StatePart::H)],
                "{id:?} pair {l}: {parts:?}"
            );
        }
        assert!(
            nodes.iter().all(|n| n.conversion.is_none()),
            "{id:?} at FP32"
        );
    }
}

/// 2026-10-03: Under the f16-sized pool, only the prefix programs convert h, and only h; the
/// ring keeps the stored bytes; the prefix cache's h is FP32-wide.
#[test]
fn an_f16_pool_converts_only_h_on_the_prefix_side() {
    let c = circuit();
    let layers = recurrent(&c);
    let u = units(&c, true);
    let sp = StatePrograms::bind(&c, &state_formats(true), &layers, u).unwrap();
    let h = |id| -> Vec<BoundStateNode> {
        sp.nodes(id)
            .unwrap()
            .iter()
            .filter(|n| n.part == StatePart::H)
            .copied()
            .collect()
    };
    for n in h(StateProgramId::PrefixSave) {
        assert_eq!(n.conversion, Some(Conversion::Widen));
        assert_eq!(n.bytes, 2 * u.h_stored);
    }
    for n in h(StateProgramId::PrefixRestore) {
        assert_eq!(n.conversion, Some(Conversion::Narrow));
        assert_eq!(n.bytes, u.h_stored);
    }
    for id in [StateProgramId::RingSave, StateProgramId::RingRestore] {
        for n in h(id) {
            assert_eq!((n.conversion, n.bytes), (None, u.h_stored), "{id:?}");
        }
    }
    let conv = sp.nodes(StateProgramId::PrefixSave).unwrap();
    assert!(
        conv.iter()
            .filter(|n| n.part == StatePart::Conv)
            .all(|n| n.conversion.is_none() && n.bytes == u.conv)
    );
}

/// 2026-10-03: A pool whose units differ from the plan's, or whose recurrent layers are not the
/// circuit's, refuses the bind.
#[test]
fn a_pool_that_disagrees_with_the_plan_is_refused() {
    let c = circuit();
    let layers = recurrent(&c);
    let mut u = units(&c, false);
    u.conv += 4;
    assert!(StatePrograms::bind(&c, &state_formats(false), &layers, u).is_err());
    let short = &layers[..layers.len() - 1];
    assert!(StatePrograms::bind(&c, &state_formats(false), short, units(&c, false)).is_err());
    let shifted: Vec<_> = layers.iter().map(|l| l + 1).collect();
    assert!(StatePrograms::bind(&c, &state_formats(false), &shifted, units(&c, false)).is_err());
}
