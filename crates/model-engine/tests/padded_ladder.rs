// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The padded batch ladder of `padded_batch_n` is the row set every circuit
//! instance plans multi-sequence decode at (kernels/circuits/INSTANCES.toml), so the golden
//! plans cover exactly the widths batched decode can launch.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use std::collections::BTreeSet;

use metrale_circuit::Mode;
use metrale_model_engine::traits::padded_batch_n;

#[test]
fn every_instance_plans_multi_seq_at_exactly_the_padded_ladder() {
    // 2026-09-28: The rungs a live batch of 2..=128 sequences pads to. Above 128 the ladder
    // returns `n` itself; multi-sequence plans stop at the widest rung.
    let ladder: BTreeSet<u64> = (2..=128).map(|n| padded_batch_n(n) as u64).collect();
    assert_eq!(ladder.iter().next_back(), Some(&128));
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/circuits/INSTANCES.toml"
    );
    let text = std::fs::read_to_string(path).expect("INSTANCES.toml");
    let instances = metrale_circuit::parse_instances(&text).expect("INSTANCES.toml parses");
    assert!(!instances.is_empty());
    for inst in instances {
        let planned: BTreeSet<u64> = inst
            .plans
            .get(&Mode::MultiSeq)
            .map(|r| r.iter().copied().collect())
            .unwrap_or_default();
        assert_eq!(
            planned, ladder,
            "{}: multi_seq rows in INSTANCES.toml differ from padded_batch_n's ladder",
            inst.recipe
        );
    }
}
