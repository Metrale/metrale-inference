// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The battle table reports the target and the ratchet side by side: a cell
//! inside its own bound that loses to the reference prints both.

use std::collections::BTreeMap;

use metrale_bench::gate::battle::{Battle, Better, Cell, MetricRule, Observed, report};
use metrale_bench::gate::record::Bound;

use super::table;

#[test]
fn a_loss_inside_its_bound_prints_both_and_the_battle_is_not_won() {
    let b = Battle {
        id: "ttft-battle".into(),
        name: "TTFT battle".into(),
        gate_prefixes: vec!["high-isl-".into()],
        metrics: vec![MetricRule {
            pattern: "median_ms".into(),
            better: Better::Lower,
        }],
        note: String::new(),
    };
    let cell = Cell {
        gate: "high-isl-ttft-cold-glm-64k".into(),
        hw: "gb10".into(),
        model: "glm".into(),
        checkpoint: "org/GLM".into(),
        metric: "median_ms".into(),
        better: Better::Lower,
        target: 10479.0,
        bound: Some(Bound {
            min: None,
            max: Some(11446.9),
            noise: None,
        }),
    };
    let seen = Observed {
        git_sha: "abc".into(),
        recorded_at: 1,
        metrics: BTreeMap::from([("median_ms".to_string(), 11103.9)]),
    };
    let text = table(&report(&b, &[cell], &|_| Some(seen.clone())));
    let line = text
        .lines()
        .find(|l| l.contains("high-isl-ttft-cold-glm-64k"))
        .expect("the cell has a row");
    assert!(
        line.contains("LOSS") && line.contains("within its bound"),
        "{line}"
    );
    assert!(line.contains("-6.0%"), "{line}");
    assert!(
        text.contains("1 lost") && text.ends_with("all cells win: false\n"),
        "{text}"
    );
}
