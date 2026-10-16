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
        caveat: None,
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

/// 2026-10-16: A caveated target is never printed bare: its row marks it and a footnote gives
/// the caveat and where the account lives; an uncaveated table carries no mark.
#[test]
fn a_caveated_target_is_marked_and_footnoted() {
    let b = Battle {
        id: "decode-battle".into(),
        name: "Decode battle".into(),
        gate_prefixes: vec!["concurrency-sweep".into()],
        metrics: vec![MetricRule {
            pattern: "c{n}_aggregate_tok_s".into(),
            better: Better::Higher,
        }],
        note: String::new(),
    };
    let cell = |caveat: Option<&str>| Cell {
        gate: "concurrency-sweep-glm".into(),
        hw: "gb10".into(),
        model: "glm-5.3-flash".into(),
        checkpoint: "org/GLM".into(),
        metric: "c16_aggregate_tok_s".into(),
        better: Better::Higher,
        target: 115.73,
        bound: None,
        caveat: caveat.map(String::from),
    };
    let flagged = table(&report(&b, &[cell(Some("likely vLLM stall"))], &|_| None));
    let row = flagged
        .lines()
        .find(|l| l.contains("c16_aggregate_tok_s") && !l.starts_with('\u{2020}'))
        .expect("the cell has a row");
    assert!(row.contains("115.73\u{2020}"), "{row}");
    assert!(
        flagged.contains(
            "\u{2020} concurrency-sweep-glm c16_aggregate_tok_s on org/GLM: likely vLLM stall. \
             The entry's note in kernels/gb10/glm-5.3-flash/BENCH.toml gives the account."
        ),
        "{flagged}"
    );
    let plain = table(&report(&b, &[cell(None)], &|_| None));
    assert!(!plain.contains('\u{2020}'), "{plain}");
}
