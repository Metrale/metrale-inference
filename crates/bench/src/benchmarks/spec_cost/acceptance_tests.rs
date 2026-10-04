// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the run's acceptance counts: the pages are Prometheus text in the
//! shape the serve renders (`telemetry export::prometheus_layers`).
//!
//! Owner: bench, spec-cost.
//! Invariants: none beyond the types.

use super::*;

fn page(lines: &[&str]) -> Scrape {
    Scrape::parse(&format!(
        "metrale_decoded_tokens_total 1\n{}",
        lines.join("\n")
    ))
    .unwrap()
}

#[test]
fn deltas_round_trip_in_ascending_edge_order() {
    let first = page(&[
        r#"metrale_spec_draft_confidence_total{le="-0.1",accepted="1"} 10"#,
        r#"metrale_spec_verify_steps_total{drafts="2",accepted="1"} 5"#,
    ]);
    let last = page(&[
        r#"metrale_spec_draft_confidence_total{le="0",accepted="0"} 1"#,
        r#"metrale_spec_draft_confidence_total{le="0",accepted="1"} 30"#,
        r#"metrale_spec_draft_confidence_total{le="-0.1",accepted="0"} 4"#,
        r#"metrale_spec_draft_confidence_total{le="-0.1",accepted="1"} 16"#,
        r#"metrale_spec_verify_steps_total{drafts="2",accepted="1"} 12"#,
        r#"metrale_spec_verify_steps_total{drafts="2",accepted="2"} 3"#,
    ]);
    let mut m = BTreeMap::new();
    record(&first, &last, &mut m).unwrap();
    let c = read(&m).unwrap().unwrap();
    assert_eq!(c.edges, vec![-0.1, 0.0]);
    assert_eq!(c.accepted, vec![6, 30]);
    assert_eq!(c.rejected, vec![4, 1]);
    assert_eq!(c.steps, BTreeMap::from([((2, 1), 7), ((2, 2), 3)]));
}

#[test]
fn a_run_without_confidences_has_no_counts() {
    let mut m = BTreeMap::new();
    record(&page(&[]), &page(&[]), &mut m).unwrap();
    assert_eq!(read(&m).unwrap(), None);
}

#[test]
fn a_counter_that_went_backwards_fails_the_run() {
    let first = page(&[r#"metrale_spec_verify_steps_total{drafts="1",accepted="0"} 9"#]);
    let last = page(&[r#"metrale_spec_verify_steps_total{drafts="1",accepted="0"} 2"#]);
    let err = record(&first, &last, &mut BTreeMap::new()).unwrap_err();
    assert!(
        format!("{err:#}").contains("the serve restarted"),
        "{err:#}"
    );
}
