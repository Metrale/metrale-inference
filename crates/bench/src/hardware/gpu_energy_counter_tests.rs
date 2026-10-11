// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Tests for the two pure functions here: the window delta
//! (including the backwards-counter case) and the metrics writer.
//! `read_mj` is not unit tested: it is a single real HTTP GET, the same
//! shape as `EnergySampler::spawn` and `fetch_hardware`, neither of which
//! is unit tested either.
//!
//! Owner: bench hardware.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn window_joules_reads_the_delta_in_joules() {
    assert_eq!(window_joules(Some(1_000.0), Some(2_500.0)), Some(1.5));
}

/// 2026-10-05: Explicitly requested: a counter that goes backwards between
/// the two bracketing scrapes (a server restart mid-window, a clock issue)
/// yields no reading — never a negative joule count, and never the
/// magnitude of the drop misread as a reading.
#[test]
fn a_counter_that_goes_backwards_yields_no_reading() {
    assert_eq!(
        window_joules(Some(5_000.0), Some(1_000.0)),
        None,
        "a drop must not read as negative or positive energy"
    );
}

/// 2026-10-05: An unmoved counter across the window is not a reading either:
/// a delivered token never costs zero energy, so a flat pair is "no
/// evidence", not "a free window" — the same rule a zero rail-joule reading
/// follows.
#[test]
fn an_unmoved_counter_is_not_a_reading() {
    assert_eq!(window_joules(Some(1_000.0), Some(1_000.0)), None);
}

#[test]
fn either_reading_missing_is_no_reading() {
    assert_eq!(window_joules(None, Some(1_000.0)), None);
    assert_eq!(window_joules(Some(1_000.0), None), None);
    assert_eq!(window_joules(None, None), None);
}

#[test]
fn metrics_writes_both_keys_under_the_prefix_and_reuses_joules_per_token() {
    let mut m = std::collections::BTreeMap::new();
    metrics("c1_", Some(2.0), 10, &mut m);
    assert_eq!(m.get("c1_gpu_energy_counter_j"), Some(&2.0));
    assert_eq!(m.get("c1_gpu_energy_counter_jpt"), Some(&0.2));
}

#[test]
fn metrics_of_no_reading_writes_nothing() {
    let mut m = std::collections::BTreeMap::new();
    metrics("", None, 10, &mut m);
    assert!(m.is_empty());
}

#[test]
fn metrics_of_zero_tokens_writes_the_joules_but_no_ratio() {
    let mut m = std::collections::BTreeMap::new();
    metrics("", Some(2.0), 0, &mut m);
    assert_eq!(m.get("gpu_energy_counter_j"), Some(&2.0));
    assert_eq!(m.get("gpu_energy_counter_jpt"), None);
}
