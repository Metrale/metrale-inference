// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the Prometheus text reader, on a page shaped like the serve's
//! `/metrics`: the `prometheus` crate's counters first, then the telemetry layers' phase
//! histogram.
//!
//! Owner: bench, spec-cost.
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-04: Shapes copied from the serve: `prometheus::TextEncoder` counters and the
/// `histogram_series` rendering in `crates/telemetry/src/export/prometheus_text.rs`.
const PAGE: &str = "\
# HELP metrale_generation_tokens_total Total tokens generated
# TYPE metrale_generation_tokens_total counter
metrale_generation_tokens_total 900
# HELP metrale_decoded_tokens_total Tokens decoded, counted as they are produced (rate-friendly)
# TYPE metrale_decoded_tokens_total counter
metrale_decoded_tokens_total 12345
# HELP metrale_sched_phase_seconds Scheduler loop phase wall time
# TYPE metrale_sched_phase_seconds histogram
metrale_sched_phase_seconds_bucket{phase=\"propose\",le=\"0.001\"} 3
metrale_sched_phase_seconds_bucket{phase=\"propose\",le=\"+Inf\"} 410
metrale_sched_phase_seconds_sum{phase=\"propose\"} 1.25
metrale_sched_phase_seconds_count{phase=\"propose\"} 410
metrale_sched_phase_seconds_bucket{phase=\"step_mtp\",le=\"0.05\"} 400
metrale_sched_phase_seconds_bucket{phase=\"step_mtp\",le=\"+Inf\"} 410
metrale_sched_phase_seconds_sum{phase=\"step_mtp\"} 16.4
metrale_sched_phase_seconds_count{phase=\"step_mtp\"} 410
metrale_spec_verify_steps_total{drafts=\"3\",accepted=\"2\"} 77
";

#[test]
fn reads_counters_and_labelled_histogram_totals() {
    let s = Scrape::parse(PAGE).unwrap();
    assert_eq!(
        s.value("metrale_decoded_tokens_total", &[]).unwrap(),
        Some(12345.0)
    );
    let step = [("phase", "step_mtp")];
    assert_eq!(
        s.value("metrale_sched_phase_seconds_count", &step).unwrap(),
        Some(410.0)
    );
    assert_eq!(
        s.value("metrale_sched_phase_seconds_sum", &step).unwrap(),
        Some(16.4)
    );
    assert_eq!(
        s.value("metrale_sched_phase_seconds_sum", &[("phase", "propose")])
            .unwrap(),
        Some(1.25)
    );
    // 2026-10-04: Label order does not matter; the label set must match exactly.
    assert_eq!(
        s.value(
            "metrale_spec_verify_steps_total",
            &[("accepted", "2"), ("drafts", "3")]
        )
        .unwrap(),
        Some(77.0)
    );
    assert_eq!(
        s.value("metrale_spec_verify_steps_total", &[("drafts", "3")])
            .unwrap(),
        None
    );
    assert_eq!(
        s.value("metrale_sched_phase_seconds_bucket", &step)
            .unwrap(),
        None,
        "a bucket carries `le` as well, so the phase label alone does not match it"
    );
    assert_eq!(
        s.value(
            "metrale_sched_phase_seconds_bucket",
            &[("phase", "step_mtp"), ("le", "+Inf")]
        )
        .unwrap(),
        Some(410.0)
    );
    assert_eq!(s.value("metrale_absent_total", &[]).unwrap(), None);
}

#[test]
fn a_name_prefix_does_not_match_a_longer_name() {
    let s = Scrape::parse("metrale_decoded_tokens_total_extra 5\n").unwrap();
    assert_eq!(s.value("metrale_decoded_tokens_total", &[]).unwrap(), None);
}

#[test]
fn quoted_braces_commas_and_escapes_stay_in_the_value() {
    let s = Scrape::parse("m{a=\"x}y,z\",b=\"q\\\"r\\\\s\\nt\",} 4 1700000000000\n").unwrap();
    assert_eq!(s.series.len(), 1);
    assert_eq!(s.series[0].labels["a"], "x}y,z");
    assert_eq!(s.series[0].labels["b"], "q\"r\\s\nt");
    assert_eq!(s.series[0].value, 4.0);
}

#[test]
fn a_duplicate_series_is_an_error_on_lookup() {
    let s = Scrape::parse("m{p=\"a\"} 1\nm{p=\"a\"} 2\nm{p=\"b\"} 3\n").unwrap();
    assert!(s.value("m", &[("p", "a")]).is_err());
    assert_eq!(s.value("m", &[("p", "b")]).unwrap(), Some(3.0));
}

#[test]
fn malformed_sample_lines_fail_the_page() {
    for bad in [
        "metrale_x\n",
        "metrale_x abc\n",
        "metrale_x 1 2 3\n",
        "metrale_x 1 notatime\n",
        "metrale_x{p=\"a\" 1\n",
        "metrale_x{p=a} 1\n",
        "metrale_x{p=\"a\",p=\"b\"} 1\n",
        "{p=\"a\"} 1\n",
    ] {
        let err = Scrape::parse(bad).unwrap_err();
        assert!(
            format!("{err:#}").contains("/metrics line 1"),
            "{bad:?}: {err:#}"
        );
    }
}

#[test]
fn comments_and_blank_lines_are_skipped() {
    let s = Scrape::parse("# HELP m x\n\n# TYPE m counter\nm 1\n").unwrap();
    assert_eq!(s.series.len(), 1);
}
