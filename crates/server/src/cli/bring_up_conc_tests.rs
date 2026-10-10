// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Tests for the bring-up concurrency section: the `--concs` grammar, energy
//! integration over a window, and rungs read from the sweep's metrics.
//!
//! Owner: server CLI (`met ml-utils`).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::*;

#[test]
fn the_default_is_the_glm_ladder_and_one_value_is_a_maximum() {
    assert_eq!(
        parse_concs(DEFAULT_CONCS_ARG).unwrap(),
        [1, 2, 4, 8, 12, 16]
    );
    assert_eq!(parse_concs("16").unwrap(), [1, 2, 4, 8, 12, 16]);
    assert_eq!(parse_concs("128").unwrap(), STANDARD_CONCS);
    assert_eq!(parse_concs("1").unwrap(), [1]);
    assert_eq!(parse_concs("1, 4,16,64,128").unwrap(), [1, 4, 16, 64, 128]);
}

#[test]
fn bad_rungs_fail_fast_with_the_reason() {
    for (input, why) in [
        ("20", "standard rung"),
        ("4,1", "strictly increasing"),
        ("4,4", "strictly increasing"),
        ("0", "outside"),
        ("1,300", "outside"),
        ("a", "not a whole number"),
        ("1,,2", "not a whole number"),
        ("", "not a whole number"),
    ] {
        let e = parse_concs(input).expect_err(input).to_string();
        assert!(e.contains(why), "{input}: {e}");
    }
}

#[test]
fn energy_is_interpolated_per_host_and_summed() {
    let a: Series = vec![(10.0, 0.0), (20.0, 10_000.0)];
    let b: Series = vec![(9.0, 500.0), (11.0, 2_500.0), (21.0, 12_500.0)];
    // a: 1 J/s over [12, 18] = 6 J; b: 1 J/s = 6 J.
    let j = energy_j(&[a.clone(), b], 12.0, 18.0).unwrap();
    assert!((j - 12.0).abs() < 1e-9, "{j}");
    assert_eq!(
        energy_j(&[a.clone()], 5.0, 18.0),
        None,
        "starts before the series"
    );
    assert_eq!(
        energy_j(&[a.clone()], 12.0, 25.0),
        None,
        "ends after the series"
    );
    assert_eq!(
        energy_j(&[], 12.0, 18.0),
        None,
        "no host is not zero joules"
    );
    assert_eq!(energy_j(&[a], 18.0, 12.0), None);
}

fn metrics() -> BTreeMap<String, f64> {
    [
        ("c1_aggregate_tok_s", 30.0),
        ("c1_ttft_p50_ms", 400.0),
        ("c1_tpot_p50_ms", 33.0),
        ("c1_completion_tokens", 1024.0),
        ("c1_window_start_unix", 100.0),
        ("c1_window_end_unix", 134.0),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

#[test]
fn a_rung_takes_the_sweeps_numbers_and_the_windows_energy() {
    let host: Series = vec![(90.0, 0.0), (140.0, 50_000.0)];
    let rs = rungs(&[1, 2], &metrics(), &[host]);
    assert_eq!(rs[0].tok_s, Some(30.0));
    assert_eq!(rs[0].ttft_p50_ms, Some(400.0));
    assert_eq!(rs[0].tpot_p50_ms, Some(33.0));
    assert!(rs[0].comparable);
    let j = rs[0].energy_j.unwrap();
    assert!((j - 34.0).abs() < 1e-9, "{j}");
    assert!((rs[0].j_per_tok.unwrap() - 34.0 / 1024.0).abs() < 1e-12);
    // 2026-10-10: C=2 was not published by the sweep: no number of any kind.
    assert!(!rs[1].comparable);
    assert_eq!(
        (rs[1].tok_s, rs[1].energy_j, rs[1].j_per_tok),
        (None, None, None)
    );
}

#[test]
fn without_energy_hosts_jtok_is_absent_not_zero() {
    let rs = rungs(&[1], &metrics(), &[]);
    assert_eq!((rs[0].energy_j, rs[0].j_per_tok), (None, None));
    assert_eq!(rs[0].tok_s, Some(30.0));
}
