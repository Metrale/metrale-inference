// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`AbsorbThreads`] against the environment reader it replaced (the two identical
//! `absorb_threads` copies), frozen here as [`old_threads`] taking the raw value.
//!
//! Owner: model-arch (MLA weight loading).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! value the integer rule cannot read (and a value past `i64::MAX`) is refused instead of read
//! as the available parallelism.

use super::*;
use metrale_model_layers::lever_equivalence::{PROBES, settings_from};

const ENV: &str = "METRALE_MLA_ABSORB_THREADS";

/// 2026-10-17: `glm5next_dsa::build::absorb_threads` / `phase_qk_absorbed::absorb_threads`.
fn old_threads(raw: Option<&str>, rows: usize) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .clamp(1, rows.max(1))
}

/// 2026-10-17: `ENV` at `probe` and nothing else, `None` when the settings refuse it.
fn new_threads(probe: Option<&str>, rows: usize) -> Option<usize> {
    let pairs: Vec<(&str, &str)> = probe.map(|v| (ENV, v)).into_iter().collect();
    settings_from(&pairs)
        .ok()
        .map(|s| AbsorbThreads::from_settings(&s).for_rows(rows))
}

#[test]
fn the_count_reads_as_before_except_unreadable_values() {
    let past_i64 = (i64::MAX as u64 + 1).to_string();
    let accepted = [
        None,
        Some("1"),
        Some("0"),
        Some("3"),
        Some(" 3 "),
        Some("+3"),
        Some("-3"),
        Some("100000"),
    ];
    let refused = [
        Some("true"),
        Some("off"),
        Some("garbage"),
        Some(""),
        Some("1.5"),
        Some("3x"),
        Some(past_i64.as_str()),
    ];
    for rows in [0, 1, 2, 7, 1 << 20] {
        for probe in accepted {
            assert_eq!(
                new_threads(probe, rows),
                Some(old_threads(probe, rows)),
                "{probe:?} at {rows} rows"
            );
        }
        for probe in refused {
            assert_eq!(new_threads(probe, rows), None, "{probe:?} at {rows} rows");
        }
    }
    // 2026-10-17: Every refused probe but the one past `i64::MAX` read as the default before.
    assert_eq!(
        old_threads(Some("garbage"), 1 << 20),
        old_threads(None, 1 << 20)
    );
    assert_eq!(old_threads(Some(past_i64.as_str()), 7), 7);
    for probe in PROBES {
        assert!(
            accepted.contains(&probe) || refused.contains(&probe),
            "{probe:?} is classified"
        );
    }
}

#[test]
fn the_count_is_always_within_one_to_rows() {
    let s = settings_from(&[(ENV, "64")]).expect("an Int resolves");
    let t = AbsorbThreads::from_settings(&s);
    assert_eq!(
        (t.for_rows(0), t.for_rows(1), t.for_rows(5), t.for_rows(65)),
        (1, 1, 5, 64)
    );
}
