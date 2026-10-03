// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Tests for the default-tier boot check's footprint arithmetic, its verdict and
//! the `/memory` document it reads.
//!
//! Owner: bench, default-tier-boot.
//! Invariants: none beyond the types.

use super::memory::MIB;
use super::{MemoryReport, footprint, verdict_for};
use crate::result::VerdictKind;

const MIB_U: u64 = 1024 * 1024;

/// 2026-10-01: A report whose device footprint is `device_mib` against a 1000 MiB budget:
/// 50 GiB available at start, 300 MiB of anonymous RSS.
fn report_with_footprint(device_mib: u64) -> MemoryReport {
    let start = 50 * 1024 * MIB_U;
    let rss = 300 * MIB_U;
    MemoryReport {
        kv_blocks: 4321,
        kv_block_bytes: 2 * MIB_U,
        max_batch_size: 128,
        budget_bytes: 1000 * MIB_U,
        ledger_live_bytes: Some(900 * MIB_U),
        mem_available_at_start_bytes: Some(start),
        mem_available_now_bytes: Some(start - rss - device_mib * MIB_U),
        rss_anon_bytes: Some(rss),
    }
}

/// 2026-10-01: Under, at and over the budget. The footprint subtracts the process's
/// anonymous RSS from the drop in `MemAvailable`, and the over-budget figure is its distance
/// from the budget, signed, with exact zero at the budget.
#[test]
fn the_footprint_is_signed_against_the_budget() {
    for (device_mib, over_mib) in [(900.0, -100.0), (1000.0, 0.0), (1250.0, 250.0)] {
        let f = footprint(&report_with_footprint(device_mib as u64)).expect("all readings");
        assert_eq!(f.device_footprint_mib, device_mib);
        assert_eq!(f.budget_mib, 1000.0);
        assert_eq!(
            f.footprint_over_budget_mib, over_mib,
            "{device_mib} MiB against 1000"
        );
        assert_eq!(f.kv_blocks, 4321);
        assert_eq!(f.max_batch_size, 128);
        assert_eq!(f.ledger_live_mib, Some(900.0));

        let mut m = std::collections::BTreeMap::new();
        f.metrics(&mut m);
        assert_eq!(
            m.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "budget_mib",
                "device_footprint_mib",
                "footprint_over_budget_mib",
                "kv_blocks",
                "ledger_live_mib",
                "max_batch_size",
            ]
        );
        assert_eq!(m["footprint_over_budget_mib"], over_mib);
        assert_eq!(m["kv_blocks"], 4321.0);
        assert_eq!(verdict_for(&Ok(f)).kind, VerdictKind::Info);
    }
}

/// 2026-10-01: A footprint that is not a whole number of MiB keeps its fraction, and the
/// over-budget figure is computed in bytes before the unit change.
#[test]
fn sub_mib_footprints_are_not_rounded() {
    let mut r = report_with_footprint(1000);
    r.mem_available_now_bytes = r.mem_available_now_bytes.map(|b| b - 1);
    let f = footprint(&r).unwrap();
    assert_eq!(f.footprint_over_budget_mib, 1.0 / MIB);
    assert!(f.footprint_over_budget_mib > 0.0);
}

/// 2026-10-01: A server without a ledger reports `null` for it; the footprint still
/// derives, and the record carries no ledger key rather than a zero.
#[test]
fn a_missing_ledger_drops_only_its_key() {
    let mut r = report_with_footprint(900);
    r.ledger_live_bytes = None;
    let f = footprint(&r).unwrap();
    let mut m = std::collections::BTreeMap::new();
    f.metrics(&mut m);
    assert!(!m.contains_key("ledger_live_mib"));
    assert_eq!(m["device_footprint_mib"], 900.0);
}

/// 2026-10-01: A document whose host readings are partly `null`, as a server sends for a
/// counter it could not read. The run names exactly the missing ones and is inconclusive,
/// never informational.
#[test]
fn null_host_readings_are_inconclusive_and_named() {
    let doc = serde_json::json!({
        "kv_blocks": 10,
        "kv_block_bytes": 4096,
        "max_batch_size": 8,
        "budget_bytes": 1_000_000,
        "ledger_live_bytes": null,
        "mem_available_at_start_bytes": null,
        "mem_available_now_bytes": 5,
        "rss_anon_bytes": null,
    });
    let r: MemoryReport = serde_json::from_value(doc).expect("nulls are valid readings");
    let err = footprint(&r).unwrap_err();
    assert!(
        err.contains("mem_available_at_start_bytes, rss_anon_bytes"),
        "{err}"
    );
    assert!(!err.contains("mem_available_now_bytes"), "{err}");
    let v = verdict_for(&Err(err));
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.starts_with("INCONCLUSIVE"), "{}", v.reason);
}
