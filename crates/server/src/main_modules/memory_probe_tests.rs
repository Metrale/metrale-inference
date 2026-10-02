// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Tests for `GET /memory`'s document and the `/proc` parsing behind it.
//!
//! Owner: server (HTTP layer).
//! Invariants: none beyond the types.

use super::{
    DeviceBudget, MemoryFacts, Readings, budget_bytes, mem_available_bytes, rss_anon_bytes,
};

const MEMINFO: &str = "MemTotal:       125366596 kB\n\
                       MemFree:         2034512 kB\n\
                       MemAvailable:   98765432 kB\n\
                       Buffers:           12345 kB\n";

const STATUS: &str = "Name:\tmet\n\
                      VmRSS:\t 3145728 kB\n\
                      RssAnon:\t 2097152 kB\n\
                      RssFile:\t 1048576 kB\n";

fn facts() -> MemoryFacts {
    MemoryFacts {
        kv_blocks: 6789,
        kv_block_bytes: 524_288,
        max_batch_size: 128,
        device: DeviceBudget {
            budget_bytes: budget_bytes(128 * 1024 * 1024 * 1024, 0.85),
            ledger: None,
        },
    }
}

/// 2026-10-01: The exact document, every field present, and a reading the server could not
/// take serialised as `null` rather than left out or zeroed.
#[test]
fn the_document_carries_every_field_and_null_for_a_missing_reading() {
    let readings = Readings {
        ledger_live_bytes: Some(70_000_000_000),
        mem_available_at_start_bytes: Some(120_000_000_000),
        mem_available_now_bytes: None,
        rss_anon_bytes: Some(2_147_483_648),
    };
    let doc = serde_json::to_value(facts().report(readings)).unwrap();
    assert_eq!(
        doc,
        serde_json::json!({
            "kv_blocks": 6789,
            "kv_block_bytes": 524_288,
            "max_batch_size": 128,
            "budget_bytes": 116_823_110_451_u64,
            "ledger_live_bytes": 70_000_000_000_u64,
            "mem_available_at_start_bytes": 120_000_000_000_u64,
            "mem_available_now_bytes": null,
            "rss_anon_bytes": 2_147_483_648_u64,
        })
    );
}

/// 2026-10-01: Without a ledger the request-time reading is `None`; with one, it is what the
/// ledger says when the request is served, not when the facts were built.
#[test]
fn the_ledger_is_read_per_request() {
    assert_eq!(facts().readings().ledger_live_bytes, None);
    let live = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(10));
    let mut with_ledger = facts();
    let reader = std::sync::Arc::clone(&live);
    with_ledger.device.ledger = Some(std::sync::Arc::new(move || {
        reader.load(std::sync::atomic::Ordering::SeqCst)
    }));
    assert_eq!(with_ledger.readings().ledger_live_bytes, Some(10));
    live.store(25, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(with_ledger.readings().ledger_live_bytes, Some(25));
}

#[test]
fn proc_fields_are_read_in_bytes() {
    assert_eq!(mem_available_bytes(MEMINFO), Some(98_765_432 * 1024));
    assert_eq!(rss_anon_bytes(STATUS), Some(2_097_152 * 1024));
}

/// 2026-10-01: A key that is absent, a prefix of another key, or not in `<n> kB` form is no
/// reading.
#[test]
fn a_malformed_or_absent_field_is_no_reading() {
    assert_eq!(rss_anon_bytes(MEMINFO), None);
    assert_eq!(mem_available_bytes("MemAvailableX:  10 kB\n"), None);
    assert_eq!(mem_available_bytes("MemAvailable:   10 MB\n"), None);
    assert_eq!(mem_available_bytes("MemAvailable:   ten kB\n"), None);
    assert_eq!(mem_available_bytes("MemAvailable:   10\n"), None);
    assert_eq!(mem_available_bytes("MemAvailable:   10 kB extra\n"), None);
    assert_eq!(
        mem_available_bytes(&format!("MemAvailable: {} kB\n", u64::MAX)),
        None,
        "a value whose byte count overflows is no reading"
    );
}

/// 2026-10-01: The parsers against this kernel's own files, so a format they do not read
/// fails here rather than as a `null` in a gate record.
#[cfg(target_os = "linux")]
#[test]
fn this_hosts_proc_files_parse() {
    let meminfo = super::read_proc("/proc/meminfo").expect("/proc/meminfo is readable");
    assert!(mem_available_bytes(&meminfo).is_some_and(|b| b > 0));
    let status = super::read_proc("/proc/self/status").expect("/proc/self/status is readable");
    assert!(rss_anon_bytes(&status).is_some_and(|b| b > 0));
}
