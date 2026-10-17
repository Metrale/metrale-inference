// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `CacheTuning` properties that need no old reader. The equivalence with the readers
//! it replaces is proven where those readers live, against their `legacy_env_*` forms (cache
//! `legacy_env_tests.rs`).
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use super::{CacheTuning, TAIL_LEASE_TTL_BUILT_IN};

#[test]
fn declared_tuning_is_every_lever_off_but_the_tail_lease() {
    assert_eq!(
        CacheTuning::declared().expect("declared defaults resolve"),
        CacheTuning {
            prefix_subblock: false,
            kv_trace: false,
            tail_lease: true,
            tail_lease_ttl: TAIL_LEASE_TTL_BUILT_IN,
            snap_evict_alpha: 0.0,
            snap_lookup_dbg: false,
            ssm_snap_stats: false,
            snap_evict_legacy: false,
            hermetic: false,
        }
    );
}
