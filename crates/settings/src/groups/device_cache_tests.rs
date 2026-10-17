// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `CacheTuning` properties: the declared tuning and the projection errors.
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use metrale_error::ErrorCode;

use super::CacheTuning;
use crate::groups::equivalence::resolved;

#[test]
fn declared_tuning_is_every_lever_off_but_the_tail_lease() {
    assert_eq!(
        CacheTuning::declared().expect("declared defaults resolve"),
        CacheTuning {
            prefix_subblock: false,
            kv_trace: false,
            tail_lease: true,
            tail_lease_ttl: 64,
            snap_evict_alpha: 0.0,
            snap_lookup_dbg: false,
            ssm_snap_stats: false,
            snap_evict_legacy: false,
            hermetic: false,
        }
    );
}

#[test]
fn a_lease_ttl_outside_u32_is_projection_out_of_range() {
    for raw in ["-1", "4294967296"] {
        let s = resolved(&[("METRALE_SSM_TAIL_LEASE_TTL", raw)]).expect("a valid Int");
        let e = CacheTuning::try_from(&s).expect_err("out of range");
        assert_eq!(e.code(), ErrorCode::ProjectionOutOfRange, "{raw}");
    }
}
