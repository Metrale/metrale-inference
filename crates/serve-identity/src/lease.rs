// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The leased server's record and the pure reuse decision: one
//! `met serve` that outlives a `met benchmark run --pull-request-gate
//! --serve-reuse` run is reused by the next only when it reports the identity
//! the next run would start.
//!
//! 2026-10-17: Moved from `met`'s `cli/bench_lease.rs`, which keeps the lease
//! file, the probe and the process control.
//!
//! Owner: serve identity (serve lease).
//! Invariants:
//! - [`mismatch`] is `None` only when the reported pid, binary, argv and lever
//!   digests and the model all match; a server that reports no lever digest is
//!   refused.

use alloc::format;
use alloc::string::String;

use crate::identity::{ServeIdentity, env_is_unknown};

/// 2026-09-26: The leased server, as recorded in `serve-lease.json`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Lease {
    pub pid: u32,
    pub port: u16,
    pub model: String,
    pub recipe_id: String,
    pub argv_sha256: String,
    pub binary_sha256: String,
    /// 2026-09-26: `env::fingerprint` of the lever set the server was
    /// started under. Recorded only: `mismatch` compares the server's own
    /// report, not this copy. Empty when the file lacks the field.
    #[serde(default)]
    pub env_sha256: String,
    /// 2026-09-26: The `--serve-lease-owner` pid (a campaign driver), or the
    /// run's own pid when none was given.
    pub owner_pid: u32,
    pub started_at: u64,
}

/// 2026-09-26: What a reused server must report: this binary, the plan's
/// rendering on the leased port, and the plan's lever set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expected {
    pub argv_sha256: String,
    pub binary_sha256: String,
    pub env_sha256: String,
}

/// 2026-09-26: Why a leased server is not the one this run needs, or `None`
/// when it is. Pure.
pub fn mismatch(
    lease: &Lease,
    reported: &ServeIdentity,
    expected: &Expected,
    model: &str,
) -> Option<String> {
    if reported.pid != lease.pid {
        return Some(format!(
            "pid {} answered on port {}, the lease names pid {}",
            reported.pid, lease.port, lease.pid
        ));
    }
    if reported.binary_sha256 != expected.binary_sha256 {
        return Some("it was built from another binary".into());
    }
    if reported.argv_sha256 != expected.argv_sha256 {
        return Some(format!(
            "it serves {} under another rendering (recipe {}, overrides or hermetic set differ)",
            lease.model, lease.recipe_id
        ));
    }
    // 2026-09-26: The recipe renders to flags, checked above; the `METRALE_*`
    // levers never reach argv, so they are checked here, separately, and the
    // message says which half differs. The server's own report is compared,
    // not the lease's copy, and a server that reports none is refused.
    if env_is_unknown(&reported.env_sha256) {
        return Some(
            "it does not report its METRALE_* serve environment (a server older than the \
             env digest), so its levers cannot be verified"
                .into(),
        );
    }
    if reported.env_sha256 != expected.env_sha256 {
        return Some(format!(
            "it was started under another METRALE_* serve environment than recipe {} is \
             measured under (a server carrying levers this run did not declare, or lacking \
             ones it did; argv and binary match, so this is env-only)",
            lease.recipe_id
        ));
    }
    if lease.model != model {
        return Some(format!("it serves {}, this run needs {model}", lease.model));
    }
    None
}
