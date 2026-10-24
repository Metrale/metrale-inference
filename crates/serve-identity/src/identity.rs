// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What makes one `met serve` "the same server" as another: the
//! bytes of the binary, the arguments it was started with, and its
//! `METRALE_*` serve levers.
//!
//! A benchmark that reuses a server somebody else started must know it is
//! measuring what it would have started itself, or its record names a config
//! it never ran. The server publishes the identity as digests
//! (`GET /serve-config`), never as the arguments themselves, because an argv
//! can carry `--auth-token`. Both sides compute the digests with the
//! functions here.
//!
//! The environment is part of the identity because levers are not recipe
//! keys: two servers with different levers can have byte-identical argv. The
//! digest is `env::fingerprint` over the process's levers, which covers every
//! `METRALE_*` variable except the harness class. The harness computes the
//! expected digest from the lever set the recipe declares. The reuse check
//! errs toward refusing: refusing a reusable server costs one server start,
//! while reusing a wrong one corrupts a record.
//!
//! 2026-10-17: Moved from metrale-bench (`serve_identity`) so `met` publishes
//! its identity without the benchmark harness; reading this process (argv,
//! executable, environment) moved to `metrale-host-probe::serve_identity`.
//!
//! Owner: serve identity.
//! Invariants:
//! - Every digest this module computes is 64 lowercase hex characters.

use alloc::string::String;

use sha2::{Digest, Sha256};

/// 2026-09-26: The SHA-256 of a server's arguments (everything after the
/// program name), each NUL-terminated so `["a b"]` and `["a", "b"]` differ.
#[must_use]
pub fn argv_fingerprint(args_after_program: &[String]) -> String {
    let mut h = Sha256::new();
    for a in args_after_program {
        h.update(a.as_bytes());
        h.update([0u8]);
    }
    crate::hex_lower(&h.finalize())
}

/// 2026-09-26: What `GET /serve-config` answers.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ServeIdentity {
    pub argv_sha256: String,
    pub binary_sha256: String,
    /// 2026-09-26: The digest of the `METRALE_*` serve levers in this
    /// server's environment (`env::fingerprint` over the process's levers).
    /// The argv says which recipe rendering the server runs; this says which
    /// levers it reads beside it.
    ///
    /// `serde(default)`, so a report without the field still parses, as the
    /// empty string, which [`env_is_unknown`] treats as unknown and the reuse
    /// check refuses: "cannot tell" must not read as "matches".
    #[serde(default)]
    pub env_sha256: String,
    pub pid: u32,
}

/// 2026-09-26: Whether a reported lever digest carries no information: the
/// empty string, which a report without [`ServeIdentity::env_sha256`] parses
/// to. A real digest is 64 hex characters even for an empty lever set.
#[must_use]
pub fn env_is_unknown(reported: &str) -> bool {
    reported.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fingerprint_separates_arguments_and_orders_them() {
        let a = argv_fingerprint(&["serve".into(), "m".into(), "--port".into(), "1".into()]);
        let b = argv_fingerprint(&["serve".into(), "m".into(), "--port".into(), "2".into()]);
        let c = argv_fingerprint(&["serve m".into(), "--port".into(), "1".into()]);
        assert_ne!(a, b, "a different port is a different server");
        assert_ne!(a, c, "joined arguments are not the same argv");
        assert_eq!(
            a,
            argv_fingerprint(&["serve".into(), "m".into(), "--port".into(), "1".into()])
        );
        assert_eq!(a.len(), 64);
    }

    /// 2026-09-26: A report without the lever digest still parses, with a
    /// digest that never equals a real one, so the reuse check reads it as
    /// unknown.
    #[test]
    fn an_older_server_reports_no_lever_digest() {
        let old: ServeIdentity =
            serde_json::from_str(r#"{"argv_sha256":"a","binary_sha256":"b","pid":1}"#).unwrap();
        assert_eq!(old.env_sha256, "");
        assert!(env_is_unknown(&old.env_sha256));
        let empty_set = crate::env::fingerprint(&Default::default());
        assert!(!env_is_unknown(&empty_set));
        assert_ne!(
            old.env_sha256, empty_set,
            "even an empty lever set has a digest an old server cannot claim"
        );
    }
}
