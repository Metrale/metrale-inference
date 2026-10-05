// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The decode-step redzone-scan env lookups used by `decode_a.rs`'s
//! `decode_dispatch_with`. Split out to keep that file under the 500-line cap; no
//! behavior changes.
//!
//! Owner: model-engine (decode).

/// 2026-09-25: `METRALE_REDZONE_RANGE_FILE=<path>`: a file holding "LO HI", re-read
/// every decode step and passed to `poison_redzones`. Unset: no bisection.
pub(super) fn redzone_range_file() -> Option<&'static str> {
    static P: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    P.get_or_init(|| std::env::var("METRALE_REDZONE_RANGE_FILE").ok())
        .as_deref()
}

/// 2026-09-25: Scan the guard bands every `n`-th decode step: 0 (never) when
/// `METRALE_REDZONE` is unset, else `METRALE_REDZONE_EVERY` (1 when unset or
/// unparsable; 0 disables the scan).
pub(super) fn redzone_every() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        if std::env::var("METRALE_REDZONE").is_err() {
            return 0;
        }
        std::env::var("METRALE_REDZONE_EVERY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
    })
}
