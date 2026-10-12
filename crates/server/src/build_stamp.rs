// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The commit this binary was built from: `METRALE_BUILD_COMMIT`, read once at
//! compile time (investor-mvp #28). Release and image builds stamp it from
//! `scripts/build-commit.sh` (`docker/build_commit_stamp_test.sh` holds every build path to
//! that); a local build is unstamped. Every reader goes through [`build_commit`]: the usage
//! records, the `/metrics` build line, `RuntimeHealth` and the issue reporter.
//!
//! Owner: server (build identity).
//! Invariants: only a full 40-hex commit counts as a stamp; anything else (an empty
//! `--build-arg`, `unknown`, a short sha) is unstamped, never passed on.

use metrale_qci_dump::contract::ids::GitCommit;

/// 2026-10-11: The raw compile-time stamp, as the build passed it.
pub(crate) const RAW: Option<&str> = option_env!("METRALE_BUILD_COMMIT");

/// 2026-10-11: The stamped commit, or `None` for an unstamped build.
pub(crate) fn build_commit() -> Option<GitCommit> {
    stamp_of(RAW)
}

/// 2026-10-11: The commit a raw stamp names, when it names one.
pub(crate) fn stamp_of(raw: Option<&str>) -> Option<GitCommit> {
    raw.and_then(|s| GitCommit::from_hex(s).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMIT: &str = "f2e18ffc17e2595362b5468aac34221e09934bf8";

    #[test]
    fn only_a_full_commit_is_a_stamp() {
        assert_eq!(
            stamp_of(Some(COMMIT)).map(|c| c.to_hex()).as_deref(),
            Some(COMMIT)
        );
        // 2026-10-11: What the Dockerfiles pass when the build arg is missing, the old label
        // default, a short sha and nothing at all.
        for raw in [Some(""), Some("unknown"), Some("f2e18ff"), None] {
            assert_eq!(stamp_of(raw), None, "{raw:?}");
        }
    }
}
