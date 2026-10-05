// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The build's own provenance, baked in by `crates/server/build.rs` and
//! printed by `met --version`'s long form — split out of `cli.rs` to keep that file
//! under the 500-line cap (`.github/scripts/check-file-size-cap.sh`).
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

/// 2026-10-05: The full git commit `build.rs` saw at compile time (`METRALE_BUILD_GIT_SHA`,
/// `crates/server/build.rs`), 40 lowercase hex chars, or the literal `"unknown"` when the
/// build tree was not a git checkout and the build ran without `METRALE_BUILD_GIT_SHA` set.
///
/// `"unknown"` is deliberate (PCND), not a silent substitute for a real sha: it never equals
/// a real `--expect-head` value, so the preflight binary-provenance check
/// (`cli::bench_preflight`) fails closed on a build that could not be attributed, instead of
/// comparing against a guessed commit.
pub const BUILD_GIT_SHA: &str = env!("METRALE_BUILD_GIT_SHA");

/// 2026-10-05: `"true"`, `"false"`, or `"unknown"` (git unavailable at build time — see
/// [`BUILD_GIT_SHA`]), from `crates/server/build.rs`. A plain `&str` rather than `bool`: the
/// three-valued "unknown" state has no safe default to collapse into, and the preflight check
/// that reads this treats `"unknown"` as a dirty tree would (never a pass).
pub const BUILD_GIT_DIRTY: &str = env!("METRALE_BUILD_GIT_DIRTY");

/// 2026-10-05: `met --version`'s long form: the package version plus the build's own
/// provenance, so a binary on disk can be identified without running it against a recipe
/// first. `-V`/`met --version` without the long form keeps printing [`super::METRALE_VERSION`]
/// alone (`Command::version`); `--version`'s long form additionally carries this
/// (`Command::long_version`), which `clap` falls back to `version` for when absent — set here
/// so the two are never silently out of sync.
pub const METRALE_LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (commit ",
    env!("METRALE_BUILD_GIT_SHA"),
    ", dirty=",
    env!("METRALE_BUILD_GIT_DIRTY"),
    ")"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, METRALE_VERSION};
    use clap::Parser;

    /// 2026-10-05: `--version` (the long form) carries the build's own commit and dirty
    /// flag, not just the package version — the thing a preflight check or a human reads to
    /// tell two binaries on disk apart. `-V` keeps printing the short `METRALE_VERSION`
    /// alone; this test is about the long flag specifically.
    #[test]
    fn long_version_flag_carries_commit_and_dirty_flag() {
        let err = Cli::try_parse_from(["met", "--version"]).expect_err("exits early");
        let printed = err.to_string();
        assert!(printed.contains(METRALE_VERSION), "{printed:?}");
        assert!(
            printed.contains(BUILD_GIT_SHA),
            "`--version` printed {printed:?}, which does not carry the build commit \
             {BUILD_GIT_SHA:?}"
        );
        assert!(
            printed.contains(BUILD_GIT_DIRTY),
            "`--version` printed {printed:?}, which does not carry the dirty flag \
             {BUILD_GIT_DIRTY:?}"
        );
    }

    /// 2026-10-05: [`BUILD_GIT_SHA`] is never empty (it is `"unknown"` at worst — see its
    /// doc), so a preflight comparison against `--expect-head` never degrades into an
    /// empty-string match.
    #[test]
    fn build_git_sha_is_never_empty() {
        assert!(!BUILD_GIT_SHA.is_empty());
    }

    /// 2026-10-05: [`BUILD_GIT_DIRTY`] is one of exactly three values — never, for example,
    /// an empty string a careless `== "true"` check would silently read as not-dirty.
    #[test]
    fn build_git_dirty_is_one_of_three_values() {
        assert!(
            matches!(BUILD_GIT_DIRTY, "true" | "false" | "unknown"),
            "{BUILD_GIT_DIRTY:?}"
        );
    }
}
