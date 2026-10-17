// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `cargo xtask file-size`: no source file over `MAX_LINES` lines. Every tracked
//! file with a source extension counts, outside `EXCLUDED`. Today's offenders are keyed by path
//! with their line count in `xtask/file-size-allow.txt`; an entry may not grow, and tightens as
//! its file shrinks.

use crate::allow::{Finding, Findings};

/// 2026-10-17: The cap.
pub const MAX_LINES: u64 = 500;

/// 2026-10-17: The extensions that count as source.
const EXTENSIONS: [&str; 11] = [
    "c", "cc", "cpp", "cu", "cuh", "h", "hpp", "metal", "py", "rs", "sh",
];

/// 2026-10-17: Directories outside the cap. `vendor/` holds forks of upstream crates (the
/// `metrale-cudarc` fork, with its own `.rustfmt.toml`): they keep upstream's file layout so
/// each upstream release still rebases, and splitting a file there would fork the layout too.
/// The root `bench/` tree holds experiment scripts and upstream source snapshots, not engine
/// code. The gate's own fixtures hold a file over the cap on purpose.
const EXCLUDED: [&str; 3] = ["bench/", "vendor/", "xtask/fixtures/"];

/// 2026-10-17: Whether `path` (relative, `/`-separated) is under the cap.
pub fn in_scope(path: &str) -> bool {
    let ext_ok = path
        .rsplit_once('.')
        .is_some_and(|(stem, ext)| !stem.ends_with('/') && EXTENSIONS.contains(&ext));
    ext_ok && !EXCLUDED.iter().any(|d| path.starts_with(d))
}

/// 2026-10-17: Lines as `wc -l` counts them, plus a last line without a newline.
pub fn line_count(bytes: &[u8]) -> u64 {
    let newlines = bytes.iter().filter(|b| **b == b'\n').count();
    let unterminated = usize::from(bytes.last().is_some_and(|b| *b != b'\n'));
    u64::try_from(newlines.saturating_add(unterminated)).unwrap_or(u64::MAX)
}

/// 2026-10-17: The files among `files` (path, contents) over the cap.
pub fn findings<'a>(files: impl IntoIterator<Item = (&'a str, &'a [u8])>) -> Findings {
    let mut out = Findings::new();
    for (path, bytes) in files {
        let lines = line_count(bytes);
        if lines > MAX_LINES {
            out.insert(
                path.to_owned(),
                Finding {
                    count: Some(lines),
                    why: format!("{lines} lines, over the {MAX_LINES}-line cap"),
                },
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{in_scope, line_count};

    #[test]
    fn lines_count_like_wc_plus_an_unterminated_tail() {
        assert_eq!(line_count(b""), 0);
        assert_eq!(line_count(b"a\n"), 1);
        assert_eq!(line_count(b"a\nb"), 2);
        assert_eq!(line_count(b"\n\n"), 2);
    }

    #[test]
    fn scope_takes_source_extensions_outside_the_excluded_trees() {
        for path in [
            "crates/a/src/lib.rs",
            "crates/bench/src/lib.rs",
            "kernels/x.cu",
            "scripts/a.sh",
            "k/y.cuh",
        ] {
            assert!(in_scope(path), "{path}");
        }
        for path in [
            "vendor/cudarc/src/lib.rs",
            "bench/run.py",
            "xtask/fixtures/big/src/lib.rs",
            "README.md",
            "crates/a/.rs",
            "Makefile",
        ] {
            assert!(!in_scope(path), "{path}");
        }
    }
}
