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

/// 2026-10-17: The files among `files` (path, contents) over the cap. Every workspace member
/// has a source file, so no file in scope means the listing failed, not that the tree is clean:
/// that is an error, never a pass.
pub fn findings<'a>(
    files: impl IntoIterator<Item = (&'a str, &'a [u8])>,
) -> Result<Findings, String> {
    let mut out = Findings::new();
    let mut scanned = 0_u64;
    for (path, bytes) in files {
        scanned = scanned.saturating_add(1);
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
    if scanned == 0 {
        return Err(
            "file-size: no source file was in scope, so the cap scanned nothing".to_owned(),
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{findings, in_scope, line_count};

    #[test]
    fn an_empty_scan_is_an_error_and_a_clean_one_is_not() {
        assert!(findings([]).is_err());
        let small: &[u8] = b"fn main() {}\n";
        assert!(findings([("a.rs", small)]).unwrap().is_empty());
        let big = b"x\n".repeat(501);
        assert_eq!(
            findings([("b.rs", big.as_slice())]).unwrap()["b.rs"].count,
            Some(501)
        );
    }

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
