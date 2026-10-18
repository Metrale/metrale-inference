// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The env-ban selftest. Each fixture under `xtask/fixtures/env-ban-*` must yield
//! exactly its listed violations (kind, path, line) and no others; a fixture that yields more,
//! fewer or different ones fails the selftest.

use super::check;
use std::path::Path;

/// 2026-10-17: A fixture and the violations it must yield: `(kind, path or entry, line)`.
struct Probe {
    fixture: &'static str,
    want: &'static [(&'static str, &'static str, usize)],
}

const LIB: &str = "crates/app/src/lib.rs";
const ALIAS: &str = "crates/app/src/alias.rs";

const PROBES: [Probe; 4] = [
    // 2026-10-17: Unlisted reads in every shape fail at their exact lines; comments, strings,
    // chars and other receivers do not; renamed imports and gpu-sys env helpers resolve; a
    // crate-root build.rs, a build_*.rs helper and the host env file pass; a src/build.rs
    // module is not a build script and fails.
    Probe {
        fixture: "env-ban-unlisted",
        want: &[
            ("unlisted", LIB, 7),
            ("unlisted", LIB, 8),
            ("unlisted", LIB, 9),
            ("unlisted", LIB, 13),
            ("unlisted", LIB, 15),
            ("unlisted", ALIAS, 3),
            ("unlisted", ALIAS, 4),
            ("unlisted", ALIAS, 7),
            ("unlisted", ALIAS, 9),
            ("unlisted", "crates/app/src/build.rs", 4),
        ],
    },
    // 2026-10-17: The same kind of reads, listed: passes.
    Probe {
        fixture: "env-ban-listed",
        want: &[],
    },
    // 2026-10-17: Entries that must go: a missing file, a file without reads, an exempt build
    // script; the reads of the file whose entry carries a line number stay unlisted.
    Probe {
        fixture: "env-ban-stale",
        want: &[
            ("stale-exempt", "crates/app/build.rs", 3),
            ("stale-missing", "crates/app/src/gone.rs", 5),
            ("stale-no-read", "crates/app/src/clean.rs", 4),
            ("malformed", "crates/app/src/lib.rs:4", 6),
            ("unlisted", LIB, 4),
        ],
    },
    // 2026-10-17: Out-of-order and duplicate entries are refused; the files are still listed.
    Probe {
        fixture: "env-ban-unsorted",
        want: &[
            ("unsorted", "crates/app/src/a.rs", 3),
            ("unsorted", "crates/app/src/b.rs", 4),
        ],
    },
];

/// 2026-10-17: Run every probe; returns how many did not end as intended. An unreadable
/// fixture is an error, not a pass.
pub fn run(fixtures: &Path) -> Result<usize, String> {
    let mut failures: usize = 0;
    for probe in &PROBES {
        let (_, _, got) = check(&fixtures.join(probe.fixture))?;
        let mut have: Vec<(&str, &str, usize)> = got.iter().map(|v| v.key()).collect();
        let mut want = probe.want.to_vec();
        have.sort_unstable();
        want.sort_unstable();
        let listing: String = got.iter().map(|v| format!("\n    {v}")).collect();
        let ok = have == want;
        let mark = if ok { "PASS" } else { "FAIL" };
        println!(
            "[{mark}] selftest {}: {} violation(s), {} expected{listing}",
            probe.fixture,
            have.len(),
            want.len()
        );
        if !ok {
            failures = failures.saturating_add(1);
        }
    }
    Ok(failures)
}
