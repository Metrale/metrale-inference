// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `cargo xtask selftest`: the deps gate runs on each fixture workspace under
//! `xtask/fixtures/`. A bad fixture must be refused under exactly its rule, at exactly the
//! dependencies listed for it, and nowhere else; the clean fixture must pass. A fixture that
//! passes, or fails for another reason or at another place, fails the selftest.

use crate::rules::{self, Rule, Violation};
use crate::shell;
use std::path::Path;

/// 2026-10-17: A dependency the gate must refuse: manifest, table header and key.
struct Hit {
    file: &'static str,
    table: &'static str,
    dep: &'static str,
}

/// 2026-10-17: A fixture, the rule it breaks (`None` for the clean one) and where it breaks it.
struct Probe {
    fixture: &'static str,
    rule: Option<Rule>,
    hits: &'static [Hit],
}

const fn hit(file: &'static str, table: &'static str, dep: &'static str) -> Hit {
    Hit { file, table, dep }
}

const APP: &str = "crates/app/Cargo.toml";

/// 2026-10-17: Each bad fixture breaks its rule in several shapes and in several kinds of table
/// (normal, dev, build, target-specific), so a rule that misses a shape or a table fails here.
const PROBES: [Probe; 4] = [
    Probe {
        fixture: "clean",
        rule: None,
        hits: &[],
    },
    Probe {
        fixture: "bad-root-default-features",
        rule: Some(Rule::RootDefaultFeatures),
        hits: &[
            hit("Cargo.toml", "workspace.dependencies", "anyhow"),
            hit("Cargo.toml", "workspace.dependencies", "cc"),
            hit("Cargo.toml", "workspace.dependencies", "fixture-lib"),
            hit("Cargo.toml", "workspace.dependencies", "serde"),
            hit("Cargo.toml", "workspace.dependencies", "thiserror"),
        ],
    },
    Probe {
        fixture: "bad-member-bypasses-workspace",
        rule: Some(Rule::MemberBypassesWorkspace),
        hits: &[
            hit(APP, "build-dependencies", "fixture-lib"),
            hit(APP, "dependencies", "anyhow"),
            hit(APP, "dependencies", "serde"),
            hit(APP, "dev-dependencies", "tempfile"),
            hit(APP, "dev-dependencies", "thiserror"),
            hit(APP, "target.'cfg(unix)'.dependencies", "libc"),
        ],
    },
    Probe {
        fixture: "bad-member-undeclared",
        rule: Some(Rule::MemberUndeclared),
        hits: &[
            hit(APP, "build-dependencies", "cc"),
            hit(APP, "dependencies", "fixture-other"),
            hit(APP, "dependencies", "regex"),
            hit(APP, "target.'cfg(unix)'.dev-dependencies", "tempfile"),
        ],
    },
];

/// 2026-10-17: Run every probe against the fixture trees under `fixtures`; returns how many did
/// not end as intended. An unreadable or malformed fixture is an error, not a pass.
pub fn run(fixtures: &Path) -> Result<usize, String> {
    let mut failures: usize = 0;
    for probe in &PROBES {
        let ws = shell::load_workspace(&fixtures.join(probe.fixture))?;
        let got = rules::check(&ws)?;
        let (ok, why) = judge(probe, &got);
        let mark = if ok { "PASS" } else { "FAIL" };
        println!("[{mark}] selftest {}: {why}", probe.fixture);
        if !ok {
            failures = failures.saturating_add(1);
        }
    }
    failures = failures.saturating_add(crate::env_ban::selftest(fixtures)?);
    Ok(failures)
}

fn judge(probe: &Probe, got: &[Violation]) -> (bool, String) {
    let listing = got
        .iter()
        .map(|v| format!("\n    {v}"))
        .collect::<Vec<_>>()
        .concat();
    let Some(rule) = probe.rule else {
        return if got.is_empty() {
            (true, "the clean fixture passes".to_owned())
        } else {
            (false, format!("the clean fixture was refused:{listing}"))
        };
    };
    if got.is_empty() {
        return (
            false,
            format!("the gate PASSED a fixture that breaks {rule}"),
        );
    }
    let mut want: Vec<(Rule, &str, &str, &str)> = probe
        .hits
        .iter()
        .map(|h| (rule, h.file, h.table, h.dep))
        .collect();
    let mut have: Vec<(Rule, &str, &str, &str)> = got
        .iter()
        .map(|v| (v.rule, v.file.as_str(), v.table.as_str(), v.dep.as_str()))
        .collect();
    want.sort_unstable();
    have.sort_unstable();
    if want == have {
        (
            true,
            format!("refused {} dependencies under {rule}{listing}", want.len()),
        )
    } else {
        let expected = probe
            .hits
            .iter()
            .map(|h| format!("\n    {} [{}] `{}`", h.file, h.table, h.dep))
            .collect::<Vec<_>>()
            .concat();
        (
            false,
            format!("expected exactly these under {rule}:{expected}\n  got:{listing}"),
        )
    }
}

#[cfg(test)]
mod tests {
    // 2026-10-17: The same probes `cargo xtask selftest` runs, so `cargo test --workspace`
    // also proves each rule refuses its fixture for its reason and the clean tree passes.
    #[test]
    fn every_fixture_is_judged_for_its_reason() {
        assert_eq!(super::run(&crate::shell::fixtures_dir()), Ok(0));
    }
}
