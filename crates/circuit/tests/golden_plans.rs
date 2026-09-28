// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The checked-in plans under kernels/circuits/plans/ must equal what the rules
//! produce today. Regenerate after an intended change with
//! `cargo test -p metrale-circuit --test golden_plans -- --ignored regenerate`.
//!
//! Owner: metrale-circuit tests.
//! Invariants: a stale, missing or extra plan file fails `golden_plans_match`.

mod common;

use std::collections::BTreeSet;

#[test]
fn golden_plans_match() {
    let dir = common::plans_dir();
    let want = common::golden_plans();
    let mut problems = Vec::new();
    for (name, text) in &want {
        match std::fs::read_to_string(dir.join(name)) {
            Ok(on_disk) if on_disk == *text => {}
            Ok(on_disk) => problems.push(describe(name, &on_disk, text)),
            Err(_) => problems.push(format!("{name}: missing")),
        }
    }
    let expected: BTreeSet<&str> = want.iter().map(|(n, _)| n.as_str()).collect();
    for entry in std::fs::read_dir(&dir).expect("plans dir").flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".txt") && !expected.contains(name.as_str()) {
            problems.push(format!("{name}: no instance produces it"));
        }
    }
    assert!(
        problems.is_empty(),
        "golden plans are stale; regenerate with \
         `cargo test -p metrale-circuit --test golden_plans -- --ignored regenerate` \
         and review the diff:\n{}",
        problems.join("\n")
    );
}

/// 2026-09-28: The first differing line of the plan body, or a note that only the digest line
/// moved (a rule or the policy changed, and this plan did not).
fn describe(name: &str, on_disk: &str, want: &str) -> String {
    let body = |t: &str| -> Vec<String> {
        t.lines()
            .filter(|l| !l.starts_with("digest: "))
            .map(str::to_string)
            .collect()
    };
    let (a, b) = (body(on_disk), body(want));
    match (0..a.len().max(b.len())).find(|&i| a.get(i) != b.get(i)) {
        Some(i) => format!(
            "{name}: plan changed at body line {}:\n  on disk:  {}\n  expected: {}",
            i + 1,
            a.get(i).map_or("<end>", String::as_str),
            b.get(i).map_or("<end>", String::as_str)
        ),
        None => format!("{name}: digest only (the plan is unchanged)"),
    }
}

#[test]
#[ignore = "writes kernels/circuits/plans/; run explicitly to regenerate"]
fn regenerate() {
    let dir = common::plans_dir();
    std::fs::create_dir_all(&dir).expect("plans dir");
    let want = common::golden_plans();
    let keep: BTreeSet<String> = want.iter().map(|(n, _)| n.clone()).collect();
    for entry in std::fs::read_dir(&dir).expect("plans dir").flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".txt") && !keep.contains(&name) {
            std::fs::remove_file(entry.path()).expect("remove stale plan");
        }
    }
    for (name, text) in want {
        std::fs::write(dir.join(name), text).expect("write plan");
    }
}
