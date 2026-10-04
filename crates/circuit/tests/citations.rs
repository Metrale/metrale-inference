// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Every rule's citation resolves, and the routing audit lists every rule with its
//! class and citation. Split from `circuits.rs`.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;

use metrale_circuit::Numerics;

#[test]
fn every_cited_path_exists_and_holds_the_cited_line() {
    let rules = common::load(&common::instances()[0]).rules;
    let prefixes = [
        ("ml/", "crates/model-layers/src/layers/"),
        ("me/", "crates/model-engine/src/model/trait_impl/"),
        ("mm/", "crates/model-engine/src/model/"),
        ("k/", "kernels/gb10/common/"),
        ("crates/", "crates/"),
        ("kernels/", "kernels/"),
    ];
    let mut checked = 0;
    for r in &rules {
        for token in r
            .cite
            .split(|c: char| c.is_whitespace() || c == ';' || c == '(' || c == ')')
        {
            let Some((path, lines)) = token.split_once(':') else {
                continue;
            };
            let Some((short, long)) = prefixes.iter().find(|(p, _)| path.starts_with(p)) else {
                continue;
            };
            let file = format!("{long}{}", &path[short.len()..]);
            let text = std::fs::read_to_string(common::root().join(&file))
                .unwrap_or_else(|_| panic!("rule `{}` cites {file}, which does not exist", r.id));
            let max = lines
                .split([',', '-'])
                .filter_map(|n| {
                    n.trim_end_matches(|c: char| !c.is_ascii_digit())
                        .parse::<usize>()
                        .ok()
                })
                .max();
            if let Some(max) = max {
                assert!(
                    max <= text.lines().count(),
                    "rule `{}` cites {file}:{max}, past its end",
                    r.id
                );
            }
            checked += 1;
        }
    }
    assert!(
        checked > 100,
        "only {checked} citations parsed; the cite parser is not seeing them"
    );
}

#[test]
fn the_routing_audit_lists_every_rule_with_its_class_and_citation() {
    let audit = common::read("kernels/circuits/ROUTING-AUDIT.md");
    let rules = common::load(&common::instances()[0]).rules;
    for r in &rules {
        let class = match &r.numerics {
            Numerics::Differs { lever } => format!("differs ({lever})"),
            other => other.class().to_string(),
        };
        let row = format!("| `{}` | {class} | {} |", r.id, r.cite);
        assert!(
            audit.contains(&row),
            "ROUTING-AUDIT.md lacks, or has a stale row for:\n{row}"
        );
    }
    let listed = audit
        .lines()
        .skip_while(|l| !l.starts_with("| Rule | Numerics |"))
        .skip(2)
        .take_while(|l| l.starts_with('|'))
        .count();
    assert_eq!(
        listed,
        rules.len(),
        "the audit lists a rule FUSIONS.toml does not have"
    );
}

// 2026-10-04: A rule's `scratch` regions are the memory model's workspaces (one source): each
// name, before any `.` sub-region, is a `[[family.workspace]]` of a family that lists one of the
// rule's kernels. Mutation: a misspelled or foreign region would order nothing against the real
// one, and the fork/join derivation would miss the hazard.
#[test]
fn every_scratch_region_is_a_workspace_of_its_kernels_family() {
    for inst in common::instances() {
        let rules = common::load(&inst).rules;
        let fams = common::families(&inst);
        for r in &rules {
            for name in &r.scratch {
                let base = name.split('.').next().unwrap_or(name);
                let known = fams.families.iter().any(|f| {
                    f.kernels.iter().any(|k| r.kernels.contains(k))
                        && f.workspace.iter().any(|w| w.name == base)
                });
                assert!(
                    known,
                    "{}: rule `{}` scratch `{name}` is no workspace of its kernels' families",
                    inst.recipe, r.id
                );
            }
        }
    }
}
