// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The group declarations against the rules `lever_group!` cannot enforce by
//! itself: path and env naming, one class per group, Presence rows, and table order.
//!
//! Owner: settings.
//! Invariants: expectations are restated here (group, class, row count), never read back from
//! the code under test.

use std::collections::BTreeSet;

use crate::groups::{self, GROUPS, RULE_GROUPS, by_name, lookup, rule_of};
use crate::spec::{Class, LeverSpec, PREFIX, Rule};

/// 2026-10-17: `(group, class, rows)` in `GROUPS` order, as the 2a contract fixes them.
const EXPECTED: [(&str, Class, usize); 14] = [
    ("build", Class::Build, 12),
    ("cache", Class::Runtime, 44),
    ("dev", Class::Dev, 33),
    ("harness", Class::Harness, 9),
    ("kernels", Class::Runtime, 280),
    ("model", Class::Runtime, 27),
    ("runtime", Class::Runtime, 49),
    ("sampling", Class::Runtime, 25),
    ("scheduler", Class::Runtime, 101),
    ("server", Class::Runtime, 14),
    ("speculative", Class::Runtime, 121),
    ("storage", Class::Runtime, 17),
    ("telemetry", Class::Runtime, 93),
    ("tool", Class::Tool, 25),
];

const PRESENCE_DOC: &str = "Any value turns it on";

#[test]
fn every_group_names_its_rows_after_itself_and_has_one_class() {
    assert_eq!(GROUPS.len(), EXPECTED.len(), "group count");
    assert_eq!(RULE_GROUPS.len(), EXPECTED.len(), "rule group count");
    for ((levers, rules), (group, class, rows)) in GROUPS.iter().zip(RULE_GROUPS).zip(EXPECTED) {
        assert_eq!(levers.len(), rows, "{group}: row count");
        assert_eq!(
            rules.len(),
            levers.len(),
            "{group}: RULES parallel to LEVERS"
        );
        let prefix = format!("{group}.");
        for l in levers.iter() {
            let field = l
                .path
                .strip_prefix(&prefix)
                .unwrap_or_else(|| panic!("{}: path {} outside group {group}", l.env, l.path));
            assert_eq!(l.class, class, "{}: class", l.env);
            assert_eq!(
                l.env,
                format!("{PREFIX}{}", field.to_ascii_uppercase()),
                "{}: env is PREFIX + the path's field in upper case",
                l.path
            );
        }
    }
}

#[test]
fn by_name_is_the_whole_table_strictly_ascending_and_unique() {
    let all = by_name();
    assert_eq!(all.len(), 850, "lever count");
    for w in all.windows(2) {
        assert!(w[0].env < w[1].env, "{} before {}", w[0].env, w[1].env);
    }
    let paths: BTreeSet<&str> = all.iter().map(|l| l.path).collect();
    assert_eq!(paths.len(), all.len(), "a path is declared twice");
    for l in &all {
        assert_eq!(lookup(l.env), Some(*l), "lookup({})", l.env);
    }
    assert_eq!(lookup(PREFIX), None, "the bare prefix is not a lever");
}

#[test]
fn presence_rows_are_exactly_the_any_value_switches_and_default_off() {
    let mut presence = 0;
    for l in by_name() {
        let rule = rule_of(l.env).unwrap_or_else(|| panic!("{}: no rule", l.env));
        assert_eq!(
            rule.ty(),
            l.ty,
            "{}: rule {} vs ty {}",
            l.env,
            rule.as_str(),
            l.ty.as_str()
        );
        let says_any = l.doc.contains(PRESENCE_DOC);
        assert_eq!(
            rule == Rule::Presence,
            says_any,
            "{}: Presence iff the doc says {PRESENCE_DOC:?}",
            l.env
        );
        if rule == Rule::Presence {
            presence += 1;
            assert_eq!(l.default, "off", "{}: a Presence lever defaults off", l.env);
        }
    }
    assert_eq!(presence, 117, "Presence row count");
    assert_eq!(rule_of(PREFIX), None);
}

#[test]
fn presence_rows_sit_where_the_contract_counts_them() {
    let count = |levers: &[LeverSpec]| {
        levers
            .iter()
            .filter(|l| rule_of(l.env) == Some(Rule::Presence))
            .count()
    };
    assert_eq!(count(groups::dev::DevLevers::LEVERS), 3, "dev");
    assert_eq!(
        count(groups::harness::HarnessSettings::LEVERS),
        1,
        "harness"
    );
    assert_eq!(count(groups::build::BuildLevers::LEVERS), 0, "build");
    assert_eq!(count(groups::tool::ToolLevers::LEVERS), 0, "tool");
}
