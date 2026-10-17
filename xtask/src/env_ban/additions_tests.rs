// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The shrink-only verdict on each history shape: a reasoned addition passes, a
//! bare one fails at its own commit, the inventory exemption fires only when the merge-base has
//! no list, and the last commit to introduce an entry is the one that must explain it.

use super::{Added, Change, Verdict, judge};

fn change(sha: &str, before: Option<&str>, after: Option<&str>, reason: &str) -> Change {
    Change {
        sha: sha.to_owned(),
        before: before.map(str::to_owned),
        after: after.map(str::to_owned),
        reason: reason.to_owned(),
    }
}

fn added(entry: &str, sha: Option<&str>) -> Added {
    Added {
        entry: entry.to_owned(),
        sha: sha.map(str::to_owned),
    }
}

#[test]
fn removals_and_comments_are_not_additions() {
    let changes = [change(
        "c1",
        Some("a.rs\nb.rs\n"),
        Some("# note\na.rs\n"),
        "",
    )];
    assert_eq!(
        judge(Some("a.rs\nb.rs\n"), &changes, "# note\na.rs\n"),
        Verdict::default()
    );
}

#[test]
fn an_addition_needs_a_reason_on_its_own_commit() {
    let changes = [
        change(
            "c1",
            Some("a.rs\n"),
            Some("a.rs\nb.rs\n"),
            "moved behind the port next",
        ),
        change("c2", Some("a.rs\nb.rs\n"), Some("a.rs\nb.rs\nc.rs\n"), ""),
    ];
    let v = judge(Some("a.rs\n"), &changes, "a.rs\nb.rs\nc.rs\n");
    assert_eq!(
        v.explained,
        vec![(
            added("b.rs", Some("c1")),
            "moved behind the port next".to_owned()
        )]
    );
    assert_eq!(v.unexplained, vec![added("c.rs", Some("c2"))]);
    assert_eq!(v.inventory, None);
}

#[test]
fn a_blank_reason_is_no_reason() {
    let changes = [change("c1", Some(""), Some("b.rs\n"), "  \n")];
    assert_eq!(
        judge(Some(""), &changes, "b.rs\n").unexplained,
        vec![added("b.rs", Some("c1"))]
    );
}

#[test]
fn the_creating_commit_is_the_inventory_only_when_the_base_has_no_list() {
    let changes = [
        change("c1", None, Some("a.rs\nb.rs\n"), ""),
        change("c2", Some("a.rs\nb.rs\n"), Some("a.rs\nb.rs\nc.rs\n"), ""),
    ];
    let v = judge(None, &changes, "a.rs\nb.rs\nc.rs\n");
    assert_eq!(v.inventory, Some(("c1".to_owned(), 2)));
    assert_eq!(v.unexplained, vec![added("c.rs", Some("c2"))]);

    let recreated = [
        change("c1", Some("a.rs\n"), None, ""),
        change("c2", None, Some("a.rs\nb.rs\n"), ""),
    ];
    let v = judge(Some("a.rs\n"), &recreated, "a.rs\nb.rs\n");
    assert_eq!(v.inventory, None);
    assert_eq!(v.unexplained, vec![added("b.rs", Some("c2"))]);
}

#[test]
fn the_last_introduction_is_the_one_judged() {
    let changes = [
        change("c1", Some(""), Some("x.rs\n"), "reasoned the first time"),
        change("c2", Some("x.rs\n"), Some(""), ""),
        change("c3", Some(""), Some("x.rs\n"), ""),
    ];
    assert_eq!(
        judge(Some(""), &changes, "x.rs\n").unexplained,
        vec![added("x.rs", Some("c3"))]
    );
}

#[test]
fn an_uncommitted_addition_has_no_commit_to_explain_it() {
    let changes = [change("c1", None, Some("a.rs\n"), "")];
    let v = judge(None, &changes, "a.rs\nd.rs\n");
    assert_eq!(v.inventory, Some(("c1".to_owned(), 1)));
    assert_eq!(v.unexplained, vec![added("d.rs", None)]);
}
