// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The git plumbing on a real throwaway repository: the merge-base, the list's
//! absence there, only the commits that changed the list, each commit's before/after text and
//! its trailer, a refusal (not an empty history) for an unknown base, and a pull-request merge
//! that leaves each entry with the commit that really added it.

use super::read;
use crate::env_ban::ALLOWLIST;
use crate::env_ban::additions::{Added, judge};
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(["-c", "user.name=t", "-c", "user.email=t@example.invalid"])
        .args([
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .current_dir(dir)
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("utf-8")
        .trim()
        .to_owned()
}

fn commit(dir: &Path, path: &str, text: &str, message: &str) -> String {
    let file = dir.join(path);
    std::fs::create_dir_all(file.parent().expect("parent")).expect("mkdir");
    std::fs::write(&file, text).expect("write");
    run(dir, &["add", path]);
    run(dir, &["commit", "-q", "-m", message]);
    run(dir, &["rev-parse", "HEAD"])
}

fn repo(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("env-ban-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    run(&dir, &["init", "-q"]);
    dir
}

#[test]
fn reads_only_the_list_changes_since_the_merge_base() {
    let dir = repo("history");
    let base = commit(&dir, "README", "r\n", "base");
    let create = commit(&dir, ALLOWLIST, "a.rs\n", "create the list");
    commit(&dir, "README", "r2\n", "unrelated");
    let add = commit(
        &dir,
        ALLOWLIST,
        "a.rs\nb.rs\n",
        "add b\n\nEnv-Ban-Allow: b moves behind the port next",
    );

    let h = read(&dir, &base).expect("history");
    assert_eq!(h.merge_base, base);
    assert_eq!(h.base_list, None);
    let got: Vec<_> = h
        .changes
        .iter()
        .map(|c| {
            (
                c.sha.as_str(),
                c.before.as_deref(),
                c.after.as_deref(),
                c.reason.as_str(),
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![
            (create.as_str(), None, Some("a.rs\n"), ""),
            (
                add.as_str(),
                Some("a.rs\n"),
                Some("a.rs\nb.rs\n"),
                "b moves behind the port next"
            ),
        ]
    );

    let later = read(&dir, &create).expect("history");
    assert_eq!(later.base_list.as_deref(), Some("a.rs\n"));
    assert_eq!(later.changes.len(), 1);

    assert!(read(&dir, "no-such-revision").is_err());
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn a_pull_request_merge_attributes_each_entry_to_the_commit_that_added_it() {
    let dir = repo("merge");
    commit(&dir, ALLOWLIST, "a.rs\nm.rs\nz.rs\n", "create the list");
    run(&dir, &["checkout", "-q", "-b", "feature"]);
    let feature = commit(
        &dir,
        ALLOWLIST,
        "a.rs\nm.rs\nz.rs\nzz.rs\n",
        "add zz\n\nEnv-Ban-Allow: zz migrates in the next lane",
    );
    run(&dir, &["checkout", "-q", "main"]);
    let main_tip = commit(&dir, ALLOWLIST, "0.rs\na.rs\nm.rs\nz.rs\n", "main adds 0");
    run(
        &dir,
        &[
            "merge",
            "-q",
            "--no-ff",
            "--no-edit",
            "-m",
            "Merge feature",
            "feature",
        ],
    );
    let merged = "0.rs\na.rs\nm.rs\nz.rs\nzz.rs\n";
    assert_eq!(
        std::fs::read_to_string(dir.join(ALLOWLIST)).expect("list"),
        merged
    );

    let h = read(&dir, &main_tip).expect("history");
    assert_eq!(
        h.changes.first().map(|c| c.sha.as_str()),
        Some(feature.as_str())
    );
    let merge = h
        .changes
        .last()
        .expect("the merge changed the list against each side");
    let before = merge.before.as_deref().expect("both parents have the list");
    assert!(before.lines().any(|l| l == "zz.rs") && before.lines().any(|l| l == "0.rs"));

    let v = judge(h.base_list.as_deref(), &h.changes, merged);
    assert_eq!(v.unexplained, vec![]);
    assert_eq!(
        v.explained,
        vec![(
            Added {
                entry: "zz.rs".to_owned(),
                sha: Some(feature.clone()),
            },
            "zz migrates in the next lane".to_owned()
        )]
    );

    run(
        &dir,
        &["checkout", "-q", "--detach", &format!("{main_tip}~1")],
    );
    run(
        &dir,
        &[
            "merge",
            "-q",
            "--no-ff",
            "--no-edit",
            "-m",
            "PR merge",
            "feature",
        ],
    );
    let h = read(&dir, &format!("{main_tip}~1")).expect("history");
    let shas: Vec<_> = h.changes.iter().map(|c| c.sha.as_str()).collect();
    assert_eq!(
        shas,
        vec![feature.as_str()],
        "a merge equal to one side is not a change"
    );
    std::fs::remove_dir_all(&dir).expect("cleanup");
}
