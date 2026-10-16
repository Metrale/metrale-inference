// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `GitRepo` reads the revision and not the working tree, and `changed_paths` is
//! what the head added since the merge base, not what the base gained since.
//!
//! Owner: bench gate tests.
//! Invariants: none beyond the types.

use std::collections::BTreeSet;

use metrale_circuit::venn::Repo;

use super::super::coverage_tests::scratch_repo::{
    branch, checkout_default, commit, current_branch, head, init,
};
use super::super::owns::Owners;
use super::super::tests::tempdir;
use super::{GitRepo, changed_paths, merge_base, owners};

#[test]
fn git_repo_reads_the_revision_not_the_working_tree() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    init(root);
    commit(root, "kernels/gb10/a.cu", "committed", "a");
    let rev = head(root);
    std::fs::write(root.join("kernels/gb10/a.cu"), "edited").unwrap();
    std::fs::write(root.join("kernels/gb10/b.cu"), "untracked").unwrap();
    let repo = GitRepo {
        root: root.to_path_buf(),
        rev,
    };
    assert_eq!(repo.read("kernels/gb10/a.cu").as_deref(), Ok("committed"));
    assert_eq!(
        repo.list("kernels"),
        Ok(vec!["kernels/gb10/a.cu".to_string()])
    );
    assert!(
        repo.read("kernels/gb10/b.cu").is_err(),
        "absent at the revision"
    );
}

#[test]
fn a_change_is_what_the_head_added_since_the_merge_base() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    init(root);
    let main = current_branch(root);
    let fork = head(root);
    branch(root, "pr");
    commit(root, "kernels/gb10/pr.cu", "x", "pr");
    checkout_default(root, &main);
    commit(root, "kernels/gb10/main.cu", "y", "main moves on");
    let got = changed_paths(root, &main, "pr").expect("diff");
    assert_eq!(got, vec!["kernels/gb10/pr.cu".to_string()]);
    // 2026-10-15: Control: the same pair the other way round is main's change alone.
    let got = changed_paths(root, "pr", &main).expect("diff");
    assert_eq!(got, vec!["kernels/gb10/main.cu".to_string()]);
    assert!(
        changed_paths(root, "no-such-rev", "pr").is_err(),
        "an unknown rev is an error"
    );
    let base = merge_base(root, &main, "pr").expect("a merge base");
    assert!(
        base.starts_with(&fork),
        "{base} is not the fork point {fork}"
    );
    assert!(merge_base(root, "no-such-rev", "pr").is_err());
}

#[test]
fn owners_are_the_valid_lists_of_registered_bench_files_at_the_revision() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    init(root);
    commit(root, "kernels/gb10/HARDWARE.toml", "", "hw");
    commit(
        root,
        "kernels/gb10/m/BENCH.toml",
        "owns = [\"crates/m\"]\n",
        "m",
    );
    commit(
        root,
        "kernels/gb10/bad/BENCH.toml",
        "owns = [\"crates/*\"]\n",
        "bad",
    );
    // 2026-10-16: No HARDWARE.toml: `bench::load_all` does not read this file, so neither do owners.
    commit(
        root,
        "kernels/nohw/n/BENCH.toml",
        "owns = [\"crates/n\"]\n",
        "nohw",
    );
    let rev = head(root);
    std::fs::write(
        root.join("kernels/gb10/m/BENCH.toml"),
        "owns = [\"crates/edited\"]\n",
    )
    .unwrap();
    let got = owners(&GitRepo {
        root: root.to_path_buf(),
        rev,
    });
    assert_eq!(
        got,
        Owners::from([("m".to_string(), BTreeSet::from(["crates/m".to_string()]))]),
        "the committed list of the registered file; the refused and unregistered ones own nothing"
    );
}
