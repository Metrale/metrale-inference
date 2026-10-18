// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `met doctor`: the home, writable and recipes checks
//! are each asserted both passing and reporting a problem. `check_identity`
//! is not covered here.
//!
//! They call the `check_*` functions directly on a home resolved from a `MapEnv` (2026-10-17;
//! before, `METRALE_HOME` was set around each under a mutex).
//!
//! Owner: server CLI (`met doctor`).
//! Invariants: none beyond the types.

use super::{check_home, check_recipes, check_writable, resolve_home};
use metrale_bench::artifacts::MetraleHome;
use metrale_io_host_mem::MapEnv;

struct Dir(std::path::PathBuf);
impl Dir {
    fn new(tag: &str) -> Self {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let p = std::env::temp_dir().join(format!("metrale-doctor-{tag}-{n}"));
        std::fs::create_dir_all(&p).expect("scratch");
        Self(p)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// 2026-10-17: The home an environment holding only `METRALE_HOME=root` resolves to.
fn home_at(root: &std::path::Path) -> anyhow::Result<MetraleHome> {
    let root = root.to_str().expect("a UTF-8 scratch path");
    resolve_home(&MapEnv::new([("METRALE_HOME", root)]))
}

#[test]
fn home_is_green_when_resolvable_and_names_its_provenance() {
    let d = Dir::new("home");
    let f = check_home(&home_at(&d.0));
    assert!(!f.problem, "{}", f.detail);
    assert!(
        f.detail.contains("from METRALE_HOME"),
        "the provenance is the point: {}",
        f.detail
    );
}

/// 2026-09-26: Problem: an empty `METRALE_HOME`, as
/// `export METRALE_HOME=$SOMETHING_UNSET` produces.
#[test]
fn home_goes_red_when_metrale_home_is_empty() {
    let f = check_home(&resolve_home(&MapEnv::new([("METRALE_HOME", "")])));
    assert!(f.problem, "an empty METRALE_HOME must be a problem");
    assert!(f.detail.contains("empty"), "{}", f.detail);
}

#[test]
fn writable_is_green_on_a_writable_home() {
    let d = Dir::new("w-ok");
    let f = check_writable(&home_at(&d.0));
    assert!(!f.problem, "{}", f.detail);
}

/// 2026-09-26: Problem: a home that exists but this process cannot write.
#[test]
fn writable_goes_red_on_a_read_only_home() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let d = Dir::new("w-ro");
        std::fs::set_permissions(&d.0, std::fs::Permissions::from_mode(0o500)).expect("chmod");
        let f = check_writable(&home_at(&d.0));
        std::fs::set_permissions(&d.0, std::fs::Permissions::from_mode(0o755)).expect("restore");
        assert!(
            f.problem,
            "a read-only home must be a problem: {}",
            f.detail
        );
        assert!(
            f.detail.contains("not writable"),
            "must say what is wrong: {}",
            f.detail
        );
    }
}

/// 2026-09-26: Problem: a home path that exists and is not a directory.
#[test]
fn writable_goes_red_when_the_home_is_a_file() {
    let d = Dir::new("w-file");
    let file = d.0.join("notadir");
    std::fs::write(&file, b"x").expect("write");
    let f = check_writable(&home_at(&file));
    assert!(f.problem, "{}", f.detail);
    assert!(f.detail.contains("not a directory"), "{}", f.detail);
}

fn a_recipe_body(n: usize) -> String {
    format!("recipe_version: \"2\"\nmodel: test/model-{n}\ncontainer: c\ndefaults:\n  port: 8888\n")
}

/// 2026-09-26: Passing: an index in the `files: { <path>: <recipe body> }`
/// schema that `parse_cache` reads, listing three recipes.
#[test]
fn recipes_is_green_when_the_files_map_lists_some() {
    let d = Dir::new("r-ok");
    std::fs::create_dir_all(d.0.join("metrale-recipes")).expect("mkdir");
    let doc = serde_json::json!({
        "tree_sha": "deadbeef",
        "fetched_at": 0,
        "files": {
            "family/one": a_recipe_body(1),
            "family/two": a_recipe_body(2),
            "family/three": a_recipe_body(3),
        },
    });
    std::fs::write(d.0.join("metrale-recipes/index.json"), doc.to_string()).expect("write");
    let f = check_recipes(&home_at(&d.0));
    assert!(!f.problem, "{}", f.detail);
    assert!(f.detail.contains('3'), "{}", f.detail);
}

/// 2026-09-26: Problem: no index file; the remedy is `sync-recipes`.
#[test]
fn recipes_goes_red_when_never_written() {
    let d = Dir::new("r-none");
    let f = check_recipes(&home_at(&d.0));
    assert!(f.problem);
    assert!(f.detail.contains("never been written"), "{}", f.detail);
    assert!(f.remedy.contains("sync-recipes"), "{}", f.remedy);
}

/// 2026-09-26: An index that exists but cannot be read is reported as
/// unreadable, not missing, and the remedy does not say to run `sync-recipes`.
#[test]
fn an_unreadable_index_is_not_reported_as_a_missing_one() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let d = Dir::new("r-perm");
        std::fs::create_dir_all(d.0.join("metrale-recipes")).expect("mkdir");
        let idx = d.0.join("metrale-recipes/index.json");
        std::fs::write(&idx, br#"{"recipes":[1]}"#).expect("write");
        std::fs::set_permissions(&idx, std::fs::Permissions::from_mode(0o000)).expect("chmod");
        let f = check_recipes(&home_at(&d.0));
        std::fs::set_permissions(&idx, std::fs::Permissions::from_mode(0o644)).expect("restore");
        assert!(f.problem);
        assert!(
            f.detail.contains("cannot be read"),
            "must distinguish unreadable from absent: {}",
            f.detail
        );
        assert!(
            !f.remedy.contains("run `met sync-recipes`"),
            "must NOT send the operator to the network for a permission fault: {}",
            f.remedy
        );
    }
}

/// 2026-09-26: Problem: an index that parses and lists no recipes.
#[test]
fn recipes_goes_red_when_the_index_is_empty() {
    let d = Dir::new("r-empty");
    std::fs::create_dir_all(d.0.join("metrale-recipes")).expect("mkdir");
    std::fs::write(d.0.join("metrale-recipes/index.json"), br#"{"files":{}}"#).expect("write");
    let f = check_recipes(&home_at(&d.0));
    assert!(f.problem);
    assert!(f.detail.contains("lists none"), "{}", f.detail);
}

/// 2026-10-17: A malformed lever is a `home` finding, not an abort: doctor still prints every
/// check.
#[test]
fn a_malformed_lever_is_a_home_problem_and_skips_the_rest() {
    let home = resolve_home(&MapEnv::new([("METRALE_NO_TUI", "maybe")]));
    let f = check_home(&home);
    assert!(
        f.problem && f.detail.contains("METRALE_NO_TUI"),
        "{}",
        f.detail
    );
    assert!(check_writable(&home).detail.starts_with("skipped"));
}
