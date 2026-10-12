// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The served model's revision and recipe, from the snapshot path and the launch
//! recipe, against the vendored `model-recipe` fixture.

use std::path::{Path, PathBuf};

use metrale_qci_dump::contract::canonical;
use metrale_qci_dump::contract::ids::GitCommit;

use super::*;

const PINNED: &str = "ad31ec163396d5cb44d8deb87c596cecb0f55bf8";

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../schemas/platform/v1/fixtures/model-recipe/valid/gpt-oss-20b-mxfp4.json")
}

fn snapshot(rev: &str) -> PathBuf {
    PathBuf::from("/cache/hub/models--openai--gpt-oss-20b/snapshots").join(rev)
}

#[test]
fn only_a_hub_snapshot_directory_names_a_revision() {
    let want = GitCommit::from_hex(PINNED).ok();
    assert_eq!(snapshot_revision(&snapshot(PINNED)), want);
    assert_eq!(
        snapshot_revision(&PathBuf::from(format!("/m/snapshots/{PINNED}/"))),
        want
    );
    for not in [
        snapshot("abc123"),
        PathBuf::from(format!("/models/{PINNED}")),
        PathBuf::from("/models/gpt-oss-20b"),
        PathBuf::from(PINNED),
    ] {
        assert_eq!(snapshot_revision(&not), None, "{}", not.display());
    }
}

#[test]
fn the_recipe_digest_is_the_canonical_digest_of_the_document_given() {
    let identity = resolve(Some(&fixture()), &snapshot(PINNED)).unwrap();
    let doc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture()).unwrap()).unwrap();
    assert_eq!(identity.recipe, Some(canonical::digest(&doc).unwrap()));
    assert_eq!(identity.revision, GitCommit::from_hex(PINNED).ok());
    let none = resolve(None, &PathBuf::from("/models/local")).unwrap();
    assert_eq!(
        none,
        ModelIdentity {
            revision: None,
            recipe: None
        }
    );
}

#[test]
fn a_recipe_pinning_another_revision_refuses_to_start() {
    let other = "0".repeat(39) + "1";
    let err = resolve(Some(&fixture()), &snapshot(&other)).unwrap_err();
    assert!(format!("{err:#}").contains(PINNED), "{err:#}");
    // 2026-10-11: An unknown loaded revision cannot be compared, so the recipe stands.
    assert!(resolve(Some(&fixture()), &PathBuf::from("/models/local")).is_ok());
}

#[test]
fn an_unreadable_or_foreign_recipe_file_refuses_to_start() {
    assert!(resolve(Some(Path::new("/no/such/recipe.json")), &snapshot(PINNED)).is_err());
    let yaml = std::env::temp_dir().join(format!("model-identity-{}.yaml", std::process::id()));
    std::fs::write(&yaml, "recipe_version: 1\nmodel: openai/gpt-oss-20b\n").unwrap();
    let err = resolve(Some(&yaml), &snapshot(PINNED)).unwrap_err();
    std::fs::remove_file(&yaml).unwrap();
    assert!(
        format!("{err:#}").contains("not a JSON document"),
        "{err:#}"
    );
}
