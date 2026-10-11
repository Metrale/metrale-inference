// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The vendored platform-contract bundle (`schemas/platform/`, investor-mvp #41)
//! is the unchanged copy its pin names, and the runtime's binding agrees with every fixture
//! of the document kinds the runtime writes or reads: each `valid` fixture parses,
//! validates and re-serializes to the canonical JSON the index records; each `invalid`
//! fixture is refused. Every valid fixture of every kind also reproduces its `jcs_sha256`,
//! which holds the canonical-JSON implementation to the platform's.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use metrale_qci_dump::contract::canonical::{digest, sha256};
use metrale_qci_dump::contract::hardware::HardwareFingerprint;
use metrale_qci_dump::contract::receipt::CompatibilityReceipt;
use metrale_qci_dump::contract::recipe::LaunchRecipe;
use metrale_qci_dump::contract::usage::InferenceUsage;
use serde_json::Value;

/// 2026-10-11: The kinds this repository binds to its own types.
const BOUND: [&str; 3] = [
    "inference-usage",
    "hardware-fingerprint",
    "compatibility-receipt",
];

fn bundle() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/platform")
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn pin(key: &str) -> String {
    let text = String::from_utf8(read(&bundle().with_extension("pin"))).unwrap();
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .find_map(|l| l.strip_prefix(&format!("{key} = ")))
        .map(|v| v.trim_matches('"').to_string())
        .unwrap_or_else(|| panic!("platform.pin has no {key}"))
}

fn files_under(dir: &Path, root: &Path, out: &mut BTreeSet<String>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files_under(&path, root, out);
        } else {
            let rel = path.strip_prefix(root).unwrap().to_string_lossy();
            out.insert(rel.replace('\\', "/"));
        }
    }
}

#[test]
fn the_copy_is_the_pinned_bundle_unchanged() {
    let root = bundle();
    let sums = read(&root.join("SHA256SUMS"));
    assert_eq!(
        sha256(&sums).to_hex(),
        pin("sha256sums"),
        "SHA256SUMS differs from the pin"
    );
    assert_eq!(pin("mfs_commit").len(), 40);
    let mut listed = BTreeSet::new();
    for line in String::from_utf8(sums).unwrap().lines() {
        let (want, file) = line.split_once("  ").expect("sha256sum line");
        let got = sha256(&read(&root.join(file))).to_hex();
        assert_eq!(got, want, "{file} was edited");
        listed.insert(file.to_string());
    }
    let mut present = BTreeSet::new();
    files_under(&root, &root, &mut present);
    present.remove("SHA256SUMS");
    assert_eq!(
        present, listed,
        "a file in the copy is missing from SHA256SUMS, or the reverse"
    );
}

/// 2026-10-11: `Ok` with the canonical digest of the re-serialized document when the
/// runtime's binding accepts `doc` as `kind`; `Err` naming why it refuses it.
fn bind(kind: &str, doc: &Value) -> Result<String, String> {
    fn take<T: serde::de::DeserializeOwned + serde::Serialize>(
        doc: &Value,
        validate: impl Fn(&T) -> Result<(), String>,
    ) -> Result<String, String> {
        let typed: T = serde_json::from_value(doc.clone()).map_err(|e| e.to_string())?;
        validate(&typed)?;
        digest(&typed)
            .map(|d| d.to_hex())
            .map_err(|e| format!("{e:?}"))
    }
    match kind {
        "inference-usage" => take(doc, |d: &InferenceUsage| {
            d.validate().map_err(|v| v.to_string())
        }),
        "hardware-fingerprint" => take(doc, |d: &HardwareFingerprint| {
            d.validate().map_err(|v| v.to_string())
        }),
        "compatibility-receipt" => take(doc, |d: &CompatibilityReceipt| {
            d.validate().map_err(|v| v.to_string())
        }),
        other => panic!("{other} is not bound"),
    }
}

#[test]
fn the_binding_agrees_with_every_fixture_verdict() {
    let root = bundle().join("v1");
    let index: Value = serde_json::from_slice(&read(&root.join("INDEX.json"))).unwrap();
    let mut seen_valid = BTreeSet::new();
    let mut seen_invalid = BTreeSet::new();
    for entry in index["valid"].as_array().unwrap() {
        let file = entry["file"].as_str().unwrap();
        let kind = entry["schema"].as_str().unwrap();
        let want = entry["jcs_sha256"].as_str().unwrap();
        let doc: Value = serde_json::from_slice(&read(&root.join(file))).unwrap();
        assert_eq!(
            digest(&doc).unwrap().to_hex(),
            want,
            "canonical JSON of {file}"
        );
        if BOUND.contains(&kind) {
            let got = bind(kind, &doc).unwrap_or_else(|e| panic!("{file} refused: {e}"));
            assert_eq!(got, want, "{file} does not round-trip through the binding");
            seen_valid.insert(kind);
        }
    }
    for entry in index["invalid"].as_array().unwrap() {
        let file = entry["file"].as_str().unwrap();
        let kind = entry["schema"].as_str().unwrap();
        if BOUND.contains(&kind) {
            let doc: Value = serde_json::from_slice(&read(&root.join(file))).unwrap();
            let rule = entry["rule"].as_str().unwrap();
            assert!(
                bind(kind, &doc).is_err(),
                "{file} breaks {rule} but was accepted"
            );
            seen_invalid.insert(kind);
        }
    }
    let all: BTreeSet<&str> = BOUND.into_iter().collect();
    assert_eq!(
        seen_valid, all,
        "a bound kind has no valid fixture in the index"
    );
    assert_eq!(
        seen_invalid, all,
        "a bound kind has no invalid fixture in the index"
    );
}

/// 2026-10-11: The rules of `model-recipe` the runtime's launch-recipe reader enforces; the
/// rest (closed objects, weights, images, env keys, precision roles) are Control's to admit.
const RECIPE_RULES: [&str; 4] = [
    "immutable-revision",
    "repository-format",
    "recipe-id-format",
    "integer-units",
];

#[test]
fn the_launch_recipe_reader_digests_every_valid_recipe_and_refuses_what_it_reads_wrong() {
    let root = bundle().join("v1");
    let index: Value = serde_json::from_slice(&read(&root.join("INDEX.json"))).unwrap();
    let mut valid = 0;
    for entry in index["valid"].as_array().unwrap() {
        if entry["schema"] != "model-recipe" {
            continue;
        }
        let file = entry["file"].as_str().unwrap();
        let recipe = LaunchRecipe::read(&read(&root.join(file)))
            .unwrap_or_else(|e| panic!("{file} refused: {e}"));
        assert_eq!(recipe.digest.to_hex(), entry["jcs_sha256"], "{file}");
        valid += 1;
    }
    assert!(valid > 0, "no valid model-recipe fixture in the index");
    let mut refused = BTreeSet::new();
    for entry in index["invalid"].as_array().unwrap() {
        let rule = entry["rule"].as_str().unwrap();
        if entry["schema"] != "model-recipe" || !RECIPE_RULES.contains(&rule) {
            continue;
        }
        let file = entry["file"].as_str().unwrap();
        assert!(
            LaunchRecipe::read(&read(&root.join(file))).is_err(),
            "{file} breaks {rule} but was read"
        );
        refused.insert(rule);
    }
    assert_eq!(
        refused,
        RECIPE_RULES.into_iter().collect(),
        "a rule the reader enforces has no fixture"
    );
}
