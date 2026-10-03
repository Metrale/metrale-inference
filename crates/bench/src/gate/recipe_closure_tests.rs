// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Tests for `recipe_closure`: which recipe edits a record survives, on scratch
//! trees whose BENCH entries name in-tree recipes, and on the repository's own entries.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::check::{Standing, record_standing};
use super::coverage_tests::{any_gate, scratch_repo};
use super::recipe_closure::{
    RecipeStanding, canonical, content_sha256, read_in_tree, recipe_path, recipe_standing,
};
use super::tests::{TEST_HW, hw, run_record, tempdir, write_baseline};
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;
use std::path::Path;

const MOE: &str = "Qwen/Qwen3.6-35B-A3B-FP8";
const DENSE: &str = "unsloth/Qwen3.8-27B-NVFP4";
const MOE_RECIPE: &str = "qwen3.6/moe";
const DENSE_RECIPE: &str = "qwen3.8/dense";

/// 2026-10-02: Two MoE gates serve one recipe, one dense gate another.
const GATES: [(&str, &str, &str); 3] = [
    ("concurrency-sweep-moe", MOE, MOE_RECIPE),
    ("high-isl-ttft-warm-moe", MOE, MOE_RECIPE),
    ("concurrency-sweep", DENSE, DENSE_RECIPE),
];

fn recipe(model: &str, activation: &str, util: &str) -> String {
    format!(
        "recipe_version: \"2\"\nmodel: {model}\nruntime: metrale\ncontainer: c:latest\n\n\
         metadata:\n  description: |\n    The measured configuration.\n  maintainer: metrale\n\n\
         defaults:\n  activation_quantization: {activation}\n  gpu_memory_utilization: {util}\n  \
         max_model_len: 32768\n"
    )
}

fn write(root: &Path, id: &str, text: &str) {
    let path = root.join(recipe_path(id));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn pins(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// 2026-10-02: A scratch tree with the three gates' entries and both recipes, each entry
/// pinning `gpu_memory_utilization`.
fn tree(root: &Path) {
    for (gate, model, id) in GATES {
        let mut models = BTreeMap::new();
        models.insert(
            model.to_string(),
            ModelBaseline {
                recipe: Some(id.to_string()),
                serve_overrides: pins(&[("gpu_memory_utilization", "0.85")]),
                metrics: [(
                    "x".to_string(),
                    Bound {
                        min: Some(1.0),
                        ..Bound::default()
                    },
                )]
                .into_iter()
                .collect(),
                ..ModelBaseline::default()
            },
        );
        let mut hardware = BTreeMap::new();
        hardware.insert(
            TEST_HW.to_string(),
            HardwareBaseline {
                default: model.to_string(),
                models,
            },
        );
        write_baseline(
            root,
            gate,
            &GateBaseline {
                schema: 2,
                hardware,
            },
        );
    }
    write(root, MOE_RECIPE, &recipe(MOE, "adaptive", "0.90"));
    write(root, DENSE_RECIPE, &recipe(DENSE, "adaptive", "0.90"));
}

/// 2026-10-02: A record of `gate` served from the tree as it is now, at commit `sha`.
fn served(root: &Path, gate: &str, sha: &str) -> GateRecord {
    let (_, model, id) = GATES.iter().find(|(g, _, _)| *g == gate).unwrap();
    let mut r = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        sha.to_string(),
        Vec::new(),
        Some(id.to_string()),
    )
    .unwrap();
    r.benchmark_id = gate.to_string();
    r.target_model = model.to_string();
    r.serve_overrides = pins(&[("gpu_memory_utilization", "0.85")]);
    let sha = content_sha256(&read_in_tree(root, id).unwrap(), &r.serve_overrides).unwrap();
    r.with_served_recipe(sha)
}

fn standings(root: &Path, records: &[GateRecord]) -> Vec<bool> {
    records
        .iter()
        .map(|r| recipe_standing(root, r) == RecipeStanding::Same)
        .collect()
}

/// 2026-10-02: Changing what the MoE recipe serves takes exactly the two MoE gates' records;
/// changing the dense recipe takes exactly the dense one.
#[test]
fn a_recipe_change_invalidates_exactly_the_gates_that_serve_it() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    tree(root);
    let records: Vec<GateRecord> = GATES.iter().map(|(g, _, _)| served(root, g, "s")).collect();
    assert_eq!(standings(root, &records), [true, true, true]);

    write(root, MOE_RECIPE, &recipe(MOE, "declared", "0.90"));
    assert_eq!(standings(root, &records), [false, false, true]);
    assert_eq!(
        recipe_standing(root, &records[0]),
        RecipeStanding::Changed("recipes/qwen3.6/moe.yaml".to_string())
    );

    write(root, MOE_RECIPE, &recipe(MOE, "adaptive", "0.90"));
    write(root, DENSE_RECIPE, &recipe(DENSE, "declared", "0.90"));
    assert_eq!(standings(root, &records), [true, true, false]);
}

/// 2026-10-02: Comments, blank lines, quoting, key order and `metadata` are not part of what is
/// served, so rewriting them keeps every record.
#[test]
fn a_reformat_or_a_metadata_edit_is_not_a_change() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    tree(root);
    let records: Vec<GateRecord> = GATES.iter().map(|(g, _, _)| served(root, g, "s")).collect();
    let rewritten = format!(
        "# the MoE flagship\nrecipe_version: '2'\nmodel: {MOE}\n\ncontainer: \"c:latest\"\n\
         runtime: metrale\nmetadata:\n  description: |\n    Reworded, with a new paragraph.\n  \
         maintainer: someone else\n  updated: 2026-10-02\ndefaults:\n  max_model_len: 32768  \
         # 32K\n  gpu_memory_utilization: \"0.90\"\n\n  activation_quantization: adaptive\n"
    );
    assert_ne!(rewritten, recipe(MOE, "adaptive", "0.90"));
    write(root, MOE_RECIPE, &rewritten);
    assert_eq!(standings(root, &records), [true, true, true]);
}

/// 2026-10-02: A key the entry's serve overrides pin is served as the pin, whatever the recipe
/// says; the same edit to an unpinned key is a change.
#[test]
fn a_key_an_override_pins_is_not_a_change() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    tree(root);
    let record = served(root, "concurrency-sweep-moe", "s");
    write(root, MOE_RECIPE, &recipe(MOE, "adaptive", "0.70"));
    assert_eq!(recipe_standing(root, &record), RecipeStanding::Same);
    let unpinned = recipe(MOE, "adaptive", "0.70").replace("32768", "40960");
    write(root, MOE_RECIPE, &unpinned);
    assert_ne!(recipe_standing(root, &record), RecipeStanding::Same);
}

/// 2026-10-02: Nothing attests what a record served when it carries no hash (it was served from
/// outside the tree), when its entry now names another recipe, or when the tree lacks the
/// recipe; each falls. An operator's own endpoint served no recipe and is unaffected.
#[test]
fn a_record_falls_when_nothing_attests_its_recipe() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    tree(root);
    let record = served(root, "concurrency-sweep-moe", "s");

    let mut unhashed = record.clone();
    unhashed.served_recipe_sha256 = None;
    assert_ne!(recipe_standing(root, &unhashed), RecipeStanding::Same);

    let mut elsewhere = record.clone();
    elsewhere.served_by = Some(DENSE_RECIPE.to_string());
    assert_ne!(recipe_standing(root, &elsewhere), RecipeStanding::Same);

    let mut endpoint = unhashed.clone();
    endpoint.served_by = None;
    assert_eq!(recipe_standing(root, &endpoint), RecipeStanding::Same);

    std::fs::remove_file(root.join(recipe_path(MOE_RECIPE))).unwrap();
    assert_ne!(recipe_standing(root, &record), RecipeStanding::Same);
    let refusal = format!("{:#}", read_in_tree(root, MOE_RECIPE).unwrap_err());
    assert!(refusal.contains("never falls back"), "{refusal}");
}

/// 2026-10-02: The verdict reads the recipe: a commit that touches only a recipe, outside
/// `PERF_PATHS`, leaves the dense record standing and invalidates the MoE one, naming the file.
#[test]
fn record_standing_names_the_recipe_that_moved() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    tree(root);
    scratch_repo::commit(root, "docs/x.md", "measured", "the measured tree");
    let measured = scratch_repo::head(root);
    let moe = served(root, "concurrency-sweep-moe", &measured);
    let dense = served(root, "concurrency-sweep", &measured);
    let gate = any_gate();
    assert_eq!(
        record_standing(root, &measured, &moe, &gate),
        Standing::Stands
    );

    scratch_repo::commit(
        root,
        &recipe_path(MOE_RECIPE),
        &recipe(MOE, "declared", "0.90"),
        "declared",
    );
    let head = scratch_repo::head(root);
    assert_eq!(
        record_standing(root, &head, &moe, &gate),
        Standing::Invalidated(vec!["recipes/qwen3.6/moe.yaml".to_string()])
    );
    assert_eq!(
        record_standing(root, &head, &dense, &gate),
        Standing::Stands
    );
}

/// 2026-10-02: The canonical form drops `metadata` and applies the overrides into `defaults`.
#[test]
fn canonical_drops_metadata_and_applies_overrides() {
    let c = canonical(
        &recipe(MOE, "adaptive", "0.90"),
        &pins(&[
            ("gpu_memory_utilization", "0.85"),
            ("kv_cache_dtype", "bf16"),
        ]),
    )
    .unwrap();
    let top = c.as_map().unwrap();
    assert!(!top.contains_key("metadata"));
    let defaults = top["defaults"].as_map().unwrap();
    assert_eq!(defaults["gpu_memory_utilization"].as_str(), Some("0.85"));
    assert_eq!(defaults["kv_cache_dtype"].as_str(), Some("bf16"));
    assert_eq!(
        defaults["activation_quantization"].as_str(),
        Some("adaptive")
    );
}

fn repo_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// 2026-10-02: Every recipe a BENCH entry in this repository names is in the tree and has a
/// canonical form, so no gate serve is refused for a missing recipe.
#[test]
fn every_bench_recipe_is_in_the_tree() {
    let root = repo_root();
    let entries = bench::load_all(&root).unwrap();
    let mut seen = 0;
    for (_, entry) in &entries {
        let Some(id) = &entry.recipe else { continue };
        let text = read_in_tree(&root, id).unwrap_or_else(|e| panic!("{e:#}"));
        canonical(&text, &entry.serve_overrides).unwrap_or_else(|e| panic!("{id}: {e:#}"));
        seen += 1;
    }
    assert!(seen > 0, "no BENCH entry names a recipe");
}

/// 2026-10-02: No BENCH entry pins a precision tier (`weight_quantization`,
/// `activation_quantization`, `expert_quantization`) to the value its recipe already serves.
/// The tier belongs to the recipe; a pin that restates it would hide a recipe edit to it from
/// [`recipe_standing`]. Other pins that restate a recipe value stay, each with a stated purpose
/// in its comment: the site's ladder fingerprint reads them from the record, or the record states
/// them for a gate that depends on them.
#[test]
fn no_bench_override_restates_its_recipe_tier() {
    const TIERS: [&str; 3] = [
        "weight_quantization",
        "activation_quantization",
        "expert_quantization",
    ];
    let root = repo_root();
    let mut restated = Vec::new();
    for (target, entry) in bench::load_all(&root).unwrap() {
        let Some(id) = &entry.recipe else { continue };
        let served = canonical(&read_in_tree(&root, id).unwrap(), &BTreeMap::new()).unwrap();
        let defaults = served.as_map().unwrap()["defaults"].as_map().unwrap();
        for (key, value) in &entry.serve_overrides {
            if TIERS.contains(&key.as_str())
                && defaults.get(key).and_then(|v| v.as_str()) == Some(value.as_str())
            {
                restated.push(format!("{target} {} {key}={value} ({id})", entry.gate));
            }
        }
    }
    assert!(
        restated.is_empty(),
        "tier pins restating their recipe:\n{}",
        restated.join("\n")
    );
}
