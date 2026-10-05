// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Tests for `recipe::explicit` — the "every recipe defines every
//! recipe-settable variable" checker.
//!
//! Owner: server (recipe).
//! Invariants: none beyond the types.

use std::collections::{BTreeMap, BTreeSet};

use super::*;

/// 2026-10-05: The real, served recipe tree at the repo root — `recipes/`, not the
/// unrelated `tests/fixtures/recipes/` snapshot the TUI tests use (that one is a hand-
/// curated fixture for dashboard tests, 28 entries including three `deepseek-v4.1`
/// recipes that do not exist in the served tree at all; this checker's whole point is to
/// police the recipes a gate or an operator actually serves).
fn real_recipes_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../recipes")
        .canonicalize()
        .expect("recipes/ exists two levels above crates/server")
}

/// 2026-10-05: Every recipe in `recipes/`, parsed with the id `family/stem` from its path.
fn load_real_recipes() -> Vec<Recipe> {
    let mut out = Vec::new();
    let mut stack = vec![real_recipes_root()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("recipes dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "yaml") {
                continue;
            }
            let family = path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let text = std::fs::read_to_string(&path).expect("read");
            out.push(
                Recipe::parse(format!("{family}/{stem}"), &text)
                    .unwrap_or_else(|e| panic!("{}: {e:#}", path.display())),
            );
        }
    }
    out
}

/// 2026-10-05: PATH A — every `runtime: metrale` recipe in the served tree sets every
/// required key. Non-metrale recipes (the two `diffusion-gemma` vLLM entries) render
/// through their own `command:` template, never `Recipe::argv`/`ServeArgs`, so this
/// checker — which is about `met serve` flags — does not apply to them, exactly as
/// `Recipe::argv_edited` itself refuses to render argv for them.
#[test]
fn every_real_recipe_defines_every_required_key() {
    let required = required_keys();
    assert!(
        required.len() > 40,
        "sanity: required_keys() looks too small ({}), check the filters in explicit.rs",
        required.len()
    );

    let recipes = load_real_recipes();
    assert!(!recipes.is_empty(), "sanity: recipes/ must not be empty");
    let metrale_count = recipes.iter().filter(|r| r.is_metrale()).count();
    assert!(
        metrale_count > 10,
        "sanity: expected more than 10 runtime: metrale recipes, found {metrale_count}"
    );

    let mut failures = Vec::new();
    for r in recipes.iter().filter(|r| r.is_metrale()) {
        let missing = missing_keys(&required, r);
        if !missing.is_empty() {
            failures.push(format!(
                "{}: missing {}",
                r.id,
                missing.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "recipe(s) leave a recipe-settable variable to its `met serve` default \
         (owner directive 2026-10-05 — see explicit.rs):\n{}",
        failures.join("\n")
    );
}

/// 2026-10-05: A fixture recipe with every required key set to some value, so removing one
/// key is the only variable in the tests below.
fn complete_fixture(required: &BTreeSet<String>) -> Recipe {
    let mut defaults = BTreeMap::new();
    for key in required {
        defaults.insert(key.clone(), "1".to_string());
    }
    Recipe {
        id: "fixture/complete".to_string(),
        version: "2".to_string(),
        model: "test/model".to_string(),
        runtime: Some("metrale".to_string()),
        container: "test".to_string(),
        min_nodes: 1,
        description: "fixture".to_string(),
        maintainer: String::new(),
        category: String::new(),
        model_params: String::new(),
        quantization: String::new(),
        kv_dtype: String::new(),
        updated: String::new(),
        defaults,
        env: BTreeMap::new(),
        starting_point: None,
    }
}

/// 2026-10-05: PATH B (part 1) — a fixture that has every required key passes, and
/// removing exactly one makes the checker fail and name that key — not a different one,
/// not all of them.
#[test]
fn removing_one_required_key_fails_and_names_it() {
    let required = required_keys();
    let complete = complete_fixture(&required);
    assert!(
        missing_keys(&required, &complete).is_empty(),
        "the fixture was built to set every required key"
    );

    let victim = required
        .iter()
        .next()
        .cloned()
        .expect("required_keys() is non-empty");
    let mut incomplete = complete.clone();
    incomplete.defaults.remove(&victim);

    let missing = missing_keys(&required, &incomplete);
    assert_eq!(
        missing,
        BTreeSet::from([victim.clone()]),
        "removing only {victim:?} must report only {victim:?} missing, not {missing:?}"
    );
}

/// 2026-10-05: PATH B (part 2) — a new clap flag that no recipe has a value for must be
/// caught, without actually adding a flag to `ServeArgs` (which would be a production
/// change, not a test). `missing_keys` takes its required set as a parameter, so a
/// test-only synthetic set — the real required set plus one key no flag has — proves the
/// checking logic itself (not `required_keys`'s derivation) catches an addition.
#[test]
fn a_new_flag_without_a_recipe_value_fails() {
    let mut synthetic = required_keys();
    let new_flag = "a_future_flag_no_recipe_has_yet";
    assert!(
        synthetic.insert(new_flag.to_string()),
        "the synthetic key must actually be new"
    );

    let complete = complete_fixture(&required_keys());
    let missing = missing_keys(&synthetic, &complete);
    assert_eq!(
        missing,
        BTreeSet::from([new_flag.to_string()]),
        "a required set that grew by one key must report exactly that key missing"
    );
}

/// 2026-10-05: PATH C — an unknown key in a recipe's `defaults:` fails, by name, instead of
/// silently rendering to nothing and falling back to the flag's default. This already holds
/// on the production path: `Recipe::serve_args` renders every `defaults:` key to argv
/// (`schema::argv_for`, which — since `schema::NOT_FLAGS` is empty — maps every key to some
/// `--flag`) and then parses that argv with clap, which refuses a flag it does not
/// recognize. This test proves that chain, not a new mechanism.
#[test]
fn an_unknown_key_is_refused_not_silently_dropped() {
    let mut defaults = BTreeMap::new();
    defaults.insert("max_batch_size".to_string(), "16".to_string());
    // A typo of `max_model_len`: one transposed character away from a real recipe key.
    defaults.insert("mx_model_len".to_string(), "4096".to_string());
    let recipe = Recipe {
        id: "fixture/typo".to_string(),
        version: "2".to_string(),
        model: "test/model".to_string(),
        runtime: Some("metrale".to_string()),
        container: "test".to_string(),
        min_nodes: 1,
        description: "fixture".to_string(),
        maintainer: String::new(),
        category: String::new(),
        model_params: String::new(),
        quantization: String::new(),
        kv_dtype: String::new(),
        updated: String::new(),
        defaults,
        env: BTreeMap::new(),
        starting_point: None,
    };

    let err = recipe
        .serve_args(&BTreeMap::new())
        .expect_err("a typo'd key must not parse as a valid serve command line");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("mx-model-len") || msg.contains("mx_model_len"),
        "the error should name the bad key so the typo is obvious: {msg}"
    );
}

/// 2026-10-05: `required_keys()` is exactly: every flag from the manifest, minus
/// presence-only flags, minus flags with no clap default, minus `EXCLUDED_NAMED`. A
/// regression here (e.g. a stray filter) would silently shrink or grow the required
/// surface without any real-recipe test catching it, since the real recipes would simply
/// be checked against the wrong set.
#[test]
fn required_keys_is_exactly_the_manifest_minus_the_three_exclusions() {
    let manifest = crate::cli::manifest::build();
    let presence_only = manifest.flags.iter().filter(|f| f.presence_only).count();
    let no_default = manifest
        .flags
        .iter()
        .filter(|f| !f.presence_only && f.default.is_none())
        .count();
    let total = manifest.flags.len();
    let expected = total - presence_only - no_default - EXCLUDED_NAMED.len();
    assert_eq!(
        required_keys().len(),
        expected,
        "total={total} presence_only={presence_only} no_default={no_default} \
         named_exclusions={}",
        EXCLUDED_NAMED.len()
    );

    // Every named exclusion must be a real flag key the manifest actually carries, and
    // must not itself be presence-only or already no-default (otherwise it is dead,
    // doubly-excluded weight in the list).
    for (key, reason) in EXCLUDED_NAMED {
        assert!(!reason.is_empty(), "{key}: exclusion needs a reason");
        let flag = manifest
            .flags
            .iter()
            .find(|f| f.key == *key)
            .unwrap_or_else(|| panic!("{key}: EXCLUDED_NAMED names a flag that does not exist"));
        assert!(
            !flag.presence_only,
            "{key}: already excluded as presence-only, drop it from EXCLUDED_NAMED"
        );
        assert!(
            flag.default.is_some(),
            "{key}: has no clap default, already excluded categorically, drop it from EXCLUDED_NAMED"
        );
    }
}
