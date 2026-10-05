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
/// `EXCLUDED_NAMED`. A regression here (e.g. a stray filter reappearing) would silently
/// shrink or grow the required surface without any real-recipe test catching it, since the
/// real recipes would simply be checked against the wrong set.
#[test]
fn required_keys_is_exactly_the_manifest_minus_the_named_exclusions() {
    let manifest = crate::cli::manifest::build();
    let total = manifest.flags.len();
    let expected = total - EXCLUDED_NAMED.len();
    assert_eq!(
        required_keys().len(),
        expected,
        "total={total} named_exclusions={}",
        EXCLUDED_NAMED.len()
    );

    // Every named exclusion must be a real flag key the manifest actually carries, give a
    // reason, and appear only once (a duplicate would silently inflate `expected` above).
    let mut seen = BTreeSet::new();
    for (key, reason) in EXCLUDED_NAMED {
        assert!(!reason.is_empty(), "{key}: exclusion needs a reason");
        assert!(
            manifest.flags.iter().any(|f| f.key == *key),
            "{key}: EXCLUDED_NAMED names a flag that does not exist"
        );
        assert!(seen.insert(*key), "{key}: listed twice in EXCLUDED_NAMED");
    }
}

/// 2026-10-05: Presence-only (bool) flags are required too — the 2026-10-05 owner
/// extension. `speculative` and `enable_prefix_caching` are two unrelated presence-only
/// flags from different structs (`ServeSchedulingArgs`); both must appear.
#[test]
fn presence_only_flags_are_required_too() {
    let required = required_keys();
    assert!(required.contains("speculative"), "{required:?}");
    assert!(required.contains("enable_prefix_caching"), "{required:?}");
    assert!(required.contains("prompt_lookup_decoding"), "{required:?}");
}

/// 2026-10-05: Flags with no clap default are required too (pinned to their resolved
/// value), not categorically excluded — the 2026-10-05 owner extension. `num_drafts`,
/// `kv_cache_dtype` and `mtp_gate` are three unrelated `Option<T>` flags with different
/// resolution chains (MODEL.toml, MODEL.toml, env-only); none are in `EXCLUDED_NAMED`.
#[test]
fn flags_with_no_clap_default_are_required_too() {
    let required = required_keys();
    assert!(required.contains("num_drafts"), "{required:?}");
    assert!(required.contains("kv_cache_dtype"), "{required:?}");
    assert!(required.contains("mtp_gate"), "{required:?}");
    assert!(required.contains("ssm_h_dtype"), "{required:?}");
    assert!(required.contains("model_name"), "{required:?}");
}

/// 2026-10-05: No recipe's `env:` block sets a GDN-family lever
/// (`METRALE_SSM_H_FP16`/`METRALE_GDN_FUSED_NORM`/`METRALE_SSM_BATCHED_RECURRENT`). This is
/// the load-bearing fact behind pinning `ssm_h_dtype`: pinning it seals
/// `KernelFlagPlan`'s GDN cell (`gdn_given` flips false->true, since `ssm_h_dtype` is an
/// `Option<T>` with no clap default), which stops a recipe's `env:` block from being able
/// to steer the cell at serve time. If any recipe relied on that, pinning would be a real
/// regression, not a no-op. None does, so it is not — this test is the control that keeps
/// that true.
#[test]
fn no_recipe_env_block_sets_a_gdn_lever() {
    let gdn_levers = [
        "METRALE_SSM_H_FP16",
        "METRALE_GDN_FUSED_NORM",
        "METRALE_SSM_BATCHED_RECURRENT",
    ];
    for r in load_real_recipes().iter().filter(|r| r.is_metrale()) {
        for lever in gdn_levers {
            assert!(
                !r.env.contains_key(lever),
                "{}: env: sets {lever}, so pinning ssm_h_dtype would remove its effect — \
                 this recipe needs a human decision before this policy can pin it",
                r.id
            );
        }
    }
}

/// 2026-10-05: Pinning the GDN quartet (`ssm_h_dtype`, `gdn_fused_norm`, `exact_verify`,
/// with `ssm_batched_recurrent` left at its already-required "auto") seals
/// `KernelFlagPlan`'s GDN cell — `gdn_given` flips false->true, an unavoidable consequence
/// of `ssm_h_dtype` being an `Option<T>` with no clap default (giving ANY value, even its
/// own no-op one, makes it `Some`). This proves the SEALED cell's bits match the UNSEALED
/// cell's bits today (`GdnFlags::from_env`, the function `gdn_flags::flags()` falls back to
/// on first read when nothing is published) — pinning changes WHEN the cell is decided,
/// not WHAT it decides, given `no_recipe_env_block_sets_a_gdn_lever` above.
///
/// `GdnFlags::from_env()` is called directly (not `gdn_flags::flags()`), which reads the
/// environment fresh every call and seals nothing — `flags()` would permanently pin the
/// process-wide `FLAGS` `OnceLock` for every other test sharing this process, the same
/// class of cross-test pollution `circuit_memory_ledger_tests` suffers from
/// `ssm_reserve::decode_ring::DECODE_RING_SLOTS` (see the PR description).
#[test]
fn the_pinned_gdn_bundle_matches_the_documented_unsealed_default() {
    use clap::Parser as _;

    let today = metrale_model_layers::layers::qwen3_ssm::gdn_flags::GdnFlags::from_env();

    let unpinned = crate::cli::ServeArgs::try_parse_from(["serve", "org/model"]).expect("parses");
    assert!(
        crate::main_modules::kernel_flag_plan::KernelFlagPlan::from_args(&unpinned)
            .gdn
            .is_none(),
        "sanity: no GDN flag given must leave the cell unsealed (gdn_given == false)"
    );

    // Exactly what every recipe now pins when it has no GDN flag of its own: `ssm_h_dtype:
    // f32` (the documented default spelling), `ssm_batched_recurrent: auto` (already
    // required and pinned as a bucket-A flag, unaffected by this policy extension).
    // `gdn_fused_norm: false` / `exact_verify: false` render to an OMITTED flag
    // (`schema::argv_for`), so they are absent here too — identical argv to a recipe that
    // never mentions them, which is the whole point of allowing an explicit `false`.
    let pinned = crate::cli::ServeArgs::try_parse_from([
        "serve",
        "org/model",
        "--ssm-h-dtype",
        "f32",
        "--ssm-batched-recurrent",
        "auto",
    ])
    .expect("parses");
    let plan = crate::main_modules::kernel_flag_plan::KernelFlagPlan::from_args(&pinned);
    let gdn = plan.gdn.expect("ssm_h_dtype given seals the GDN cell");

    assert_eq!(
        gdn.h_f16, today.h_f16,
        "h_f16 (ssm_h_dtype=f32) must match env resolution"
    );
    assert_eq!(gdn.h_f16_pool, today.h_f16_pool);
    assert_eq!(
        gdn.fused_norm, today.fused_norm,
        "gdn_fused_norm omitted == env unset"
    );
    assert_eq!(
        gdn.exact_verify, today.exact_verify,
        "exact_verify omitted == env unset"
    );
    let batched_recurrent = gdn.batched_recurrent.unwrap_or(
        metrale_model_layers::layers::ops::target_defaults::resolved()
            .ssm_batched_recurrent
            .value,
    );
    assert_eq!(
        batched_recurrent, today.batched_recurrent,
        "ssm_batched_recurrent=auto still defers to the HARDWARE.toml target default inside \
         a sealed cell (serve_flags.rs's own unwrap_or), so it matches either way"
    );
}

/// 2026-10-05: Every real `runtime: metrale` recipe parses AND produces a valid
/// `ServeArgs` (`Recipe::serve_args`, which runs `validate_serve_args`) with no
/// overrides. Being in `required_keys()` only proves a key is present; it says nothing
/// about whether the VALUE a recipe was given is one `validate_serve_args` accepts in
/// combination with the recipe's other settings (e.g. `--num-drafts` above 1 without a
/// speculative method, or `--fp8-kv-calibration-tokens` above 0 with a non-fp8
/// `--kv-cache-dtype`) — this is the check that catches that class of mistake.
#[test]
fn every_real_recipe_produces_a_valid_serve_config() {
    let recipes = load_real_recipes();
    let metrale: Vec<&Recipe> = recipes.iter().filter(|r| r.is_metrale()).collect();
    assert!(metrale.len() > 10, "sanity: {}", metrale.len());
    let mut failures = Vec::new();
    for r in &metrale {
        if let Err(e) = r.serve_args(&BTreeMap::new()) {
            failures.push(format!("{}: {e:#}", r.id));
        }
    }
    assert!(
        failures.is_empty(),
        "{} real recipe(s) do not produce a valid serve config:\n{}",
        failures.len(),
        failures.join("\n---\n")
    );
}
