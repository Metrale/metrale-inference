// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Tests for `bench_preflight::core::evaluate`. Each check gets its
//! pass/fail/skip paths by flipping exactly one `Facts` field, the convention
//! `bench_certify/preflight.rs` and `hardware/calibration_tests.rs` already use. A
//! `baseline()` fixture that is entirely `Pass`/`Skipped` lets every test start from a
//! known-clean state and change only what it is testing.
//!
//! Owner: server CLI (`met bench preflight`).
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-05: A `Facts` that reports clean on every check: an optimized, clean,
/// head-matching build with a fully-attested, un-staled kernel set, and every optional
/// check omitted. Tests mutate exactly one field away from this.
fn baseline() -> Facts {
    Facts {
        is_debug_build: false,
        embedded_head: "a".repeat(40),
        embedded_dirty: "false".to_string(),
        expect_head: Some(ExpectHeadFacts {
            requested: "HEAD".to_string(),
            resolved: Ok("a".repeat(40)),
        }),
        kernel_attestation_targets: 2,
        kernel_mismatches: Vec::new(),
        vllm: None,
        compare: None,
        recipe_explicit: None,
        env_vars: Vec::new(),
        recipe_env_mismatches: Vec::new(),
        recipe_load_error: None,
    }
}

fn status_of<'a>(results: &'a [CheckResult], id: &str) -> &'a CheckStatus {
    &results
        .iter()
        .find(|r| r.id == id)
        .unwrap_or_else(|| {
            panic!(
                "no check with id {id:?}; ids present: {:?}",
                results.iter().map(|r| r.id).collect::<Vec<_>>()
            )
        })
        .status
}

/// 2026-10-05: PATH A — a fully clean `Facts` passes every gating check and reports
/// `Skipped` for every optional one that was not supplied, never `Pass` for something
/// not actually checked.
#[test]
fn baseline_passes_every_gating_check_and_skips_every_optional_one() {
    let results = evaluate(&baseline());
    assert_eq!(results.len(), 8, "one row per check, always");
    assert_eq!(exit_code(&results), 0);
    assert_eq!(*status_of(&results, "release_build"), CheckStatus::Pass);
    assert_eq!(*status_of(&results, "binary_head"), CheckStatus::Pass);
    assert_eq!(*status_of(&results, "binary_clean"), CheckStatus::Pass);
    assert_eq!(*status_of(&results, "kernel_freshness"), CheckStatus::Pass);
    assert_eq!(
        *status_of(&results, "vllm_image_pinned"),
        CheckStatus::Skipped
    );
    assert_eq!(
        *status_of(&results, "comparison_non_vacuous"),
        CheckStatus::Skipped
    );
    assert_eq!(
        *status_of(&results, "recipe_explicit"),
        CheckStatus::Skipped
    );
    assert_eq!(
        *status_of(&results, "environment_record"),
        CheckStatus::Info
    );
}

// --- 1. RELEASE BUILD ---------------------------------------------------------------

#[test]
fn a_debug_build_fails_and_blocks() {
    let mut f = baseline();
    f.is_debug_build = true;
    let results = evaluate(&f);
    assert_eq!(*status_of(&results, "release_build"), CheckStatus::Fail);
    assert_eq!(exit_code(&results), 1);
}

// --- 2a. BINARY PROVENANCE: head -----------------------------------------------------

#[test]
fn no_expect_head_is_skipped_not_passed() {
    let mut f = baseline();
    f.expect_head = None;
    let results = evaluate(&f);
    assert_eq!(*status_of(&results, "binary_head"), CheckStatus::Skipped);
    assert_eq!(
        exit_code(&results),
        0,
        "an unrequested check must not block"
    );
}

#[test]
fn an_unresolvable_expect_head_fails_and_blocks() {
    let mut f = baseline();
    f.expect_head = Some(ExpectHeadFacts {
        requested: "not-a-ref".to_string(),
        resolved: Err("unknown revision".to_string()),
    });
    let results = evaluate(&f);
    assert_eq!(*status_of(&results, "binary_head"), CheckStatus::Fail);
    assert_eq!(exit_code(&results), 1);
}

#[test]
fn a_mismatched_expect_head_fails_and_names_both_shas() {
    let mut f = baseline();
    f.expect_head = Some(ExpectHeadFacts {
        requested: "HEAD".to_string(),
        resolved: Ok("b".repeat(40)),
    });
    let results = evaluate(&f);
    let row = results.iter().find(|r| r.id == "binary_head").unwrap();
    assert_eq!(row.status, CheckStatus::Fail);
    assert!(row.detail.contains(&"a".repeat(40)), "{}", row.detail);
    assert!(row.detail.contains(&"b".repeat(40)), "{}", row.detail);
    assert_eq!(exit_code(&results), 1);
}

#[test]
fn a_matching_expect_head_passes() {
    let results = evaluate(&baseline());
    assert_eq!(*status_of(&results, "binary_head"), CheckStatus::Pass);
}

// --- 2b. BINARY PROVENANCE: clean tree -----------------------------------------------

#[test]
fn a_dirty_build_fails_and_blocks() {
    let mut f = baseline();
    f.embedded_dirty = "true".to_string();
    let results = evaluate(&f);
    assert_eq!(*status_of(&results, "binary_clean"), CheckStatus::Fail);
    assert_eq!(exit_code(&results), 1);
}

/// 2026-10-05: An unknown dirty-flag value (git unavailable at build time) is treated as
/// dirty, never as a silent pass — PCND: no implicit "assume clean" default.
#[test]
fn an_unknown_dirty_flag_is_treated_as_dirty_not_as_clean() {
    let mut f = baseline();
    f.embedded_dirty = "unknown".to_string();
    let results = evaluate(&f);
    assert_eq!(*status_of(&results, "binary_clean"), CheckStatus::Fail);
}

// --- 3. KERNEL FRESHNESS --------------------------------------------------------------

#[test]
fn zero_attested_targets_fails_rather_than_passing_vacuously() {
    let mut f = baseline();
    f.kernel_attestation_targets = 0;
    let results = evaluate(&f);
    assert_eq!(*status_of(&results, "kernel_freshness"), CheckStatus::Fail);
    assert_eq!(exit_code(&results), 1);
}

#[test]
fn a_stale_target_fails_and_names_it() {
    let mut f = baseline();
    f.kernel_mismatches = vec!["gb10/qwen3.8-27b/nvfp4".to_string()];
    let results = evaluate(&f);
    let row = results.iter().find(|r| r.id == "kernel_freshness").unwrap();
    assert_eq!(row.status, CheckStatus::Fail);
    assert!(
        row.detail.contains("gb10/qwen3.8-27b/nvfp4"),
        "{}",
        row.detail
    );
}

#[test]
fn every_target_matching_passes() {
    let results = evaluate(&baseline());
    assert_eq!(*status_of(&results, "kernel_freshness"), CheckStatus::Pass);
}

// --- 4. VLLM IMAGE PINNED -------------------------------------------------------------

#[test]
fn no_vllm_image_is_skipped() {
    let results = evaluate(&baseline());
    assert_eq!(
        *status_of(&results, "vllm_image_pinned"),
        CheckStatus::Skipped
    );
}

#[test]
fn a_bare_tag_reference_fails() {
    let mut f = baseline();
    f.vllm = Some(VllmFacts {
        given_ref: "vllm/vllm-openai:latest".to_string(),
        repo_digests: Ok(vec!["vllm/vllm-openai@sha256:deadbeef".to_string()]),
    });
    let results = evaluate(&f);
    assert_eq!(*status_of(&results, "vllm_image_pinned"), CheckStatus::Fail);
}

#[test]
fn a_digest_the_local_image_does_not_carry_fails() {
    let mut f = baseline();
    f.vllm = Some(VllmFacts {
        given_ref: "vllm/vllm-openai@sha256:deadbeef".to_string(),
        repo_digests: Ok(vec!["vllm/vllm-openai@sha256:otherdigest".to_string()]),
    });
    let results = evaluate(&f);
    assert_eq!(*status_of(&results, "vllm_image_pinned"), CheckStatus::Fail);
}

#[test]
fn an_uninspectable_image_fails() {
    let mut f = baseline();
    f.vllm = Some(VllmFacts {
        given_ref: "vllm/vllm-openai@sha256:deadbeef".to_string(),
        repo_digests: Err("no such image".to_string()),
    });
    let results = evaluate(&f);
    assert_eq!(*status_of(&results, "vllm_image_pinned"), CheckStatus::Fail);
}

#[test]
fn a_digest_pin_matching_the_local_image_passes() {
    let mut f = baseline();
    f.vllm = Some(VllmFacts {
        given_ref: "vllm/vllm-openai@sha256:deadbeef".to_string(),
        repo_digests: Ok(vec!["vllm/vllm-openai@sha256:deadbeef".to_string()]),
    });
    let results = evaluate(&f);
    assert_eq!(*status_of(&results, "vllm_image_pinned"), CheckStatus::Pass);
}

// --- 5. NON-VACUOUS COMPARISON --------------------------------------------------------

#[test]
fn no_compare_is_skipped() {
    let results = evaluate(&baseline());
    assert_eq!(
        *status_of(&results, "comparison_non_vacuous"),
        CheckStatus::Skipped
    );
}

#[test]
fn two_empty_directories_fail_rather_than_reading_identical() {
    let mut f = baseline();
    f.compare = Some(CompareFacts {
        dir_a: "/tmp/a".to_string(),
        dir_b: "/tmp/b".to_string(),
        count_a: Ok(0),
        count_b: Ok(0),
    });
    let results = evaluate(&f);
    assert_eq!(
        *status_of(&results, "comparison_non_vacuous"),
        CheckStatus::Fail
    );
}

#[test]
fn one_empty_side_fails_even_if_the_other_is_not() {
    let mut f = baseline();
    f.compare = Some(CompareFacts {
        dir_a: "/tmp/a".to_string(),
        dir_b: "/tmp/b".to_string(),
        count_a: Ok(3),
        count_b: Ok(0),
    });
    let results = evaluate(&f);
    assert_eq!(
        *status_of(&results, "comparison_non_vacuous"),
        CheckStatus::Fail
    );
}

#[test]
fn different_file_counts_fail() {
    let mut f = baseline();
    f.compare = Some(CompareFacts {
        dir_a: "/tmp/a".to_string(),
        dir_b: "/tmp/b".to_string(),
        count_a: Ok(3),
        count_b: Ok(4),
    });
    let results = evaluate(&f);
    assert_eq!(
        *status_of(&results, "comparison_non_vacuous"),
        CheckStatus::Fail
    );
}

#[test]
fn an_unreadable_directory_fails_with_its_own_error() {
    let mut f = baseline();
    f.compare = Some(CompareFacts {
        dir_a: "/tmp/a".to_string(),
        dir_b: "/tmp/b".to_string(),
        count_a: Err("permission denied".to_string()),
        count_b: Ok(4),
    });
    let results = evaluate(&f);
    let row = results
        .iter()
        .find(|r| r.id == "comparison_non_vacuous")
        .unwrap();
    assert_eq!(row.status, CheckStatus::Fail);
    assert!(row.detail.contains("permission denied"), "{}", row.detail);
}

#[test]
fn equal_nonzero_counts_pass_and_the_report_prints_n() {
    let mut f = baseline();
    f.compare = Some(CompareFacts {
        dir_a: "/tmp/a".to_string(),
        dir_b: "/tmp/b".to_string(),
        count_a: Ok(7),
        count_b: Ok(7),
    });
    let results = evaluate(&f);
    let row = results
        .iter()
        .find(|r| r.id == "comparison_non_vacuous")
        .unwrap();
    assert_eq!(row.status, CheckStatus::Pass);
    assert!(row.detail.contains("N=7"), "{}", row.detail);
}

// --- 6. RECIPE FULLY EXPLICIT ----------------------------------------------------------

#[test]
fn no_recipe_is_skipped() {
    let results = evaluate(&baseline());
    assert_eq!(
        *status_of(&results, "recipe_explicit"),
        CheckStatus::Skipped
    );
}

/// 2026-10-05: With a recipe given but PR #124 not on main, the check is `Unavailable`
/// — never silently `Pass`, and distinct from `Skipped` (which means "not requested").
#[test]
fn a_requested_recipe_check_is_unavailable_not_passed_or_skipped() {
    let mut f = baseline();
    f.recipe_explicit = Some(RecipeExplicitFacts::Unavailable {
        reason: "PR #124 is not yet on main",
    });
    let results = evaluate(&f);
    let row = results.iter().find(|r| r.id == "recipe_explicit").unwrap();
    assert_eq!(row.status, CheckStatus::Unavailable);
    assert_ne!(row.status, CheckStatus::Pass);
    assert_ne!(row.status, CheckStatus::Skipped);
    assert!(row.detail.contains("#124"), "{}", row.detail);
    // 2026-10-05: Unavailable does not itself block — see the module doc.
    assert_eq!(exit_code(&results), 0);
}

// --- 7. ENVIRONMENT RECORD -------------------------------------------------------------

/// 2026-10-05: The environment record is always `Info` and never blocks, however many
/// vars or mismatches it carries — it is a record, not a gate.
#[test]
fn environment_record_never_blocks() {
    let mut f = baseline();
    f.env_vars = vec![("METRALE_TARGET_HW".to_string(), "gb10".to_string())];
    f.recipe_env_mismatches = vec![EnvMismatch {
        key: "METRALE_TARGET_HW".to_string(),
        recipe_value: "gb10".to_string(),
        process_value: None,
    }];
    let results = evaluate(&f);
    let row = results
        .iter()
        .find(|r| r.id == "environment_record")
        .unwrap();
    assert_eq!(row.status, CheckStatus::Info);
    assert!(row.detail.contains('1'), "{}", row.detail);
    assert_eq!(exit_code(&results), 0);
}

// --- exit_code ---------------------------------------------------------------------

#[test]
fn exit_code_is_nonzero_iff_any_check_fails() {
    assert_eq!(exit_code(&evaluate(&baseline())), 0);
    let mut f = baseline();
    f.is_debug_build = true;
    assert_eq!(exit_code(&evaluate(&f)), 1);
}

/// 2026-10-05: A recipe that failed to load reports the load error in the environment
/// record instead of a mismatch count that would otherwise read as "nothing differs".
#[test]
fn a_recipe_load_error_is_surfaced_not_swallowed_into_a_clean_count() {
    let mut f = baseline();
    f.recipe_load_error =
        Some("recipes/does-not-exist.yaml: No such file or directory".to_string());
    let results = evaluate(&f);
    let row = results
        .iter()
        .find(|r| r.id == "environment_record")
        .unwrap();
    assert_eq!(row.status, CheckStatus::Info);
    assert!(row.detail.contains("unavailable"), "{}", row.detail);
    assert!(row.detail.contains("does-not-exist.yaml"), "{}", row.detail);
}

// --- diff_recipe_env -------------------------------------------------------------------

#[test]
fn a_recipe_env_var_matching_the_process_is_not_a_mismatch() {
    let recipe_env =
        std::collections::BTreeMap::from([("METRALE_TARGET_HW".to_string(), "gb10".to_string())]);
    let process_env = vec![("METRALE_TARGET_HW".to_string(), "gb10".to_string())];
    assert!(diff_recipe_env(&recipe_env, &process_env).is_empty());
}

#[test]
fn an_unset_recipe_env_var_is_a_mismatch_with_no_process_value() {
    let recipe_env = std::collections::BTreeMap::from([(
        "METRALE_NVFP4_GEMM_BACKEND".to_string(),
        "marlin".to_string(),
    )]);
    let mismatches = diff_recipe_env(&recipe_env, &[]);
    assert_eq!(mismatches.len(), 1);
    assert_eq!(mismatches[0].key, "METRALE_NVFP4_GEMM_BACKEND");
    assert_eq!(mismatches[0].process_value, None);
}

#[test]
fn a_recipe_env_var_set_to_a_different_value_is_a_mismatch_naming_both() {
    let recipe_env =
        std::collections::BTreeMap::from([("METRALE_TARGET_HW".to_string(), "gb10".to_string())]);
    let process_env = vec![("METRALE_TARGET_HW".to_string(), "h100".to_string())];
    let mismatches = diff_recipe_env(&recipe_env, &process_env);
    assert_eq!(mismatches.len(), 1);
    assert_eq!(mismatches[0].recipe_value, "gb10");
    assert_eq!(mismatches[0].process_value, Some("h100".to_string()));
}

#[test]
fn exit_code_ignores_unavailable_and_skipped() {
    let mut f = baseline();
    f.recipe_explicit = Some(RecipeExplicitFacts::Unavailable { reason: "x" });
    f.expect_head = None;
    f.vllm = None;
    f.compare = None;
    assert_eq!(exit_code(&evaluate(&f)), 0);
}
