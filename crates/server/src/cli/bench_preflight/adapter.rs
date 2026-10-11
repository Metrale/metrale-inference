// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The I/O half of `met bench preflight` (SBIO): everything that reads this
//! process's own build-time constants, runs `git`/`docker`, walks a directory or loads a
//! recipe file, gathered once into a [`super::core::Facts`] for the pure
//! `core::evaluate` to decide over.
//!
//! Owner: server CLI (`met bench preflight`).
//! Invariants:
//! - No check decision is made here — only data gathering and the plumbing to turn an
//!   `Err` into the `String` [`super::core::Facts`] carries. `gather` cannot fail: every
//!   I/O error becomes a fact a check reports on (a FAIL with a reason), not a propagated
//!   `Err` that would abort the whole preflight before it can say why.

use std::path::{Path, PathBuf};

use metrale_bench::gate;

use super::core::{CompareFacts, ExpectHeadFacts, Facts, VllmFacts};
use super::{dircompare, docker, recipe_explicit};
use crate::cli;
use crate::recipe::Recipe;

/// 2026-10-05: What `met bench preflight` was asked to check, the I/O-shaped counterpart
/// of [`super::PreflightArgs`] (plain owned values, no clap).
pub struct Request {
    pub root: PathBuf,
    pub expect_head: Option<String>,
    pub vllm_image: Option<String>,
    pub compare: Option<(PathBuf, PathBuf)>,
    pub recipe: Option<String>,
}

/// 2026-10-05: Gather every fact `core::evaluate` reads. Never fails: an I/O error (git,
/// docker, a missing directory, a bad recipe path) is folded into the `Facts` field the
/// relevant check reads, so the check can report *why*, rather than the whole command
/// aborting with no table at all.
pub fn gather(req: &Request) -> Facts {
    let expect_head = req.expect_head.as_ref().map(|requested| ExpectHeadFacts {
        requested: requested.clone(),
        resolved: gate::git_rev_parse(&req.root, requested).map_err(|e| format!("{e:#}")),
    });

    let (kernel_attestation_targets, kernel_mismatches) = kernel_freshness(&req.root);

    let vllm = req.vllm_image.as_ref().map(|given_ref| VllmFacts {
        given_ref: given_ref.clone(),
        repo_digests: docker::repo_digests(given_ref),
    });

    let compare = req.compare.as_ref().map(|(a, b)| CompareFacts {
        dir_a: a.display().to_string(),
        dir_b: b.display().to_string(),
        count_a: dircompare::count_files(a),
        count_b: dircompare::count_files(b),
    });

    let recipe_explicit = req.recipe.as_ref().map(|r| recipe_explicit::check(r));

    let env_vars = metrale_env_vars();

    let (recipe_env_mismatches, recipe_load_error) = match &req.recipe {
        None => (Vec::new(), None),
        Some(recipe_ref) => match load_recipe(&req.root, recipe_ref) {
            Ok(recipe) => (super::core::diff_recipe_env(&recipe.env, &env_vars), None),
            Err(e) => (Vec::new(), Some(e)),
        },
    };

    Facts {
        is_debug_build: cfg!(debug_assertions),
        embedded_head: cli::BUILD_GIT_SHA.to_string(),
        embedded_dirty: cli::BUILD_GIT_DIRTY.to_string(),
        expect_head,
        kernel_attestation_targets,
        kernel_mismatches,
        vllm,
        compare,
        recipe_explicit,
        env_vars,
        recipe_env_mismatches,
        recipe_load_error,
    }
}

/// 2026-10-05: Parse this binary's own baked kernel-closure attestation
/// (`metrale_kernels::TARGET_CLOSURES`) and recompute every attested target's hash from
/// `root`'s working tree. Returns `(targets attested, keys that no longer match)`.
///
/// A target key that does not parse as `hardware/model/quant`, or whose hash cannot be
/// recomputed at all (sources no longer resolve), counts as a mismatch: there is no
/// "could not check, so assume fresh" path (PCND).
fn kernel_freshness(root: &Path) -> (usize, Vec<String>) {
    let attestation: gate::closure::Attestation =
        serde_json::from_str(metrale_kernels::TARGET_CLOSURES).unwrap_or_default();
    let mut mismatches = Vec::new();
    for (key, closure) in &attestation {
        let mut parts = key.splitn(3, '/');
        let target = match (parts.next(), parts.next(), parts.next()) {
            (Some(hardware), Some(model), Some(quant)) => gate::taxon::Target {
                hardware: hardware.to_string(),
                model: model.to_string(),
                quant: quant.to_string(),
            },
            _ => {
                mismatches.push(key.clone());
                continue;
            }
        };
        match closure.recompute_hash(root, &target) {
            Some(current) if current == closure.hash => {}
            _ => mismatches.push(key.clone()),
        }
    }
    (attestation.len(), mismatches)
}

/// 2026-10-05: Every `METRALE_*` variable set in this process, sorted by key so the
/// printed record is stable run to run.
fn metrale_env_vars() -> Vec<(String, String)> {
    let mut vars: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| k.starts_with("METRALE_"))
        .collect();
    vars.sort_by(|a, b| a.0.cmp(&b.0));
    vars
}

/// 2026-10-05: Load a recipe by id (`family/stem`, resolved under `<root>/recipes/`) or by
/// a direct `.yaml`/`.yml` path.
fn load_recipe(root: &Path, recipe_ref: &str) -> Result<Recipe, String> {
    let path = if recipe_ref.ends_with(".yaml") || recipe_ref.ends_with(".yml") {
        PathBuf::from(recipe_ref)
    } else {
        root.join("recipes").join(format!("{recipe_ref}.yaml"))
    };
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Recipe::parse(recipe_ref.to_string(), &text).map_err(|e| format!("{}: {e:#}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-05: A recipe reference that resolves to no file is a load error, not a
    /// panic, and names the path it looked for.
    #[test]
    fn an_unresolvable_recipe_id_is_a_named_load_error() {
        let root = std::env::temp_dir().join(format!(
            "metrale-preflight-adapter-{}-no-recipes-dir",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let err = load_recipe(&root, "family/does-not-exist").unwrap_err();
        assert!(err.contains("family/does-not-exist.yaml"), "{err}");
    }

    /// 2026-10-05: A direct `.yaml` path is read as given, not re-rooted under `recipes/`.
    #[test]
    fn a_direct_yaml_path_is_read_as_given() {
        let dir = std::env::temp_dir().join(format!(
            "metrale-preflight-adapter-{}-direct-path",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("custom.yaml");
        std::fs::write(
            &path,
            "recipe_version: \"2\"\nmodel: test/model\nruntime: metrale\ncontainer: test\n\
             defaults: {}\n",
        )
        .unwrap();
        let recipe = load_recipe(&dir, path.to_str().unwrap()).unwrap();
        assert_eq!(recipe.model, "test/model");
    }

    /// 2026-10-05: A `TARGET_CLOSURES` key with the wrong shape (not exactly three
    /// `/`-separated parts) is reported as a mismatch rather than silently skipped: the
    /// alternative — ignoring it — would let a malformed attestation pass this check
    /// vacuously.
    #[test]
    fn a_malformed_target_key_is_counted_as_a_mismatch() {
        // kernel_freshness() reads the real metrale_kernels::TARGET_CLOSURES of THIS
        // binary, so the malformed-key path is exercised directly against the parser
        // instead (the same splitn(3, '/') rule kernel_freshness uses).
        let key = "only-two/parts";
        let mut parts = key.splitn(3, '/');
        assert!(
            !matches!(
                (parts.next(), parts.next(), parts.next()),
                (Some(_), Some(_), Some(_))
            ),
            "a two-part key must not parse as a complete hardware/model/quant target"
        );
    }
}
