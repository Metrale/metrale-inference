// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The pure decision core of `met bench preflight` — one check per
//! measurement-method failure mode the lever journal (PR #126,
//! `.claude/skills/new-hardware/references/lever-patterns.md`) catalogued as cheap to
//! catch and expensive to miss:
//!
//! 1. a debug build mistaken for a hardware hang (`release_build`);
//! 2. a stale pre-head binary in a timed A/B (`binary_head`, `binary_clean`);
//! 3. stale PTX from a `CARGO_TARGET_DIR` shared across worktrees (`kernel_freshness`);
//! 4. vLLM `:latest` silently moving versions (`vllm_image_pinned`);
//! 5. an "IDENTICAL" comparison of two empty directories (`comparison_non_vacuous`);
//! 6. a recipe silently inheriting a `met serve` default such as `max_batch_size=8`
//!    (`recipe_explicit`);
//! plus a seventh item the journal's table does not name a failure for but the owner
//! asked for directly: an environment record, so a lever is provably armed rather than
//! assumed (`environment_record`).
//!
//! `evaluate` is a pure function over [`Facts`]: every I/O (reading the binary's own
//! build-time constants, running `git`/`docker`, listing directories, loading a recipe)
//! happens in `adapter.rs`, which gathers a `Facts` and hands it here. Each check is
//! tested by constructing the `Facts` that should trip it, not by running a real binary,
//! git repo or docker daemon.
//!
//! Owner: server CLI (`met bench preflight`).
//! Invariants:
//! - [`evaluate`] always returns exactly one [`CheckResult`] per check, in a fixed order,
//!   whether or not that check's inputs were supplied — an unsupplied, optional check
//!   (no `--expect-head`, no `--vllm-image`, no `--compare`, no `--recipe`) is
//!   [`CheckStatus::Skipped`], which is reported on the table like everything else, never
//!   omitted. A table with eight rows every time is what makes "it passed" and "it was
//!   never checked" impossible to confuse at a glance.
//! - Only [`CheckStatus::Fail`] makes [`exit_code`] non-zero. `Skipped` and `Unavailable`
//!   are loud (always a visible row) but not blocking by themselves: a benchmark that
//!   measures nothing comparison-shaped should not be refused for not passing
//!   `comparison_non_vacuous`. `recipe_explicit` is the one check whose `Unavailable` a
//!   caller may choose to additionally gate on until PR #124 lands (see its doc).

/// 2026-10-05: One check's outcome. `Pass`/`Fail`/`Warn` are a verdict on real evidence;
/// `Skipped` means the caller did not supply this check's inputs (an optional check that
/// does not apply to this run); `Unavailable` means the check's real implementation does
/// not exist yet on this branch (PR #124, see [`RecipeExplicitFacts`]) and a human must
/// not read the row as a pass. `Info` is a non-gating record (`environment_record`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Pass,
    Fail,
    Warn,
    Skipped,
    Unavailable,
    Info,
}

impl CheckStatus {
    /// 2026-10-05: Only `Fail` blocks. See the module doc for why `Skipped`/`Unavailable`
    /// do not.
    pub fn blocks(self) -> bool {
        matches!(self, CheckStatus::Fail)
    }

    /// 2026-10-05: The four-to-eleven character tag the table prints in its status
    /// column, padded by the caller.
    pub fn tag(self) -> &'static str {
        match self {
            CheckStatus::Pass => "PASS",
            CheckStatus::Fail => "FAIL",
            CheckStatus::Warn => "WARN",
            CheckStatus::Skipped => "SKIP",
            CheckStatus::Unavailable => "UNAVAILABLE",
            CheckStatus::Info => "INFO",
        }
    }
}

/// 2026-10-05: One row of the preflight table.
#[derive(Clone, Debug, serde::Serialize)]
pub struct CheckResult {
    /// 2026-10-05: Stable machine id (`"release_build"`, ...), for scripts and for
    /// `lever-patterns.md`'s gates table to point at.
    pub id: &'static str,
    pub title: &'static str,
    pub status: CheckStatus,
    pub detail: String,
}

impl CheckResult {
    fn new(
        id: &'static str,
        title: &'static str,
        status: CheckStatus,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            id,
            title,
            status,
            detail: detail.into(),
        }
    }
}

/// 2026-10-05: `--expect-head`'s resolution. Gathered by the adapter via
/// `gate::git_rev_parse(root, requested)`, so `requested` can be a sha, `HEAD`, a branch
/// or a remote ref — anything git itself accepts.
#[derive(Clone, Debug)]
pub struct ExpectHeadFacts {
    pub requested: String,
    /// 2026-10-05: The full 40-hex commit `requested` resolves to, or the error text
    /// when it does not resolve at all (unknown ref, not a git checkout).
    pub resolved: Result<String, String>,
}

/// 2026-10-05: `--vllm-image`'s local provenance, gathered by `docker inspect`.
#[derive(Clone, Debug)]
pub struct VllmFacts {
    pub given_ref: String,
    /// 2026-10-05: The local image's `RepoDigests`, or the error text when `docker
    /// inspect` failed (image not pulled locally, docker not running).
    pub repo_digests: Result<Vec<String>, String>,
}

/// 2026-10-05: `--compare DIR_A DIR_B`'s file counts, gathered by listing each directory.
#[derive(Clone, Debug)]
pub struct CompareFacts {
    pub dir_a: String,
    pub dir_b: String,
    pub count_a: Result<usize, String>,
    pub count_b: Result<usize, String>,
}

/// 2026-10-05: The recipe-explicitness check's outcome. Only one variant exists today:
/// PR #124 (`feat/recipes-fully-explicit`, "recipe: require every recipe-settable serve
/// flag to be explicit") is not yet on main, so there is no real checker to call. This
/// type is the interface PR #124 lands into: add a `Checked { missing: Vec<String> }`
/// variant backed by `recipe::explicit::{required_keys, missing_keys}` once it does, and
/// `evaluate`'s `recipe_explicit` arm gains a case for it. Until then every requested
/// check reports `Unavailable`, loudly — never `Pass`, never silently dropped.
#[derive(Clone, Debug)]
pub enum RecipeExplicitFacts {
    Unavailable { reason: &'static str },
}

/// 2026-10-05: One `METRALE_*` environment variable a recipe declares that the live
/// process either does not have set, or has set to a different value — proof the lever
/// the recipe asks for either is, or is not, actually armed (memory: prove-the-lever-
/// moved).
#[derive(Clone, Debug)]
pub struct EnvMismatch {
    pub key: String,
    pub recipe_value: String,
    pub process_value: Option<String>,
}

/// 2026-10-05: Everything [`evaluate`] reads, gathered once by `adapter::gather`.
/// Optional fields are `None` when the corresponding flag was not given — [`evaluate`]
/// reports `Skipped`, never `Pass`, for those.
#[derive(Clone, Debug, Default)]
pub struct Facts {
    pub is_debug_build: bool,
    /// 2026-10-05: `cli::BUILD_GIT_SHA` — this process's own build-time commit.
    pub embedded_head: String,
    /// 2026-10-05: `cli::BUILD_GIT_DIRTY` — `"true"`, `"false"` or `"unknown"`.
    pub embedded_dirty: String,
    pub expect_head: Option<ExpectHeadFacts>,
    /// 2026-10-05: How many targets `metrale_kernels::TARGET_CLOSURES` attests. Zero
    /// means the binary carries no kernel attestation at all (a `METRALE_SKIP_BUILD=1`
    /// build, or one whose compute target reported no compiler) — `evaluate` fails this
    /// rather than passing vacuously, since a benchmark that touches kernels cannot be
    /// verified fresh against an attestation that does not exist.
    pub kernel_attestation_targets: usize,
    /// 2026-10-05: `hardware/model/quant` keys whose recomputed hash did not match the
    /// binary's attested hash (or could not be recomputed at all — the adapter folds
    /// "could not recompute" into "mismatch", never into "match").
    pub kernel_mismatches: Vec<String>,
    pub vllm: Option<VllmFacts>,
    pub compare: Option<CompareFacts>,
    pub recipe_explicit: Option<RecipeExplicitFacts>,
    /// 2026-10-05: Every `METRALE_*` variable set in this process, sorted by key.
    pub env_vars: Vec<(String, String)>,
    pub recipe_env_mismatches: Vec<EnvMismatch>,
    /// 2026-10-05: Set when `--recipe` was given but the adapter could not load it (bad
    /// path, unparseable YAML) — [`diff_recipe_env`] never ran, so `recipe_env_mismatches`
    /// is an empty "nothing differs" that would otherwise read as good news it is not.
    pub recipe_load_error: Option<String>,
}

/// 2026-10-05: The sha-like prefix of a `name@sha256:...` image reference, or `None` for
/// a bare tag (`name:latest`, or no tag at all, which Docker reads as `:latest`).
fn digest_pin(image_ref: &str) -> Option<&str> {
    image_ref.split_once('@').map(|(_, digest)| digest)
}

/// 2026-10-05: Every check, in a fixed order, one [`CheckResult`] each. See the module
/// doc for why every check always produces a row.
pub fn evaluate(f: &Facts) -> Vec<CheckResult> {
    let mut out = Vec::with_capacity(8);

    // 1. RELEASE BUILD.
    out.push(
        match crate::cli::debug_build_guard::refuse_debug_build(f.is_debug_build) {
            Ok(()) => CheckResult::new(
                "release_build",
                "release build",
                CheckStatus::Pass,
                "built with --release",
            ),
            Err(msg) => CheckResult::new("release_build", "release build", CheckStatus::Fail, msg),
        },
    );

    // 2a. BINARY PROVENANCE — head.
    out.push(match &f.expect_head {
        None => CheckResult::new(
            "binary_head",
            "binary head == --expect-head",
            CheckStatus::Skipped,
            "no --expect-head given",
        ),
        Some(e) => match &e.resolved {
            Err(msg) => CheckResult::new(
                "binary_head",
                "binary head == --expect-head",
                CheckStatus::Fail,
                format!("--expect-head {:?} did not resolve: {msg}", e.requested),
            ),
            Ok(sha) if sha == &f.embedded_head => CheckResult::new(
                "binary_head",
                "binary head == --expect-head",
                CheckStatus::Pass,
                format!("{sha} (--expect-head {:?})", e.requested),
            ),
            Ok(sha) => CheckResult::new(
                "binary_head",
                "binary head == --expect-head",
                CheckStatus::Fail,
                format!(
                    "this binary was built from {}, but --expect-head {:?} resolves to \
                     {sha} — a stale pre-head binary would measure the wrong code",
                    f.embedded_head, e.requested
                ),
            ),
        },
    });

    // 2b. BINARY PROVENANCE — clean tree.
    out.push(match f.embedded_dirty.as_str() {
        "false" => CheckResult::new(
            "binary_clean",
            "build tree was not dirty",
            CheckStatus::Pass,
            "clean at build time",
        ),
        "true" => CheckResult::new(
            "binary_clean",
            "build tree was not dirty",
            CheckStatus::Fail,
            "this binary was built from a dirty tree — a gate number from it attests to \
             no reviewable commit",
        ),
        other => CheckResult::new(
            "binary_clean",
            "build tree was not dirty",
            CheckStatus::Fail,
            format!(
                "build-time dirty flag is {other:?}, not a known value (git was \
                 unavailable at build time) — treated as dirty, not as clean"
            ),
        ),
    });

    // 3. KERNEL FRESHNESS.
    out.push(if f.kernel_attestation_targets == 0 {
        CheckResult::new(
            "kernel_freshness",
            "kernel closure matches the tree",
            CheckStatus::Fail,
            "this binary carries no kernel closure attestation at all \
             (metrale_kernels::TARGET_CLOSURES is empty) — PTX freshness cannot be verified",
        )
    } else if f.kernel_mismatches.is_empty() {
        CheckResult::new(
            "kernel_freshness",
            "kernel closure matches the tree",
            CheckStatus::Pass,
            format!(
                "{} target(s) attested, all match the tree",
                f.kernel_attestation_targets
            ),
        )
    } else {
        CheckResult::new(
            "kernel_freshness",
            "kernel closure matches the tree",
            CheckStatus::Fail,
            format!(
                "stale PTX: {} of {} attested target(s) no longer match the tree \
                 (a CARGO_TARGET_DIR shared across worktrees is the known cause): {}",
                f.kernel_mismatches.len(),
                f.kernel_attestation_targets,
                f.kernel_mismatches.join(", ")
            ),
        )
    });

    // 4. VLLM IMAGE PINNED.
    out.push(match &f.vllm {
        None => CheckResult::new(
            "vllm_image_pinned",
            "vLLM image pinned by digest",
            CheckStatus::Skipped,
            "no --vllm-image given",
        ),
        Some(v) => match digest_pin(&v.given_ref) {
            None => CheckResult::new(
                "vllm_image_pinned",
                "vLLM image pinned by digest",
                CheckStatus::Fail,
                format!(
                    "{:?} is not pinned by digest (no @sha256:...) — a tag can silently \
                     move to a different build, the way :latest moved 0.27.1 -> 0.31.0",
                    v.given_ref
                ),
            ),
            Some(digest) => match &v.repo_digests {
                Err(msg) => CheckResult::new(
                    "vllm_image_pinned",
                    "vLLM image pinned by digest",
                    CheckStatus::Fail,
                    format!("could not inspect {:?} locally: {msg}", v.given_ref),
                ),
                Ok(digests) if digests.iter().any(|d| d.contains(digest)) => CheckResult::new(
                    "vllm_image_pinned",
                    "vLLM image pinned by digest",
                    CheckStatus::Pass,
                    format!("{:?} matches a local RepoDigest", v.given_ref),
                ),
                Ok(digests) => CheckResult::new(
                    "vllm_image_pinned",
                    "vLLM image pinned by digest",
                    CheckStatus::Fail,
                    format!(
                        "{:?} names digest {digest}, but the local image's RepoDigests are \
                         {digests:?} — pull the pinned digest before measuring",
                        v.given_ref
                    ),
                ),
            },
        },
    });

    // 5. NON-VACUOUS COMPARISON.
    out.push(match &f.compare {
        None => CheckResult::new(
            "comparison_non_vacuous",
            "comparison is non-vacuous",
            CheckStatus::Skipped,
            "no --compare given",
        ),
        Some(c) => match (&c.count_a, &c.count_b) {
            (Err(e), _) => CheckResult::new(
                "comparison_non_vacuous",
                "comparison is non-vacuous",
                CheckStatus::Fail,
                format!("{}: {e}", c.dir_a),
            ),
            (_, Err(e)) => CheckResult::new(
                "comparison_non_vacuous",
                "comparison is non-vacuous",
                CheckStatus::Fail,
                format!("{}: {e}", c.dir_b),
            ),
            (Ok(a), Ok(b)) if *a == 0 || *b == 0 => CheckResult::new(
                "comparison_non_vacuous",
                "comparison is non-vacuous",
                CheckStatus::Fail,
                format!(
                    "{} has {a} file(s), {} has {b} file(s) — an empty side can prove \
                     nothing \"IDENTICAL\"",
                    c.dir_a, c.dir_b
                ),
            ),
            (Ok(a), Ok(b)) if a != b => CheckResult::new(
                "comparison_non_vacuous",
                "comparison is non-vacuous",
                CheckStatus::Fail,
                format!(
                    "{a} file(s) in {} vs {b} in {} — different counts cannot be an identity \
                     claim",
                    c.dir_a, c.dir_b
                ),
            ),
            (Ok(n), Ok(_)) => CheckResult::new(
                "comparison_non_vacuous",
                "comparison is non-vacuous",
                CheckStatus::Pass,
                format!("N={n} file(s) compared on both sides"),
            ),
        },
    });

    // 6. RECIPE FULLY EXPLICIT.
    out.push(match &f.recipe_explicit {
        None => CheckResult::new(
            "recipe_explicit",
            "recipe sets every serve flag explicitly",
            CheckStatus::Skipped,
            "no --recipe given",
        ),
        Some(RecipeExplicitFacts::Unavailable { reason }) => CheckResult::new(
            "recipe_explicit",
            "recipe sets every serve flag explicitly",
            CheckStatus::Unavailable,
            *reason,
        ),
    });

    // 7. ENVIRONMENT RECORD (always informational; never blocks).
    out.push(CheckResult::new(
        "environment_record",
        "environment record",
        CheckStatus::Info,
        match &f.recipe_load_error {
            Some(err) => format!(
                "{} METRALE_* var(s) set in this process; recipe env diff unavailable: {err}",
                f.env_vars.len()
            ),
            None => format!(
                "{} METRALE_* var(s) set in this process; {} differ from the recipe's \
                 declared env",
                f.env_vars.len(),
                f.recipe_env_mismatches.len()
            ),
        },
    ));

    out
}

/// 2026-10-05: 1 when any check [`CheckStatus::blocks`], else 0 — `main` exits with this.
pub fn exit_code(results: &[CheckResult]) -> i32 {
    i32::from(results.iter().any(|r| r.status.blocks()))
}

/// 2026-10-05: Which of a recipe's declared `env:` entries the live process either does
/// not have set, or has set to a different value — the environment-record check's proof
/// that a lever the recipe asks for is actually armed (memory: prove-the-lever-moved),
/// not merely declared in YAML. Pure: takes the recipe's `env:` map and the process's
/// `METRALE_*` snapshot as plain data, so the adapter is the only place that reads a
/// recipe file or `std::env`.
pub fn diff_recipe_env(
    recipe_env: &std::collections::BTreeMap<String, String>,
    process_env: &[(String, String)],
) -> Vec<EnvMismatch> {
    recipe_env
        .iter()
        .filter_map(|(key, recipe_value)| {
            let process_value = process_env
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone());
            if process_value.as_deref() == Some(recipe_value.as_str()) {
                None
            } else {
                Some(EnvMismatch {
                    key: key.clone(),
                    recipe_value: recipe_value.clone(),
                    process_value,
                })
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "core_tests.rs"]
mod core_tests;
