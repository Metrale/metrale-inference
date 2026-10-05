// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: `met bench preflight` — one AUTOMATIC gate run before every timed A/B,
//! ladder or benchmark, turning the lever journal's (PR #126,
//! `.claude/skills/new-hardware/references/lever-patterns.md`) measurement-method failure
//! table into checks that make those failures impossible rather than merely documented.
//!
//! `core` is the pure decision core (SBIO): [`core::evaluate`] takes a [`core::Facts`] and
//! returns the table, with no I/O of its own. `adapter` is the only place that runs git,
//! docker, reads a directory or loads a recipe. `docker`, `dircompare` and
//! `recipe_explicit` are the smaller I/O/interface pieces `adapter` composes.
//!
//! Owner: server CLI (`met bench preflight`).
//! Invariants:
//! - `preflight_cmd` never panics on a missing optional flag: every check it cannot run is
//!   `Skipped`, not an error that aborts before the table prints.
//! - The table always has one row per check (`core::evaluate`'s invariant); this module
//!   only renders it and computes the process exit code.

mod adapter;
mod dircompare;
mod docker;
mod recipe_explicit;

pub mod core;

use std::path::PathBuf;

use anyhow::Result;

/// `met bench preflight` — run every measurement-method check before a timed run.
#[derive(clap::Args, Debug)]
pub struct PreflightArgs {
    /// Git ref (a sha, `HEAD`, a branch or remote ref) this binary's embedded commit must
    /// match, resolved with `git rev-parse`. Omit to skip the binary-head check (reported
    /// SKIP, never a silent pass).
    #[arg(long = "expect-head")]
    pub expect_head: Option<String>,

    /// A vLLM image reference that must be pinned by digest (`name@sha256:...`); verified
    /// against the local image's RepoDigests via `docker inspect`. Omit to skip.
    #[arg(long = "vllm-image")]
    pub vllm_image: Option<String>,

    /// Two directories a byte-identity claim is about to be made over: refused when
    /// either is empty or they hold different file counts. Omit to skip.
    #[arg(long = "compare", num_args = 2, value_names = ["DIR_A", "DIR_B"])]
    pub compare: Option<Vec<PathBuf>>,

    /// A recipe to check for full explicitness (PR #124) and to diff the live
    /// `METRALE_*` environment against: an id (`family/stem`, under `recipes/`) or a
    /// path to a recipe YAML file. Omit to skip the recipe-explicit check (reported
    /// SKIP) and the environment diff (reported with zero mismatches, since there is
    /// nothing to diff against).
    #[arg(long)]
    pub recipe: Option<String>,

    /// Print the machine-readable report as one JSON object on stdout, in addition to the
    /// human table.
    #[arg(long)]
    pub json: bool,
}

/// 2026-10-05: `met bench preflight`'s entry point: gather, decide, render, exit code.
pub async fn preflight_cmd(args: PreflightArgs) -> Result<i32> {
    let root = super::bench_run::repo_root()?;
    let req = adapter::Request {
        root,
        expect_head: args.expect_head,
        vllm_image: args.vllm_image,
        compare: args.compare.map(|v| (v[0].clone(), v[1].clone())),
        recipe: args.recipe,
    };
    let facts = adapter::gather(&req);
    let results = core::evaluate(&facts);
    print_table(&results);
    if args.json {
        println!("{}", serde_json::to_string(&results)?);
    }
    Ok(core::exit_code(&results))
}

/// 2026-10-05: The one-screen PASS/FAIL table, in `core::evaluate`'s fixed check order —
/// the same `"  TAG  id"` + indented reason convention `bench_gate_check::print_statuses`
/// uses for `met benchmark --pull-request-gate-check`.
fn print_table(results: &[core::CheckResult]) {
    println!("met bench preflight");
    for r in results {
        println!("  {:<11} {}", r.status.tag(), r.title);
        println!("        {}", r.detail);
    }
    let failed: Vec<&str> = results
        .iter()
        .filter(|r| r.status.blocks())
        .map(|r| r.id)
        .collect();
    if failed.is_empty() {
        println!("preflight: clean to measure");
    } else {
        println!("preflight: REFUSED — {}", failed.join(", "));
    }
}

#[cfg(test)]
mod mod_tests {
    use super::*;

    /// 2026-10-05: `--compare` requires exactly two paths; clap enforces this before
    /// `preflight_cmd` ever sees it, so `req.compare`'s `v[0]`/`v[1]` indexing cannot
    /// panic on a short vector in production use. This test proves the enforcement, not
    /// the indexing.
    #[test]
    fn compare_rejects_one_path() {
        use clap::Parser;
        #[derive(clap::Parser, Debug)]
        struct Wrapper {
            #[command(flatten)]
            preflight: PreflightArgs,
        }
        let err = Wrapper::try_parse_from(["met", "--compare", "/tmp/a"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::WrongNumberOfValues);
    }

    #[test]
    fn compare_accepts_exactly_two_paths() {
        use clap::Parser;
        #[derive(clap::Parser)]
        struct Wrapper {
            #[command(flatten)]
            preflight: PreflightArgs,
        }
        let parsed = Wrapper::try_parse_from(["met", "--compare", "/tmp/a", "/tmp/b"]).unwrap();
        assert_eq!(
            parsed.preflight.compare,
            Some(vec![PathBuf::from("/tmp/a"), PathBuf::from("/tmp/b")])
        );
    }
}
