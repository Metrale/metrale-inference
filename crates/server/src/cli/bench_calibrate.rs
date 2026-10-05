// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `met benchmark calibrate`: the box-calibration probe.
//!
//! Drives three existing gates as `run --pull-request-gate --serve-reuse`
//! would — `decode-floor`, `high-isl-ttft-cold`, `high-isl-ttft-warm` — one
//! model load shared across all three, reads back each leg's own committed
//! gate record for the raw metric it contributes, and writes
//! `kernels/<hardware>/BOX_PROFILES.toml`.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants:
//! - A leg's PASS/FAIL verdict is not consulted: `--no-fail-on-verdict` is
//!   always set, because calibration reads the raw metric, not the gate's own
//!   ceiling.
//! - The leased server is released at the end, on every path (success or
//!   error), so a failed calibration never leaves a server running that a
//!   later `run --serve-reuse` would silently inherit.
//! - Live GPU verification note (2026-10-04): the three-leg drive has not
//!   been run end to end on a live box as of this commit — see the PR
//!   description. The pure profile math (`hardware::calibration`) is fully
//!   unit tested without a GPU.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use metrale_bench::gate::{self, GateRecord};
use metrale_bench::hardware::Hardware;
use metrale_bench::hardware::calibration::{self, BoxProfile, LegMetrics, ProfileEntry, Raw};

use super::bench_args::{CalibrateArgs, OutputFormat, RunArgs};
use super::bench_run;

/// 2026-10-04: One leg of the probe: its benchmark id and which raw field(s)
/// of [`calibration::LegMetrics`] its record feeds.
const LEGS: [&str; 3] = ["decode-floor", "high-isl-ttft-cold", "high-isl-ttft-warm"];

pub async fn calibrate_cmd(args: CalibrateArgs) -> Result<i32> {
    let root = bench_run::repo_root()?;
    let hardware = args
        .hardware
        .clone()
        .unwrap_or_else(|| Hardware::probe().gate_key());

    if args.dry_run {
        println!(
            "calibrate: would run {} against checkpoint {:?} on hardware {hardware:?}, \
             sharing one leased server, then write kernels/{hardware}/BOX_PROFILES.toml",
            LEGS.join(", "),
            args.checkpoint
        );
        return Ok(0);
    }
    if let Err(msg) = super::debug_build_guard::refuse_debug_build(cfg!(debug_assertions)) {
        bail!("{msg}");
    }

    let owner = std::process::id();
    let sha = gate::git_sha(&root).context("resolving the commit under test")?;

    let mut records: BTreeMap<&str, GateRecord> = BTreeMap::new();
    let mut first_error: Option<anyhow::Error> = None;
    for &id in &LEGS {
        match run_leg(id, &hardware, &args.checkpoint, owner).await {
            Ok(()) => match newest_record_for(&root, id, &sha) {
                Ok(rec) => {
                    records.insert(id, rec);
                }
                Err(e) => {
                    eprintln!("calibrate: {id} ran but its record could not be read: {e:#}");
                    first_error.get_or_insert(e);
                }
            },
            Err(e) => {
                eprintln!("calibrate: {id} did not complete: {e:#}");
                first_error.get_or_insert(e);
            }
        }
    }

    // 2026-10-04: Release the leased server before acting on any leg's
    // outcome: a calibration that errors out must not leave a server behind
    // for the next unrelated `--serve-reuse` run to inherit.
    if let Err(e) = super::bench_lease::release_cmd() {
        eprintln!("calibrate: releasing the leased server: {e:#}");
    }

    let legs = LegMetrics {
        decode_floor: records.get("decode-floor").map(|r| &r.metrics),
        high_isl_cold: records.get("high-isl-ttft-cold").map(|r| &r.metrics),
        high_isl_warm: records.get("high-isl-ttft-warm").map(|r| &r.metrics),
    };
    let raw = calibration::extract(legs);
    if raw == Raw::default() {
        bail!(
            "calibrate: no leg produced a usable reading — nothing to write to \
             BOX_PROFILES.toml{}",
            first_error
                .as_ref()
                .map(|e| format!(" (first error: {e:#})"))
                .unwrap_or_default()
        );
    }

    let mut profiles = calibration::load(&root, &hardware)?;
    let perf_class = perf_class_of(&records).unwrap_or_else(|| format!("{hardware}@unknown"));
    let reference = Raw::fleet_mean(&profiles.profiles, Some(&raw));
    let profile = BoxProfile::new(gate::now_secs(), raw, &reference);
    profiles.profiles.insert(
        perf_class.clone(),
        ProfileEntry {
            git_sha: sha.clone(),
            profile,
        },
    );
    calibration::save(&root, &hardware, &profiles)?;
    println!(
        "calibrate: wrote kernels/{hardware}/BOX_PROFILES.toml [{perf_class}] \
         decode={:?} tok/s (x{:?}) prefill_cold32k={:?} ms (x{:?}) \
         restore_warm32k={:?} ms (x{:?}) energy_c1={:?} J/tok (x{:?}) \
         idle={:?} W (x{:?})",
        profile.decode_tok_s,
        profile.decode_tok_s_ratio,
        profile.prefill_cold32k_ms,
        profile.prefill_cold32k_ms_ratio,
        profile.restore_warm32k_ms,
        profile.restore_warm32k_ms_ratio,
        profile.energy_c1_j_per_tok,
        profile.energy_c1_ratio,
        profile.idle_power_w,
        profile.idle_power_w_ratio,
    );

    match first_error {
        // 2026-10-04: A partial profile (some legs missing) is written and
        // reported, but still exits non-zero: a campaign's automatic
        // calibration step should notice and re-run rather than silently
        // trust an incomplete profile.
        Some(_) => Ok(1),
        None => Ok(0),
    }
}

/// 2026-10-04: One leg: `run --pull-request-gate --serve-reuse
/// --no-fail-on-verdict`, sharing `owner`'s lease across legs.
async fn run_leg(id: &str, hardware: &str, checkpoint: &str, owner: u32) -> Result<()> {
    let args = RunArgs {
        id: id.to_string(),
        url: String::new(),
        model: None,
        hardware: Some(hardware.to_string()),
        checkpoint: Some(checkpoint.to_string()),
        params: Vec::new(),
        format: OutputFormat::Json,
        poll_ms: 250,
        no_save: true,
        yes: false,
        quiet: true,
        // 2026-10-04: The module invariant: a leg's own ceiling never aborts
        // calibration.
        no_fail_on_verdict: true,
        skip_coherence_probe: false,
        pull_request_gate: true,
        serve_override: Vec::new(),
        serve_reuse: true,
        serve_lease_owner: Some(owner),
        output_image: None,
        output_image_args: None,
    };
    let code = bench_run::run(args).await?;
    if code != 0 {
        // 2026-10-04: With `no_fail_on_verdict`, `run` itself already exits 0
        // on a FAIL verdict; a non-zero code here is a harness-level problem
        // (e.g. an inconclusive run with nothing to measure), not a verdict.
        bail!("{id} exited {code}");
    }
    Ok(())
}

/// 2026-10-04: The newest record in `.benchmarks/<id>/` whose file name
/// carries `sha` — handles same-day re-runs, which sort lexically after the
/// canonical name (`gate::record_path`'s doc).
fn newest_record_for(root: &Path, id: &str, sha: &str) -> Result<GateRecord> {
    let dir = gate::gate_dir(root, id);
    let newest = std::fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "json")
                && p.file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.contains(sha))
        })
        .max()
        .with_context(|| format!("no {id} record for {sha} under {}", dir.display()))?;
    gate::read_record(&newest)
}

/// 2026-10-04: The `perf_class` (`"gb10@dgx2"`) the legs' own captures agree
/// on, from whichever leg has a `hardware_state`. `None` when no leg captured
/// one (every leg's precheck was refused before any capture), which the
/// caller falls back from rather than failing calibration outright.
fn perf_class_of(records: &BTreeMap<&str, GateRecord>) -> Option<String> {
    records
        .values()
        .find_map(|r| r.hardware_state.as_ref())
        .map(|s| s.perf_class.clone())
}
