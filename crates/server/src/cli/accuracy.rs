// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `met accuracy points|check|calibrate`: the I/O side of metrale-accuracy. It reads
//! the repository's INSTANCES.toml, circuits, KERNEL_FAMILIES.toml and ACCURACY.toml from the
//! working tree, plans the checks, runs them on this binary's kernels through the GPU runner,
//! prints one line per check and writes the record.
//!
//! Owner: server CLI.
//! Invariants:
//! - Nothing here judges a kernel: `metrale_accuracy::check` does, over the runner this file
//!   supplies.
//! - The exit status is 0 only when every planned check passed; a contract problem, an
//!   unrunnable check or an empty plan is a failure, never a pass.
//! - Every `METRALE_*` variable set in the environment is printed and recorded (a launcher lever
//!   can route a kernel elsewhere).

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use metrale_accuracy::check::{Job, Outcome, Verdict, run};
use metrale_accuracy::contract::parse_contracts;
use metrale_accuracy::jobs::{self, Scope};
use metrale_accuracy::points::sweep;
use metrale_accuracy::record::{Record, calibration_rows, margin, render};
use metrale_circuit::venn::{Repo, parse_families};
use sha2::{Digest, Sha256};

use super::accuracy_gpu::GpuRunner;
use super::circuit_venn::{FsRepo, find_root};
use super::{AccuracyAction, AccuracyArgs, AccuracyRunArgs, AccuracySelectArgs};

fn sha(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn repo(sel: &AccuracySelectArgs) -> Result<FsRepo> {
    let root = match &sel.root {
        Some(r) => r.clone(),
        None => find_root(&std::env::current_dir()?)?,
    };
    Ok(FsRepo { root })
}

/// 2026-10-09: Run `met accuracy`; the process exit status.
pub(crate) fn dispatch(args: AccuracyArgs) -> Result<i32> {
    match args.action {
        AccuracyAction::Points(sel) => points(&sel),
        AccuracyAction::Check(a) => check(&a, false),
        AccuracyAction::Calibrate(a) => check(&a, true),
        AccuracyAction::Model(a) => super::accuracy_model::run(&a),
    }
}

struct Loaded {
    contracts_text: String,
    families_text: String,
}

fn load(repo: &FsRepo, hw: &str) -> Result<Loaded> {
    let read = |rel: &str| repo.read(rel).map_err(anyhow::Error::msg);
    Ok(Loaded {
        contracts_text: read(&format!("kernels/{hw}/common/ACCURACY.toml"))?,
        families_text: read(&format!("kernels/{hw}/common/KERNEL_FAMILIES.toml"))?,
    })
}

fn points(sel: &AccuracySelectArgs) -> Result<i32> {
    let repo = repo(sel)?;
    let l = load(&repo, &sel.hardware)?;
    let contracts = parse_contracts(&l.contracts_text)?;
    let fams = parse_families(&l.families_text)?;
    let problems = jobs::validate(&contracts, &fams);
    let s = sweep(&repo, &sel.hardware)?;
    let (planned, cov) = jobs::plan(
        &s,
        &contracts,
        &fams,
        Scope::Full,
        sel.family.as_deref(),
        sel.model.as_deref(),
    );
    let mut by: BTreeMap<(String, String), usize> = BTreeMap::new();
    for p in &planned {
        *by.entry((p.family.id.clone(), p.kernel.clone()))
            .or_default() += 1;
    }
    println!(
        "# {} recipes, {} swept points, {} covered",
        s.recipes.len(),
        cov.swept_points,
        cov.covered_points
    );
    for ((f, k), n) in &by {
        println!("covered  {f:<24} {k:<56} {n} checks (full)");
    }
    let mut gaps: BTreeMap<(String, String), (Vec<String>, String)> = BTreeMap::new();
    for ((f, k, op), why) in &cov.uncovered {
        let e = gaps
            .entry((f.clone(), why.clone()))
            .or_insert_with(|| (Vec::new(), k.clone()));
        e.0.push(op.clone());
    }
    for ((f, why), (ops, k)) in &gaps {
        println!(
            "UNCOVERED {f:<23} {} op(s) e.g. {} via {k} — {why}",
            ops.len(),
            ops[0]
        );
    }
    for (f, k) in &cov.unused {
        println!("UNSWEPT  {f:<24} {k} — contracted, no described model runs it");
    }
    let mut problems = problems;
    for c in &contracts.contracts {
        let sibling = match &c.class {
            metrale_accuracy::contract::Class::BitIdentical { against } => Some(against),
            metrale_accuracy::contract::Class::Derived => None,
        };
        for k in c.kernels.iter().chain(sibling) {
            if !super::accuracy_adapters::has_adapter(k) {
                problems.push(format!("`{}`: no launch adapter for `{k}`", c.family));
            }
        }
    }
    for p in &problems {
        println!("CONTRACT {p}");
    }
    Ok(i32::from(!problems.is_empty()))
}

fn metrale_env() -> Vec<String> {
    let mut v: Vec<String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("METRALE_"))
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    v.sort();
    v
}

fn check(a: &AccuracyRunArgs, calibrate: bool) -> Result<i32> {
    let sel = &a.select;
    let scope =
        Scope::parse(&a.scope).with_context(|| format!("--scope {} (quick | full)", a.scope))?;
    let repo = repo(sel)?;
    let l = load(&repo, &sel.hardware)?;
    let contracts = parse_contracts(&l.contracts_text)?;
    let fams = parse_families(&l.families_text)?;
    let problems = jobs::validate(&contracts, &fams);
    if !problems.is_empty() {
        bail!(
            "ACCURACY.toml does not fit the families:\n  {}",
            problems.join("\n  ")
        );
    }
    let env = metrale_env();
    for e in &env {
        eprintln!("env: {e}");
    }
    let s = sweep(&repo, &sel.hardware)?;
    let (planned, cov) = jobs::plan(
        &s,
        &contracts,
        &fams,
        scope,
        sel.family.as_deref(),
        sel.model.as_deref(),
    );
    if planned.is_empty() {
        bail!(
            "no check planned (family {:?}, model {:?}): nothing would be proven",
            sel.family,
            sel.model
        );
    }
    let mut runner = GpuRunner::new(&sel.hardware)?;
    let started = std::time::Instant::now();
    let mut outcomes: Vec<Outcome> = Vec::with_capacity(planned.len());
    for p in &planned {
        let job = Job {
            contract: p.contract,
            family: p.family,
            kernel: &p.kernel,
            point: &p.point,
            shape: &p.shape,
            input: p.input,
            seed: contracts.seed,
        };
        let t = std::time::Instant::now();
        let o = match runner.select(&p.targets) {
            Ok(()) => run(&job, &mut runner),
            Err(e) => unrunnable(&job, e.to_string()),
        };
        print_line(&o, t.elapsed());
        outcomes.push(o);
    }
    let failed = outcomes
        .iter()
        .filter(|o| o.verdict != Verdict::Pass)
        .count();
    println!(
        "# {} checks, {} failed, {:.1} s; {} of {} swept points covered",
        outcomes.len(),
        failed,
        started.elapsed().as_secs_f64(),
        cov.covered_points,
        cov.swept_points
    );
    let record = Record {
        hardware: sel.hardware.clone(),
        closures: runner.closures_used(),
        device: metrale_accuracy::runner::KernelRunner::device(&runner),
        commit: metrale_bench::gate::git_sha(&repo.root)
            .unwrap_or_else(|e| format!("unknown: {e}")),
        contracts_sha256: sha(&l.contracts_text),
        families_sha256: sha(&l.families_text),
        scope: a.scope.clone(),
        outcomes,
        coverage: cov,
    };
    let path = write_record(&a.out, &record, &env)?;
    println!("# record: {}", path.display());
    if calibrate {
        print!(
            "{}",
            calibration_rows(&record.outcomes, &record.closures.join(","))
        );
    }
    Ok(i32::from(failed > 0))
}

fn unrunnable(job: &Job<'_>, why: String) -> Outcome {
    Outcome {
        family: job.family.id.clone(),
        kernel: job.kernel.to_string(),
        key: job.key(),
        input: job.input,
        good: None,
        floor: None,
        mutations: Vec::new(),
        output_sha256: String::new(),
        verdict: Verdict::Error(why),
    }
}

fn print_line(o: &Outcome, t: std::time::Duration) {
    let good = o.good.as_ref().map_or("-".to_string(), |g| {
        format!("ratio {:.2e} (margin {:.1e})", g.ratio, margin(g.ratio))
    });
    let floor = o
        .floor
        .as_ref()
        .map_or(String::new(), |f| format!(" floor {:.2e}", f.ratio));
    let muts = o
        .mutations
        .iter()
        .map(|m| format!("{}={:.1e}", m.name, m.ratio))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "{:<28} {:<14} {:<70} {good}{floor} | {muts} [{} ms]",
        o.verdict.name(),
        o.input.name(),
        o.key,
        t.as_millis()
    );
}

fn write_record(out: &std::path::Path, r: &Record, env: &[String]) -> Result<PathBuf> {
    let dir = out.join(&r.hardware);
    std::fs::create_dir_all(&dir).with_context(|| dir.display().to_string())?;
    let key = sha(&r.closures.join(","));
    let path = dir.join(format!("{}.toml", &key[..16]));
    let env_lines: String = env.iter().map(|e| format!("# env: {e}\n")).collect();
    std::fs::write(&path, format!("{env_lines}{}", render(r)))
        .with_context(|| path.display().to_string())?;
    Ok(path)
}
