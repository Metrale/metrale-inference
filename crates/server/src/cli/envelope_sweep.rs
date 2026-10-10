// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `met accuracy envelope grid|sweep`: the I/O side of
//! `metrale_accuracy::envelope`. `grid` prints this shard's cells; `sweep` runs them on this
//! box's GPU. Per cell: the shared operands (one case per digest class, built once so every
//! candidate reads the same bytes), then per candidate the output digests, its own accuracy
//! contract (seeded inputs, no mutations: the contracts' mutation coverage is proven by
//! `met accuracy check`), and, if it passed, its timing; one JSON record each.
//!
//! Owner: server CLI.
//! Invariants:
//! - A candidate that is unavailable, faults or leaves its contract is recorded with no time and
//!   can never win; nothing is skipped silently.
//! - Each record carries the clock-event reasons NVML reported around its timing; a power or
//!   thermal slowdown marks it `throttled` so a later run re-measures it (the resume skips only
//!   non-throttled records).
//! - The sweep stops cleanly at `--deadline-unix`, between candidates.

use std::collections::BTreeSet;
use std::io::Write as _;

use anyhow::{Context, Result, bail};
use metrale_accuracy::case::Case;
use metrale_accuracy::check::{Job, Verdict as CheckVerdict, case_of, run};
use metrale_accuracy::contract::{Contract, Contracts, parse_contracts};
use metrale_accuracy::envelope::grid::{GridCell, LADDER, grid, point_for, shard, uncovered};
use metrale_accuracy::envelope::record::{Cell, Measurement, SCHEMA, Verdict, parse_records};
use metrale_accuracy::inputs::InputClass;
use metrale_accuracy::points::{Shape, Sweep, sweep};
use metrale_accuracy::runner::{KernelRunner, RunError};
use metrale_circuit::venn::{Families, Repo, parse_families};
use sha2::{Digest, Sha256};

use crate::cli::accuracy_adapters_tc::{sweep_max_rows, sweep_min_rows};
use crate::cli::accuracy_args::{
    EnvelopeAction, EnvelopeArgs, EnvelopeGridArgs, EnvelopeSweepArgs,
};
use crate::cli::accuracy_gpu::GpuRunner;
use crate::cli::circuit_venn::{FsRepo, find_root};

/// 2026-10-10: The input classes every candidate's output bytes are digested on (each in every
/// projection contract): the selection compares a winner's digests with the default's.
const DIGEST_CLASSES: [InputClass; 2] = [InputClass::Gaussian, InputClass::Outliers];

/// 2026-10-10: NVML clock-event reasons that mean a slowdown: SW power cap (0x4), HW slowdown
/// (0x8), SW thermal (0x20), HW thermal (0x40), HW power brake (0x80).
const SLOWDOWN: u64 = 0x4 | 0x8 | 0x20 | 0x40 | 0x80;

pub(crate) fn dispatch(a: EnvelopeArgs) -> Result<i32> {
    match a.action {
        EnvelopeAction::Grid(g) => print_grid(&g),
        EnvelopeAction::Sweep(s) => sweep_gpu(&s),
    }
}

struct Inputs {
    repo: FsRepo,
    sweep: Sweep,
    contracts: Contracts,
    fams: Families,
}

fn inputs(g: &EnvelopeGridArgs) -> Result<Inputs> {
    let root = match &g.root {
        Some(r) => r.clone(),
        None => find_root(&std::env::current_dir()?)?,
    };
    let repo = FsRepo { root };
    let read = |rel: &str| repo.read(rel).map_err(anyhow::Error::msg);
    let contracts = parse_contracts(&read(&format!(
        "kernels/{}/common/ACCURACY.toml",
        g.hardware
    ))?)?;
    let fams = parse_families(&read(&format!(
        "kernels/{}/common/KERNEL_FAMILIES.toml",
        g.hardware
    ))?)?;
    let sweep = sweep(&repo, &g.hardware)?;
    Ok(Inputs {
        repo,
        sweep,
        contracts,
        fams,
    })
}

fn cells_of(i: &Inputs, g: &EnvelopeGridArgs) -> Result<Vec<GridCell>> {
    if g.shards == 0 || g.shard >= g.shards {
        bail!("--shard {} of --shards {}", g.shard, g.shards);
    }
    let mut all = grid(&i.sweep, &i.contracts, &i.fams, &LADDER, g.margin);
    if !g.families.is_empty() {
        for c in &mut all {
            c.candidates.retain(|x| g.families.contains(&x.family));
        }
        all.retain(|c| !c.candidates.is_empty());
    }
    Ok(shard(&all, g.shard, g.shards)
        .into_iter()
        .cloned()
        .collect())
}

fn print_grid(g: &EnvelopeGridArgs) -> Result<i32> {
    let i = inputs(g)?;
    let cells = cells_of(&i, g)?;
    let mut est = 0.0;
    for c in &cells {
        let cand: Vec<&str> = c.candidates.iter().map(|x| x.kernel.as_str()).collect();
        println!(
            "{} {} {} {}x{} rows={} margin={} default={} floor_us={:.1} candidates={}",
            c.cell.op,
            c.cell.weight,
            c.cell.activation,
            c.cell.n,
            c.cell.k,
            c.cell.rows,
            c.margin,
            c.default.as_deref().unwrap_or("-"),
            c.floor_us,
            cand.join(",")
        );
        est += c.floor_us * c.candidates.len() as f64;
    }
    for (op, w, a, k, n) in uncovered(&i.sweep, &i.contracts, &i.fams) {
        println!("uncovered: {op} {w} {a} {n}x{k}: no contracted kernel runs these formats");
    }
    println!(
        "# shard {}/{}: {} cells, {} candidate runs, {:.1} s of kernel floor per launch set",
        g.shard,
        g.shards,
        cells.len(),
        cells.iter().map(|c| c.candidates.len()).sum::<usize>(),
        est / 1e6
    );
    Ok(0)
}

/// 2026-10-10: The NVML device of GPU 0, read around every timing.
struct Gpu0 {
    nvml: Option<metrale_gpu_sys::nvml::Nvml>,
}

impl Gpu0 {
    fn open() -> Self {
        let nvml = metrale_gpu_sys::nvml::Nvml::open().ok();
        if nvml.is_none() {
            eprintln!("warning: NVML unavailable: no temperature gate or slowdown flags");
        }
        Gpu0 { nvml }
    }

    fn reasons(&self) -> u64 {
        self.nvml
            .as_ref()
            .and_then(|n| n.device(0).ok())
            .and_then(|d| d.clocks_event_reasons().ok().flatten())
            .unwrap_or(0)
    }

    fn temp_c(&self) -> f64 {
        self.nvml
            .as_ref()
            .and_then(|n| n.device(0).ok())
            .and_then(|d| d.temperature_c().ok().flatten())
            .map_or(f64::NAN, f64::from)
    }

    /// 2026-10-10: Wait while hotter than `max`, until at most `resume`.
    fn cool(&self, max: u32, resume: u32) {
        let t = self.temp_c();
        if !(t > f64::from(max)) {
            return;
        }
        eprintln!("cool-down: {t:.0} C > {max} C, waiting for {resume} C");
        while self.temp_c() > f64::from(resume) {
            std::thread::sleep(std::time::Duration::from_secs(5));
        }
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 2026-10-10: `secs` since the epoch as RFC 3339 UTC (civil-from-days).
fn rfc3339(secs: u64) -> String {
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn host() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

fn shape_of(cell: &Cell) -> Shape {
    Shape {
        op: cell.op.clone(),
        weight: Some(cell.weight.clone()),
        activation: Some(cell.activation.clone()),
        output: Some("bf16".into()),
        in_dim: cell.k,
        out_dim: cell.n,
        rows: cell.rows,
        runtime: Default::default(),
    }
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// 2026-10-10: The contract with no mutation arms: the sweep judges the candidate's own output.
fn without_mutations(c: &Contract) -> Contract {
    let mut c = c.clone();
    c.mutations.clear();
    c
}

fn done_keys(out: &std::path::Path) -> Result<BTreeSet<(Cell, String)>> {
    if !out.exists() {
        return Ok(BTreeSet::new());
    }
    let text = std::fs::read_to_string(out).with_context(|| out.display().to_string())?;
    let recs = parse_records(&text).map_err(anyhow::Error::msg)?;
    Ok(recs
        .into_iter()
        .filter(|r| !r.throttled)
        .map(|r| (r.cell, r.kernel))
        .collect())
}

fn sweep_gpu(a: &EnvelopeSweepArgs) -> Result<i32> {
    let i = inputs(&a.grid)?;
    let cells = cells_of(&i, &a.grid)?;
    let done = done_keys(&a.out)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&a.out)
        .with_context(|| a.out.display().to_string())?;
    let mut runner = GpuRunner::for_sweep(&a.grid.hardware)?;
    runner.select_any()?;
    let gpu = Gpu0::open();
    let host = host();
    let started = std::time::Instant::now();
    let (mut written, mut stopped) = (0usize, false);
    'cells: for (ci, gc) in cells.iter().enumerate() {
        gpu.cool(a.max_temp_c, a.resume_temp_c);
        let todo: Vec<_> = gc
            .candidates
            .iter()
            .filter(|c| !done.contains(&(gc.cell.clone(), c.kernel.clone())))
            .collect();
        if todo.is_empty() {
            continue;
        }
        let shared = shared_cases(&i, gc)?;
        for cand in todo {
            if now_unix() >= a.deadline_unix {
                stopped = true;
                break 'cells;
            }
            let m = measure(&i, gc, cand, &shared, &mut runner, &gpu, a, &host);
            let line = serde_json::to_string(&m)?;
            writeln!(file, "{line}")?;
            file.flush()?;
            written += 1;
            println!(
                "[{}/{}] {} {} {}x{} rows={} {} {:?} {}",
                ci + 1,
                cells.len(),
                gc.cell.op,
                gc.cell.weight,
                gc.cell.n,
                gc.cell.k,
                gc.cell.rows,
                m.kernel,
                m.verdict,
                m.median_us()
                    .map_or_else(|| m.detail.clone(), |t| format!("{t:.2} us"))
            );
        }
    }
    println!(
        "# {} records in {:.0} s{}",
        written,
        started.elapsed().as_secs_f64(),
        if stopped {
            " (stopped at the deadline)"
        } else {
            ""
        }
    );
    Ok(0)
}

/// 2026-10-10: One case per digest class, built from the default's job when the cell has a
/// default among its candidates, else from the first candidate's: every candidate then runs the
/// same operands (the canonical layouts of the weight format).
fn shared_cases(i: &Inputs, gc: &GridCell) -> Result<Vec<(InputClass, Case)>> {
    let lead = gc
        .default
        .as_ref()
        .and_then(|d| gc.candidates.iter().find(|c| &c.kernel == d))
        .unwrap_or(&gc.candidates[0]);
    let contract = &i.contracts.contracts[lead.contract];
    let family = i
        .fams
        .families
        .iter()
        .find(|f| f.id == lead.family)
        .with_context(|| format!("family {}", lead.family))?;
    let point = point_for(&i.sweep, &i.fams, &lead.family, &gc.cell)
        .with_context(|| format!("no point of {} at {:?}", lead.family, gc.cell))?;
    let shape = shape_of(&gc.cell);
    let mut out = Vec::new();
    for input in DIGEST_CLASSES {
        let job = Job {
            contract,
            family,
            kernel: &lead.kernel,
            point: &point,
            shape: &shape,
            input,
            seed: i.contracts.seed,
        };
        out.push((input, case_of(&job).map_err(anyhow::Error::msg)?));
    }
    Ok(out)
}

/// 2026-10-10: The rows at which a candidate is judged on every input class of its contract:
/// the fewest and the most it runs in the ladder (between them, the gaussian class).
fn full_check_row(kernel: &str, rows: u64) -> bool {
    let cap = sweep_max_rows(kernel).map_or(u64::MAX, |c| c as u64);
    let floor = sweep_min_rows(kernel) as u64;
    let runs: Vec<u64> = LADDER
        .iter()
        .copied()
        .filter(|&r| r >= floor && r <= cap)
        .collect();
    runs.first() == Some(&rows) || runs.last() == Some(&rows)
}

#[allow(clippy::too_many_arguments)]
fn measure(
    i: &Inputs,
    gc: &GridCell,
    cand: &metrale_accuracy::envelope::grid::Candidate,
    shared: &[(InputClass, Case)],
    runner: &mut GpuRunner,
    gpu: &Gpu0,
    a: &EnvelopeSweepArgs,
    host: &str,
) -> Measurement {
    let mut m = Measurement {
        schema: SCHEMA,
        hardware: a.grid.hardware.clone(),
        host: host.to_string(),
        cell: gc.cell.clone(),
        kernel: cand.kernel.clone(),
        family: cand.family.clone(),
        default: gc.default.clone(),
        verdict: Verdict::Pass,
        detail: String::new(),
        digests: Default::default(),
        time_us: Vec::new(),
        floor_us: gc.floor_us,
        throttled: false,
        temp_c: f64::NAN,
        closure: runner.closure(),
        at: rfc3339(now_unix()),
    };
    let fail = |m: &mut Measurement, v: Verdict, why: String| {
        m.verdict = v;
        m.detail = why;
    };
    // 2026-10-10: The digests on the shared operands.
    for (input, case) in shared {
        let mut c = case.clone();
        c.kernel = cand.kernel.clone();
        c.launcher = cand.kernel.clone();
        match runner.run(&c) {
            Ok(bytes) => {
                m.digests
                    .insert(input.name().to_string(), hex(&Sha256::digest(&bytes)));
            }
            Err(RunError::Unavailable(w)) => {
                fail(&mut m, Verdict::Unavailable, w);
                return m;
            }
            Err(RunError::Fault(w)) => {
                fail(&mut m, Verdict::Fail, format!("fault: {w}"));
                return m;
            }
        }
    }
    // 2026-10-10: The candidate's own contract on its own seeded inputs.
    let contract = without_mutations(&i.contracts.contracts[cand.contract]);
    let Some(family) = i.fams.families.iter().find(|f| f.id == cand.family) else {
        fail(
            &mut m,
            Verdict::Unavailable,
            "family not in the manifest".into(),
        );
        return m;
    };
    let Some(point) = point_for(&i.sweep, &i.fams, &cand.family, &gc.cell) else {
        fail(
            &mut m,
            Verdict::Unavailable,
            "no family point at these formats".into(),
        );
        return m;
    };
    let shape = shape_of(&gc.cell);
    let classes: Vec<InputClass> = if full_check_row(&cand.kernel, gc.cell.rows) {
        contract.inputs.clone()
    } else {
        vec![InputClass::Gaussian]
    };
    for input in classes {
        let job = Job {
            contract: &contract,
            family,
            kernel: &cand.kernel,
            point: &point,
            shape: &shape,
            input,
            seed: i.contracts.seed,
        };
        let o = run(&job, runner);
        if o.verdict != CheckVerdict::Pass {
            fail(
                &mut m,
                Verdict::Fail,
                format!("{} on {}", o.verdict.name(), input.name()),
            );
            return m;
        }
    }
    // 2026-10-10: Timing on the shared gaussian operands.
    let mut c = shared[0].1.clone();
    c.kernel = cand.kernel.clone();
    c.launcher = cand.kernel.clone();
    let before = gpu.reasons();
    match runner.time(&c, a.warmup, a.iters, a.reps) {
        Ok(t) => m.time_us = t,
        Err(e) => {
            fail(&mut m, Verdict::Fail, format!("timing: {e}"));
            return m;
        }
    }
    let after = gpu.reasons();
    m.throttled = (before | after) & SLOWDOWN != 0;
    m.temp_c = gpu.temp_c();
    m.detail = format!("clock_event_reasons=0x{:x}", before | after);
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_is_civil_utc() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_791_662_400), "2026-10-10T20:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
    }

    #[test]
    fn the_full_check_rows_are_the_ends_of_what_a_candidate_runs() {
        assert!(full_check_row("w4a16_gemv_tc::w4a16_gemv_tc8", 1));
        assert!(full_check_row("w4a16_gemv_tc::w4a16_gemv_tc8", 8));
        assert!(!full_check_row("w4a16_gemv_tc::w4a16_gemv_tc8", 4));
        assert!(full_check_row("gemv::dense_gemv_bf16", 128));
        assert!(!full_check_row("gemv::dense_gemv_bf16", 16));
        assert!(full_check_row(
            "dense_gemv_bf16_tc::dense_gemv_bf16_tc32",
            2
        ));
        assert!(!full_check_row(
            "dense_gemv_bf16_tc::dense_gemv_bf16_tc32",
            1
        ));
    }
}
