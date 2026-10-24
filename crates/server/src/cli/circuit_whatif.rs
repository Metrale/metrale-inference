// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `met circuit whatif`: the I/O side of `metrale_circuit::hardware::critical`. It
//! takes what `met circuit headroom` takes (the plan, its trace TOML) plus an event capture
//! (`--events`: an nsys SQLite file it exports through `scripts/critical_trace.py`, or that
//! export's JSON), and prints the critical path, per-site slack and exact upside beside the
//! headroom's Amdahl upside, the rewrites of a spec and their backtest, and a KV breakpoint
//! against another analysis.
//!
//! Owner: server CLI.
//! Invariants:
//! - Nothing here models, infers or sums; the crate does. This file reads files and runs the
//!   exporter.
//! - Every number that changes a prediction is on the command line, in the spec or in the
//!   capture (`--max-ctas-per-sm` is required: DEVICES.toml does not state it).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use metrale_circuit::hardware::critical::analyze::{Inputs, WhatIf, analyze};
use metrale_circuit::hardware::critical::backtest::{Breakpoint, breakpoint, parse_spec, verdicts};
use metrale_circuit::hardware::critical::capture::{Capture, parse_capture};
use metrale_circuit::hardware::critical::render::render_whatif;
use metrale_circuit::hardware::critical::sim::{Resources, SmLimits};
use metrale_circuit::hardware::{self};

use super::circuit_headroom::{CircuitHeadroomArgs, HeadroomFormat, with_inputs};
use super::circuit_venn::find_root;

/// `met circuit whatif` options.
#[derive(clap::Args, Debug, Clone)]
pub struct CircuitWhatIfArgs {
    /// The plan and its headroom trace, as `met circuit headroom` takes them (`--format` md or
    /// json).
    #[command(flatten)]
    pub base: CircuitHeadroomArgs,
    /// The event capture: an nsys SQLite file (exported with `--step-kernel` and
    /// `--attribution`) or its `critical-trace/v1` JSON.
    #[arg(long)]
    pub events: PathBuf,
    /// CTAs one SM holds at most (the SM residency limit; not in DEVICES.toml).
    #[arg(long)]
    pub max_ctas_per_sm: u32,
    /// A graph node starting more than this after its stream predecessor finished gets an
    /// inferred fork/join, microseconds. 1.0: twice the median in-graph node gap on GB10.
    #[arg(long, default_value_t = 1.0)]
    pub join_tau_us: f64,
    /// An unexplained graph node launched within this of a running node's start on another
    /// stream, and finishing after it, is that node's programmatic dependent, microseconds.
    /// 3.0: the combine-on-sweep launch offset on GB10 is ~2.1 µs.
    #[arg(long, default_value_t = 3.0)]
    pub programmatic_us: f64,
    /// A `whatif-spec/v1` TOML of rewrite sets (and their serve measurements).
    #[arg(long)]
    pub spec: Option<PathBuf>,
    /// Which of the spec's measurements apply to this capture.
    #[arg(long, requires = "spec")]
    pub tag: Option<String>,
    /// A `whatif/v1` JSON of the same plan at another KV length: report `--breakpoint-site`'s
    /// KV breakpoint between the two.
    #[arg(long, requires = "breakpoint_site")]
    pub breakpoint_with: Option<PathBuf>,
    /// `run · site` as the report names it (e.g. `verify n=2 · dsa [select]`).
    #[arg(long)]
    pub breakpoint_site: Option<String>,
    /// Sites listed in the comparison tables.
    #[arg(long, default_value_t = 15)]
    pub top: usize,
}

fn read_events(root: &Path, a: &CircuitWhatIfArgs) -> Result<Capture> {
    let path = root.join(&a.events);
    let text = if path.extension().is_some_and(|e| e == "sqlite") {
        let Some(step) = &a.base.step_kernel else {
            bail!("an nsys SQLite --events needs --step-kernel");
        };
        let mut cmd = std::process::Command::new("python3");
        cmd.arg(root.join("scripts/critical_trace.py"))
            .arg(&path)
            .args(["--step-kernel", step])
            .args(["--source", &format!("nsys {}", a.events.display())]);
        if let Some(m) = &a.base.attribution {
            cmd.arg("--attribution").arg(root.join(m));
        }
        let out = cmd.output().context("running scripts/critical_trace.py")?;
        if !out.status.success() {
            bail!(
                "scripts/critical_trace.py failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        String::from_utf8(out.stdout).context("exporter output is not UTF-8")?
    } else {
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?
    };
    Ok(parse_capture(&text)?)
}

/// 2026-10-23: The analysis `a` asks for.
pub(crate) fn compute(
    root: &Path,
    a: &CircuitWhatIfArgs,
) -> Result<(
    WhatIf,
    Option<Breakpoint>,
    Vec<hardware::critical::backtest::Verdict>,
)> {
    let capture = read_events(root, a)?;
    let spec = match &a.spec {
        Some(p) => Some(parse_spec(
            &std::fs::read_to_string(root.join(p))
                .with_context(|| format!("reading {}", p.display()))?,
        )?),
        None => None,
    };
    let sets: Vec<(String, Vec<_>)> = spec
        .iter()
        .flat_map(|s| s.set.iter().map(|x| (x.id.clone(), x.rewrite.clone())))
        .collect();
    let w = with_inputs(root, &a.base, |reg, tree, model, runs, inputs| {
        let dev = reg
            .devices
            .iter()
            .find(|d| d.id == a.base.hardware)
            .with_context(|| format!("no device `{}`", a.base.hardware))?;
        let resources = Resources {
            sms: f64::from(dev.sms),
            sm: SmLimits {
                regs: dev.regfile_per_sm_kb * 1024 / 4,
                threads: dev.max_threads_per_sm,
                smem: dev.smem_per_sm_kb * 1024,
                ctas: a.max_ctas_per_sm,
            },
            dram_bytes_per_us: 0.0,
            dram_peak_bytes_per_us: dev.bandwidth_gbps * 1e3,
        };
        let (h, group_site) = hardware::build_whatif_sites(
            reg,
            &a.base.hardware,
            tree,
            model,
            runs,
            inputs,
            &capture.groups,
        )?;
        let inp = Inputs {
            group_site: &group_site,
            headroom: &h,
            resources,
            tau_ns: (a.join_tau_us * 1e3).round() as i64,
            prog_ns: (a.programmatic_us * 1e3).round() as i64,
            rewrites: &sets,
        };
        Ok(analyze(&capture, &inp)?)
    })?;
    let bp = match (&a.breakpoint_with, &a.breakpoint_site) {
        (Some(p), Some(site)) => {
            let other: WhatIf = serde_json::from_str(
                &std::fs::read_to_string(root.join(p))
                    .with_context(|| format!("reading {}", p.display()))?,
            )?;
            Some(breakpoint(&w, &other, site)?)
        }
        _ => None,
    };
    let v = match (&spec, &a.tag) {
        (Some(s), Some(t)) => verdicts(s, &w, t),
        _ => Vec::new(),
    };
    Ok((w, bp, v))
}

/// 2026-10-23: The command line that regenerates `a`'s output.
fn command(a: &CircuitWhatIfArgs) -> String {
    let mut c = super::circuit_headroom::command(&a.base).replacen(
        "met circuit headroom",
        "met circuit whatif",
        1,
    );
    c.push_str(&format!(
        " --events {} --max-ctas-per-sm {} --join-tau-us {} --programmatic-us {}",
        a.events.display(),
        a.max_ctas_per_sm,
        a.join_tau_us,
        a.programmatic_us
    ));
    if let Some(s) = &a.spec {
        c.push_str(&format!(" --spec {}", s.display()));
    }
    if let Some(t) = &a.tag {
        c.push_str(&format!(" --tag {t}"));
    }
    if let (Some(p), Some(s)) = (&a.breakpoint_with, &a.breakpoint_site) {
        c.push_str(&format!(
            " --breakpoint-with {} --breakpoint-site '{s}'",
            p.display()
        ));
    }
    c
}

/// 2026-10-23: Run `met circuit whatif`.
pub(crate) fn run(a: CircuitWhatIfArgs) -> Result<()> {
    let root = match &a.base.root {
        Some(r) => r.clone(),
        None => find_root(&std::env::current_dir()?)?,
    };
    let (w, bp, v) = compute(&root, &a)?;
    let text = match a.base.format {
        HeadroomFormat::Md => render_whatif(&w, &v, bp.as_ref(), &command(&a), a.top),
        HeadroomFormat::Json => {
            let mut s = serde_json::to_string_pretty(&w)?;
            s.push('\n');
            s
        }
        f => bail!("met circuit whatif prints md or json, not {f:?}"),
    };
    match &a.base.out {
        Some(rel) => {
            super::circuit_hw::write_or_check(&root, rel, &text, a.base.check)?;
            eprintln!(
                "{rel}: {}",
                if a.base.check { "current" } else { "written" }
            );
            Ok(())
        }
        None => {
            use std::io::Write;
            match std::io::stdout().lock().write_all(text.as_bytes()) {
                Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
                other => Ok(other?),
            }
        }
    }
}
