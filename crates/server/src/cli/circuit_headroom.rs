// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `met circuit headroom --recipe <id> --hardware <device>`: the I/O side of
//! `metrale_circuit::hardware::headroom`. It reads the instance and the kernel tree from the
//! working tree, an optional capture (`--trace`: a `headroom/v1` trace TOML, or an nsys SQLite file
//! it exports through `scripts/headroom_trace.py`), and writes, prints or checks the report as
//! Markdown, `headroom/v1` JSON, or one of its two SVG figures.
//!
//! Owner: server CLI.
//! Invariants:
//! - Nothing here costs, attributes or sums; the crate does.
//! - Every input that changes a number is stated on the command line or in the capture, never
//!   guessed: the capture's KV length and shard dims win only when the flags do not contradict
//!   them, and a contradiction is an error.
//! - `--check` compares and never writes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use metrale_circuit::hardware::headroom::{Headroom, Inputs};
use metrale_circuit::hardware::headroom_trace::{Stated, Trace, parse_trace, pool};
use metrale_circuit::hardware::{self, PrecisionChoice};
use metrale_circuit::venn::{Repo, Run};
use metrale_circuit::{Mode, parse_instances};

use super::CircuitMode;
use super::circuit_hw::{FsTree, mode_of, registry};
use super::circuit_venn::find_root;

/// `met circuit headroom` options.
#[derive(clap::Args, Debug, Clone)]
pub struct CircuitHeadroomArgs {
    /// The recipe id (kernels/circuits/INSTANCES.toml) whose formats and policy are planned.
    #[arg(long)]
    pub recipe: String,
    /// Target device id from kernels/DEVICES.toml.
    #[arg(long)]
    pub hardware: String,
    /// The step's main forward.
    #[arg(long, value_enum)]
    pub mode: CircuitMode,
    /// Padded rows of the main forward (required for multi_seq and verify).
    #[arg(long)]
    pub rows: Option<u64>,
    /// Another forward the same step runs, `<mode>:<rows>` (e.g. `draft:1` for the MTP head).
    #[arg(long = "also", value_parser = parse_run)]
    pub also: Vec<Run>,
    /// A capture: a trace TOML (`scripts/headroom_trace.py`) or an nsys SQLite export. Repeat it
    /// to pool captures of one configuration (their steps are bootstrapped together, blocks never
    /// crossing captures).
    #[arg(long)]
    pub trace: Vec<PathBuf>,
    /// With an nsys SQLite `--trace`: the kernel launched once per step.
    #[arg(long)]
    pub step_kernel: Option<String>,
    /// With an nsys SQLite `--trace`: the launch-shape to site map (`scripts/headroom/*.toml`).
    #[arg(long)]
    pub attribution: Option<PathBuf>,
    /// KV tokens per sequence; without it the capture's, else the class's assumption.
    #[arg(long)]
    pub kv_tokens: Option<u64>,
    /// Tokens a step emits; without it the capture's, else the plan's rows (every draft accepted).
    #[arg(long)]
    pub tokens_per_step: Option<f64>,
    /// The time one empty launch takes on the device, microseconds; without it the capture's.
    #[arg(long)]
    pub launch_floor_us: Option<f64>,
    /// The fraction of its floor the second upside column assumes a site reaches, percent.
    #[arg(long, default_value_t = 80.0)]
    pub target_pct: f64,
    /// What to print.
    #[arg(long, value_enum, default_value_t = HeadroomFormat::Md)]
    pub format: HeadroomFormat,
    /// Write the output here (relative to the repository root) instead of stdout.
    #[arg(long)]
    pub out: Option<String>,
    /// Compare with the file on disk instead of writing; a stale or missing one is an error.
    #[arg(long, requires = "out")]
    pub check: bool,
    /// Repository root; by default the nearest directory above the working directory that has
    /// kernels/circuits/INSTANCES.toml.
    #[arg(long)]
    pub root: Option<PathBuf>,
}

/// `met circuit headroom --format`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadroomFormat {
    /// The Markdown report.
    #[value(name = "md")]
    Md,
    /// `headroom/v1` JSON (schemas/headroom.v1.json).
    #[value(name = "json")]
    Json,
    /// The roofline figure.
    #[value(name = "svg-roofline")]
    SvgRoofline,
    /// The step waterfall figure.
    #[value(name = "svg-waterfall")]
    SvgWaterfall,
}

fn parse_run(s: &str) -> Result<Run, String> {
    let (m, r) = s
        .split_once(':')
        .ok_or_else(|| format!("`{s}` is not <mode>:<rows>"))?;
    let mode = Mode::ALL
        .into_iter()
        .find(|x| x.name() == m)
        .ok_or_else(|| format!("unknown mode `{m}`"))?;
    let rows: u64 = r
        .parse()
        .map_err(|_| format!("rows `{r}` is not a count"))?;
    if rows == 0 {
        return Err("rows must be at least 1".into());
    }
    Ok(Run { mode, rows })
}

/// 2026-10-22: Read the capture: a trace TOML as is, an nsys SQLite file through the exporter.
fn read_trace(root: &Path, a: &CircuitHeadroomArgs, path: &Path) -> Result<Trace> {
    let text = if path.extension().is_some_and(|e| e == "sqlite") {
        let Some(step) = &a.step_kernel else {
            bail!("an nsys SQLite --trace needs --step-kernel (the kernel launched once per step)");
        };
        let mut cmd = std::process::Command::new("python3");
        cmd.arg(root.join("scripts/headroom_trace.py"))
            .arg(path)
            .args(["--step-kernel", step])
            .args(["--source", &format!("nsys {}", path.display())]);
        if let Some(m) = &a.attribution {
            cmd.arg("--attribution").arg(m);
        }
        let out = cmd.output().context("running scripts/headroom_trace.py")?;
        if !out.status.success() {
            bail!(
                "scripts/headroom_trace.py failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        String::from_utf8(out.stdout).context("exporter output is not UTF-8")?
    } else {
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
    };
    parse_trace(&text).map_err(anyhow::Error::msg)
}

/// 2026-10-22: A flag and the capture's statement of the same quantity: the flag, the capture's,
/// or an error when both are given and differ.
fn agree(flag: Option<f64>, captured: Option<&Stated>, what: &str) -> Result<Option<Stated>> {
    match (flag, captured) {
        (Some(f), Some(c)) if (f - c.value).abs() > 1e-9 * f.abs().max(1.0) => bail!(
            "--{what} {f} contradicts the capture's {} ({})",
            c.value,
            c.basis
        ),
        (Some(f), _) => Ok(Some(Stated {
            value: f,
            basis: "stated on the command line".into(),
        })),
        (None, c) => Ok(c.cloned()),
    }
}

/// 2026-10-22: The headroom `a` asks for, and the command that regenerates it.
pub(crate) fn compute(root: &Path, a: &CircuitHeadroomArgs) -> Result<Headroom> {
    with_inputs(root, a, |reg, tree, model, runs, inputs| {
        Ok(hardware::build_headroom(reg, &a.hardware, tree, model, runs, inputs)?)
    })
}

/// 2026-10-23: Everything `a` states, resolved (the instance with the capture's shard dims, the
/// runs, the inputs whose flags agree with the capture), handed to `f`: the headroom and the
/// what-if plan from the same statement.
pub(crate) fn with_inputs<T>(
    root: &Path,
    a: &CircuitHeadroomArgs,
    f: impl FnOnce(
        &hardware::Registry,
        &FsTree,
        &hardware::ModelUnderPlan,
        &[Run],
        &Inputs<'_>,
    ) -> Result<T>,
) -> Result<T> {
    let tree = FsTree::new(root.to_path_buf());
    let reg = registry(&tree)?;
    let text = tree
        .read("kernels/circuits/INSTANCES.toml")
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut inst = parse_instances(&text)?
        .into_iter()
        .find(|i| i.recipe == a.recipe)
        .with_context(|| format!("no circuit instance for recipe `{}`", a.recipe))?;
    let captures = a
        .trace
        .iter()
        .map(|p| read_trace(root, a, &root.join(p)))
        .collect::<Result<Vec<_>>>()?;
    let trace = if captures.is_empty() {
        None
    } else {
        Some(pool(&captures).map_err(anyhow::Error::msg)?)
    };
    let tp = trace.as_ref().map(|t| t.plan.clone()).unwrap_or_default();
    for (k, v) in &tp.dims {
        let Some(slot) = inst.shape.dims.get_mut(k) else {
            bail!(
                "the capture states dim `{k}`, which recipe {} does not have",
                a.recipe
            );
        };
        *slot = *v;
    }
    let model = hardware::model::model_of(
        &tree,
        &inst,
        format!("recipe {}", inst.recipe),
        PrecisionChoice::Recipe,
    )?;
    let mode = mode_of(a.mode);
    let rows = match (a.rows, mode) {
        (Some(0), _) => bail!("--rows must be at least 1"),
        (Some(r), _) => r,
        (None, Mode::Decode | Mode::Draft) => 1,
        (None, _) => bail!("--mode {} needs --rows", mode.name()),
    };
    let mut runs = vec![Run { mode, rows }];
    runs.extend(a.also.iter().copied());
    let kv_captured = tp.kv_tokens.map(|v| Stated {
        value: v as f64,
        basis: tp
            .kv_basis
            .clone()
            .unwrap_or_else(|| "the capture's".into()),
    });
    let kv = agree(
        a.kv_tokens.map(|v| v as f64),
        kv_captured.as_ref(),
        "kv-tokens",
    )?;
    let tokens = agree(
        a.tokens_per_step,
        trace.as_ref().and_then(|t| t.tokens_per_step.as_ref()),
        "tokens-per-step",
    )?
    .unwrap_or_else(|| Stated {
        value: rows as f64,
        basis: format!(
            "the main forward's {rows} row(s){}",
            if mode == Mode::Verify {
                ": every draft accepted"
            } else {
                ""
            }
        ),
    });
    let launch = agree(
        a.launch_floor_us,
        trace.as_ref().and_then(|t| t.launch_floor_us.as_ref()),
        "launch-floor-us",
    )?;
    let inputs = Inputs {
        model: format!("recipe {}", a.recipe),
        kv_tokens: kv,
        tokens_per_step: tokens,
        launch_floor_us: launch,
        dims: (
            tp.dims
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect::<BTreeMap<_, _>>(),
            tp.dims_basis.clone(),
        ),
        target_pct: a.target_pct,
        trace: trace.as_ref(),
        energy: None,
    };
    f(&reg, &tree, &model, &runs, &inputs)
}

/// 2026-10-22: The command line that regenerates `a`'s output.
pub(crate) fn command(a: &CircuitHeadroomArgs) -> String {
    let mut c = format!(
        "met circuit headroom --recipe {} --hardware {} --mode {}",
        a.recipe,
        a.hardware,
        mode_of(a.mode).name()
    );
    if let Some(r) = a.rows {
        c.push_str(&format!(" --rows {r}"));
    }
    for r in &a.also {
        c.push_str(&format!(" --also {}:{}", r.mode.name(), r.rows));
    }
    for t in &a.trace {
        c.push_str(&format!(" --trace {}", t.display()));
    }
    c.push_str(&format!(" --target-pct {}", a.target_pct));
    c
}

/// 2026-10-22: The output text of `h` in `format`.
pub(crate) fn render(h: &Headroom, format: HeadroomFormat, command: &str) -> Result<String> {
    Ok(match format {
        HeadroomFormat::Md => hardware::render_headroom(h, command),
        HeadroomFormat::Json => {
            let mut s = serde_json::to_string_pretty(h)?;
            s.push('\n');
            s
        }
        HeadroomFormat::SvgRoofline => hardware::roofline_svg(h),
        HeadroomFormat::SvgWaterfall => hardware::waterfall_svg(h),
    })
}

/// 2026-10-22: Run `met circuit headroom`.
pub(crate) fn run(a: CircuitHeadroomArgs) -> Result<()> {
    let root = match &a.root {
        Some(r) => r.clone(),
        None => find_root(&std::env::current_dir()?)?,
    };
    let h = compute(&root, &a)?;
    let text = render(&h, a.format, &command(&a))?;
    match &a.out {
        Some(rel) => {
            super::circuit_hw::write_or_check(&root, rel, &text, a.check)?;
            eprintln!("{rel}: {}", if a.check { "current" } else { "written" });
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

#[cfg(test)]
#[path = "circuit_headroom_tests.rs"]
mod circuit_headroom_tests;
