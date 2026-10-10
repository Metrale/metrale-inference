// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `met accuracy envelope schedules|fusions`: the I/O side of the selection. Reads the
//! sweep's record files and the repository through [`FsRepo`], writes SCHEDULES.toml (the kernel
//! build bakes it) and a Markdown report listing every winner as byte-identical to today's
//! default (enabled by default) or numerics-changing (opt-in only), every cell without a winner
//! and every record that must be rerun.
//!
//! Owner: server CLI.
//! Invariants: nothing here selects or classifies; `metrale_accuracy::envelope` does.

use std::fmt::Write as _;

use anyhow::{Context, Result};
use metrale_accuracy::envelope::record::parse_records;
use metrale_accuracy::envelope::schedules::{Enabled, Numerics, parse, render};
use metrale_accuracy::envelope::select::{Selection, families, schedules as build, select};
use metrale_accuracy::envelope::{fusions, sources};
use metrale_circuit::venn::Repo;

use crate::cli::accuracy_args::{EnvelopeFusionsArgs, EnvelopeSchedulesArgs};
use crate::cli::circuit_venn::{FsRepo, find_root};

fn repo(root: &Option<std::path::PathBuf>) -> Result<FsRepo> {
    let root = match root {
        Some(r) => r.clone(),
        None => find_root(&std::env::current_dir()?)?,
    };
    Ok(FsRepo { root })
}

pub(super) fn schedules(a: &EnvelopeSchedulesArgs) -> Result<i32> {
    let repo = repo(&a.root)?;
    let mut records = Vec::new();
    for f in &a.records {
        let text = std::fs::read_to_string(f).with_context(|| f.display().to_string())?;
        records.extend(parse_records(&text).map_err(anyhow::Error::msg)?);
    }
    let sel = select(&records).map_err(|e| anyhow::anyhow!("{e}"))?;
    let fams = families(&sel);
    let ids: Vec<&str> = fams.iter().map(String::as_str).collect();
    let srcs = sources::sources_of(&repo, &a.hardware, &ids).map_err(|e| anyhow::anyhow!("{e}"))?;
    let generated_by = format!(
        "met accuracy envelope schedules --hardware {} --records {}",
        a.hardware,
        a.records
            .iter()
            .map(|p| p.file_name().map_or_else(String::new, |n| n.to_string_lossy().into()))
            .collect::<Vec<_>>()
            .join(",")
    );
    let file = build(&sel, &generated_by, srcs)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    std::fs::write(&a.out, render(&file)).with_context(|| a.out.display().to_string())?;
    std::fs::write(&a.report, report(&sel, records.len()))
        .with_context(|| a.report.display().to_string())?;
    println!(
        "# {} records -> {} decisions ({} schedules), {} undecided, {} reruns; wrote {} and {}",
        records.len(),
        sel.decisions.len(),
        file.schedule.len(),
        sel.undecided.len(),
        sel.reruns.len(),
        a.out.display(),
        a.report.display()
    );
    Ok(0)
}

/// 2026-10-10: The selection as Markdown: counts, then the winners by class.
pub(super) fn report(sel: &Selection, records: usize) -> String {
    let mut s = String::new();
    let count = |n: Numerics| sel.decisions.iter().filter(|d| d.numerics == n).count();
    let _ = writeln!(
        s,
        "# Envelope sweep selection ({})\n\n{records} records, {} decisions: {} default kept \
         (same), {} byte-identical winners (enabled by default), {} numerics-changing winners \
         (opt-in only), {} new (no default; opt-in only). {} cells undecided, {} records to \
         rerun.\n",
        sel.hardware,
        sel.decisions.len(),
        count(Numerics::Same),
        count(Numerics::BitIdentical),
        count(Numerics::Differs),
        count(Numerics::New),
        sel.undecided.len(),
        sel.reruns.len()
    );
    for (title, n) in [
        ("Byte-identical winners (enabled by default)", Numerics::BitIdentical),
        ("Numerics-changing winners (opt-in only)", Numerics::Differs),
        ("New cells (no default; opt-in only)", Numerics::New),
    ] {
        let _ = writeln!(
            s,
            "## {title}\n\n| Cell | Winner | Default | Winner us | Default us | Speed-up | Floor us | Enabled |\n|---|---|---|---:|---:|---:|---:|---|"
        );
        for d in sel.decisions.iter().filter(|d| d.numerics == n) {
            let c = &d.cell;
            let up = d.default_us.map_or("-".into(), |u| format!("{:.2}x", u / d.median_us));
            let _ = writeln!(
                s,
                "| {} {} {} {}x{} rows={} | `{}` | `{}` | {:.2} | {} | {up} | {:.2} | {} |",
                c.op,
                c.weight,
                c.activation,
                c.n,
                c.k,
                c.rows,
                d.kernel,
                d.default.as_deref().unwrap_or("-"),
                d.median_us,
                d.default_us.map_or("-".into(), |u| format!("{u:.2}")),
                d.floor_us,
                match d.enabled {
                    Enabled::Default => "default",
                    Enabled::OptIn => "opt-in",
                }
            );
        }
        s.push('\n');
    }
    let _ = writeln!(s, "## Undecided cells\n");
    for u in &sel.undecided {
        let c = &u.cell;
        let _ = writeln!(
            s,
            "- {} {} {} {}x{} rows={}: {:?}",
            c.op, c.weight, c.activation, c.n, c.k, c.rows, u.why
        );
    }
    let _ = writeln!(s, "\n## Reruns (every record throttled)\n");
    for r in &sel.reruns {
        let c = &r.cell;
        let _ = writeln!(
            s,
            "- {} {} {} {}x{} rows={} `{}` on {}",
            c.op,
            c.weight,
            c.activation,
            c.n,
            c.k,
            c.rows,
            r.kernel,
            r.hosts.join(", ")
        );
    }
    s
}

pub(super) fn fusions(a: &EnvelopeFusionsArgs) -> Result<i32> {
    let repo = repo(&a.root)?;
    let text = std::fs::read_to_string(&a.schedules)
        .with_context(|| a.schedules.display().to_string())?;
    let file = parse(&text).map_err(|e| anyhow::anyhow!("{e}"))?;
    let rules = repo
        .read(&format!("kernels/{}/common/FUSIONS.toml", a.hardware))
        .map_err(anyhow::Error::msg)?;
    let r = fusions::check_fusions(&rules, &file).map_err(|e| anyhow::anyhow!("{}", e.0))?;
    print!("{}", r.render());
    Ok(0)
}
