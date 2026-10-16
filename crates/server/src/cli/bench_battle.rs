// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `met benchmark battle <id>`: a named battle's cells, each the newest committed
//! record against the reference its BENCH.toml entry declares, beside the cell's own
//! regression bound.
//!
//! Owner: server CLI (`met benchmark`).
//! GPU-free: it reads `bench/battles.toml`, the BENCH.toml files and `.benchmarks/`.
//! Invariants: exit code 0 only when there is a cell and every cell shown is a win.

use std::fmt::Write as _;

use anyhow::{Result, bail};
use metrale_bench::gate::battle::{self, Cell, Judged, Ratchet, Report, Verdict};
use metrale_bench::gate::battle_load;

use super::bench_args::{BattleArgs, OutputFormat};

pub fn battle_cmd(args: BattleArgs) -> Result<i32> {
    let root = super::bench_run::repo_root()?;
    let loaded = battle_load::load(&root)?;
    let Some((b, cells)) = loaded.iter().find(|(b, _)| b.id == args.id) else {
        let known: Vec<&str> = loaded.iter().map(|(b, _)| b.id.as_str()).collect();
        bail!(
            "{} is not a battle — the battles are: {}",
            args.id,
            known.join(", ")
        );
    };
    let shown: Vec<Cell> = cells
        .iter()
        .filter(|c| {
            args.model
                .as_ref()
                .is_none_or(|m| *m == c.checkpoint || *m == c.model)
                && args.hardware.as_ref().is_none_or(|h| *h == c.hw)
        })
        .cloned()
        .collect();
    let r = battle::report(b, &shown, &|c| battle_load::newest(&root, c));
    match args.format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&r)?),
        OutputFormat::Text => print!("{}", table(&r)),
    }
    Ok(i32::from(!r.all_cells_win))
}

fn verdict(v: Verdict) -> &'static str {
    match v {
        Verdict::Win => "WIN",
        Verdict::Loss => "LOSS",
        Verdict::Tie => "TIE",
        Verdict::NoRecord => "no record",
        Verdict::NotReported => "not reported",
    }
}

fn ratchet(r: &Ratchet) -> String {
    match r {
        Ratchet::Within => "within its bound".into(),
        Ratchet::Breach(why) => format!("BREACH {why}"),
        Ratchet::Unbounded => "no bound".into(),
        Ratchet::NoValue => "-".into(),
    }
}

fn number(v: Option<f64>) -> String {
    v.map_or_else(|| "-".into(), |v| format!("{v:.2}"))
}

/// 2026-10-16: The target, marked when its entry carries a caveat on it.
fn target(j: &Judged) -> String {
    let mark = if j.cell.caveat.is_some() {
        "\u{2020}"
    } else {
        ""
    };
    format!("{}{mark}", number(Some(j.cell.target)))
}

fn row(j: &Judged) -> String {
    format!(
        "  {:<30} {:<22} {:>10} {:>10}  {:<12} {:>8}  {}\n",
        j.cell.gate,
        j.cell.metric,
        number(j.ours),
        target(j),
        verdict(j.verdict),
        j.margin_pct
            .map_or_else(|| "-".into(), |m| format!("{m:+.1}%")),
        ratchet(&j.ratchet),
    )
}

/// 2026-10-16: The battle as text: one block per (hardware, checkpoint), one row per cell,
/// then the overall verdict.
pub(super) fn table(r: &Report) -> String {
    let count = |v: Verdict| r.cells.iter().filter(|j| j.verdict == v).count();
    let mut out = format!(
        "{} ({}): {} cell(s), {} won, {} lost, {} tied, {} without a value\n",
        r.name,
        r.id,
        r.cells.len(),
        count(Verdict::Win),
        count(Verdict::Loss),
        count(Verdict::Tie),
        count(Verdict::NoRecord) + count(Verdict::NotReported),
    );
    let mut subject = None;
    for j in &r.cells {
        let here = (&j.cell.hw, &j.cell.checkpoint);
        if subject != Some(here) {
            subject = Some(here);
            let _ = write!(
                out,
                "\n{} on {}\n  {:<30} {:<22} {:>10} {:>10}  {:<12} {:>8}  ratchet\n",
                j.cell.checkpoint,
                j.cell.hw,
                "gate",
                "metric",
                "ours",
                "target",
                "verdict",
                "margin"
            );
        }
        out.push_str(&row(j));
    }
    for j in &r.cells {
        if let Some(caveat) = &j.cell.caveat {
            let _ = write!(
                out,
                "\n\u{2020} {} {} on {}: {caveat}. The entry's note in kernels/{}/{}/BENCH.toml gives the account.",
                j.cell.gate, j.cell.metric, j.cell.checkpoint, j.cell.hw, j.cell.model,
            );
        }
    }
    let _ = writeln!(out, "\nall cells win: {}", r.all_cells_win);
    out
}

#[cfg(test)]
#[path = "bench_battle_tests.rs"]
mod bench_battle_tests;
