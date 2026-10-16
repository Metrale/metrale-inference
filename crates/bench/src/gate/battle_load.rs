// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The I/O half of `battle`: reads `bench/battles.toml`, every BENCH.toml entry
//! and the committed records, and hands them to the pure core.
//!
//! Owner: bench gate.
//! Invariants:
//! - [`load`] fails when the file does not parse or any `targets` value is not a battle cell,
//!   so a typo cannot drop a cell from the scoreboard without a word.
//! - [`newest`] is the newest committed record of the cell's gate whose `target_model` is the
//!   cell's checkpoint and whose hardware key is the cell's hardware; an unreadable record is
//!   skipped, as `records_newest_first` sorts it last.

use std::path::Path;

use anyhow::{Context, Result, bail};

use super::battle::{self, Battle, Cell, Entry, Observed};
use super::check::records_newest_first;
use super::record::read_record;

/// 2026-10-16: The battles, validated against the registered gate ids.
pub fn battles(root: &Path) -> Result<Vec<Battle>> {
    let path = root.join(battle::PATH);
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let ids: Vec<&str> = crate::registry::all().iter().map(|d| d.id).collect();
    battle::parse(&text, &ids).map_err(anyhow::Error::msg)
}

/// 2026-10-16: Every BENCH.toml entry, as a battle reads it.
pub fn entries(root: &Path) -> Result<Vec<Entry>> {
    Ok(super::bench::load_all(root)?
        .into_iter()
        .map(|(target, e)| Entry {
            gate: e.gate,
            hw: target.hardware,
            model: target.model,
            checkpoint: e.checkpoint,
            bounds: e.metrics.unwrap_or_default(),
            targets: e.targets,
            caveats: e.target_caveats,
        })
        .collect())
}

/// 2026-10-16: Each battle with its cells; fails on any target that is not a cell.
pub fn load(root: &Path) -> Result<Vec<(Battle, Vec<Cell>)>> {
    let battles = battles(root)?;
    let entries = entries(root)?;
    let problems = battle::problems(&battles, &entries);
    if !problems.is_empty() {
        bail!("battle targets:\n  {}", problems.join("\n  "));
    }
    Ok(battles
        .into_iter()
        .map(|b| {
            let cells = battle::cells(&b, &entries);
            (b, cells)
        })
        .collect())
}

/// 2026-10-16: The newest committed record of `cell`'s subject.
pub fn newest(root: &Path, cell: &Cell) -> Option<Observed> {
    records_newest_first(root, &cell.gate)
        .iter()
        .filter_map(|p| read_record(p).ok())
        .find(|r| {
            r.benchmark_id == cell.gate
                && r.target_model == cell.checkpoint
                && r.hardware.gate_key() == cell.hw
        })
        .map(|r| Observed {
            git_sha: r.git_sha,
            recorded_at: r.recorded_at,
            metrics: r.metrics,
        })
}
