// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Named battles (`bench/battles.toml`): a family of gates read together against
//! the reference figure each BENCH.toml entry declares beside its bounds (`targets`), one cell
//! per (entry, target metric). `met benchmark battle <id>` prints the table, and
//! `met benchmark certify --gates <id>` runs the member gates that are still open.
//!
//! A battle is a scoreboard, not a verdict: no gate check reads it. A cell can pass its own
//! regression ratchet (the entry's bound) and still lose its battle, so both are reported.
//!
//! Owner: bench gate.
//! Invariants:
//! - Pure: the file text, the registered gate ids, the BENCH.toml entries and the newest
//!   record of each cell are passed in (`battle_load` reads them).
//! - [`parse`] refuses a battle id that is a gate id, a gate two battles claim, a battle no
//!   registered gate belongs to, and a metric pattern other than a literal with at most one
//!   `{n}` (one or more digits).
//! - Every `targets` key in the tree belongs to exactly one battle cell: [`problems`] names a
//!   target on a gate in no battle, a key no rule of its battle matches, and a value that is
//!   not finite and positive.
//! - A cell whose subject has no record is `NoRecord` and one whose record lacks the metric is
//!   `NotReported`, never a number; `all_cells_win` is false while either exists.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::check_models::Owed;
use super::record::Bound;
use super::scoring::{Comparison, compare};

/// 2026-10-16: The only `schema` [`parse`] accepts.
pub const SCHEMA: u32 = 1;
/// 2026-10-16: The battle file, relative to the repository root. Outside `PERF_PATHS`, so
/// editing it invalidates no record.
pub const PATH: &str = "bench/battles.toml";

/// 2026-10-16: Which direction of a metric wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Better {
    Lower,
    Higher,
}

/// 2026-10-16: The metric keys a battle scores: `pattern` is a literal key, or one with a
/// single `{n}` standing for one or more digits (`median_ms_{n}`, `c{n}_aggregate_tok_s`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricRule {
    pub pattern: String,
    pub better: Better,
}

/// 2026-10-16: One `[[battles]]` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Battle {
    pub id: String,
    pub name: String,
    /// 2026-10-16: A gate is a member when its id starts with one of these.
    pub gate_prefixes: Vec<String>,
    pub metrics: Vec<MetricRule>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BattleFile {
    schema: u32,
    battles: Vec<Battle>,
}

impl Battle {
    pub fn member(&self, gate: &str) -> bool {
        self.gate_prefixes
            .iter()
            .any(|p| gate.starts_with(p.as_str()))
    }

    /// 2026-10-16: The direction of the first rule that matches `metric`; `None` when none does.
    pub fn better(&self, metric: &str) -> Option<Better> {
        self.metrics
            .iter()
            .find(|r| matches(&r.pattern, metric))
            .map(|r| r.better)
    }
}

fn matches(pattern: &str, metric: &str) -> bool {
    match pattern.split_once("{n}") {
        None => pattern == metric,
        Some((pre, post)) => metric
            .strip_prefix(pre)
            .and_then(|rest| rest.strip_suffix(post))
            .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())),
    }
}

fn pattern_problem(pattern: &str) -> Option<String> {
    let rest = pattern.replacen("{n}", "", 1);
    if pattern.is_empty() {
        Some("an empty metric pattern".into())
    } else if rest.contains('{') || rest.contains('}') {
        Some(format!(
            "metric pattern {pattern:?} may hold one `{{n}}` and no other brace"
        ))
    } else {
        None
    }
}

/// 2026-10-16: The battles in `text`, checked against the registered gate ids.
pub fn parse(text: &str, gate_ids: &[&str]) -> Result<Vec<Battle>, String> {
    let file: BattleFile = toml::from_str(text).map_err(|e| format!("{PATH}: {e}"))?;
    if file.schema != SCHEMA {
        return Err(format!("{PATH}: schema {} is not {SCHEMA}", file.schema));
    }
    if file.battles.is_empty() {
        return Err(format!("{PATH}: no [[battles]]"));
    }
    let mut ids = BTreeSet::new();
    for b in &file.battles {
        if !ids.insert(b.id.as_str()) {
            return Err(format!("{PATH}: battle {} is declared twice", b.id));
        }
        if b.id.is_empty() || b.name.is_empty() {
            return Err(format!("{PATH}: a battle needs an id and a name"));
        }
        if gate_ids.contains(&b.id.as_str()) {
            return Err(format!(
                "{PATH}: battle id {} is a gate id; `--gates {}` could not tell them apart",
                b.id, b.id
            ));
        }
        if b.gate_prefixes.is_empty() || b.gate_prefixes.iter().any(String::is_empty) {
            return Err(format!(
                "{PATH}: battle {} needs non-empty gate_prefixes",
                b.id
            ));
        }
        if b.metrics.is_empty() {
            return Err(format!("{PATH}: battle {} scores no metric", b.id));
        }
        if let Some(p) = b.metrics.iter().find_map(|r| pattern_problem(&r.pattern)) {
            return Err(format!("{PATH}: battle {}: {p}", b.id));
        }
        if !gate_ids.iter().any(|g| b.member(g)) {
            return Err(format!(
                "{PATH}: no registered gate belongs to battle {} ({})",
                b.id,
                b.gate_prefixes.join(", ")
            ));
        }
    }
    for g in gate_ids {
        let claims: Vec<&str> = file
            .battles
            .iter()
            .filter(|b| b.member(g))
            .map(|b| b.id.as_str())
            .collect();
        if claims.len() > 1 {
            return Err(format!(
                "{PATH}: gate {g} belongs to more than one battle ({})",
                claims.join(", ")
            ));
        }
    }
    Ok(file.battles)
}

/// 2026-10-16: The battle a gate belongs to, if any.
pub fn battle_of<'a>(battles: &'a [Battle], gate: &str) -> Option<&'a Battle> {
    battles.iter().find(|b| b.member(gate))
}

/// 2026-10-16: One BENCH.toml entry, as far as a battle reads it.
#[derive(Debug, Clone, Default)]
pub struct Entry {
    pub gate: String,
    pub hw: String,
    pub model: String,
    pub checkpoint: String,
    /// 2026-10-16: The entry's bounds; empty for an `unmeasured` entry.
    pub bounds: BTreeMap<String, Bound>,
    pub targets: BTreeMap<String, f64>,
}

/// 2026-10-16: Every `targets` value in `entries` that is not a battle cell, one line each.
pub fn problems(battles: &[Battle], entries: &[Entry]) -> Vec<String> {
    let mut out = Vec::new();
    for e in entries.iter().filter(|e| !e.targets.is_empty()) {
        let at = format!(
            "kernels/{}/{}/BENCH.toml {} / {}",
            e.hw, e.model, e.gate, e.checkpoint
        );
        let Some(b) = battle_of(battles, &e.gate) else {
            out.push(format!(
                "{at}: declares targets, and {} is in no battle",
                e.gate
            ));
            continue;
        };
        for (metric, target) in &e.targets {
            if b.better(metric).is_none() {
                out.push(format!(
                    "{at}: target {metric} matches no metric rule of battle {}",
                    b.id
                ));
            }
            if !(target.is_finite() && *target > 0.0) {
                out.push(format!(
                    "{at}: target {metric} must be finite and positive, got {target}"
                ));
            }
        }
    }
    out
}

/// 2026-10-16: One scored (gate, subject, metric) of a battle.
#[derive(Debug, Clone, Serialize)]
pub struct Cell {
    pub gate: String,
    pub hw: String,
    pub model: String,
    pub checkpoint: String,
    pub metric: String,
    pub better: Better,
    pub target: f64,
    /// 2026-10-16: The entry's own bound on the metric: its regression ratchet.
    pub bound: Option<Bound>,
}

/// 2026-10-16: The cells of `battle`: every target its member gates' entries declare. Run
/// [`problems`] first: a target no rule matches is left out here.
pub fn cells(battle: &Battle, entries: &[Entry]) -> Vec<Cell> {
    let mut out = Vec::new();
    for e in entries.iter().filter(|e| battle.member(&e.gate)) {
        for (metric, target) in &e.targets {
            let Some(better) = battle.better(metric) else {
                continue;
            };
            out.push(Cell {
                gate: e.gate.clone(),
                hw: e.hw.clone(),
                model: e.model.clone(),
                checkpoint: e.checkpoint.clone(),
                metric: metric.clone(),
                better,
                target: *target,
                bound: e.bounds.get(metric).cloned(),
            });
        }
    }
    out.sort_by(|a, b| {
        (&a.hw, &a.checkpoint, &a.gate)
            .cmp(&(&b.hw, &b.checkpoint, &b.gate))
            .then_with(|| natural(&a.metric, &b.metric))
    });
    out
}

/// 2026-10-16: Digits compare by value, so `median_ms_256` sorts before `median_ms_1024`.
fn natural(a: &str, b: &str) -> std::cmp::Ordering {
    let key = |s: &str| {
        let digits: String = s.chars().filter(char::is_ascii_digit).collect();
        let rest: String = s.chars().filter(|c| !c.is_ascii_digit()).collect();
        (rest, digits.parse::<u64>().unwrap_or(0), s.to_string())
    };
    key(a).cmp(&key(b))
}

/// 2026-10-16: The newest record of a cell's subject, as far as a battle reads it.
#[derive(Debug, Clone)]
pub struct Observed {
    pub git_sha: String,
    pub recorded_at: u64,
    pub metrics: BTreeMap<String, f64>,
}

/// 2026-10-16: Against the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    Win,
    Loss,
    Tie,
    NoRecord,
    NotReported,
}

/// 2026-10-16: Against the entry's own bound, judged as the gate judges it (`scoring::compare`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", content = "why", rename_all = "kebab-case")]
pub enum Ratchet {
    Within,
    Breach(String),
    /// 2026-10-16: The entry bounds no such metric (an `unmeasured` entry has no bounds yet).
    Unbounded,
    NoValue,
}

/// 2026-10-16: One cell's standing.
#[derive(Debug, Clone, Serialize)]
pub struct Judged {
    pub cell: Cell,
    pub ours: Option<f64>,
    pub record_sha: Option<String>,
    pub recorded_at: Option<u64>,
    pub verdict: Verdict,
    /// 2026-10-16: How far ahead of the target, in percent of it; negative is behind.
    pub margin_pct: Option<f64>,
    pub ratchet: Ratchet,
}

/// 2026-10-16: One cell against its target and its own bound.
pub fn judge(cell: &Cell, newest: Option<&Observed>) -> Judged {
    let ours = newest
        .and_then(|o| o.metrics.get(&cell.metric))
        .copied()
        .filter(|v| v.is_finite());
    let verdict = match (newest, ours) {
        (None, _) => Verdict::NoRecord,
        (Some(_), None) => Verdict::NotReported,
        (Some(_), Some(v)) => {
            let ahead = match cell.better {
                Better::Lower => cell.target - v,
                Better::Higher => v - cell.target,
            };
            if ahead > 0.0 {
                Verdict::Win
            } else if ahead < 0.0 {
                Verdict::Loss
            } else {
                Verdict::Tie
            }
        }
    };
    let margin_pct = ours.map(|v| match cell.better {
        Better::Lower => (cell.target - v) / cell.target * 100.0,
        Better::Higher => (v - cell.target) / cell.target * 100.0,
    });
    let ratchet = match (ours, &cell.bound) {
        (None, _) => Ratchet::NoValue,
        (Some(_), None) => Ratchet::Unbounded,
        (Some(v), Some(b)) => match compare(&cell.metric, v, b) {
            Comparison::Pass => Ratchet::Within,
            Comparison::Fail(why) => Ratchet::Breach(why),
            Comparison::Skip(_) => Ratchet::Unbounded,
        },
    };
    Judged {
        cell: cell.clone(),
        ours,
        record_sha: newest.map(|o| o.git_sha.clone()),
        recorded_at: newest.map(|o| o.recorded_at),
        verdict,
        margin_pct,
        ratchet,
    }
}

/// 2026-10-16: A battle's cells, judged.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub id: String,
    pub name: String,
    pub cells: Vec<Judged>,
    /// 2026-10-16: True only when there is at least one cell and every cell is a `Win`.
    pub all_cells_win: bool,
}

/// 2026-10-16: Judge each cell against the newest record of its subject.
pub fn report(
    battle: &Battle,
    cells: &[Cell],
    newest: &dyn Fn(&Cell) -> Option<Observed>,
) -> Report {
    let judged: Vec<Judged> = cells.iter().map(|c| judge(c, newest(c).as_ref())).collect();
    let all_cells_win = !judged.is_empty() && judged.iter().all(|j| j.verdict == Verdict::Win);
    Report {
        id: battle.id.clone(),
        name: battle.name.clone(),
        cells: judged,
        all_cells_win,
    }
}

/// 2026-10-16: The member gates on `hw` with a cell whose subject `is_pass` says is not
/// current, sorted and each once: what `--gates <battle>` runs.
pub fn open_members(cells: &[Cell], hw: &str, is_pass: &dyn Fn(&Cell) -> bool) -> Vec<String> {
    let open: BTreeSet<&str> = cells
        .iter()
        .filter(|c| c.hw == hw && !is_pass(c))
        .map(|c| c.gate.as_str())
        .collect();
    open.into_iter().map(str::to_string).collect()
}

/// 2026-10-16: Per model directory, the battles a change leaves owed: a battle is owed by a
/// model when one of its cells' (gate, checkpoint, hw) is owed.
pub fn owed_battles(
    battles: &[(Battle, Vec<Cell>)],
    owed: &[Owed],
) -> BTreeMap<String, BTreeSet<String>> {
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (battle, cells) in battles {
        for o in owed {
            let hit = cells.iter().any(|c| {
                c.gate == o.gate && c.checkpoint == o.subject.checkpoint && c.hw == o.subject.hw
            });
            if hit {
                out.entry(o.subject.model.clone())
                    .or_default()
                    .insert(battle.id.clone());
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "battle_tests.rs"]
mod battle_tests;
