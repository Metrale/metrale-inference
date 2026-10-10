// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Winner selection: from the sweep's records (several boxes, reruns), one decision
//! per cell, the cells that have none (and why), and the (cell, kernel) pairs that must be
//! rerun; then the row merge into SCHEDULES.toml entries.
//!
//! Owner: metrale-accuracy (envelope).
//! Invariants:
//! - Per (cell, kernel) the latest non-throttled record counts; a pair with only throttled
//!   records is a rerun and never a candidate.
//! - Only a `pass` candidate with a time can win. A served cell whose default did not pass, was
//!   not measured or must be rerun gets no decision: today's routing stays.
//! - The noise rule keeps the default unless the winner is faster by at least
//!   max([`NOISE_FLOOR`], [`SPREAD_FACTOR`] x the larger repetition spread).
//! - Bit-identical by default: [`Enabled::of`] decides, from the numerics alone.

use std::collections::{BTreeMap, BTreeSet};

use super::record::{Cell, Measurement, Verdict};
use super::schedules::{Enabled, Numerics, Shape};

/// 2026-10-10: The smallest relative gain that replaces a passing default: below 3% the
/// difference is inside run-to-run noise on GB10 even when the repetitions agree.
pub const NOISE_FLOOR: f64 = 0.03;

/// 2026-10-10: The gain must also exceed twice the larger repetition spread (max - min over
/// median) of the winner and the default, so a noisy timing never displaces the default.
pub const SPREAD_FACTOR: f64 = 2.0;

/// 2026-10-10: The decision at one cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// 2026-10-10: The cell.
    pub cell: Cell,
    /// 2026-10-10: The winning entry point.
    pub kernel: String,
    /// 2026-10-10: Its family.
    pub family: String,
    /// 2026-10-10: Today's routed entry point, if a served plan has one.
    pub default: Option<String>,
    /// 2026-10-10: Numerics class.
    pub numerics: Numerics,
    /// 2026-10-10: [`Enabled::of`] the numerics.
    pub enabled: Enabled,
    /// 2026-10-10: The winner's median, microseconds.
    pub median_us: f64,
    /// 2026-10-10: The default's median, when there is a default.
    pub default_us: Option<f64>,
    /// 2026-10-10: Roofline floor, microseconds.
    pub floor_us: f64,
    /// 2026-10-10: `<host> <at>` of the winner's record.
    pub measured: String,
}

/// 2026-10-10: Why a cell has no decision.
#[derive(Debug, Clone, PartialEq)]
pub enum NoWinner {
    /// 2026-10-10: No candidate passed with a time: each counted candidate's (kernel, verdict,
    /// detail).
    NoPassingCandidate(Vec<(String, Verdict, String)>),
    /// 2026-10-10: The cell's default has no record.
    DefaultNotMeasured(String),
    /// 2026-10-10: The default has only throttled records.
    DefaultNeedsRerun(String),
    /// 2026-10-10: The default failed or could not run its contract (or passed with no time).
    DefaultDidNotPass {
        /// 2026-10-10: The default.
        default: String,
        /// 2026-10-10: Its verdict.
        verdict: Verdict,
        /// 2026-10-10: Its detail.
        detail: String,
    },
}

/// 2026-10-10: A cell without a decision.
#[derive(Debug, Clone, PartialEq)]
pub struct Undecided {
    /// 2026-10-10: The cell.
    pub cell: Cell,
    /// 2026-10-10: Why.
    pub why: NoWinner,
}

/// 2026-10-10: A (cell, kernel) whose every record was throttled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rerun {
    /// 2026-10-10: The cell.
    pub cell: Cell,
    /// 2026-10-10: The candidate.
    pub kernel: String,
    /// 2026-10-10: The boxes whose records were throttled.
    pub hosts: Vec<String>,
}

/// 2026-10-10: The selection over a sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    /// 2026-10-10: Hardware class.
    pub hardware: String,
    /// 2026-10-10: Decisions, by cell.
    pub decisions: Vec<Decision>,
    /// 2026-10-10: Cells without one, by cell.
    pub undecided: Vec<Undecided>,
    /// 2026-10-10: Throttled-only pairs, by cell then kernel.
    pub reruns: Vec<Rerun>,
    /// 2026-10-10: The row counts swept per shape (every cell with any record): the ladder the
    /// row merge walks.
    pub swept: BTreeMap<Shape, BTreeSet<u64>>,
}

/// 2026-10-10: Records that cannot be selected from.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SelectError {
    /// 2026-10-10: No records.
    #[error("no envelope records")]
    Empty,
    /// 2026-10-10: Records of two hardware classes.
    #[error("records of two hardware classes: `{0}` and `{1}`")]
    MixedHardware(String, String),
    /// 2026-10-10: Two records of one cell name different defaults.
    #[error("cell {cell:?}: records disagree on the default ({a:?} vs {b:?})")]
    DefaultDisagrees {
        /// 2026-10-10: The cell.
        cell: Box<Cell>,
        /// 2026-10-10: One default.
        a: Option<String>,
        /// 2026-10-10: The other.
        b: Option<String>,
    },
    /// 2026-10-10: Two records of one kernel name different families.
    #[error("kernel `{kernel}`: records name families `{a}` and `{b}`")]
    FamilyDisagrees {
        /// 2026-10-10: The kernel.
        kernel: String,
        /// 2026-10-10: One family.
        a: String,
        /// 2026-10-10: The other.
        b: String,
    },
    /// 2026-10-10: A record breaks a rule the selection relies on.
    #[error("cell {cell:?} kernel `{kernel}`: {problem}")]
    BadRecord {
        /// 2026-10-10: The cell.
        cell: Box<Cell>,
        /// 2026-10-10: The kernel.
        kernel: String,
        /// 2026-10-10: What is wrong.
        problem: String,
    },
}

fn bad(m: &Measurement, problem: impl Into<String>) -> SelectError {
    SelectError::BadRecord {
        cell: Box::new(m.cell.clone()),
        kernel: m.kernel.clone(),
        problem: problem.into(),
    }
}

/// 2026-10-10: The shape of a cell.
pub fn shape_of(c: &Cell) -> Shape {
    Shape {
        op: c.op.clone(),
        weight: c.weight.clone(),
        activation: c.activation.clone(),
        k: c.k,
        n: c.n,
    }
}

/// 2026-10-10: An RFC 3339 UTC instant `YYYY-MM-DDTHH:MM:SS[.f]Z` as an orderable key (the
/// seconds part, then the fraction in nanoseconds); any other spelling is refused, so string
/// order never stands in for time order.
fn instant(m: &Measurement) -> Result<(String, u64), SelectError> {
    let at = m.at.as_str();
    let refuse = || bad(m, format!("`at` = `{at}` is not RFC 3339 UTC (…Z)"));
    let body = at.strip_suffix('Z').ok_or_else(refuse)?;
    let (secs, frac) = body.split_once('.').unwrap_or((body, ""));
    let shape_ok = secs.len() == 19
        && secs.bytes().enumerate().all(|(i, b)| match i {
            4 | 7 => b == b'-',
            10 => b == b'T',
            13 | 16 => b == b':',
            _ => b.is_ascii_digit(),
        });
    if !shape_ok || frac.len() > 9 || !frac.bytes().all(|b| b.is_ascii_digit()) {
        return Err(refuse());
    }
    if body.contains('.') && frac.is_empty() {
        return Err(refuse());
    }
    let nanos = format!("{frac:0<9}").parse::<u64>().map_err(|_| refuse())?;
    Ok((secs.to_string(), nanos))
}

/// 2026-10-10: Per (cell, kernel) the counted record, or the throttled hosts.
enum Pick<'a> {
    Counted(&'a Measurement),
    Throttled(Vec<String>),
}

/// 2026-10-10: Per cell, its default and its picks by kernel.
type Cells<'a> = BTreeMap<Cell, (Option<String>, BTreeMap<String, Pick<'a>>)>;

fn group(records: &[Measurement]) -> Result<(String, Cells<'_>), SelectError> {
    let first = records.first().ok_or(SelectError::Empty)?;
    let mut families: BTreeMap<&str, &str> = BTreeMap::new();
    let mut defaults: BTreeMap<&Cell, &Option<String>> = BTreeMap::new();
    let mut latest: BTreeMap<(&Cell, &str), ((String, u64), &Measurement)> = BTreeMap::new();
    let mut throttled: BTreeMap<(&Cell, &str), BTreeSet<String>> = BTreeMap::new();
    for m in records {
        if m.hardware != first.hardware {
            return Err(SelectError::MixedHardware(
                first.hardware.clone(),
                m.hardware.clone(),
            ));
        }
        if m.default.as_deref() == Some("") {
            return Err(bad(m, "`default` is empty: absent is `null`"));
        }
        if let Some(f) = families
            .insert(&m.kernel, &m.family)
            .filter(|f| *f != m.family)
        {
            return Err(SelectError::FamilyDisagrees {
                kernel: m.kernel.clone(),
                a: f.to_string(),
                b: m.family.clone(),
            });
        }
        if let Some(d) = defaults
            .insert(&m.cell, &m.default)
            .filter(|d| **d != m.default)
        {
            return Err(SelectError::DefaultDisagrees {
                cell: Box::new(m.cell.clone()),
                a: d.clone(),
                b: m.default.clone(),
            });
        }
        let at = instant(m)?;
        let key = (&m.cell, m.kernel.as_str());
        if m.throttled {
            throttled.entry(key).or_default().insert(m.host.clone());
            continue;
        }
        match latest.get(&key) {
            Some((prev, p)) if *prev == at && *p != m => {
                return Err(bad(
                    m,
                    format!("two different records at the same instant {}", m.at),
                ));
            }
            Some((prev, _)) if *prev >= at => {}
            _ => {
                latest.insert(key, (at, m));
            }
        }
    }
    let mut cells: Cells = defaults
        .into_iter()
        .map(|(c, d)| (c.clone(), (d.clone(), BTreeMap::new())))
        .collect();
    for ((cell, kernel), hosts) in throttled {
        cells.entry(cell.clone()).or_default().1.insert(
            kernel.to_string(),
            Pick::Throttled(hosts.into_iter().collect()),
        );
    }
    for ((cell, kernel), (_, m)) in latest {
        cells
            .entry(cell.clone())
            .or_default()
            .1
            .insert(kernel.to_string(), Pick::Counted(m));
    }
    Ok((first.hardware.clone(), cells))
}

/// 2026-10-10: A counted record's median when it passed with a time.
fn timed(m: &Measurement) -> Result<Option<f64>, SelectError> {
    if m.verdict != Verdict::Pass {
        return Ok(None);
    }
    match m.median_us() {
        Some(t) if t > 0.0 => Ok(Some(t)),
        Some(t) => Err(bad(m, format!("median {t} us is not positive"))),
        None if m.time_us.is_empty() => Ok(None),
        None => Err(bad(m, "a repetition time is not finite")),
    }
}

/// 2026-10-10: The winner's bytes equal the default's: equal, non-empty class sets with equal
/// digests.
fn bit_identical(winner: &Measurement, default: &Measurement) -> bool {
    !winner.digests.is_empty() && winner.digests == default.digests
}

fn spread(m: &Measurement) -> f64 {
    m.spread().unwrap_or(0.0)
}

fn decide(
    cell: &Cell,
    default: &Option<String>,
    picks: &BTreeMap<String, Pick<'_>>,
) -> Result<Result<Decision, NoWinner>, SelectError> {
    let counted: Vec<&Measurement> = picks
        .values()
        .filter_map(|p| match p {
            Pick::Counted(m) => Some(*m),
            Pick::Throttled(_) => None,
        })
        .collect();
    let mut best: Option<(&Measurement, f64)> = None;
    for &m in &counted {
        if let Some(t) = timed(m)?
            && best.is_none_or(|(b, bt)| t < bt || (t == bt && m.kernel < b.kernel))
        {
            best = Some((m, t));
        }
    }
    let default_rec = match default {
        None => None,
        Some(d) => match picks.get(d) {
            None => return Ok(Err(NoWinner::DefaultNotMeasured(d.clone()))),
            Some(Pick::Throttled(_)) => return Ok(Err(NoWinner::DefaultNeedsRerun(d.clone()))),
            Some(Pick::Counted(m)) => match timed(m)? {
                Some(t) => Some((*m, t)),
                None => {
                    return Ok(Err(NoWinner::DefaultDidNotPass {
                        default: d.clone(),
                        verdict: m.verdict,
                        detail: m.detail.clone(),
                    }));
                }
            },
        },
    };
    let Some((mut win, mut win_t)) = best else {
        return Ok(Err(NoWinner::NoPassingCandidate(
            counted
                .iter()
                .map(|m| (m.kernel.clone(), m.verdict, m.detail.clone()))
                .collect(),
        )));
    };
    let numerics = match default_rec {
        None => Numerics::New,
        Some((d, d_t)) => {
            let margin = NOISE_FLOOR.max(SPREAD_FACTOR * spread(win).max(spread(d)));
            if win.kernel == d.kernel || (d_t - win_t) / d_t < margin {
                (win, win_t) = (d, d_t);
                Numerics::Same
            } else if bit_identical(win, d) {
                Numerics::BitIdentical
            } else {
                Numerics::Differs
            }
        }
    };
    if !win.floor_us.is_finite() || win.floor_us < 0.0 {
        return Err(bad(win, "floor_us is not a finite non-negative time"));
    }
    Ok(Ok(Decision {
        cell: cell.clone(),
        kernel: win.kernel.clone(),
        family: win.family.clone(),
        default: default.clone(),
        numerics,
        enabled: Enabled::of(numerics),
        median_us: win_t,
        default_us: default_rec.map(|(_, t)| t),
        floor_us: win.floor_us,
        measured: format!("{} {}", win.host, win.at),
    }))
}

/// 2026-10-10: Select over `records` (any number of boxes and reruns, any order).
pub fn select(records: &[Measurement]) -> Result<Selection, SelectError> {
    let (hardware, cells) = group(records)?;
    let mut sel = Selection {
        hardware,
        decisions: Vec::new(),
        undecided: Vec::new(),
        reruns: Vec::new(),
        swept: BTreeMap::new(),
    };
    for (cell, (default, picks)) in &cells {
        sel.swept
            .entry(shape_of(cell))
            .or_default()
            .insert(cell.rows);
        for (kernel, p) in picks {
            if let Pick::Throttled(hosts) = p {
                sel.reruns.push(Rerun {
                    cell: cell.clone(),
                    kernel: kernel.clone(),
                    hosts: hosts.clone(),
                });
            }
        }
        match decide(cell, default, picks)? {
            Ok(d) => sel.decisions.push(d),
            Err(why) => sel.undecided.push(Undecided {
                cell: cell.clone(),
                why,
            }),
        }
    }
    Ok(sel)
}

#[path = "select_merge.rs"]
mod merge;
pub use merge::{families, merge_rows, schedules};

#[cfg(test)]
#[path = "select_tests.rs"]
mod tests;
