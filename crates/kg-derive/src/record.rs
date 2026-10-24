// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `kg_record`'s pure half (design §3.2 tool 4, §4.1 invariant 4): build the one line
//! a record appends to its SSOT file from the caller's fields, and the new file text. The host
//! validates the new text (the campaign's `ledger.py check`, or this crate's adapter), writes a
//! temp copy and renames it into place; on any rejection the file is untouched.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - Every column is a required field (PCND): a missing or unknown field is `KgToolArgument` with
//!   its name; nothing is defaulted, not even the timestamp (the caller passes `ts`).
//! - A field with a tab, a newline or a carriage return is refused (it would shift columns).
//! - A field that matches the secrets denylist is refused (records are committed files).
//! - The record is appended after the last line; a file without a final newline gains one first.

use crate::hash::sha256;
use crate::prelude::*;
use crate::secrets::secret_pattern;
use metrale_error::{ErrorCode, Result, error};

/// 2026-10-23: The record kinds `kg_record` accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Ledger,
    Friction,
    Decision,
    Run,
}

impl Kind {
    /// 2026-10-23: The kind of a tool argument.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "ledger" => Ok(Self::Ledger),
            "friction" => Ok(Self::Friction),
            "decision" => Ok(Self::Decision),
            "run" => Ok(Self::Run),
            _ => Err(error!(
                ErrorCode::KgToolArgument,
                "kind",
                format!("`{s}` is not ledger, friction, decision or run")
            )),
        }
    }
}

/// 2026-10-23: The RUNS.tsv columns (design §3.3 row "Run fingerprints"); `run_id` is derived.
pub const RUNS_COLUMNS: [&str; 10] = [
    "run_id",
    "binary_sha",
    "recipe_env_sha",
    "recipe_flags_sha",
    "prompt_sha",
    "box",
    "sm_mhz",
    "nsys",
    "date",
    "ledger_ids",
];

/// 2026-10-23: The TOOLING.md table columns.
pub const TOOLING_COLUMNS: [&str; 5] = ["phase", "friction", "cost_min", "proposed_change", "done"];

fn field<'a>(fields: &'a BTreeMap<String, String>, name: &str) -> Result<&'a str> {
    let v = fields
        .get(name)
        .ok_or_else(|| error!(ErrorCode::KgToolArgument, name.to_string(), "is required"))?;
    if v.contains(['\t', '\n', '\r']) {
        return Err(error!(
            ErrorCode::KgToolArgument,
            name.to_string(),
            "holds a tab or a line break"
        ));
    }
    if let Some(p) = secret_pattern(v) {
        return Err(error!(
            ErrorCode::KgToolArgument,
            name.to_string(),
            format!("matches the secrets denylist ({p})")
        ));
    }
    Ok(v)
}

fn exactly(fields: &BTreeMap<String, String>, columns: &[&str]) -> Result<Vec<String>> {
    if let Some(extra) = fields.keys().find(|k| !columns.contains(&k.as_str())) {
        return Err(error!(
            ErrorCode::KgToolArgument,
            extra.clone(),
            "is not a column of this record"
        ));
    }
    columns
        .iter()
        .map(|c| field(fields, c).map(str::to_string))
        .collect()
}

/// 2026-10-23: A ledger row in the column order of `header` (the file's own first line).
pub fn ledger_line(header: &str, fields: &BTreeMap<String, String>) -> Result<String> {
    let cols: Vec<&str> = header.trim_end_matches(['\n', '\r']).split('\t').collect();
    Ok(exactly(fields, &cols)?.join("\t"))
}

/// 2026-10-23: A TOOLING.md row; `|` in a cell is escaped so the table keeps its columns.
pub fn friction_line(fields: &BTreeMap<String, String>) -> Result<String> {
    let cells = exactly(fields, &TOOLING_COLUMNS)?;
    let cells: Vec<String> = cells.iter().map(|c| c.replace('|', "\\|")).collect();
    Ok(format!("| {} |", cells.join(" | ")))
}

/// 2026-10-23: A run's id: the first 8 hex of sha256 over its other columns, tab-joined.
pub fn run_id(columns: &str) -> String {
    sha256(columns.as_bytes()).hex().chars().take(8).collect()
}

/// 2026-10-23: A RUNS.tsv row; `run_id` is [`run_id`] of the other columns.
pub fn run_line(fields: &BTreeMap<String, String>) -> Result<(String, String)> {
    let cells = exactly(fields, &RUNS_COLUMNS[1..])?;
    let joined = cells.join("\t");
    let id = run_id(&joined);
    Ok((id.clone(), format!("{id}\t{joined}")))
}

/// 2026-10-23: The RUNS.tsv header line.
pub fn runs_header() -> String {
    RUNS_COLUMNS.join("\t")
}

/// 2026-10-23: `text` with `line` appended.
pub fn append(text: &str, line: &str) -> String {
    let mut s = String::with_capacity(text.len() + line.len() + 2);
    s.push_str(text);
    if !s.is_empty() && !s.ends_with('\n') {
        s.push('\n');
    }
    s.push_str(line);
    s.push('\n');
    s
}

/// 2026-10-23: CHARTER.md with a decision bullet under `## Decisions <date>`, creating the section
/// at the end when the date has none. Fields: `date` (YYYY-MM-DD) and `text`.
pub fn decision_text(charter: &str, fields: &BTreeMap<String, String>) -> Result<String> {
    let v = exactly(fields, &["date", "text"])?;
    let (date, text) = (&v[0], &v[1]);
    let ok = date.len() == 10
        && date.bytes().enumerate().all(|(i, c)| {
            if i == 4 || i == 7 {
                c == b'-'
            } else {
                c.is_ascii_digit()
            }
        });
    if !ok {
        return Err(error!(
            ErrorCode::KgToolArgument,
            "date", "is not YYYY-MM-DD"
        ));
    }
    let head = format!("## Decisions {date}");
    let lines: Vec<&str> = charter.lines().collect();
    let Some(start) = lines.iter().position(|l| l.trim_end() == head) else {
        return Ok(append(
            &append(charter, &format!("\n{head}")),
            &format!("- {text}"),
        ));
    };
    let end = lines[start + 1..]
        .iter()
        .position(|l| l.starts_with("## "))
        .map_or(lines.len(), |i| start + 1 + i);
    let mut at = end;
    while at > start + 1 && lines[at - 1].trim().is_empty() {
        at -= 1;
    }
    let mut out: Vec<String> = lines.iter().map(|l| (*l).to_string()).collect();
    out.insert(at, format!("- {text}"));
    let mut s = out.join("\n");
    if charter.ends_with('\n') {
        s.push('\n');
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(kv: &[(&str, &str)]) -> BTreeMap<String, String> {
        kv.iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn a_ledger_row_follows_the_header_and_refuses_shifting_or_missing_fields() {
        let h = "ts\tid\tgate\n";
        assert_eq!(
            ledger_line(h, &f(&[("ts", "t"), ("id", "x"), ("gate", "-")])).unwrap(),
            "t\tx\t-"
        );
        let tab = ledger_line(h, &f(&[("ts", "t"), ("id", "x\ty"), ("gate", "-")])).unwrap_err();
        assert!(
            tab.to_string().contains("id") && tab.to_string().contains("tab"),
            "{tab}"
        );
        assert!(
            ledger_line(h, &f(&[("ts", "t"), ("id", "x")]))
                .unwrap_err()
                .to_string()
                .contains("gate")
        );
        assert!(
            ledger_line(
                h,
                &f(&[("ts", "t"), ("id", "x"), ("gate", "-"), ("extra", "")])
            )
            .is_err()
        );
        assert!(
            ledger_line(
                h,
                &f(&[("ts", "t"), ("id", "sk-or-v1-zzzz"), ("gate", "-")])
            )
            .is_err()
        );
    }

    #[test]
    fn a_run_id_is_content_addressed() {
        let cols: Vec<(&str, &str)> = RUNS_COLUMNS[1..].iter().map(|c| (*c, "v")).collect();
        let (a, line) = run_line(&f(&cols)).unwrap();
        assert_eq!(line.split('\t').count(), RUNS_COLUMNS.len());
        let mut other = cols.clone();
        other[0].1 = "w";
        assert_ne!(run_line(&f(&other)).unwrap().0, a);
    }

    #[test]
    fn a_decision_lands_in_its_dated_section_or_opens_one() {
        let c = "# C\n## Decisions 2026-10-22\n- a\n\n## Re-plan\nx\n";
        let out = decision_text(c, &f(&[("date", "2026-10-22"), ("text", "b")])).unwrap();
        assert_eq!(
            out,
            "# C\n## Decisions 2026-10-22\n- a\n- b\n\n## Re-plan\nx\n"
        );
        let new = decision_text(c, &f(&[("date", "2026-10-24"), ("text", "z")])).unwrap();
        assert!(
            new.ends_with("x\n\n## Decisions 2026-10-24\n- z\n"),
            "{new}"
        );
        assert!(decision_text(c, &f(&[("date", "24-10-2026"), ("text", "z")])).is_err());
    }

    #[test]
    fn append_adds_a_missing_final_newline_first() {
        assert_eq!(append("a", "b"), "a\nb\n");
        assert_eq!(append("a\n", "b"), "a\nb\n");
    }
}
