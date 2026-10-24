// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `kg_record` end to end over two ports (DESIGN-latlas.md §4 `la_record`, §5.2.4):
//! build the new file text, run every check, preview the taint the record would cause, then write
//! through the host. The core never touches a file: [`RecordFs`] reads and replaces, [`Validator`]
//! runs the file's own validator of record (`ledger.py check` for LEDGER.tsv).
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - Fail closed: a rejection by any check returns before [`RecordFs::replace`] is called.
//! - `dry_run` runs every check, returns the preview and never writes.
//! - A record that would taint `n > 0` ledger rows writes only with `acknowledge_taint == n`.
//! - The Rust ledger check must agree with the validator of record; a disagreement is a refusal.

use crate::bundle::Builder;
use crate::claims;
use crate::config::Campaign;
use crate::ledger::{Table, tainted};
use crate::prelude::*;
use crate::record::{
    Kind, append, decision_text, friction_line, ledger_line, run_line, runs_header,
};
use metrale_error::{ErrorCode, Result, error};

/// 2026-10-23: The file port `kg_record` writes through (a host implements it).
#[expect(
    async_fn_in_trait,
    reason = "2026-10-23: as metrale-abstract-io's ports: the host's executor drives the future"
)]
pub trait RecordFs {
    /// 2026-10-23: The file's text, `None` when it does not exist.
    async fn read(&self, path: &str) -> Result<Option<String>>;
    /// 2026-10-23: Replace the file atomically if it still holds `expected`, else refuse.
    async fn replace(&self, path: &str, expected: &str, new: &str) -> Result<()>;
}

/// 2026-10-23: The validator of record for a file kind (a host implements it).
#[expect(
    async_fn_in_trait,
    reason = "2026-10-23: as metrale-abstract-io's ports: the host's executor drives the future"
)]
pub trait Validator {
    /// 2026-10-23: Validate `new`, the whole new text of `path`: `Ok(what it said)` or a refusal.
    async fn validate(&self, kind: Kind, path: &str, new: &str) -> Result<String>;
}

/// 2026-10-23: What one record call asks.
#[derive(Clone, Debug)]
pub struct Request {
    pub kind: Kind,
    pub fields: BTreeMap<String, String>,
    pub dry_run: bool,
    pub acknowledge_taint: Option<u32>,
}

/// 2026-10-23: What it did (or, for `dry_run`, would do).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub id: String,
    pub file: String,
    pub line: usize,
    pub validator: String,
    pub claims: Option<String>,
    pub would_taint: Vec<String>,
    pub warnings: Vec<String>,
    pub written: bool,
}

fn reject(file: &str, why: String) -> metrale_error::Error {
    error!(ErrorCode::KgValidatorRejected, file.to_string(), why)
}

fn findings(f: impl FnOnce(&mut Builder)) -> usize {
    let mut b = Builder::default();
    f(&mut b);
    b.seal(Vec::new()).errors.len()
}

/// 2026-10-23: The ledger rows `new` taints that `old` did not, sorted.
pub fn taint_preview(old: &str, new: &str) -> Vec<String> {
    let before: BTreeSet<String> = tainted(&Table::parse(old).links())
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    tainted(&Table::parse(new).links())
        .into_iter()
        .map(|(id, _)| id)
        .filter(|id| !before.contains(id))
        .collect()
}

/// 2026-10-23: Build, check, preview and (unless `dry_run`) write one record of `c`.
pub async fn record<F: RecordFs, V: Validator>(
    fs: &F,
    validator: &V,
    c: &Campaign,
    req: &Request,
) -> Result<Outcome> {
    let file = match req.kind {
        Kind::Ledger => c.ledger.clone(),
        Kind::Friction => c.tooling.clone(),
        Kind::Decision => c.charter.clone(),
        Kind::Run => c.runs.clone(),
    };
    let old = fs.read(&file).await?.unwrap_or_default();
    let mut warnings = Vec::new();
    let mut claims_out = None;
    let mut would_taint = Vec::new();
    let (new, id) = match req.kind {
        Kind::Ledger => {
            let header = old.lines().next().unwrap_or("");
            let new = append(&old, &ledger_line(header, &req.fields)?);
            let rust = Table::parse(&new).check();
            if !rust.is_empty() {
                return Err(reject(&file, format!("ledger check: {}", rust.join("; "))));
            }
            let gate = req.fields.get("gate").map_or("-", String::as_str);
            match claims::parse(gate) {
                Ok(cl) if cl.is_empty() => {
                    warnings.push("no gate parsed; claim lacks a gate".into())
                }
                Ok(cl) => claims_out = Some(claims::canonical(&cl)),
                Err(r) => return Err(reject(&file, format!("gate: {r}"))),
            }
            would_taint = taint_preview(&old, &new);
            let id = req.fields.get("id").cloned().unwrap_or_default();
            (new, format!("ledger:{id}"))
        }
        Kind::Friction => {
            let new = append(&old, &friction_line(&req.fields)?);
            let known = BTreeSet::new();
            let n0 = findings(|b| crate::tooling::derive(b, &c.id, &file, &old, &known));
            let n1 = findings(|b| crate::tooling::derive(b, &c.id, &file, &new, &known));
            if n1 != n0 {
                return Err(reject(
                    &file,
                    "the row does not parse as a TOOLING.md table row".into(),
                ));
            }
            (new, "friction".into())
        }
        Kind::Decision => {
            let new = decision_text(&old, &req.fields)?;
            let ids = BTreeSet::new();
            let n0 = findings(|b| crate::decisions::charter(b, &file, &old, &ids));
            let n1 = findings(|b| crate::decisions::charter(b, &file, &new, &ids));
            if n1 != n0 {
                return Err(reject(
                    &file,
                    "the decision does not parse under its dated section".into(),
                ));
            }
            (new, "decision".into())
        }
        Kind::Run => {
            let base = if old.is_empty() {
                format!("{}\n", runs_header())
            } else {
                old.clone()
            };
            if base.lines().next() != Some(runs_header().as_str()) {
                return Err(reject(
                    &file,
                    "the header is not the RUNS.tsv header".into(),
                ));
            }
            let (rid, line) = run_line(&req.fields)?;
            if base
                .lines()
                .skip(1)
                .any(|l| l.split('\t').next() == Some(rid.as_str()))
            {
                return Err(reject(&file, format!("run {rid} is already recorded")));
            }
            let new = append(&base, &line);
            if findings(|b| crate::runs::derive(b, &file, &new)) != 0 {
                return Err(reject(&file, "the new RUNS.tsv does not parse".into()));
            }
            (new, format!("run:{rid}"))
        }
    };
    let said = validator.validate(req.kind, &file, &new).await?;
    let n = u32::try_from(would_taint.len()).unwrap_or(u32::MAX);
    let mut written = false;
    if !req.dry_run {
        if n > 0 && req.acknowledge_taint != Some(n) {
            return Err(error!(
                ErrorCode::KgToolArgument,
                "acknowledge_taint",
                format!(
                    "the record taints {n} rows ({}); pass acknowledge_taint: {n}",
                    would_taint.join(", ")
                )
            ));
        }
        fs.replace(&file, &old, &new).await?;
        written = true;
    }
    Ok(Outcome {
        id,
        line: new.lines().count(),
        file,
        validator: said,
        claims: claims_out,
        would_taint,
        warnings,
        written,
    })
}
