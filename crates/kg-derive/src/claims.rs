// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The claim grammar of LEDGER.tsv column 15 `gate` (queries part §3): `;`-separated
//! comparator statements `kind(args)[@{scope,...}]`, a declared absence `none(<reason>)` (alone,
//! unscoped: the row makes no measured claim, e.g. `none(enabling)`), or `-` for no gate stated. The Python half lives in
//! `tools/ledger.py`; both parse `crates/kg-derive/tests/ledger_gate_cases.tsv` to identical
//! canonical forms, so the two parsers cannot drift.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - [`parse`] accepts exactly the grammar; a refusal carries one of the shared reason words
//!   (`empty`, `syntax`, `unknown_kind`, `bad_args`, `bad_scope`, `none_mixed`).
//! - `none(<reason>)` is a declared absence: a `Claim` node with no gate edge, valid for any
//!   verdict; a bare `-` with a verdict outside the no-claim set stays a `KgBadClaim`.
//! - `-` is a claim-free gate, allowed only for the verdicts in
//!   [`crate::ledger_adapter::NO_CLAIM_VERDICTS`].

use crate::bundle::Builder;
use crate::id::{KgId, Label, Rel};
use crate::ledger_adapter::NO_CLAIM_VERDICTS;
use crate::prelude::*;
use metrale_error::{ErrorCode, error};
use serde_json::Value;

/// 2026-10-23: One parsed comparator statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    pub kind: &'static str,
    pub args: Vec<String>,
    pub scope: Vec<String>,
}

const KINDS: &[&str] = &[
    "bitid", "multiset", "tost", "superior", "test", "agentic", "needle", "bfcl", "bounded", "none",
];

fn is_word(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'))
}

/// 2026-10-23: 1 to 9 ASCII digits (both parsers; `str::parse` would accept `+5`, Python `int`
/// would accept `5_0` and padding).
fn digits(s: &str) -> bool {
    (1..=9).contains(&s.len()) && s.bytes().all(|c| c.is_ascii_digit())
}

/// 2026-10-23: A non-negative decimal `d+(.d+)?` (no sign, exponent, `inf` or `nan`).
fn dec(s: &str) -> bool {
    match s.split_once('.') {
        Some((i, f)) => digits(i) && digits(f),
        None => digits(s),
    }
}

fn positive(s: &str) -> bool {
    digits(s) && s.bytes().any(|c| c != b'0')
}

fn check_args(kind: &str, a: &[String]) -> bool {
    match (kind, a) {
        ("bitid", [x]) => {
            x == "texts"
                || x == "launches"
                || x == "values"
                || x.strip_prefix("values(")
                    .and_then(|v| v.strip_suffix(')'))
                    .is_some_and(digits)
        }
        ("multiset", [x]) | ("test", [x]) | ("none", [x]) => is_word(x),
        ("tost", [m, n]) => {
            m.strip_suffix('%').is_some_and(dec) && n.strip_prefix("n=").is_some_and(positive)
        }
        ("superior", [x]) => x == "ci_excludes_0" || x.strip_prefix("p<").is_some_and(dec),
        ("agentic", [x]) => x
            .split_once('/')
            .is_some_and(|(k, n)| digits(k) && positive(n)),
        ("needle", [x]) => x.strip_suffix('k').is_some_and(digits),
        ("bfcl", [s, d]) => {
            s.strip_prefix(">=").is_some_and(dec)
                && d.strip_prefix("draw=").is_some_and(|h| {
                    h.len() == 8 && h.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'))
                })
        }
        ("bounded", [x]) => x.strip_prefix("overlap>=").is_some_and(dec),
        _ => false,
    }
}

/// 2026-10-23: Split `a,b(c,d),e` on top-level commas.
fn split_top(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    out.push(cur.trim().to_string());
    out
}

fn statement(s: &str) -> Result<Claim, &'static str> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty");
    }
    let (body, scope) = match s.split_once('@') {
        Some((b, sc)) => (b.trim(), Some(sc.trim())),
        None => (s, None),
    };
    let open = body.find('(').ok_or("syntax")?;
    let inner = body[open + 1..].strip_suffix(')').ok_or("syntax")?;
    let name = body[..open].trim();
    let kind = KINDS
        .iter()
        .copied()
        .find(|k| *k == name)
        .ok_or("unknown_kind")?;
    let args = split_top(inner);
    if !check_args(kind, &args) {
        return Err("bad_args");
    }
    let scope = match scope {
        None => Vec::new(),
        Some(sc) => {
            let list = sc
                .strip_prefix('{')
                .and_then(|x| x.strip_suffix('}'))
                .ok_or("bad_scope")?;
            let items = split_top(list);
            if items.iter().any(|i| !is_word(i)) {
                return Err("bad_scope");
            }
            items
        }
    };
    if kind == "none" && !scope.is_empty() {
        return Err("bad_scope");
    }
    Ok(Claim { kind, args, scope })
}

/// 2026-10-23: Parse a `gate` cell: `-` (no claim) or the statements, or the first refusal's
/// reason word.
pub fn parse(cell: &str) -> Result<Vec<Claim>, &'static str> {
    let cell = cell.trim();
    if cell == "-" {
        return Ok(Vec::new());
    }
    if cell.is_empty() {
        return Err("empty");
    }
    let claims: Vec<Claim> = cell.split(';').map(statement).collect::<Result<_, _>>()?;
    if claims.len() > 1 && claims.iter().any(|c| c.kind == "none") {
        return Err("none_mixed");
    }
    Ok(claims)
}

/// 2026-10-23: The canonical form both parsers print: `kind|arg|...|@scope,...` per statement,
/// `;`-joined; `-` for no claim.
pub fn canonical(claims: &[Claim]) -> String {
    if claims.is_empty() {
        return "-".to_string();
    }
    let parts: Vec<String> = claims
        .iter()
        .map(|c| format!("{}|{}|@{}", c.kind, c.args.join("|"), c.scope.join(",")))
        .collect();
    parts.join(";")
}

/// 2026-10-23: The gate node a statement is decided by.
pub fn gate_of(c: &Claim) -> String {
    let a0 = c.args.first().map_or("", String::as_str);
    match c.kind {
        "bitid" => format!("instrument/bitid-{}", a0.split('(').next().unwrap_or(a0)),
        "multiset" => "test/every_captured_run_matches_its_capture".to_string(),
        "tost" | "superior" => "instrument/paired-ab".to_string(),
        "test" => format!("test/{a0}"),
        "bounded" => "test/bounded_selection_overlap".to_string(),
        "none" => String::new(),
        other => format!("bench/{other}"),
    }
}

/// 2026-10-23: Derive the `Claim` and `Gate` nodes of one ledger row's `gate` cell.
pub fn derive_row(b: &mut Builder, row: &KgId, at: &str, cell: &str, verdict: &str) {
    let claims = match parse(cell) {
        Ok(c) => c,
        Err(reason) => {
            b.error(
                &error!(
                    ErrorCode::KgBadClaim,
                    at.to_string(),
                    cell.to_string(),
                    reason
                ),
                at,
            );
            return;
        }
    };
    if claims.is_empty() && !NO_CLAIM_VERDICTS.contains(&verdict) {
        let e = error!(
            ErrorCode::KgBadClaim,
            at.to_string(),
            "-",
            "a `-` gate needs verdict BLOCKED, ABORT or n/a"
        );
        b.error(&e, at);
    }
    for (n, c) in claims.iter().enumerate() {
        let Ok(cid) = KgId::new(Label::Claim, format!("{}/{}", row.key(), n + 1)) else {
            continue;
        };
        let mut p = BTreeMap::new();
        p.insert("kind".into(), Value::String(c.kind.to_string()));
        p.insert("args".into(), Value::from(c.args.clone()));
        p.insert("scope".into(), Value::from(c.scope.clone()));
        let raw = canonical(core::slice::from_ref(c));
        p.insert("raw".into(), Value::String(raw.clone()));
        b.define(cid.clone(), at.to_string(), &raw, p);
        b.edge(row, Rel::Claims, &cid, 1.0, at);
        if c.kind == "none" {
            continue;
        }
        if let Ok(g) = KgId::new(Label::Gate, gate_of(c)) {
            b.edge(&cid, Rel::GatedBy, &g, 1.0, at);
        }
    }
}
