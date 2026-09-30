// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The FUSIONS.toml file format and its parse into [`super::Rule`]s, split out of
//! `rules.rs` unchanged apart from the `[[runtime]]` table ([`crate::runtime`]).
//!
//! Owner: metrale-circuit.
//! Invariants: those of `rules.rs`.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use super::{KernelId, Mode, Numerics, PatternOp, Repeat, Rule, RuleError};
use crate::format::Format;
use crate::ir::{LayerKind, LinearRole, OpKind};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RulesFile {
    schema: u32,
    rule: Vec<RuleFile>,
    #[serde(default)]
    runtime: Vec<crate::runtime::RouteFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleFile {
    id: String,
    pattern: Vec<PatternFile>,
    kernels: Vec<KernelFile>,
    repeat: String,
    copies: Option<String>,
    emitter: String,
    rows: [u64; 2],
    modes: Vec<String>,
    #[serde(default)]
    requires: Vec<String>,
    #[serde(default)]
    when: BTreeMap<String, String>,
    numerics: String,
    microtest: Option<String>,
    lever: Option<String>,
    priority: i64,
    cite: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KernelFile {
    module: String,
    func: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PatternFile {
    op: String,
    role: Option<String>,
    #[serde(default)]
    roles: Vec<String>,
    layer_kind: Option<String>,
    format: Option<String>,
    local: Option<String>,
    weight: Option<String>,
    input: Option<String>,
    writes: Option<String>,
    #[serde(default)]
    keep: bool,
    #[serde(default)]
    sibling: bool,
}

/// 2026-09-28: Parse FUSIONS.toml text into rules, in file order. 2026-09-30: A file that
/// declares `[[runtime]]` routes is refused: a caller that plans reads it with
/// [`crate::runtime::parse_rule_set`], so no route is dropped unseen.
pub fn parse_rules(text: &str) -> Result<Vec<Rule>, RuleError> {
    let (rules, routes) = parse_file(text)?;
    if let Some(r) = routes.first() {
        return Err(RuleError::Runtime {
            route: r.id.clone(),
            detail: "a rule set with runtime routes is read with `parse_rule_set`".into(),
        });
    }
    Ok(rules)
}

/// 2026-09-30: The rules, and the `[[runtime]]` entries unchecked.
pub(crate) fn parse_file(
    text: &str,
) -> Result<(Vec<Rule>, Vec<crate::runtime::RouteFile>), RuleError> {
    let file: RulesFile = toml::from_str(text).map_err(|e| RuleError::Parse(e.to_string()))?;
    if file.schema != 1 {
        return Err(RuleError::Parse(format!(
            "schema {} (this build reads 1)",
            file.schema
        )));
    }
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(file.rule.len());
    for r in file.rule {
        if !seen.insert(r.id.clone()) {
            return Err(RuleError::DuplicateId(r.id));
        }
        out.push(rule(r)?);
    }
    Ok((out, file.runtime))
}

fn rule(r: RuleFile) -> Result<Rule, RuleError> {
    let shape = |detail: &str| RuleError::Shape {
        rule: r.id.clone(),
        detail: detail.to_string(),
    };
    if r.pattern.is_empty() {
        return Err(shape("empty pattern"));
    }
    if r.modes.is_empty() {
        return Err(shape("empty modes"));
    }
    if r.rows[0] == 0 || r.rows[0] > r.rows[1] {
        return Err(shape("rows must be [lo, hi] with 1 <= lo <= hi"));
    }
    if r.cite.trim().is_empty() {
        return Err(shape("empty cite"));
    }
    let mut modes = BTreeSet::new();
    for m in &r.modes {
        modes.insert(Mode::parse(m).ok_or_else(|| shape(&format!("unknown mode `{m}`")))?);
    }
    let repeat = Repeat::parse(&r.repeat).ok_or_else(|| {
        shape(&format!(
            "unknown repeat `{}` (once | per_row | per_row_but_last | chunk<n>)",
            r.repeat
        ))
    })?;
    let copies = r
        .copies
        .as_deref()
        .map(|c| {
            Repeat::parse(c).ok_or_else(|| {
                shape(&format!(
                    "unknown copies `{c}` (once | per_row | per_row_but_last | chunk<n>)"
                ))
            })
        })
        .transpose()?;
    if r.pattern[0].sibling {
        return Err(shape("the first pattern element cannot be a sibling"));
    }
    let numerics = numerics(&r)?;
    let mut pattern = Vec::with_capacity(r.pattern.len());
    for p in &r.pattern {
        pattern.push(pattern_op(&r.id, p)?);
    }
    Ok(Rule {
        pattern,
        kernels: r
            .kernels
            .into_iter()
            .map(|k| KernelId {
                module: k.module,
                func: k.func,
            })
            .collect(),
        repeat,
        copies,
        emitter: r.emitter,
        rows: (r.rows[0], r.rows[1]),
        modes,
        requires: r.requires.into_iter().collect(),
        when: r.when,
        numerics,
        priority: r.priority,
        cite: r.cite,
        id: r.id,
    })
}

fn numerics(r: &RuleFile) -> Result<Numerics, RuleError> {
    let stray = |what: &str| RuleError::Numerics {
        rule: r.id.clone(),
        detail: format!("`{what}` is not allowed on a `{}` rule", r.numerics),
    };
    match r.numerics.as_str() {
        "bit_identical" => {
            if r.lever.is_some() {
                return Err(stray("lever"));
            }
            match &r.microtest {
                Some(m) if !m.trim().is_empty() => Ok(Numerics::BitIdentical {
                    microtest: m.clone(),
                }),
                _ => Err(RuleError::MissingMicrotest(r.id.clone())),
            }
        }
        "reference" => {
            if r.lever.is_some() {
                return Err(stray("lever"));
            }
            if r.microtest.is_some() {
                return Err(stray("microtest"));
            }
            Ok(Numerics::Reference)
        }
        "differs" => {
            if r.microtest.is_some() {
                return Err(stray("microtest"));
            }
            match &r.lever {
                Some(l) if !l.trim().is_empty() => Ok(Numerics::Differs { lever: l.clone() }),
                _ => Err(RuleError::MissingLever(r.id.clone())),
            }
        }
        other => Err(RuleError::Numerics {
            rule: r.id.clone(),
            detail: format!("unknown numerics class `{other}`"),
        }),
    }
}

fn pattern_op(rule: &str, p: &PatternFile) -> Result<PatternOp, RuleError> {
    let err = |detail: String| RuleError::Op {
        rule: rule.to_string(),
        detail,
    };
    let fmt = |s: &Option<String>| -> Result<Option<Format>, RuleError> {
        s.as_deref()
            .map(Format::parse)
            .transpose()
            .map_err(|e| err(e.to_string()))
    };
    let mut roles = BTreeSet::new();
    for r in &p.roles {
        roles
            .insert(LinearRole::parse(r).ok_or_else(|| err(format!("unknown linear role `{r}`")))?);
    }
    let role = match (p.role.as_deref(), p.roles.first()) {
        (Some(_), Some(_)) => return Err(err("give `role` or `roles`, not both".into())),
        (Some(r), None) => Some(r),
        (None, first) => first.map(String::as_str),
    };
    let op = OpKind::parse(&p.op, role, fmt(&p.format)?).map_err(|e| err(e.to_string()))?;
    let layer_kind = p
        .layer_kind
        .as_deref()
        .map(|k| LayerKind::parse(k).ok_or_else(|| err(format!("unknown layer kind `{k}`"))))
        .transpose()?;
    Ok(PatternOp {
        op,
        roles,
        layer_kind,
        local: p.local.clone(),
        weight: fmt(&p.weight)?,
        input: fmt(&p.input)?,
        writes: fmt(&p.writes)?,
        keep: p.keep,
        sibling: p.sibling,
    })
}
