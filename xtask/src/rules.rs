// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The workspace dependency convention, as three rules over parsed manifests.
//! The root `[workspace.dependencies]` declares every dependency once, as
//! `name = { version = "...", default-features = false }` (path crates likewise). A member names
//! it as `name = { workspace = true, features = [...] }` and nothing else.

use crate::manifest::{self, Manifest, Workspace};
use std::fmt;
use toml::{Table, Value};

/// 2026-10-17: Which part of the convention a violation breaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rule {
    /// 2026-10-17: (a) a root `[workspace.dependencies]` entry that enables features: it lacks
    /// `default-features = false`, or names a `features` list every member would inherit.
    RootDefaultFeatures,
    /// 2026-10-17: (b) a member dependency on a crate the root declares that carries its own
    /// version or source, does not inherit it with `workspace = true`, or names the `default`
    /// feature and so turns the crate's defaults back on.
    MemberBypassesWorkspace,
    /// 2026-10-17: (c) a member dependency on a crate the root does not declare.
    MemberUndeclared,
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Rule::RootDefaultFeatures => "rule a: root entry enables features",
            Rule::MemberBypassesWorkspace => "rule b: member bypasses the workspace declaration",
            Rule::MemberUndeclared => "rule c: member depends on a crate the root does not declare",
        })
    }
}

/// 2026-10-17: One refused dependency: the manifest, the table and the key it sits under.
#[derive(Debug, PartialEq, Eq)]
pub struct Violation {
    pub rule: Rule,
    pub file: String,
    pub table: String,
    pub dep: String,
    pub detail: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] `{}`: {} ({})",
            self.file, self.table, self.dep, self.detail, self.rule
        )
    }
}

/// 2026-10-17: The only keys a member may write beside `workspace = true`. Version, source
/// (`path`, `git`, `registry`, ...), `package` and `default-features` all belong to the root.
const MEMBER_KEYS: [&str; 3] = ["workspace", "features", "optional"];

/// 2026-10-17: Every violation in `ws`: the root's declarations first, then each manifest's
/// dependency tables (the root's own, if it has any, then the members' in `members` order).
pub fn check(ws: &Workspace) -> Result<Vec<Violation>, String> {
    let declared = manifest::workspace_deps(&ws.root)?;
    let mut out = Vec::new();
    for (dep, spec) in declared {
        if let Some(detail) = root_entry_problem(spec) {
            out.push(Violation {
                rule: Rule::RootDefaultFeatures,
                file: ws.root.file.clone(),
                table: "workspace.dependencies".to_owned(),
                dep: dep.clone(),
                detail,
            });
        }
    }
    for m in std::iter::once(&ws.root).chain(&ws.members) {
        member_violations(m, declared, &mut out)?;
    }
    Ok(out)
}

fn member_violations(
    m: &Manifest,
    declared: &Table,
    out: &mut Vec<Violation>,
) -> Result<(), String> {
    for table in manifest::dep_tables(m)? {
        for (dep, spec) in table.entries {
            if let Some((rule, detail)) = member_entry_problem(dep, spec, declared) {
                out.push(Violation {
                    rule,
                    file: m.file.clone(),
                    table: table.header.clone(),
                    dep: dep.clone(),
                    detail,
                });
            }
        }
    }
    Ok(())
}

fn root_entry_problem(spec: &Value) -> Option<String> {
    match spec {
        Value::String(version) => Some(format!(
            "is the bare version string \"{version}\"; \
             write {{ version = \"{version}\", default-features = false }}"
        )),
        Value::Table(t) => match t.get("default-features") {
            Some(Value::Boolean(false)) if t.contains_key("features") => {
                Some("names features; the root declares none, each member names its own".to_owned())
            }
            Some(Value::Boolean(false)) => None,
            Some(Value::Boolean(true)) => {
                Some("sets default-features = true; it must be false".to_owned())
            }
            Some(other) => Some(format!(
                "sets default-features to a {}; it must be the boolean false",
                other.type_str()
            )),
            None if t.contains_key("default_features") => {
                Some("spells it default_features; write default-features = false".to_owned())
            }
            None => Some("has no default-features = false".to_owned()),
        },
        other => Some(format!(
            "is a {}, not a version string or a table",
            other.type_str()
        )),
    }
}

fn member_entry_problem(dep: &str, spec: &Value, declared: &Table) -> Option<(Rule, String)> {
    if !declared.contains_key(dep) {
        return Some((
            Rule::MemberUndeclared,
            "is not in the root [workspace.dependencies]; declare it there with \
             default-features = false and name it here with workspace = true"
                .to_owned(),
        ));
    }
    let bypass = |detail: String| Some((Rule::MemberBypassesWorkspace, detail));
    match spec {
        Value::String(version) => bypass(format!(
            "declares its own version \"{version}\"; write {{ workspace = true }} instead"
        )),
        Value::Table(t) if t.get("workspace") != Some(&Value::Boolean(true)) => {
            bypass(match t.get("version").and_then(Value::as_str) {
                Some(version) => format!(
                    "declares its own version \"{version}\" without workspace = true; \
                     the root declares it"
                ),
                None => "has no workspace = true; the root declares it".to_owned(),
            })
        }
        Value::Table(t) => {
            let extra: Vec<&str> = t
                .keys()
                .map(String::as_str)
                .filter(|k| !MEMBER_KEYS.contains(k))
                .collect();
            if !extra.is_empty() {
                bypass(format!(
                    "sets `{}` beside workspace = true; a member may write only `{}`",
                    extra.join("`, `"),
                    MEMBER_KEYS.join("`, `")
                ))
            } else if names_default(t) {
                bypass("names the `default` feature; list the features it uses instead".to_owned())
            } else {
                None
            }
        }
        other => bypass(format!(
            "is a {}, not a version string or a table",
            other.type_str()
        )),
    }
}

/// 2026-10-17: Whether a member entry's `features` list names `default`, which restores every
/// default feature the root's `default-features = false` turned off.
fn names_default(t: &Table) -> bool {
    t.get("features")
        .and_then(Value::as_array)
        .is_some_and(|features| features.iter().any(|f| f.as_str() == Some("default")))
}
