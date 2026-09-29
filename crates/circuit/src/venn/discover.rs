// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Rediscover each family's instantiated points from the kernel sources and report
//! where the manifest and the sources disagree (drift).
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - Pure: the caller lists the repo's kernel files and supplies the text of every file a
//!   `macro` rule reads ([`macro_files`]); a file a rule needs and the caller did not supply is
//!   itself drift, never skipped.
//! - Drift is two-way: a point the sources realise that the manifest lacks, and a point the
//!   manifest declares at a file whose rule does not find it there.

use std::collections::{BTreeMap, BTreeSet};

use super::families::{Discover, Families, Values};
use crate::precision::glob;

/// 2026-09-29: The kernel sources discovery reads.
#[derive(Debug, Clone, Default)]
pub struct KernelSources {
    /// 2026-09-29: Every repo-relative file under `kernels/` (the file rules glob these).
    pub paths: BTreeSet<String>,
    /// 2026-09-29: Path to text, for the files [`macro_files`] names.
    pub texts: BTreeMap<String, String>,
}

/// 2026-09-29: The files whose text a `macro` rule reads.
pub fn macro_files(families: &Families) -> BTreeSet<String> {
    families
        .families
        .iter()
        .flat_map(|f| f.discover.iter())
        .filter_map(|d| match d {
            Discover::Macro { file, .. } => Some(file.clone()),
            Discover::File { .. } => None,
        })
        .collect()
}

/// 2026-09-29: One point a rule found: its family, the values the rule fixes, and the file.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Found {
    /// 2026-09-29: Family id.
    pub family: String,
    /// 2026-09-29: The values the rule fixes (a subset of the family's point parameters).
    pub values: Values,
    /// 2026-09-29: Repo-relative file.
    pub file: String,
}

/// 2026-09-29: Every point the discovery rules find in `src`, and the problems met on the way
/// (a macro file not supplied, an argument missing, an argument no `map` entry spells).
pub fn discover(families: &Families, src: &KernelSources) -> (BTreeSet<Found>, Vec<String>) {
    let mut found = BTreeSet::new();
    let mut problems = Vec::new();
    for f in &families.families {
        for rule in &f.discover {
            match rule {
                Discover::File { glob: g, values } => {
                    for p in src.paths.iter().filter(|p| glob(g, p)) {
                        found.insert(Found {
                            family: f.id.clone(),
                            values: values.clone(),
                            file: p.clone(),
                        });
                    }
                }
                Discover::Macro {
                    file,
                    name,
                    args,
                    map,
                } => {
                    let Some(text) = src.texts.get(file) else {
                        problems.push(format!("{}: macro file {file} was not supplied", f.id));
                        continue;
                    };
                    for call in invocations(text, name) {
                        match point_of(&call, args, map) {
                            Ok(values) => {
                                found.insert(Found {
                                    family: f.id.clone(),
                                    values,
                                    file: file.clone(),
                                });
                            }
                            Err(e) => problems.push(format!("{}: {file}: {name}: {e}", f.id)),
                        }
                    }
                }
            }
        }
    }
    (found, problems)
}

/// 2026-09-29: Every disagreement between the manifest's points and the sources; empty when
/// they agree.
pub fn drift(families: &Families, src: &KernelSources) -> Vec<String> {
    let (found, mut problems) = discover(families, src);
    for f in &families.families {
        for p in &f.points {
            for file in &p.files {
                if !src.paths.contains(file) {
                    problems.push(format!(
                        "{}: point {:?} names {file}, which does not exist",
                        f.id, p.values
                    ));
                }
            }
        }
        for d in found.iter().filter(|d| d.family == f.id) {
            let declared = f.points.iter().any(|p| {
                p.files.contains(&d.file)
                    && d.values.iter().all(|(k, v)| p.values.get(k) == Some(v))
            });
            if !declared {
                problems.push(format!(
                    "{}: {} realises {:?}, which no declared point lists at that file",
                    f.id, d.file, d.values
                ));
            }
        }
        // 2026-09-29: A declared point at a file some rule covers must be found there.
        for p in &f.points {
            for file in &p.files {
                let covered = f.discover.iter().any(|r| match r {
                    Discover::File { glob: g, .. } => glob(g, file),
                    Discover::Macro { file: m, .. } => m == file,
                });
                let seen = found.iter().any(|d| {
                    d.family == f.id
                        && &d.file == file
                        && d.values.iter().all(|(k, v)| p.values.get(k) == Some(v))
                });
                if covered && !seen {
                    problems.push(format!(
                        "{}: point {:?} is declared at {file}, but no rule finds it there",
                        f.id, p.values
                    ));
                }
            }
        }
    }
    problems
}

/// 2026-09-29: The argument lists of every line that starts (after indentation) with
/// `name(`: one per invocation, split at top-level commas and trimmed.
fn invocations(text: &str, name: &str) -> Vec<Vec<String>> {
    let open = format!("{name}(");
    text.lines()
        .filter_map(|l| l.trim_start().strip_prefix(open.as_str()))
        .map(|rest| {
            let mut depth = 0usize;
            let mut args = vec![String::new()];
            for c in rest.chars() {
                match c {
                    '(' | '<' => depth += 1,
                    ')' if depth == 0 => break,
                    ')' | '>' => depth = depth.saturating_sub(1),
                    ',' if depth == 0 => {
                        args.push(String::new());
                        continue;
                    }
                    _ => {}
                }
                if let Some(last) = args.last_mut() {
                    last.push(c);
                }
            }
            args.into_iter().map(|a| a.trim().to_string()).collect()
        })
        .collect()
}

fn point_of(
    call: &[String],
    args: &BTreeMap<String, usize>,
    map: &BTreeMap<String, BTreeMap<String, String>>,
) -> Result<Values, String> {
    let mut values = Values::new();
    for (param, &i) in args {
        let raw = call
            .get(i)
            .ok_or_else(|| format!("argument {i} missing in {call:?}"))?;
        let v = match map.get(param) {
            Some(m) => m
                .get(raw)
                .cloned()
                .ok_or_else(|| format!("argument `{raw}` of `{param}` has no map entry"))?,
            None => raw.clone(),
        };
        values.insert(param.clone(), v);
    }
    Ok(values)
}

#[cfg(test)]
#[path = "discover_tests.rs"]
mod discover_tests;
