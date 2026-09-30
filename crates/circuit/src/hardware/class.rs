// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: A kernel class (`kernels/<class>/`) as the planner sees it: its inheritance chain
//! and build defines from `HARDWARE.toml`, its fusion rules (`FUSIONS.toml`, inherited and
//! overridden along the chain) and its kernel families (`KERNEL_FAMILIES.toml`, with points
//! rediscovered from the class's own resolved sources and evidence kept per class).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Rules: a class's `common/FUSIONS.toml` either stands alone or names `inherits = "<class>"`,
//!   in which case its rules replace the parent's rules of the same id, add new ones, and
//!   `remove` drops parent rules by id. A class without the file takes its HARDWARE.toml
//!   parent's rules; a class with neither has no rules ([`ClassRules::None`]), which the planner
//!   reports, never fills in.
//! - Families: evidence is a measurement on one class. Families inherited from another class's
//!   manifest keep their points (re-resolved against this class's sources) and lose their
//!   evidence, so a report on this class says "unmeasured", never "optimized".

use std::collections::{BTreeMap, BTreeSet};

use super::HwError;
use super::sources::ClassSources;
use crate::rules::{Rule, parse_rules};
use crate::venn::families::{Families, Point, parse_families};
use crate::venn::repo::Repo;

/// 2026-09-30: What one `HARDWARE.toml` says about the class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassInfo {
    /// 2026-09-30: Directory name.
    pub name: String,
    /// 2026-09-30: `[hardware] arch`.
    pub arch: String,
    /// 2026-09-30: `[hardware] inherits`.
    pub inherits: Option<String>,
    /// 2026-09-30: Macros `[build] extra_nvcc_flags` defines (`-DNAME`).
    pub defines: BTreeSet<String>,
    /// 2026-09-30: `[defaults]`, values as text.
    pub defaults: BTreeMap<String, String>,
}

fn toml_err(rel: &str) -> impl Fn(toml::de::Error) -> HwError + '_ {
    move |e| HwError::Class(format!("{rel}: {e}"))
}

/// 2026-09-30: Parse `kernels/<name>/HARDWARE.toml`.
pub fn parse_class(name: &str, text: &str) -> Result<ClassInfo, HwError> {
    let rel = format!("kernels/{name}/HARDWARE.toml");
    let t: toml::Table = toml::from_str(text).map_err(toml_err(&rel))?;
    let hw = t
        .get("hardware")
        .and_then(|v| v.as_table())
        .ok_or_else(|| HwError::Class(format!("{rel}: no [hardware]")))?;
    let arch = hw
        .get("arch")
        .and_then(|v| v.as_str())
        .ok_or_else(|| HwError::Class(format!("{rel}: no [hardware] arch")))?
        .to_string();
    let inherits = hw
        .get("inherits")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let defines = t
        .get("build")
        .and_then(|b| b.get("extra_nvcc_flags"))
        .and_then(|f| f.as_array())
        .into_iter()
        .flatten()
        .filter_map(|f| f.as_str()?.strip_prefix("-D"))
        .map(|d| d.split('=').next().unwrap_or(d).to_string())
        .collect();
    let defaults = t
        .get("defaults")
        .and_then(|d| d.as_table())
        .into_iter()
        .flatten()
        .map(|(k, v)| {
            let text = v.as_str().map_or_else(|| v.to_string(), str::to_string);
            (k.clone(), text)
        })
        .collect();
    Ok(ClassInfo {
        name: name.to_string(),
        arch,
        inherits,
        defines,
        defaults,
    })
}

/// 2026-09-30: The class and its ancestors, own class first.
pub fn chain(repo: &dyn Repo, class: &str) -> Result<Vec<ClassInfo>, HwError> {
    let mut out: Vec<ClassInfo> = Vec::new();
    let mut next = Some(class.to_string());
    while let Some(name) = next {
        if out.iter().any(|c| c.name == name) {
            return Err(HwError::Class(format!("inheritance cycle through `{name}`")));
        }
        let text = repo
            .read(&format!("kernels/{name}/HARDWARE.toml"))
            .map_err(|e| HwError::Class(format!("class `{name}`: {e}")))?;
        let info = parse_class(&name, &text)?;
        next = info.inherits.clone();
        out.push(info);
    }
    Ok(out)
}

/// 2026-09-30: The classes a plan on `class` reads rules and families from: its HARDWARE.toml
/// chain, continued through the `inherits` of the last class's FUSIONS.toml (a class that does
/// not inherit gb10's sources may still declare that it plans with gb10's rules).
pub fn planning_chain(repo: &dyn Repo, class: &str) -> Result<Vec<ClassInfo>, HwError> {
    let mut out = chain(repo, class)?;
    loop {
        let last = out.last().map(|c| c.name.clone()).unwrap_or_default();
        let Ok(text) = repo.read(&fusions_rel(&last)) else {
            return Ok(out);
        };
        let t: toml::Table = toml::from_str(&text).map_err(toml_err(&fusions_rel(&last)))?;
        let Some(next) = t.get("inherits").and_then(|v| v.as_str()) else {
            return Ok(out);
        };
        if out.iter().any(|c| c.name == next) {
            return Ok(out);
        }
        out.extend(chain(repo, next)?);
    }
}

/// 2026-09-30: A class's resolved rule set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassRules {
    /// 2026-09-30: The rules, and the files they were resolved from, base first.
    Rules {
        /// 2026-09-30: Rules after every override.
        rules: Vec<Rule>,
        /// 2026-09-30: `kernels/<class>/common/FUSIONS.toml` paths read, base first.
        files: Vec<String>,
    },
    /// 2026-09-30: Neither the class nor any ancestor declares a FUSIONS.toml.
    None,
}

fn fusions_rel(class: &str) -> String {
    format!("kernels/{class}/common/FUSIONS.toml")
}

/// 2026-09-30: The rules `class` plans with.
pub fn class_rules(repo: &dyn Repo, class: &str) -> Result<ClassRules, HwError> {
    rules_of(repo, class, &mut Vec::new())
}

fn rules_of(repo: &dyn Repo, class: &str, seen: &mut Vec<String>) -> Result<ClassRules, HwError> {
    if seen.iter().any(|c| c == class) {
        return Err(HwError::Class(format!(
            "FUSIONS inheritance cycle through `{class}`"
        )));
    }
    seen.push(class.to_string());
    let rel = fusions_rel(class);
    let Ok(text) = repo.read(&rel) else {
        let info = chain(repo, class)?;
        return match info[0].inherits.as_deref() {
            Some(parent) => rules_of(repo, parent, seen),
            None => Ok(ClassRules::None),
        };
    };
    let mut table: toml::Table = toml::from_str(&text).map_err(toml_err(&rel))?;
    let inherits = table.remove("inherits");
    let remove = table.remove("remove");
    table
        .entry("rule")
        .or_insert_with(|| toml::Value::Array(Vec::new()));
    let own_text = toml::to_string(&table).map_err(|e| HwError::Class(format!("{rel}: {e}")))?;
    let own = parse_rules(&own_text).map_err(|e| HwError::Class(format!("{rel}: {e}")))?;
    let Some(parent) = inherits else {
        if remove.is_some() {
            return Err(HwError::Class(format!(
                "{rel}: `remove` needs `inherits`: a standalone rule set has nothing to remove"
            )));
        }
        return Ok(ClassRules::Rules {
            rules: own,
            files: vec![rel],
        });
    };
    let parent = parent
        .as_str()
        .ok_or_else(|| HwError::Class(format!("{rel}: `inherits` is not a class name")))?;
    let ClassRules::Rules {
        rules: mut base,
        mut files,
    } = rules_of(repo, parent, seen)?
    else {
        return Err(HwError::Class(format!(
            "{rel}: inherits `{parent}`, which has no rules"
        )));
    };
    let drop: Vec<String> = remove
        .as_ref()
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .map(|v| v.as_str().map(str::to_string))
        .collect::<Option<_>>()
        .ok_or_else(|| HwError::Class(format!("{rel}: `remove` is not a list of rule ids")))?;
    for id in &drop {
        let before = base.len();
        base.retain(|r| &r.id != id);
        if base.len() == before {
            return Err(HwError::Class(format!(
                "{rel}: removes `{id}`, which `{parent}` does not define"
            )));
        }
    }
    for r in own {
        match base.iter_mut().find(|b| b.id == r.id) {
            Some(slot) => *slot = r,
            None => base.push(r),
        }
    }
    files.push(rel);
    Ok(ClassRules::Rules { rules: base, files })
}

/// 2026-09-30: The kernel families `class` is classified against: the nearest
/// `KERNEL_FAMILIES.toml` along the chain as the base (evidence dropped unless it is the class's
/// own), the class's own manifest overriding families by id, and every point re-resolved against
/// the class's sources ([`class_points`]).
pub fn class_families(
    repo: &dyn Repo,
    chain: &[ClassInfo],
    sources: &ClassSources,
) -> Result<Families, HwError> {
    let rel = |c: &str| format!("kernels/{c}/common/KERNEL_FAMILIES.toml");
    let mut found: Vec<(String, Families)> = Vec::new();
    for c in chain {
        if let Ok(text) = repo.read(&rel(&c.name)) {
            let f = parse_families(&text).map_err(|e| HwError::Class(e.to_string()))?;
            found.push((c.name.clone(), f));
        }
    }
    let own = chain
        .first()
        .map(|c| c.name.clone())
        .ok_or_else(|| HwError::Class("empty class chain".into()))?;
    let (base_class, mut out) = found.pop().ok_or_else(|| {
        HwError::Class(format!(
            "no KERNEL_FAMILIES.toml along `{own}`'s planning chain: nothing to classify against"
        ))
    })?;
    if base_class != own {
        for f in &mut out.families {
            f.evidence.clear();
        }
    }
    while let Some((class, overlay)) = found.pop() {
        for f in overlay.families {
            let mut f = f;
            if class != own {
                f.evidence.clear();
            }
            match out.families.iter_mut().find(|g| g.id == f.id) {
                Some(slot) => *slot = f,
                None => out.families.push(f),
            }
        }
        out.legacy.extend(overlay.legacy);
    }
    out.hardware = own;
    for f in &mut out.families {
        f.points = class_points(&f.points, chain, sources);
    }
    Ok(out)
}

/// 2026-09-30: The points of a family that `sources` realises: a point whose kernel files the
/// class compiles as they are, or through its own copy of the same file name (a shadow), with
/// the files rewritten to what the class compiles. Host-side files (`crates/...`) are shared by
/// every class. A point one of whose kernel files the class does not compile is dropped.
pub fn class_points(points: &[Point], chain: &[ClassInfo], sources: &ClassSources) -> Vec<Point> {
    let by_name: BTreeMap<&str, &str> = sources
        .files
        .iter()
        .map(|p| (file_name(p), p.as_str()))
        .collect();
    let compiled: BTreeSet<&str> = sources.files.iter().map(String::as_str).collect();
    let in_chain = |p: &str| chain.iter().any(|c| p.starts_with(&format!("kernels/{}/", c.name)));
    points
        .iter()
        .filter_map(|p| {
            let files = p
                .files
                .iter()
                .map(|f| {
                    if !f.starts_with("kernels/") || compiled.contains(f.as_str()) {
                        return Some(f.clone());
                    }
                    let own = by_name.get(file_name(f)).copied()?;
                    in_chain(own).then(|| own.to_string())
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Point {
                values: p.values.clone(),
                how: p.how,
                files,
            })
        })
        .collect()
}

fn file_name(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}
