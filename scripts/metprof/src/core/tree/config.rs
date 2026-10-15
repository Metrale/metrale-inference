// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the `[tree]` section of the family table file (pure).
//!
//! ```toml
//! [tree]
//! coverage_min = 0.95                 # required: attributed share of kernel time
//!
//! [tree.circuit_names]                # optional: plan kernel -> profiler short name,
//! "module::func" = "cupti_short_name" # only where the two differ
//!
//! [tree.model.<name>]                 # optional: structure over the families
//! fuses = { family = ["absorbed", ...] }
//! [[tree.model.<name>.node]]
//! path = "a/b"                        # interior node, created where first used
//! families = ["^regex", ...]          # families placed under it (Rust regex syntax)
//! fuses = ["..."]                     # optional
//! ```
//!
//! The family table stays the only place a kernel name is matched; an overlay
//! adds structure over family names and nothing else. A pattern that matches
//! no declared family, a family claimed by two nodes, an unknown family in
//! `fuses` and an unknown key are all errors. No key has a default.

use std::collections::BTreeMap;

use indexmap::IndexMap;
use regex::Regex;

use crate::core::error::Result;
use crate::core::families::{FamilyTable, check_keys};
use crate::core::repr::{quote, toml_value};
use crate::fail;

#[derive(Debug, Clone)]
pub struct OverlayNode {
    pub path: Vec<String>,
    pub families: Vec<Regex>,
    pub fuses: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Overlay {
    pub nodes: Vec<OverlayNode>,
    pub fuses: IndexMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct TreeConfig {
    pub coverage_min: f64,
    /// `module::func` -> the profiler's short name.
    pub circuit_names: BTreeMap<String, String>,
    pub models: BTreeMap<String, Overlay>,
}

/// An overlay checked against a family table: the node each claimed family
/// sits under, and the declared fuses of nodes (by path) and families.
#[derive(Debug, Clone, Default)]
pub struct Resolved {
    pub claim: BTreeMap<String, Vec<String>>,
    pub node_fuses: BTreeMap<String, Vec<String>>,
    pub family_fuses: BTreeMap<String, Vec<String>>,
}

fn str_list(place: &str, v: &toml::Value, non_empty: bool) -> Result<Vec<String>> {
    let items: Option<Vec<String>> = v.as_array().and_then(|a| {
        a.iter()
            .map(|x| x.as_str().filter(|s| !s.is_empty()).map(str::to_string))
            .collect()
    });
    match items {
        Some(l) if !(non_empty && l.is_empty()) => Ok(l),
        _ => fail!(
            "{place}: must be a list of non-empty strings, got {}",
            toml_value(v)
        ),
    }
}

fn table<'t>(place: &str, v: &'t toml::Value) -> Result<&'t toml::Table> {
    match v.as_table() {
        Some(t) => Ok(t),
        None => fail!("{place}: must be a table"),
    }
}

pub fn tree_config(doc: &toml::Table) -> Result<TreeConfig> {
    let Some(raw) = doc.get("tree") else {
        fail!("families: [tree] is required by the kernel tree (coverage_min has no default)");
    };
    let t = table("families [tree]", raw)?;
    check_keys(
        "families [tree]",
        t,
        &["coverage_min"],
        &["circuit_names", "model"],
    )?;
    let cov = match &t["coverage_min"] {
        toml::Value::Integer(i) => Some(*i as f64),
        toml::Value::Float(f) => Some(*f),
        _ => None,
    };
    let Some(coverage_min) = cov.filter(|c| *c > 0.0 && *c <= 1.0) else {
        fail!(
            "families [tree]: coverage_min must be a number in (0, 1], got {}",
            toml_value(&t["coverage_min"])
        );
    };
    let mut circuit_names = BTreeMap::new();
    if let Some(v) = t.get("circuit_names") {
        for (k, name) in table("families [tree.circuit_names]", v)? {
            let ok = k
                .split_once("::")
                .is_some_and(|(m, f)| !m.is_empty() && !f.is_empty());
            let Some(name) = name.as_str().filter(|s| !s.is_empty() && ok) else {
                fail!(
                    "families [tree.circuit_names]: {} must be \"module::func\" = \"short name\"",
                    quote(k)
                );
            };
            circuit_names.insert(k.clone(), name.to_string());
        }
    }
    let mut models = BTreeMap::new();
    if let Some(v) = t.get("model") {
        for (name, m) in table("families [tree.model]", v)? {
            models.insert(
                name.clone(),
                overlay(&format!("families [tree.model.{name}]"), m)?,
            );
        }
    }
    Ok(TreeConfig {
        coverage_min,
        circuit_names,
        models,
    })
}

fn overlay(place: &str, v: &toml::Value) -> Result<Overlay> {
    let m = table(place, v)?;
    check_keys(place, m, &[], &["node", "fuses"])?;
    let mut out = Overlay::default();
    if let Some(f) = m.get("fuses") {
        for (fam, list) in table(&format!("{place} fuses"), f)? {
            out.fuses.insert(
                fam.clone(),
                str_list(&format!("{place} fuses.{fam}"), list, true)?,
            );
        }
    }
    let nodes = match m.get("node") {
        None => &Vec::new(),
        Some(n) => match n.as_array() {
            Some(a) => a,
            None => fail!("{place}: node must be an array of tables ([[...node]])"),
        },
    };
    for (i, n) in nodes.iter().enumerate() {
        let at = format!("{place} node #{}", i + 1);
        let n = table(&at, n)?;
        check_keys(&at, n, &["path", "families"], &["fuses"])?;
        let Some(path) = n["path"].as_str() else {
            fail!("{at}: path must be a string");
        };
        let segs: Vec<String> = path.split('/').map(str::to_string).collect();
        if segs.iter().any(String::is_empty) {
            fail!("{at}: path {} has an empty segment", quote(path));
        }
        let mut families = Vec::new();
        for p in str_list(&format!("{at} families"), &n["families"], true)? {
            match Regex::new(&p) {
                Ok(re) => families.push(re),
                Err(e) => fail!(
                    "{at}: families pattern {} is not a valid regex: {e}",
                    quote(&p)
                ),
            }
        }
        let fuses = match n.get("fuses") {
            None => Vec::new(),
            Some(f) => str_list(&format!("{at} fuses"), f, true)?,
        };
        out.nodes.push(OverlayNode {
            path: segs,
            families,
            fuses,
        });
    }
    Ok(out)
}

impl Overlay {
    /// Check the overlay against `table` and resolve which node each family
    /// sits under.
    pub fn resolve(&self, table: &FamilyTable) -> Result<Resolved> {
        let fams = table.families();
        let mut r = Resolved::default();
        for (i, n) in self.nodes.iter().enumerate() {
            let path = n.path.join("/");
            for re in &n.families {
                let hits: Vec<&&str> = fams.iter().filter(|f| re.is_match(f)).collect();
                if hits.is_empty() {
                    fail!(
                        "[tree] overlay node #{} ({}): pattern {} matches no declared family",
                        i + 1,
                        quote(&path),
                        quote(re.as_str())
                    );
                }
                for f in hits {
                    if let Some(prev) = r.claim.get(*f).filter(|p| p.join("/") != path) {
                        fail!(
                            "[tree] overlay: family {} is claimed by both {} and {}",
                            quote(f),
                            quote(&prev.join("/")),
                            quote(&path)
                        );
                    }
                    r.claim.insert(f.to_string(), n.path.clone());
                }
            }
            if !n.fuses.is_empty() {
                r.node_fuses.insert(path, n.fuses.clone());
            }
        }
        for (f, list) in &self.fuses {
            if !fams.contains(&f.as_str()) {
                fail!(
                    "[tree] overlay fuses: {} is not a declared family",
                    quote(f)
                );
            }
            r.family_fuses.insert(f.clone(), list.clone());
        }
        Ok(r)
    }
}
