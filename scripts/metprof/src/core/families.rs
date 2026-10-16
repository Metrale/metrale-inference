// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the family table (SSOT): validation and kernel classification.
//!
//! The table is an ordered list of rules; the first rule that matches a kernel
//! names its family. A rule may carry a call-site condition
//! (`next_on_stream_family`): it then matches only when the next kernel on the
//! same stream was classified into one of the listed families. This is the
//! launch-site proxy used when a capture has no NVTX ranges, e.g. to tell the
//! router GEMM (and its split-K reduce) from the dense GEMMs that run the same
//! vendor kernel. The references must form a DAG and may not name the rule's
//! own family, so a call-site chain is bounded by the table, never by the
//! capture.
//!
//! No key has a default. A missing or unknown key is a hard error.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use regex::Regex;

use crate::core::error::Result;
use crate::core::model::Kernel;
use crate::core::repr::{float, quote, quote_list, toml_value};
use crate::core::union::Class;
use crate::fail;

pub const SCHEMA: i64 = 1;
const RULE_REQUIRED: [&str; 4] = ["family", "group", "class", "name_regex"];
const RULE_OPTIONAL: [&str; 1] = ["next_on_stream_family"];
const TOP_REQUIRED: [&str; 4] = ["schema", "coverage_min", "api", "rule"];
const API_REQUIRED: [&str; 1] = ["host_sync_regex"];

#[derive(Debug, Clone)]
pub struct Rule {
    pub family: String,
    pub group: String,
    pub class: Class,
    pub pattern: Regex,
    pub next_on_stream: Option<BTreeSet<String>>,
}

impl Rule {
    pub fn matches(&self, name: &str, next_family: Option<&str>) -> bool {
        if !self.pattern.is_match(name) {
            return false;
        }
        match &self.next_on_stream {
            None => true,
            Some(set) => next_family.is_some_and(|f| set.contains(f)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FamilyTable {
    pub rules: Vec<Rule>,
    pub coverage_min: f64,
    pub host_sync: Regex,
}

impl FamilyTable {
    fn rule_of(&self, family: &str) -> Result<&Rule> {
        match self.rules.iter().find(|r| r.family == family) {
            Some(r) => Ok(r),
            None => fail!("unknown family {}", quote(family)),
        }
    }

    pub fn class(&self, family: &str) -> Result<Class> {
        Ok(self.rule_of(family)?.class)
    }

    pub fn group(&self, family: &str) -> Result<&str> {
        Ok(&self.rule_of(family)?.group)
    }

    /// Declared families in first-seen order.
    pub fn families(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for r in &self.rules {
            if !out.contains(&r.family.as_str()) {
                out.push(&r.family);
            }
        }
        out
    }
}

pub fn parse_class(v: &toml::Value) -> Option<Class> {
    let s = v.as_str()?;
    Class::ALL.into_iter().find(|c| c.as_str() == s)
}

fn compile(place: &str, key: &str, value: &toml::Value) -> Result<Regex> {
    let Some(text) = value.as_str().filter(|s| !s.is_empty()) else {
        fail!("{place}: {key} must be a non-empty string");
    };
    match Regex::new(text) {
        Ok(re) => Ok(re),
        Err(e) => fail!("{place}: {key} {} is not a valid regex: {e}", quote(text)),
    }
}

pub fn check_keys(
    place: &str,
    got: &toml::Table,
    required: &[&str],
    optional: &[&str],
) -> Result<()> {
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|k| !got.contains_key(*k))
        .collect();
    let unknown: Vec<&str> = got
        .keys()
        .map(String::as_str)
        .filter(|k| !required.contains(k) && !optional.contains(k))
        .collect();
    if !missing.is_empty() {
        fail!("{place}: missing required key(s) {}", quote_list(&missing));
    }
    if !unknown.is_empty() {
        fail!("{place}: unknown key(s) {}", quote_list(&unknown));
    }
    Ok(())
}

fn non_empty_str<'a>(place: &str, r: &'a toml::Table, key: &str) -> Result<&'a str> {
    match r[key].as_str().filter(|s| !s.is_empty()) {
        Some(s) => Ok(s),
        None => fail!("{place}: {key} must be a non-empty string"),
    }
}

fn parse_rule(
    place: &str,
    raw: &toml::Value,
    seen: &mut BTreeMap<String, (String, Class)>,
) -> Result<Rule> {
    let Some(r) = raw.as_table() else {
        fail!("{place}: must be a table");
    };
    check_keys(place, r, &RULE_REQUIRED, &RULE_OPTIONAL)?;
    let family = non_empty_str(place, r, "family")?;
    let group = non_empty_str(place, r, "group")?;
    let Some(class) = parse_class(&r["class"]) else {
        let names: Vec<&str> = Class::ALL.iter().map(|c| c.as_str()).collect();
        fail!(
            "{place}: class must be one of {}, got {}",
            quote_list(&names),
            toml_value(&r["class"])
        );
    };
    let decl = (group.to_string(), class);
    let prior = seen
        .entry(family.to_string())
        .or_insert_with(|| decl.clone())
        .clone();
    if prior != decl {
        fail!(
            "{place}: family {} is declared with group/class ({}, {}) by an earlier rule and ({}, {}) here",
            quote(family),
            quote(&prior.0),
            quote(prior.1.as_str()),
            quote(group),
            quote(class.as_str())
        );
    }
    let next_on_stream = match r.get("next_on_stream_family") {
        None => None,
        Some(v) => {
            let list: Option<Vec<&str>> = v.as_array().filter(|a| !a.is_empty()).and_then(|a| {
                a.iter()
                    .map(|x| x.as_str().filter(|s| !s.is_empty()))
                    .collect()
            });
            match list {
                Some(names) => Some(names.into_iter().map(str::to_string).collect()),
                None => {
                    fail!("{place}: next_on_stream_family must be a non-empty list of family names")
                }
            }
        }
    };
    Ok(Rule {
        family: family.to_string(),
        group: group.to_string(),
        class,
        pattern: compile(place, "name_regex", &r["name_regex"])?,
        next_on_stream,
    })
}

pub fn table_from_toml(doc: &toml::Table) -> Result<FamilyTable> {
    check_keys("families", doc, &TOP_REQUIRED, &["tree", "slope"])?;
    if doc["schema"].as_integer() != Some(SCHEMA) {
        fail!(
            "families: schema {} is not supported (expected {SCHEMA})",
            toml_value(&doc["schema"])
        );
    }
    let cov = match &doc["coverage_min"] {
        toml::Value::Integer(i) => Some(*i as f64),
        toml::Value::Float(f) => Some(*f),
        _ => None,
    };
    let Some(coverage_min) = cov.filter(|c| *c > 0.0 && *c <= 1.0) else {
        fail!(
            "families: coverage_min must be a number in (0, 1], got {}",
            toml_value(&doc["coverage_min"])
        );
    };
    let Some(api) = doc["api"].as_table() else {
        fail!("families: [api] must be a table");
    };
    check_keys("families [api]", api, &API_REQUIRED, &[])?;
    let Some(raw) = doc["rule"].as_array().filter(|a| !a.is_empty()) else {
        fail!("families: at least one [[rule]] is required");
    };
    let mut seen = BTreeMap::new();
    let mut rules = Vec::with_capacity(raw.len());
    for (i, r) in raw.iter().enumerate() {
        rules.push(parse_rule(
            &format!("families [[rule]] #{}", i + 1),
            r,
            &mut seen,
        )?);
    }
    check_call_sites(&rules)?;
    Ok(FamilyTable {
        rules,
        coverage_min,
        host_sync: compile("families [api]", "host_sync_regex", &api["host_sync_regex"])?,
    })
}

fn check_call_sites(rules: &[Rule]) -> Result<()> {
    let declared: BTreeSet<&str> = rules.iter().map(|r| r.family.as_str()).collect();
    let mut refs: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (i, r) in rules.iter().enumerate() {
        for f in r.next_on_stream.iter().flatten() {
            if !declared.contains(f.as_str()) {
                fail!(
                    "families [[rule]] #{}: next_on_stream_family names undeclared family {}",
                    i + 1,
                    quote(f)
                );
            }
            if *f == r.family {
                fail!(
                    "families [[rule]] #{}: next_on_stream_family may not name its own family {}",
                    i + 1,
                    quote(f)
                );
            }
            refs.entry(r.family.as_str())
                .or_default()
                .insert(f.as_str());
        }
    }
    let mut state: HashMap<&str, u8> = HashMap::new();
    for f in refs.keys() {
        visit(f, &mut Vec::new(), &refs, &mut state)?;
    }
    Ok(())
}

fn visit<'a>(
    f: &'a str,
    path: &mut Vec<&'a str>,
    refs: &BTreeMap<&'a str, BTreeSet<&'a str>>,
    state: &mut HashMap<&'a str, u8>,
) -> Result<()> {
    match state.get(f) {
        Some(2) => return Ok(()),
        Some(1) => {
            path.push(f);
            fail!(
                "families: next_on_stream_family references form a cycle: {}",
                path.join(" -> ")
            );
        }
        _ => {}
    }
    state.insert(f, 1);
    path.push(f);
    for g in refs.get(f).into_iter().flatten() {
        visit(g, path, refs, state)?;
    }
    path.pop();
    state.insert(f, 2);
    Ok(())
}

/// The family of each kernel (aligned with `kernels`), `None` = unclassified.
///
/// Each stream is walked from its last kernel backwards, so the next kernel's
/// family is known when a call-site rule is tried.
pub fn classify<'t>(kernels: &[Kernel], table: &'t FamilyTable) -> Vec<Option<&'t str>> {
    let mut order: Vec<usize> = (0..kernels.len()).collect();
    order.sort_by_key(|&i| (kernels[i].stream, kernels[i].start, kernels[i].end));
    let mut out: Vec<Option<&'t str>> = vec![None; kernels.len()];
    let mut cache: HashMap<(&str, Option<&'t str>), Option<&'t str>> = HashMap::new();
    let mut next_stream: Option<i64> = None;
    let mut next_family: Option<&'t str> = None;
    for &i in order.iter().rev() {
        let k = &kernels[i];
        if next_stream != Some(k.stream) {
            next_stream = Some(k.stream);
            next_family = None;
        }
        let fam = *cache
            .entry((k.name.as_str(), next_family))
            .or_insert_with(|| {
                table
                    .rules
                    .iter()
                    .find(|r| r.matches(&k.name, next_family))
                    .map(|r| r.family.as_str())
            });
        out[i] = fam;
        next_family = fam;
    }
    out
}

/// Classified share of busy time; below the table's coverage_min is an error.
pub fn coverage_gate(classified_ns: f64, total_ns: f64, table: &FamilyTable) -> Result<f64> {
    if total_ns <= 0.0 {
        fail!("coverage: the window has no busy time");
    }
    let share = classified_ns / total_ns;
    if share < table.coverage_min {
        fail!(
            "coverage {share:.4} is below coverage_min {}: {:.3} ms of busy time is unclassified; \
             add the kernels to the family table",
            float(table.coverage_min),
            (total_ns - classified_ns) / 1e6
        );
    }
    Ok(share)
}
