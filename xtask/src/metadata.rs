// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The resolved dependency graph `cargo metadata` prints, as values: which package
//! ids are workspace members, each package's name, and each node's dependency edges with their
//! kinds. `purity_io.rs` runs cargo; this module only reads the JSON it returns.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// 2026-10-17: One resolved dependency edge: the target package id and the kinds it is used as.
pub struct Edge {
    pub to: String,
    pub normal: bool,
    pub dev: bool,
}

/// 2026-10-17: A resolved graph for one platform filter.
pub struct Graph {
    /// What the graph was resolved for (`wasm32v1-none`, the host triple), for messages.
    pub platform: String,
    pub names: BTreeMap<String, String>,
    pub members: BTreeSet<String>,
    pub edges: BTreeMap<String, Vec<Edge>>,
}

impl Graph {
    /// 2026-10-17: The package name of `id`; ids cargo printed always have one.
    pub fn name<'a>(&'a self, id: &'a str) -> &'a str {
        self.names.get(id).map_or(id, String::as_str)
    }

    /// 2026-10-17: The member package id named `name`.
    pub fn member_id(&self, name: &str) -> Option<&str> {
        self.members
            .iter()
            .find(|id| self.name(id) == name)
            .map(String::as_str)
    }
}

fn field<'a>(v: &'a Value, key: &str, at: &str) -> Result<&'a Value, String> {
    v.get(key)
        .ok_or_else(|| format!("cargo metadata: {at} has no `{key}`"))
}

fn str_field<'a>(v: &'a Value, key: &str, at: &str) -> Result<&'a str, String> {
    field(v, key, at)?
        .as_str()
        .ok_or_else(|| format!("cargo metadata: {at}.{key} is not a string"))
}

fn array<'a>(v: &'a Value, key: &str, at: &str) -> Result<&'a Vec<Value>, String> {
    field(v, key, at)?
        .as_array()
        .ok_or_else(|| format!("cargo metadata: {at}.{key} is not an array"))
}

/// 2026-10-17: Parse `cargo metadata --format-version 1` output resolved for `platform`.
pub fn parse(platform: &str, json: &str) -> Result<Graph, String> {
    let root: Value =
        serde_json::from_str(json).map_err(|e| format!("cargo metadata: not JSON: {e}"))?;
    let mut names = BTreeMap::new();
    for p in array(&root, "packages", "metadata")? {
        names.insert(
            str_field(p, "id", "package")?.to_owned(),
            str_field(p, "name", "package")?.to_owned(),
        );
    }
    let members = array(&root, "workspace_members", "metadata")?
        .iter()
        .map(|m| {
            m.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "cargo metadata: a workspace member is not a string".to_owned())
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let resolve = field(&root, "resolve", "metadata")?;
    let mut edges = BTreeMap::new();
    for node in array(resolve, "nodes", "resolve")? {
        let id = str_field(node, "id", "node")?;
        let mut out = Vec::new();
        for dep in array(node, "deps", id)? {
            let mut edge = Edge {
                to: str_field(dep, "pkg", id)?.to_owned(),
                normal: false,
                dev: false,
            };
            for kind in array(dep, "dep_kinds", id)? {
                match field(kind, "kind", id)? {
                    Value::Null => edge.normal = true,
                    Value::String(k) if k == "dev" => edge.dev = true,
                    Value::String(k) if k == "build" => {}
                    other => return Err(format!("cargo metadata: {id}: unknown dep kind {other}")),
                }
            }
            out.push(edge);
        }
        edges.insert(id.to_owned(), out);
    }
    Ok(Graph {
        platform: platform.to_owned(),
        names,
        members,
        edges,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn edges_keep_normal_and_dev_kinds_and_drop_build_only_ones() {
        let json = r#"{
            "packages": [
                {"id": "a#a@1", "name": "a"}, {"id": "b#b@1", "name": "b"},
                {"id": "c#c@1", "name": "c"}
            ],
            "workspace_members": ["a#a@1"],
            "resolve": {"nodes": [
                {"id": "a#a@1", "deps": [
                    {"pkg": "b#b@1", "dep_kinds": [{"kind": null}, {"kind": "dev"}]},
                    {"pkg": "c#c@1", "dep_kinds": [{"kind": "build"}]}
                ]},
                {"id": "b#b@1", "deps": []}, {"id": "c#c@1", "deps": []}
            ]}
        }"#;
        let g = super::parse("host", json).unwrap();
        assert_eq!(g.member_id("a"), Some("a#a@1"));
        assert_eq!(g.member_id("b"), None);
        let a = &g.edges["a#a@1"];
        assert_eq!(
            a.iter()
                .map(|e| (g.name(&e.to), e.normal, e.dev))
                .collect::<Vec<_>>(),
            vec![("b", true, true), ("c", false, false)]
        );
    }

    #[test]
    fn an_unknown_dep_kind_is_an_error() {
        let json = r#"{"packages": [], "workspace_members": [], "resolve": {"nodes": [
            {"id": "a", "deps": [{"pkg": "b", "dep_kinds": [{"kind": "weird"}]}]}]}}"#;
        assert!(super::parse("host", json).is_err());
    }
}
